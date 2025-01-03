#![feature(allocator_api)]
#![feature(pointer_is_aligned_to)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]
#![feature(slice_ptr_get)]
#![feature(let_chains)]
#![feature(ptr_sub_ptr)]
#![feature(ptr_as_ref_unchecked)]
#![feature(unchecked_shifts)]
//
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(internal_features)]
#![allow(incomplete_features)]
//

use core::panic;
use smallvec::SmallVec;
use std::alloc::{Allocator, Global, Layout};
use std::char::MAX;
use std::intrinsics::{assume, cold_path, likely, unlikely};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::result;

pub mod base_conv;
pub mod base_conv_gen;
pub mod blocks;
pub mod buf;
pub mod error;
pub mod fixed_size;
pub mod limb_buf;
pub mod ll;
pub mod tagged_ptr;

//use buf::{Buffer, InlineBuffer};
use base_conv::BaseConv;
use error::{assert, Error, ErrorKind};
use limb_buf::LimbBuf;
use ll::{numcpy_est, Limb};
use tagged_ptr::TaggedPtr;

#[macro_export]
macro_rules! testvec {
	($($x:expr),* $(,)?) => {
		vec![$(Limb { val: $x }),*]
	};
}

pub enum ViewKind {
	Small,
	Large,
}

pub struct IntView<'a> {
	pub kind: ViewKind,
	pub neg: bool,
	len: usize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
}

impl<'a> Deref for IntView<'a> {
	type Target = [ll::Limb];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.limbs.as_ptr(), self.len) }
	}
}

pub struct NonZeroIntView<'a> {
	pub kind: ViewKind,
	pub neg: bool,
	len: NonZeroUsize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
}

impl<'a> Deref for NonZeroIntView<'a> {
	type Target = [ll::Limb];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.limbs.as_ptr(), self.len.get()) }
	}
}

pub struct BufView<'a> {
	pub kind: ViewKind,
	cap: NonZeroUsize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
}

impl<'a> Deref for BufView<'a> {
	type Target = [ll::Limb];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.limbs.as_ptr(), self.cap.get()) }
	}
}

impl<'a> DerefMut for BufView<'a> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		unsafe { std::slice::from_raw_parts_mut(self.limbs.as_ptr(), self.cap.get()) }
	}
}

pub struct Int {
	/// `vec` is a pointer to an array with limbs.
	/// vec[-1] is used to store the capacity of the array.
	///
	/// Additionally, the pointer is tagged with 1 bit, which is used to store the sign. We use the
	/// least significant bit for the tag, so for negative numbers, the pointer is not aligned.
	///
	/// When the number is small, `vec` points to a sentinel value returned by `__null()`,
	/// which should be treated as a real `null`. Specifically, we can check the tag and check
	/// equality, but we should not do pointer arithmetic or dereference it.
	///
	/// We use a sentinel instead of `null` so `vec` can be a NonNull pointer and
	/// the compiler can optimize `Option<Int>`.
	vec: TaggedPtr<Limb>,

	/// If the number is small, `magn` contains its value.
	/// If the number is large, `magn` is the number of limbs.
	magn: ll::Limb,
}

impl Int {
	pub fn new_zero() -> Self {
		Self::new_inline(ll::Limb::ZERO, false)
	}

	pub fn new_inline(value: ll::Limb, neg: bool) -> Self {
		let vec = TaggedPtr::new_null(neg);
		let magn = value;
		Self { vec, magn }
	}

	pub fn new_with_cap(cap: usize) -> Result<Self, Error> {
		let buf = LimbBuf::new(cap).map_err(|e| {
			cold_path();
			e
		})?;
		Self::new_with_buf(buf, 0, false)
	}

	pub fn new_with_buf(buf: LimbBuf, len: usize, neg: bool) -> Result<Self, Error> {
		if len <= buf.cap() {
			let buf = ManuallyDrop::new(buf);
			Ok(Self {
				vec: TaggedPtr::new(buf.as_non_null_ptr(), neg),
				magn: ll::Limb { val: len },
			})
		} else {
			cold_path();
			Err(Error::new_buffer_too_small("new_with_buf"))
		}
	}

	fn new_with_small_buf(
		buf: &[Limb; LimbBuf::MIN_ALLOC_SIZE], len: usize, neg: bool,
	) -> Result<Self, Error> {
		if len <= Self::ONE_LIMB {
			Ok(Self::new_inline(buf[0], neg))
		} else {
			let mut heap_buf = LimbBuf::new(LimbBuf::MIN_ALLOC_SIZE).map_err(|e| {
				cold_path();
				e
			})?;
			unsafe {
				std::ptr::copy_nonoverlapping(
					buf.as_ptr(),
					heap_buf.as_mut_slice().as_mut_ptr(),
					LimbBuf::MIN_ALLOC_SIZE,
				);
			}
			Self::new_with_buf(heap_buf, len, neg)
		}
	}

	#[inline(never)]
	fn __from_digits(neg: bool, digits: &[u8], base: &BaseConv) -> Result<Self, Error> {
		/*let limbs = base.digits_to_int_est(digits.len());
		if limbs <= Int::ONE_LIMB + 1 {
			const _: () = assert!(Int::ONE_LIMB + 1 <= MIN_ALLOC_SIZE);
			let mut buf = [Limb::default(); MIN_ALLOC_SIZE];
			let len = base.digits_to_int(&mut buf, digits).map_err(|e| {
				cold_path();
				e
			})?;
			Self::new_with_small_buf(&buf, len, neg)
		} else {
			let mut buf = Self::alloc_buf(limbs).map_err(|e| {
				cold_path();
				e
			})?;
			let len = base.digits_to_int(&mut buf, digits).map_err(|e| {
				cold_path();
				e
			})?;
			Self::new_with_buf(buf, len, neg)
		}*/
		Err(Error::new_alloc_failed("Int::__from_digits")) // TODO
	}

	/*	pub fn __from_str(str: &str, base: &BaseConv) -> Result<Self, Error> {
		const SMALL_BUF_SIZE: usize = 64;
		if str.len() <= SMALL_BUF_SIZE {
			let mut digits = [0; SMALL_BUF_SIZE];
			let (neg, ndigits) = base.str_to_digits(str, &mut digits)?;
			Self::__from_digits(neg, &digits[..ndigits], base)
		} else {
			let mut vec = Vec::<u8>::with_capacity(str.len());
			let digits = unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr(), str.len()) };
			let (neg, ndigits) = base.str_to_digits(str, digits)?;
			Self::__from_digits(neg, &digits[..ndigits], base)
		}
	}*/

	#[inline(never)]
	pub fn from_str_slow(
		neg: bool, first_limb: Limb, bytes: &[u8], pos: usize,
	) -> Result<Self, Error> {
		// first convert the string to a list of digits
		let mut digits_buf = SmallVec::<[u8; 128]>::new();
		let pos3 =
			base_conv::parse_digits::<10>(bytes, &base_conv_gen::SHORT_MAPPING, &mut digits_buf);
		if pos3 != bytes.len() {
			cold_path();
			return Err(Error::new_parse_error("Int::from_str_slow"));
		}

		let mut limb_buf = [Limb::default(); 8];
		let (limb_buf, len) =
			base_conv::digits_to_limbs::<10>(digits_buf.as_slice(), first_limb, &mut limb_buf)?;

		Ok(Self::new_zero()) // TODO
	}

	#[inline(never)]
	pub fn from_str(str: &str) -> Result<Self, Error> {
		let bytes = str.as_bytes();
		if bytes.is_empty() {
			cold_path();
			return Ok(Self::new_zero());
		}

		let (neg, pos1) = //.
			match bytes[0] {
				b'-' => (true, 1),
				b'+' => (false, 1),
				_ => (false, 0),
			};
		let bytes = &bytes[pos1..];

		let (short, pos2) = base_conv::parse_short::<10>(bytes, &base_conv_gen::SHORT_MAPPING);
		if pos2 == bytes.len() {
			Ok(Self::new_inline(short, neg))
		} else {
			let bytes = &bytes[pos2..];
			Self::from_str_slow(neg, short, bytes, pos1 + pos2)
		}
	}

	/*	pub fn from_str_with_base(str: &str, base: usize) -> Result<Self, Error> {
		let base = BaseConv::get(base).ok_or_else(|| {
			cold_path();
			Error::new_invalid_base("Int::from_str")
		})?;
		Self::__from_str(str, base)
	}*/

	pub fn extract_buf(self) -> Option<LimbBuf> {
		let this = ManuallyDrop::new(self);
		Some(unsafe { LimbBuf::from_non_null_ptr(this.vec.ptr()?) })
	}

	pub fn is_negative(&self) -> bool {
		self.vec.tag()
	}

	pub fn is_zero(&self) -> bool {
		// Note that this works for both small and large numbers.
		self.magn.is_zero()
	}

	pub fn is_positive(&self) -> bool {
		let zero = self.is_zero();
		let neg = self.is_negative();
		!zero && !neg
	}

	/// A number is considered "small" if it stores all the data locally and doesn't use the heap.
	fn is_small(&self) -> bool {
		self.vec.ptr().is_none()
	}

	fn are_both_small(a: &Int, b: &Int) -> bool {
		TaggedPtr::are_both_null(a.vec, b.vec)
	}

	fn __buf_view<'a>(&'a self) -> BufView<'a> {
		if let Some(large) = self.vec.ptr() {
			let cap = unsafe { large.offset(-1).read().val };
			debug_assert!(cap > 0);
			let cap = unsafe { NonZeroUsize::new_unchecked(cap) };
			BufView {
				kind: ViewKind::Large,
				limbs: large,
				cap,
				phantom: std::marker::PhantomData,
			}
		} else {
			BufView {
				kind: ViewKind::Small,
				limbs: NonNull::from(&self.magn),
				cap: NonZeroUsize::new(1).unwrap(),
				phantom: std::marker::PhantomData,
			}
		}
	}

	/// The capacity of the inline buffer is 1 limb. I use this constant to be able
	/// to easily find places where the capacity is used.
	const ONE_LIMB: usize = 1;

	fn view<'a>(&'a self) -> IntView<'a> {
		if let Some(large) = self.vec.ptr() {
			IntView {
				kind: ViewKind::Large,
				neg: self.is_negative(),
				limbs: large,
				len: self.magn.val,
				phantom: std::marker::PhantomData,
			}
		} else {
			IntView {
				kind: ViewKind::Small,
				neg: self.is_negative(),
				limbs: NonNull::from(&self.magn),
				len: self.magn.is_not_zero() as usize,
				phantom: std::marker::PhantomData,
			}
		}
	}

	fn nonzero_view<'a>(&'a self) -> Option<NonZeroIntView<'a>> {
		if self.is_zero() {
			return None;
		}

		let view = self.view();
		Some(NonZeroIntView {
			kind: view.kind,
			neg: view.neg,
			// SAFETY: We checked that the number is not zero.
			len: unsafe { NonZeroUsize::new(view.len).unwrap_unchecked() },
			limbs: view.limbs,
			phantom: view.phantom,
		})
	}

	pub fn bit_capacity(&self) -> usize {
		let view = self.__buf_view();
		let cap = match view.kind {
			ViewKind::Small => Self::ONE_LIMB,
			ViewKind::Large => view.cap.get(),
		};
		cap * Limb::BITS
	}

	pub fn bit_width(&self) -> usize {
		self.nonzero_view().map_or(0, |view| ll::bit_width(&view))
	}

	pub fn clone(&self) -> Result<Self, Error> {
		let view = self.view();
		match view.kind {
			ViewKind::Small => Ok(Self { vec: self.vec, magn: self.magn }),
			ViewKind::Large => {
				let mut r = LimbBuf::new(ll::numcpy_est(&view))?;
				let len = ll::numcpy(&mut r, &view);
				unsafe { assume(len <= r.cap()) };
				Self::new_with_buf(r, len, view.neg)
			},
		}
	}

	#[inline(always)]
	fn try_add_small(a: &Int, b: &Int) -> Option<Int> {
		if !Self::are_both_small(a, b) {
			return None;
		}
		let (val, carry) = fixed_size::add(&[a.magn], &[b.magn]);
		if carry {
			return None;
		}
		// `vec: a.vec` copies the sign of a
		Some(Int { vec: a.vec, magn: val[0] })
	}

	#[inline(always)]
	fn try_sub_small(a: &Int, b: &Int) -> Option<Int> {
		if !Self::are_both_small(a, b) {
			return None;
		}
		let (val, neg) = fixed_size::sub(&[a.magn], &[b.magn]);
		Some(Self::new_inline(val[0], a.is_negative() ^ neg))
	}

	#[inline(never)]
	fn try_add_or_sub_slow(a: &Int, sub: bool, b: &Int) -> Result<Int, Error> {
		// slow path
		let a = a.view();
		let b = b.view();
		if sub {
			let mut r = LimbBuf::new(ll::sub_est(&a, &b))?;
			let (neg, len) = ll::sub(&mut r, &a, &b)?;
			if len > Self::ONE_LIMB {
				unsafe { assume(len <= r.cap()) };
				Self::new_with_buf(r, len, a.neg ^ neg)
			} else {
				let val = if len == 0 { 0 } else { r[0].val };
				Ok(Self::new_inline(Limb { val }, a.neg ^ neg))
			}
		} else {
			let mut r = LimbBuf::new(ll::add_est(&a, &b))?;
			let len = ll::add(&mut r, &a, &b)?;
			unsafe { assume(len <= r.cap()) };
			Self::new_with_buf(r, len, a.neg)
		}
	}

	#[inline(never)]
	pub fn try_add_or_sub(a: &Int, sub: bool, b: &Int) -> Result<Int, Error> {
		let sub = a.is_negative() ^ (sub ^ b.is_negative());
		if sub {
			if let Some(small) = Self::try_sub_small(a, b) {
				return Ok(small);
			}
		} else {
			if let Some(small) = Self::try_add_small(a, b) {
				return Ok(small);
			}
		}

		Self::try_add_or_sub_slow(a, sub, b)
	}

	#[inline]
	pub fn try_add(a: &Int, b: &Int) -> Result<Int, Error> {
		Self::try_add_or_sub(a, false, b)
	}

	#[inline]
	pub fn try_sub(a: &Int, b: &Int) -> Result<Int, Error> {
		Self::try_add_or_sub(a, true, b)
	}

	pub fn add_or_sub(a: &Int, sub: bool, b: &Int) -> Int {
		Self::try_add_or_sub(a, sub, b).unwrap()
	}

	pub fn add(a: &Int, b: &Int) -> Int {
		Self::try_add_or_sub(a, false, b).unwrap()
	}

	pub fn sub(a: &Int, b: &Int) -> Int {
		Self::try_add_or_sub(a, true, b).unwrap()
	}

	pub fn try_mul(a: &Int, b: &Int) -> Result<Int, Error> {
		if a.is_zero() || b.is_zero() {
			return Ok(Self::new_zero());
		}

		let a = a.view();
		let b = b.view();
		let mut r = LimbBuf::new(ll::mul_est(&a, &b))?;
		let len = ll::mul(&mut r, &a, &b, &std::alloc::Global::default())?;
		unsafe { assume(len <= r.cap()) };
		Self::new_with_buf(r, len, a.neg ^ b.neg)
	}
}

impl Drop for Int {
	fn drop(&mut self) {
		let t = Self { vec: self.vec, magn: self.magn };
		let _buf = t.extract_buf();
		// _buf will be dropped here
	}
}

impl Default for Int {
	fn default() -> Self {
		Self::new_zero()
	}
}

#[inline(never)]
pub fn test_new_zero() -> Int {
	Int::new_zero()
}

#[inline(never)]
pub fn test_new_zero2() -> Option<Int> {
	Some(Int::new_zero())
}

#[inline(never)]
pub fn test_new_zero3() -> Option<Int> {
	None
}

#[inline(never)]
#[allow(private_interfaces)]
pub fn test_view(i: &Int) -> IntView {
	i.view()
}

#[inline(never)]
pub fn t1() -> usize {
	std::println!("t1");
	1
}

#[inline(never)]
pub fn t2() -> usize {
	std::println!("t2");
	2
}

#[inline(never)]
pub fn t3() -> usize {
	std::println!("t3");
	3
}

#[inline(never)]
pub fn test_buf_view(i: &Int) -> BufView {
	i.__buf_view()
}

#[inline(never)]
pub fn test_nonzero_view(i: &Int) -> Option<NonZeroIntView> {
	i.nonzero_view()
}

#[inline(never)]
pub fn ttt(A: bool, B: bool) -> usize {
	if A {
		t1()
	} else {
		cold_path();
		if B {
			t2()
		} else {
			cold_path();
			t3()
		}
	}
}
