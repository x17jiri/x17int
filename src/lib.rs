#![feature(allocator_api)]
#![feature(pointer_is_aligned_to)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]
#![feature(slice_ptr_get)]
#![feature(let_chains)]
#![feature(ptr_sub_ptr)]
//
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(internal_features)]
#![allow(incomplete_features)]
//
#![feature(ptr_as_ref_unchecked)]

use core::panic;
use std::alloc::{Allocator, Global, Layout};
use std::char::MAX;
use std::intrinsics::{assume, cold_path, likely, unlikely};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::result;

pub mod blocks;
pub mod buf;
pub mod error;
pub mod fixed_size;
pub mod ll;
pub mod tagged_ptr;

//use buf::{Buffer, InlineBuffer};
use error::{assert, Error, ErrorKind};
use ll::{numcpy_est, Limb};
use tagged_ptr::TaggedPtr;

#[macro_export]
macro_rules! testvec {
	($($x:expr),* $(,)?) => {
		vec![$(Limb { value: $x }),*]
	};
}

const MIN_ALLOC_SIZE: usize = 3;

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

pub struct OwnedBuffer {
	ptr: NonNull<ll::Limb>,
	cap: NonZeroUsize,
}

impl OwnedBuffer {
	fn as_mut_slice(&mut self) -> &mut [ll::Limb] {
		unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap.get()) }
	}
}

impl Deref for OwnedBuffer {
	type Target = [ll::Limb];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap.get()) }
	}
}

impl DerefMut for OwnedBuffer {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.as_mut_slice()
	}
}

impl Drop for OwnedBuffer {
	fn drop(&mut self) {
		unsafe {
			let ptr = self.ptr.offset(-1);
			let cap = self.cap.get();
			debug_assert!(cap == ptr.read().value);

			let size = (cap + 1) * std::mem::size_of::<ll::Limb>();
			let align = std::mem::align_of::<ll::Limb>();
			debug_assert!(size > 0);

			assume(size > 0);
			Global.deallocate(ptr.cast::<u8>(), Layout::from_size_align_unchecked(size, align));
		}
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
		Self::new_inline(ll::Limb { value: 0 }, false)
	}

	pub fn new_inline(value: ll::Limb, neg: bool) -> Self {
		let vec = TaggedPtr::new_null(neg);
		let magn = value;
		Self { vec, magn }
	}

	pub fn new_with_cap(cap: usize) -> Result<Self, Error> {
		let buf = Self::alloc_buf(cap).map_err(|e| {
			cold_path();
			e
		})?;
		Self::new_with_buf(buf, 0, false)
	}

	pub fn new_with_buf(buf: OwnedBuffer, len: usize, neg: bool) -> Result<Self, Error> {
		if len <= buf.cap.get() {
			let buf = ManuallyDrop::new(buf);
			Ok(Self {
				vec: TaggedPtr::new(buf.ptr, neg),
				magn: ll::Limb { value: len },
			})
		} else {
			cold_path();
			Err(Error::new_buffer_too_small("new_with_buf"))
		}
	}

	pub fn extract_buf(self) -> Option<OwnedBuffer> {
		let this = ManuallyDrop::new(self);
		let buf = this.__buf_view();
		match buf.kind {
			ViewKind::Small => None,
			ViewKind::Large => Some(OwnedBuffer { ptr: buf.limbs, cap: buf.cap }),
		}
	}

	pub fn is_negative(&self) -> bool {
		self.vec.tag()
	}

	pub fn is_zero(&self) -> bool {
		// Note that this works for both small and large numbers.
		self.magn.value == 0
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
			let cap = unsafe { large.offset(-1).read().value };
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

	const MAX_LIMBS: usize = usize::MAX / Limb::BITS;
	pub const MAX_BITS: usize = Self::MAX_LIMBS * Limb::BITS;

	/// The capacity of the inline buffer is 1 limb. I use this constant to be able
	/// to easily find places where the capacity is used.
	const ONE_LIMB: usize = 1;

	#[inline(never)]
	fn __alloc_buf(n: usize) -> Result<OwnedBuffer, Error> {
		let n = MIN_ALLOC_SIZE.max(n);

		assert(n <= Self::MAX_LIMBS, || {
			Error::new_alloc_failed("Number of limbs exceeds the maximum.")
		})?;

		let layout = std::alloc::Layout::array::<ll::Limb>(n).map_err(|_| {
			cold_path();
			Error::new_alloc_failed("Cannot create Layout instance.")
		})?;

		let new_buf = Global.allocate(layout).map_err(|_| {
			cold_path();
			Error::new_alloc_failed("Cannot allocate memory.")
		})?;

		// Verify the buffer is at least as big as we requested so we can
		// assume it in `alloc_buf()`. This branch should really be dead code.
		if new_buf.len() < n {
			panic!("Allocator returned a buffer that is too small.");
		}

		let ptr = new_buf.as_non_null_ptr();
		let ptr = ptr.cast::<ll::Limb>();

		let cap = (new_buf.len() / std::mem::size_of::<ll::Limb>()) - 1;
		let cap = if likely(cap <= Self::MAX_LIMBS) { cap } else { Self::MAX_LIMBS };
		let cap = unsafe { NonZeroUsize::new_unchecked(cap) };

		unsafe { ptr.write(ll::Limb { value: cap.get() }) };
		let ptr = unsafe { ptr.offset(1) };

		Ok(OwnedBuffer { ptr, cap })
	}

	pub fn alloc_buf(n: usize) -> Result<OwnedBuffer, Error> {
		let buf = Self::__alloc_buf(n)?;
		// SAFETY: We verify this in `__alloc_buf()`.
		unsafe { assume(buf.cap.get() >= n) };
		Ok(buf)
	}

	fn view<'a>(&'a self) -> IntView<'a> {
		if let Some(large) = self.vec.ptr() {
			IntView {
				kind: ViewKind::Large,
				neg: self.is_negative(),
				limbs: large,
				len: self.magn.value,
				phantom: std::marker::PhantomData,
			}
		} else {
			IntView {
				kind: ViewKind::Small,
				neg: self.is_negative(),
				limbs: NonNull::from(&self.magn),
				len: (self.magn.value != 0) as usize,
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
				let mut r = Self::alloc_buf(ll::numcpy_est(&view))?;
				let len = ll::numcpy(&mut r, &view);
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

		// slow path
		let a = a.view();
		let b = b.view();
		if sub {
			let mut r = Self::alloc_buf(ll::sub_est(&a, &b))?;
			let (neg, len) = ll::sub(&mut r, &a, &b);
			if len > Self::ONE_LIMB {
				Self::new_with_buf(r, len, a.neg ^ neg)
			} else {
				let value = if len == 0 { 0 } else { r[0].value };
				Ok(Self::new_inline(Limb { value }, a.neg ^ neg))
			}
		} else {
			let mut r = Self::alloc_buf(ll::add_est(&a, &b))?;
			let len = ll::add(&mut r, &a, &b);
			Self::new_with_buf(r, len, a.neg)
		}
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
