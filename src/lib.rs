#![feature(allocator_api)]
#![feature(pointer_is_aligned_to)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]
#![feature(slice_ptr_get)]
#![feature(let_chains)]
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
pub mod ll;

//use buf::{Buffer, InlineBuffer};
use error::{assert, Error, ErrorKind};
use ll::{numcpy_est, Limb};

#[macro_export]
macro_rules! testvec {
	($($x:expr),* $(,)?) => {
		vec![$(Limb { value: $x }),*]
	};
}

enum ViewType {
	Small,
	Large,
}

struct IntView<'a> {
	view_type: ViewType,
	neg: bool,
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

struct NonZeroIntView<'a> {
	view_type: ViewType,
	neg: bool,
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

struct BufView<'a> {
	view_type: ViewType,
	cap: NonZeroUsize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
}

impl<'a> BufView<'a> {
	fn as_mut_slice(&self) -> &'a mut [ll::Limb] {
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
	vec: NonNull<u8>,

	/// If the number is small, `magn` contains its value.
	/// If the number is large, `magn` is the number of limbs.
	magn: ll::Limb,
}

impl Int {
	pub fn new_zero() -> Self {
		Self::new_inline(ll::Limb { value: 0 }, false)
	}

	pub fn new_inline(value: ll::Limb, neg: bool) -> Self {
		let vec = Self::__null(neg);
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

	unsafe fn new_with_buf_unchecked(buf: OwnedBuffer, len: usize, neg: bool) -> Self {
		let buf = ManuallyDrop::new(buf);
		Self {
			vec: buf.ptr.cast::<u8>().offset(neg as isize),
			magn: ll::Limb { value: len },
		}
	}

	pub fn new_with_buf(buf: OwnedBuffer, len: usize, neg: bool) -> Result<Self, Error> {
		if len <= buf.cap.get() {
			Ok(unsafe { Self::new_with_buf_unchecked(buf, len, neg) })
		} else {
			cold_path();
			Err(Error::new_buffer_too_small("new_with_buf"))
		}
	}

	pub fn extract_buf(self) -> Option<OwnedBuffer> {
		let this = ManuallyDrop::new(self);
		if this.is_small() {
			None
		} else {
			let ptr = this.vec.cast::<ll::Limb>();

			let cap = unsafe { ptr.offset(-1).read() }.value;
			debug_assert!(cap > 0);
			let cap = unsafe { NonZeroUsize::new_unchecked(cap) };

			Some(OwnedBuffer { ptr, cap })
		}
	}

	pub fn is_negative(&self) -> bool {
		!self.vec.is_aligned_to(2)
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
	pub fn is_small(&self) -> bool {
		self.vec.addr().get() < 4
	}

	pub fn are_both_small(a: &Int, b: &Int) -> bool {
		(a.vec.addr().get() | b.vec.addr().get()) < 4
	}

	pub fn are_both_zero(a: &Int, b: &Int) -> bool {
		(a.magn.value | b.magn.value) == 0
	}

	fn __null(neg: bool) -> NonNull<u8> {
		unsafe {
			// I'm not aware of any contemporary architecture where the null pointer
			// is not at address 0 and adresses 0-3 are valid.
			//
			// However, if there is such an architecture, this implementation would fail.
			// In that case, `__null()` could return a pointer to a static variable
			// and we'd need to update the test in `is_small()`.
			// ```rust
			//     static NULL_LIMB: ll::Limb = ll::Limb { value: 0 };
			// ```
			//
			// For the moment, I stick with this implementation, because it leads to good assembly.
			NonNull::new_unchecked(std::ptr::without_provenance_mut(2 | (neg as usize)))
		}
	}

	fn __large_buf_view<'a>(&'a self) -> Option<BufView<'a>> {
		if self.is_small() {
			return None;
		}
		unsafe {
			let neg = self.is_negative();
			let limbs = self.vec.offset(-(neg as isize)).cast::<ll::Limb>();
			let cap = limbs.offset(-1).read().value;
			debug_assert!(cap > 0);
			let cap = NonZeroUsize::new_unchecked(cap);
			let phantom = std::marker::PhantomData;
			Some(BufView {
				view_type: ViewType::Large,
				cap,
				limbs,
				phantom,
			})
		}
	}

	fn __small_buf_view<'a>(&'a self) -> Option<BufView<'a>> {
		if !self.is_small() {
			return None;
		}
		let limbs = NonNull::from(&self.magn);
		let cap = unsafe { NonZeroUsize::new_unchecked(1) };
		let phantom = std::marker::PhantomData;
		Some(BufView {
			view_type: ViewType::Small,
			cap,
			limbs,
			phantom,
		})
	}

	fn __buf_view<'a>(&'a self) -> BufView<'a> {
		if self.is_small() {
			self.__small_buf_view().unwrap()
		} else {
			self.__large_buf_view().unwrap()
		}
	}

	const MAX_LIMBS: usize = usize::MAX / Limb::BITS;
	pub const MAX_BITS: usize = Self::MAX_LIMBS * Limb::BITS;

	/// The capacity of the inline buffer is 1 limb. I use this constant to be able
	/// to easily find places where the capacity is used.
	const ONE_LIMB: usize = 1;

	#[inline(never)]
	fn __alloc_buf(n: usize) -> Result<OwnedBuffer, Error> {
		let n = n.max(3); // TODO - create named constant for the '3'

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
		unsafe { assume(buf.cap.get() >= n) };
		Ok(buf)
	}

	fn __small_view<'a>(&'a self) -> IntView<'a> {
		let neg = self.is_negative();
		let limbs = NonNull::from(&self.magn);

		// Many operations work with the assumption that the highest limb is non-zero.
		// Therefore when the value stored in magn is zero, we should return 0 limbs.
		let len = (self.magn.value != 0) as usize;

		let phantom = std::marker::PhantomData;
		IntView {
			view_type: ViewType::Small,
			neg,
			len,
			limbs,
			phantom,
		}
	}

	fn __large_view<'a>(&'a self) -> IntView<'a> {
		let neg = self.is_negative();
		let len = self.magn.value;

		// TODO: The `offset()` below generates `sub` instruction.
		// Could we improve it to generate `and`?
		let limbs = unsafe { self.vec.offset(-(neg as isize)).cast::<ll::Limb>() };

		let phantom = std::marker::PhantomData;
		IntView {
			view_type: ViewType::Large,
			neg,
			len,
			limbs,
			phantom,
		}
	}

	fn view<'a>(&'a self) -> IntView<'a> {
		if self.is_small() {
			self.__small_view()
		} else {
			self.__large_view()
		}
	}

	fn nonzero_view<'a>(&'a self) -> Option<NonZeroIntView<'a>> {
		if self.is_zero() {
			return None;
		}
		let view = self.view();
		Some(NonZeroIntView {
			view_type: view.view_type,
			neg: view.neg,
			// SAFETY: We know that the number is not zero.
			len: unsafe { NonZeroUsize::new_unchecked(view.len) },
			limbs: view.limbs,
			phantom: view.phantom,
		})
	}

	pub fn bit_capacity(&self) -> usize {
		let view = self.__buf_view();
		let cap = match view.view_type {
			ViewType::Small => Self::ONE_LIMB,
			ViewType::Large => view.cap.get(),
		};
		cap * Limb::BITS
	}

	pub fn bit_width(&self) -> usize {
		self.nonzero_view().map_or(0, |view| ll::bit_width(&view))
	}

	pub fn clone(&self) -> Result<Self, Error> {
		let view = self.view();
		match view.view_type {
			ViewType::Small => Ok(Self { vec: self.vec, magn: self.magn }),
			ViewType::Large => {
				let len = ll::numcpy_est(&view);
				let mut r = Self::alloc_buf(len)?;
				let len = ll::numcpy(&mut r, &view)?;
				Self::new_with_buf(r, len, view.neg)
			},
		}
	}

	/*	#[inline(never)]
	#[must_use]
	pub fn try_assign(&mut self, a: &Int) -> Result<(), Error> {
		let A = a.view();
		let buf_size = numcpy_est(A.as_slice());
		let mut inline_buf = buf::InlineBuffer::default();
		let mut buf = buf::Buffer::new(self, buf_size, &mut inline_buf)?;

		let r_len = ll::numcpy(buf.as_slice(), A.as_slice())?;
		let r_neg = A.neg;

		buf.commit(r_len, r_neg)
	}*/

	/*	#[inline(never)]
	pub fn assign(&mut self, a: &Int) {
		self.try_assign(a).unwrap()
	}*/

	#[inline(always)]
	pub fn try_add_small(a: &Int, b: &Int) -> Option<Int> {
		if !Self::are_both_small(a, b) {
			return None;
		}
		let (val, carry) = ll::add_small(&[a.magn], &[b.magn]);
		if carry {
			return None;
		}
		// `vec: a.vec` copies the sign of a
		Some(Int { vec: a.vec, magn: val[0] })
	}

	#[inline(always)]
	pub fn try_sub_small(a: &Int, b: &Int) -> Option<Int> {
		if !Self::are_both_small(a, b) {
			return None;
		}
		let (val, neg) = ll::sub_small(&[a.magn], &[b.magn]);
		Some(Self::new_inline(val[0], a.is_negative() ^ neg))
	}

	#[inline(never)]
	fn __add(a: &Int, b: &Int) -> Result<Int, Error> {
		let (a, b) = (a.view(), b.view());
		let buf_size = ll::add_est(&a, &b);
		let mut r = Self::alloc_buf(buf_size)?;
		let len = ll::add(&mut r, &a, &b)?;
		Self::new_with_buf(r, len, a.neg)
	}

	#[inline(never)]
	fn __sub(a: &Int, b: &Int) -> Result<Int, Error> {
		let (a, b) = (a.view(), b.view());
		let buf_size = ll::sub_est(&a, &b);
		let mut r = Self::alloc_buf(buf_size)?;
		let (neg, len) = ll::sub(&mut r, &a, &b)?;
		Self::new_with_buf(r, len, a.neg ^ neg)
	}

	#[inline(never)]
	pub fn try_add_or_sub(a: &Int, sub: bool, b: &Int) -> Result<Int, Error> {
		let sub = a.is_negative() ^ (sub ^ b.is_negative());
		if sub {
			if let Some(small) = Self::try_sub_small(a, b) {
				Ok(small)
			} else {
				Self::__sub(a, b)
			}
		} else {
			if let Some(small) = Self::try_add_small(a, b) {
				Ok(small)
			} else {
				Self::__add(a, b)
			}
		}
	}

	/*	#[inline(never)]
	#[must_use]
	pub fn try_add_or_sub_(&mut self, a: &Int, sub: bool, b: &Int) -> Result<(), Error> {
		let A = a.view();
		let B = b.view();
		let sub = A.neg ^ (sub ^ B.neg);
		let buf_size = if sub {
			ll::sub_est(A.as_slice(), B.as_slice())
		} else {
			ll::add_est(A.as_slice(), B.as_slice())
		};
		let mut inline_buf = buf::InlineBuffer::default();
		let mut buf = buf::Buffer::new(self, buf_size, &mut inline_buf)?;

		let mut r_neg = A.neg;
		let r_len;
		if sub {
			let (neg, len) = ll::sub(buf.as_slice(), A.as_slice(), B.as_slice())?;
			r_neg ^= neg;
			r_len = len;
		} else {
			r_len = ll::add(buf.as_slice(), A.as_slice(), B.as_slice())?;
		};

		buf.commit(r_len, r_neg)
	}*/
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

/*#[inline(never)]
pub fn test_commit(b: &mut Buffer, len: usize, neg: bool) -> Result<(), Error> {
	b.commit(len, neg)
} */

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
