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

enum ViewKind {
	Small,
	Large,
}

struct IntView<'a> {
	kind: ViewKind,
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
	kind: ViewKind,
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
	kind: ViewKind,
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

#[derive(Clone, Copy)]
struct TaggedPtr<T> {
	__ptr: NonNull<u8>,
	phantom: std::marker::PhantomData<T>,
}

impl<T> TaggedPtr<T> {
	#[inline]
	fn new_null(tag: bool) -> Self {
		unsafe {
			// I'm not aware of any contemporary architecture where the null pointer
			// is not at address 0 and adresses 0-3 are valid.
			//
			// However, if there is such an architecture, this implementation would fail.
			// In that case, we could return a pointer to a static variable and we'd need to
			// update the test in `is_null()`.
			// ```rust
			//     static NULL_LIMB: ll::Limb = ll::Limb { value: 0 };
			// ```
			//
			// For the moment, I stick with this implementation, because it leads to good assembly.
			TaggedPtr::<T> {
				__ptr: NonNull::new_unchecked(std::ptr::without_provenance_mut(2 | (tag as usize))),
				phantom: std::marker::PhantomData,
			}
		}
	}

	#[inline]
	fn new(ptr: NonNull<T>, tag: bool) -> Self {
		Self {
			__ptr: unsafe { ptr.cast::<u8>().offset(tag as isize).cast() },
			phantom: std::marker::PhantomData,
		}
	}

	#[inline]
	fn is_null(&self) -> bool {
		self.__ptr.addr().get() < 4
	}

	#[inline]
	fn are_both_null(a: Self, b: Self) -> bool {
		(a.__ptr.addr().get() | b.__ptr.addr().get()) < 4
	}

	#[inline]
	fn tag(&self) -> bool {
		!self.__ptr.is_aligned_to(2)
	}

	#[inline]
	fn ptr(&self) -> NonNull<T> {
		let tag = self.tag();
		unsafe { self.__ptr.offset(-(tag as isize)).cast::<T>() }
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
	pub fn is_small(&self) -> bool {
		self.vec.is_null()
	}

	pub fn are_both_small(a: &Int, b: &Int) -> bool {
		TaggedPtr::are_both_null(a.vec, b.vec)
	}

	pub fn are_both_zero(a: &Int, b: &Int) -> bool {
		(a.magn.value | b.magn.value) == 0
	}

	unsafe fn __small_buf_view<'a>(&'a self) -> BufView<'a> {
		BufView {
			kind: ViewKind::Small,
			limbs: NonNull::from(&self.magn),
			cap: NonZeroUsize::new_unchecked(1),
			phantom: std::marker::PhantomData,
		}
	}

	unsafe fn __large_buf_view<'a>(&'a self) -> BufView<'a> {
		let limbs = self.vec.ptr();
		let cap = limbs.offset(-1).read().value;
		debug_assert!(cap > 0);
		BufView {
			kind: ViewKind::Large,
			limbs,
			cap: NonZeroUsize::new_unchecked(cap),
			phantom: std::marker::PhantomData,
		}
	}

	fn __buf_view<'a>(&'a self) -> BufView<'a> {
		if self.is_small() {
			unsafe { self.__small_buf_view() }
		} else {
			unsafe { self.__large_buf_view() }
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

	unsafe fn __small_view<'a>(&'a self) -> IntView<'a> {
		// Many operations work with the assumption that the highest limb is non-zero.
		// Therefore when the value stored in magn is zero, we should return len == 0.
		IntView {
			kind: ViewKind::Small,
			neg: self.is_negative(),
			limbs: NonNull::from(&self.magn),
			len: (self.magn.value != 0) as usize,
			phantom: std::marker::PhantomData,
		}
	}

	unsafe fn __large_view<'a>(&'a self) -> IntView<'a> {
		IntView {
			kind: ViewKind::Large,
			neg: self.is_negative(),
			limbs: self.vec.ptr(),
			len: self.magn.value,
			phantom: std::marker::PhantomData,
		}
	}

	fn view<'a>(&'a self) -> IntView<'a> {
		if self.is_small() {
			unsafe { self.__small_view() }
		} else {
			unsafe { self.__large_view() }
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
			// SAFETY: We know that the number is not zero.
			len: unsafe { NonZeroUsize::new_unchecked(view.len) },
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
