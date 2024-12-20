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
use std::num::NonZeroUsize;
use std::ptr::NonNull;

pub mod blocks;
pub mod buf;
pub mod error;
pub mod ll;

use buf::{Buffer, InlineBuffer};
use error::{Error, ErrorKind, assert};
use ll::Limb;

#[macro_export]
macro_rules! testvec {
	($($x:expr),* $(,)?) => {
		vec![$(Limb { value: $x }),*]
	};
}

struct IntView<'a> {
	neg: bool,
	len: usize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
}

impl<'a> IntView<'a> {
	fn as_slice(&self) -> &'a [ll::Limb] {
		unsafe { std::slice::from_raw_parts(self.limbs.as_ptr(), self.len) }
	}
}

struct NonZeroIntView<'a> {
	neg: bool,
	len: NonZeroUsize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
}

impl<'a> NonZeroIntView<'a> {
	fn as_slice(&self) -> &'a [ll::Limb] {
		unsafe { std::slice::from_raw_parts(self.limbs.as_ptr(), self.len.get()) }
	}
}

struct BufView<'a> {
	cap: NonZeroUsize,
	limbs: NonNull<ll::Limb>,
	phantom: std::marker::PhantomData<&'a ll::Limb>,
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
		Self {
			vec: Self::__null(false),
			magn: ll::Limb { value: 0 },
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
			Some(BufView { cap, limbs, phantom })
		}
	}

	fn __small_buf_view<'a>(&'a self) -> Option<BufView<'a>> {
		if !self.is_small() {
			return None;
		}
		let limbs = NonNull::from(&self.magn);
		let cap = unsafe { NonZeroUsize::new_unchecked(1) };
		let phantom = std::marker::PhantomData;
		Some(BufView { cap, limbs, phantom })
	}

	fn __buf_view<'a>(&'a self) -> BufView<'a> {
		self.__small_buf_view().unwrap_or_else(|| {
			self.__large_buf_view().unwrap_or_else(|| {
				// SAFETY: The number has to be either small or large.
				unsafe { std::hint::unreachable_unchecked() }
			})
		})
	}

	fn __set_inline(&mut self, value: ll::Limb, neg: bool) {
		self.vec = Self::__null(neg);
		self.magn = value;
	}

	/// The allocated buffer has to be at least as big as the inline buffer, i.e., one limb.
	unsafe fn __set_allocated(&mut self, buffer: NonNull<[ll::Limb]>, len: usize, neg: bool) {
		debug_assert!(buffer.len() >= Self::ONE_LIMB);
		debug_assert!(len <= buffer.len());

		// SAFETY: We still stay within the allocated buffer.
		// The `offset` just makes the pointer unaligned to represent the sign.
		self.vec = buffer.cast::<u8>().offset(neg as isize);
		self.magn = ll::Limb { value: len };
	}

	unsafe fn __set_len_neg(&mut self, len: usize, neg: bool) {
		debug_assert!(!self.is_small());
		if let Some(buf_view) = self.__large_buf_view() {
			debug_assert!(len <= buf_view.cap.get());
		}

		let prev_neg = self.is_negative();
		self.vec = self.vec.offset(-(prev_neg as isize) + (neg as isize));
		self.magn = ll::Limb { value: len };
	}

	const MAX_LIMBS: usize = usize::MAX / Limb::BITS;
	pub const MAX_BITS: usize = Self::MAX_LIMBS * Limb::BITS;

	/// The capacity of the inline buffer is 1 limb. I use this constant to be able
	/// to easily find places where the capacity is used.
	const ONE_LIMB: usize = 1;

	#[must_use]
	#[inline(never)]
	fn __alloc(n: usize) -> Result<NonNull<[ll::Limb]>, Error> {
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
		let ptr = ptr.cast::<ll::Limb>().as_ptr();
		let cap = (new_buf.len() / std::mem::size_of::<ll::Limb>()) - 1;
		let cap = if likely(cap <= Self::MAX_LIMBS) { cap } else { Self::MAX_LIMBS };
		unsafe {
			ptr.write(ll::Limb { value: cap });
			let ptr = ptr.offset(1);
			Ok(NonNull::new_unchecked(std::slice::from_raw_parts_mut(ptr, cap)))
		}
	}

	unsafe fn __free(buf: NonNull<ll::Limb>) {
		let ptr = buf.offset(-1);
		let cap = ptr.read().value;
		let size = (cap + 1) * std::mem::size_of::<ll::Limb>();
		let align = std::mem::align_of::<ll::Limb>();
		debug_assert!(size > 0);
		assume(size > 0);
		Global.deallocate(ptr.cast::<u8>(), Layout::from_size_align_unchecked(size, align));
	}

	fn __large_view<'a>(&'a self) -> Option<IntView<'a>> {
		if self.is_small() {
			return None;
		}
		// SAFETY: We know that the number is not small.
		unsafe {
			let neg = self.is_negative();
			let len = self.magn.value;

			// TODO: The `offset()` below generates `sub` instruction.
			// Could we improve it to generate `and`?
			let limbs = self.vec.offset(-(neg as isize)).cast::<ll::Limb>();

			let phantom = std::marker::PhantomData;
			Some(IntView { neg, len, limbs, phantom })
		}
	}

	fn __small_view<'a>(&'a self) -> Option<IntView<'a>> {
		if !self.is_small() {
			return None;
		}
		let neg = self.is_negative();
		let limbs = NonNull::from(&self.magn);

		// Many operations work with the assumption that the highest limb is non-zero.
		// Therefore when the value stored in magn is zero, we should return 0 limbs.
		let len = (self.magn.value != 0) as usize;

		let phantom = std::marker::PhantomData;
		Some(IntView { neg, len, limbs, phantom })
	}

	fn view<'a>(&'a self) -> IntView<'a> {
		self.__small_view().unwrap_or_else(|| {
			self.__large_view().unwrap_or_else(|| {
				// SAFETY: The number has to be either small or large.
				unsafe { std::hint::unreachable_unchecked() }
			})
		})
	}

	fn nonzero_view<'a>(&'a self) -> Option<NonZeroIntView<'a>> {
		if self.is_zero() {
			return None;
		}
		let view = self.view();
		Some(NonZeroIntView {
			neg: view.neg,
			// SAFETY: We know that the number is not zero.
			len: unsafe { NonZeroUsize::new_unchecked(view.len) },
			limbs: view.limbs,
			phantom: view.phantom,
		})
	}

	pub fn bit_capacity(&self) -> usize {
		let limbs = //.
			if let Some(view) = self.__large_view() {
				view.len
			} else {
				Self::ONE_LIMB
			};
		limbs * Limb::BITS
	}

	pub fn bit_width(&self) -> usize {
		if let Some(view) = self.nonzero_view() {
			ll::bit_width(view.as_slice()) //
		} else {
			0
		}
	}

	#[inline(never)]
	#[must_use]
	pub fn try_assign(&mut self, a: &Int) -> Result<(), Error> {
		let A = a.view();
		let mut inline_buf = buf::InlineBuffer::default();
		let mut buf = buf::Buffer::new(self, A.len, &mut inline_buf)?;

		unsafe {
			std::intrinsics::assume(buf.cap.get() >= A.len);
		}
		ll::numcpy(buf.as_slice(), A.as_slice())?;

		buf.commit(A.len, A.neg)
	}

	#[inline(never)]
	pub fn assign(&mut self, a: &Int) {
		self.try_assign(a).unwrap()
	}

	/*#[inline(never)]
	fn add_or_sub(&mut self, a: &Int, sub: bool, b: &Int) {
		let A = a.view();
		let mut B = b.view();
		B.neg ^= sub;

		let sub = A.neg ^ B.neg;

		let (mut A, mut B) = if A.limbs.len() < B.limbs.len() { (B, A) } else { (A, B) };
		let mut out_neg = A.neg;

		//if B.limbs.is_empty() {
		//	A.limbs = if out_sign < 0 { &[] } else { A.limbs };
		//	return;
		//}
		//
		//let out_len = A.limbs.len() + (op >= 0) as usize;
		//let mut out = Vec::with_capacity(out_len);
		//unsafe { out.set_len(out_len) };
		//
		//if op >= 0 {
		//	ll::m_abs_add(&mut out, A.limbs, B.limbs);
		//} else {
		//	ll::m_abs_sub(&mut out, A.limbs, B.limbs, out_sign);
		//}
		//
		//R.vec = NonNull::new(out.as_mut_ptr()).unwrap();
		//R.magn = ll::Limb { value: out_sign };
	}*/
}

/*[[X17_NO_INLINE]]
Integer &add_or_sub(Integer &R, Int_view A_, Count op, Int_view B_) {
	Index A_sign = A_.m_len;
	Abs_int_view A = A_;

	Index B_sign = op ^ B_.m_len; // NOLINT(*-signed-bitwise)
	Abs_int_view B = B_;

	op = A_sign ^ B_sign; // NOLINT(*-signed-bitwise)

	if (A.m_len < B.m_len) {
		boost::swap(A, B);
		boost::swap(A_sign, B_sign);
	}
	Index out_sign = A_sign;

	if (B.m_len == 0) {
		A.m_len = out_sign < 0 ? -A.m_len : A.m_len;
		return assign(R, A);
	}

	// subtraction cannot overflow, so no need to add 1 for op < 0
	Out_buf::Local_buffer out_loc_buf;
	Out_buf out(R, A.m_len + (op >= 0), out_loc_buf);

	if (op >= 0) {
		m_abs_add(out, A, B);
	} else {
		m_abs_sub(out, A, B, out_sign);
	}

	out.swap(out_sign);
	return R;
}
*/

impl Drop for Int {
	fn drop(&mut self) {
		if let Some(view) = self.__large_view() {
			unsafe { Self::__free(view.limbs) };
		}
	}
}

impl Default for Int {
	fn default() -> Self {
		Self::new_zero()
	}
}

#[inline(never)]
pub fn test_set_inline(i: &mut Int, value: usize, neg: bool) {
	i.__set_inline(ll::Limb { value }, neg);
}

#[inline(never)]
pub fn test_commit(b: &mut Buffer, len: usize, neg: bool) -> Result<(), Error> {
	b.commit(len, neg)
}

#[inline(never)]
pub fn test_alloc(n: usize) -> Result<NonNull<[ll::Limb]>, Error> {
	Int::__alloc(n)
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
