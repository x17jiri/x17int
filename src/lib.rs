#![feature(allocator_api)]
#![feature(strict_provenance)]
#![feature(pointer_is_aligned_to)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]
#![feature(slice_ptr_get)]

use std::alloc::{Allocator, Global, Layout};
use std::intrinsics::unlikely;
use std::num::NonZeroUsize;
use std::ptr::{dangling, NonNull};

use ll::Limb;

mod ll;

pub struct Error {
	pub message: &'static str,
}

pub struct IntView<'a> {
	neg: bool,
	limbs: &'a [ll::Limb],
}

pub enum BufferOwneership {
	Owned,
	Borrowed,
	Inline,
}

struct Buffer<'a> {
	neg: bool,
	len: usize,
	cap: usize,
	limbs: *mut ll::Limb,
	ownership: BufferOwneership,
	R: &'a mut Int,
}

const INLINE_BUF_SIZE: usize = 3;
type InlineBuffer = [ll::Limb; INLINE_BUF_SIZE];

impl<'a> Buffer<'a> {
	fn new(R: &'a mut Int, size: usize, inline: &'a mut InlineBuffer) -> Result<Self, Error> {
		if R.is_small() {
			// R uses inline buffer, try to use OUR inline buffer instead.
			// R's inline buffer is 1 limb long and it is unlikely to be big enough
			if size <= INLINE_BUF_SIZE {
				return Ok(Self {
					neg: false,
					len: 0,
					cap: INLINE_BUF_SIZE,
					limbs: inline.as_mut_ptr(),
					ownership: BufferOwneership::Inline,
					R,
				});
			}
		} else {
			// R has a buffer allocated, try to use it
			let old_cap = R.__limbs_cap_large();
			if size <= old_cap {
				return Ok(Self {
					neg: false,
					len: 0,
					cap: old_cap,
					limbs: R.__limbs_ptr_large_mut(),
					ownership: BufferOwneership::Borrowed,
					R,
				});
			}
		}

		let new_buf = Int::__alloc(size)?;
		Ok(Self {
			neg: false,
			len: 0,
			cap: new_buf.len(),
			limbs: new_buf.as_mut_ptr(),
			ownership: BufferOwneership::Owned,
			R,
		})
	}

	fn swap(&mut self) -> Result<(), Error> {
		match self.ownership {
			BufferOwneership::Owned => {
				if self.len <= 1 {
					let value = unsafe { self.limbs.read() };
					self.R.__set_inline(value, self.neg);
					// TODO - deallocate
				} else {
				}
			},
			BufferOwneership::Borrowed => {
				// We used R's buffer, just update the length and sign
				self.R.__set_len_neg_large(self.len, self.neg);
			},
			BufferOwneership::Inline => {
				if self.len <= 1 {
					let value = unsafe { self.limbs.read() };
					self.R.__set_inline(value, self.neg);
				} else {
					let new_buf = Int::__alloc(INLINE_BUF_SIZE)?;
					unsafe {
						std::ptr::copy_nonoverlapping(
							self.limbs,
							new_buf.as_mut_ptr(),
							INLINE_BUF_SIZE,
						);
					}
					self.R.__set_allocated(new_buf, self.len, self.neg);
				}
			},
		}
		Ok(())
	}
}

pub struct Int {
	// The representation is a bit complicated, but hopefully we can hide the complexity behind a few public methods.

	// `vec` is a pointer tagged with 1 or 2 bits. If, without the tag, it is null, the number is small.
	// - bit 0: sign bit. 0 for positive, 1 for negative.
	// - bit 1:
	//   - if `vec != null`, this tag bit is not used
	//   - if `vec == null`, this tag bit is always 1
	// This way, the tagged pointer is never null and Option<Int> can be the same size as Int.
	vec: NonNull<u8>,

	// If the number is small, `magn` is the value.
	// If the number is large, `magn` is the number of limbs.
	magn: ll::Limb,
}

impl Int {
	pub fn is_negative(&self) -> bool {
		!self.vec.is_aligned_to(2)
	}

	pub fn is_zero(&self) -> bool {
		self.magn.value == 0
	}

	pub fn is_positive(&self) -> bool {
		let neg = self.is_negative();
		self.magn.value != 0 && !neg
	}

	pub fn is_small(&self) -> bool {
		self.vec.addr().get() < 4
	}

	fn __limbs_small(&self) -> &[ll::Limb] {
		debug_assert!(self.is_small());
		unsafe {
			std::slice::from_raw_parts(
				&self.magn as *const ll::Limb,
				(self.magn.value != 0) as usize,
			)
		}
	}

	fn __mut_limbs_small(&mut self) -> &mut [ll::Limb] {
		self.__limbs_small().as_mut()
	}

	fn __limbs_large(&self) -> &[ll::Limb] {
		debug_assert!(!self.is_small());
		unsafe {
			let neg = self.is_negative();
			let bytes = self.vec.offset(-(neg as isize));
			let ptr = bytes.cast::<ll::Limb>().as_ptr();
			let len = self.magn.value;
			std::slice::from_raw_parts(ptr, len)
		}
	}

	fn __limbs(&self) -> &[ll::Limb] {
		if self.is_small() { self.__limbs_small() } else { self.__limbs_large() }
	}

	fn __buf_small(&self) -> &[ll::Limb] {
		debug_assert!(self.is_small());
		unsafe {
			let ptr = &self.magn as *const ll::Limb;
			let cap = 1;
			std::slice::from_raw_parts(ptr, cap)
		}
	}

	fn __buf_large(&self) -> &[ll::Limb] {
		debug_assert!(!self.is_small());
		unsafe {
			let neg = self.is_negative();
			let bytes = self.vec.offset(-(neg as isize));
			let ptr = bytes.cast::<ll::Limb>().as_ptr();
			let cap = ptr.offset(-1).read().value;
			std::slice::from_raw_parts(ptr, cap)
		}
	}

	fn __buf(&self) -> &[ll::Limb] {
		if self.is_small() { self.__buf_small() } else { self.__buf_large() }
	}

	fn __set_inline(&mut self, value: ll::Limb, neg: bool) {
		let tag = 2 | (neg as usize);
		let bytes = std::ptr::without_provenance_mut::<u8>(tag);
		self.vec = unsafe { NonNull::new_unchecked(bytes) };
		self.magn = value;
	}

	fn __set_allocated(&mut self, buffer: NonNull<[ll::Limb]>, len: usize, neg: bool) {
		let bytes = buffer.cast::<u8>();
		self.vec = unsafe { bytes.offset(neg as isize) };
		self.magn = ll::Limb { value: len };
	}

	fn __set_len_neg_large(&mut self, len: usize, neg: bool) {
		debug_assert!(!self.is_small());
		debug_assert!(len <= self.__buf_large().len());
		let prev_neg = self.is_negative();
		let bytes = unsafe { self.vec.offset(-(prev_neg as isize)) };
		let new_bytes = unsafe { bytes.offset(neg as isize) };
		self.vec = new_bytes;
		self.magn = ll::Limb { value: len };
	}

	const MAX_LIMBS: usize = usize::MAX / Limb::BITS;
	const MAX_BITS: usize = Self::MAX_LIMBS * Limb::BITS;

	fn __alloc(n: usize) -> Result<NonNull<[ll::Limb]>, Error> {
		if n > Self::MAX_LIMBS {
			return Err(Error {
				message: "Allocation failed. Number of limbs exceeds the maximum.",
			});
		}

		let layout = //.
			match std::alloc::Layout::array::<ll::Limb>(n) {
				Ok(layout) => layout,
				_ => {
					return Err(Error {
						message: "Allocation failed. Cannot create Layout instance.",
					});
				},
			};

		let new_buf = //.
			match Global.allocate(layout) {
				Ok(new_buf) => new_buf,
				_ => {
					return Err(Error {
						message: "Allocation failed. Cannot allocate memory.",
					});
				},
			};

		let ptr = new_buf.as_mut_ptr();
		let ptr = ptr as *mut ll::Limb;
		let cap = Self::MAX_LIMBS.min(new_buf.len() / std::mem::size_of::<ll::Limb>());
		Ok(unsafe { NonNull::new_unchecked(std::slice::from_raw_parts_mut(ptr, cap)) })
	}

	fn __free(buf: NonNull<[ll::Limb]>) {
		unsafe {
			let ptr = buf.as_non_null_ptr().offset(-1);
			let bytes = ptr.cast::<u8>();
			let size = (buf.len() + 1) * std::mem::size_of::<ll::Limb>();
			std::intrinsics::assume(size > 0);
			let align = std::mem::align_of::<ll::Limb>();
			let layout = Layout::from_size_align(size, align).unwrap_unchecked();
			Global.deallocate(bytes, layout);
		}
	}

	pub fn view<'a>(&'a self) -> IntView<'a> {
		let neg = self.is_negative();
		if self.is_small() {
			IntView { neg, limbs: self.__limbs_small() }
		} else {
			IntView { neg, limbs: self.__limbs_large() }
		}
	}

	pub fn bit_capacity(&self) -> usize {
		self.__buf().len() * ll::Limb::BITS
	}

	pub fn bit_width(&self) -> usize {
		if self.is_zero() {
			0
		} else {
			let view = self.view();
			ll::bit_width(view.limbs.as_ptr(), view.limbs.len())
		}
	}

	/*#[inline(never)]
	pub fn assign(&mut self, a: &Int) {
		let A = a.view();
		Count n = std::abs(A.m_len);
		Out_buf::Local_buffer out_loc_buf;
		Out_buf out(R, n, out_loc_buf);

		my_mpn::numcpy(out.m_limbs, A.m_limbs, n);

		out.m_len = A.m_len;
		out.swap(0);
	}*/

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
		if !self.is_small() {
			unsafe {
				let buf = self.__buf_large();
				let ptr = buf.as_ptr();
				let cap = buf.len();
				let slice = std::ptr::slice_from_raw_parts(ptr, cap)
				let mut_slice = slice as *mut [ll::Limb];
				Self::__free(NonNull::new_unchecked(mut_slice));
			}
		}
	}
}
