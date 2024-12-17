#![feature(allocator_api)]
#![feature(strict_provenance)]
#![feature(pointer_is_aligned_to)]
#![feature(core_intrinsics)]
#![feature(inherent_associated_types)]
#![feature(generic_const_exprs)]

use std::alloc::{Allocator, Global, Layout};
use std::intrinsics::unlikely;
use std::num::NonZeroUsize;
use std::ptr::{dangling, NonNull};

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
	fn new(R: &'a mut Int, size: usize, inline: &'a InlineBuffer) -> Result<Self, Error> {
		if R.is_small() {
			// R uses inline buffer, try to use OUR inline buffer instead.
			// R's inline buffer is 1 limb long and it is unlikely to be big enough
			if size <= INLINE_BUF_SIZE {
				return Ok(Self {
					neg: false,
					len: 0,
					cap: INLINE_BUF_SIZE,
					limbs: inline.as_ptr() as *mut ll::Limb,
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
					limbs: R.__limbs_ptr_large() as *mut ll::Limb,
					ownership: BufferOwneership::Borrowed,
					R,
				});
			}
		}

		let layout = //.
			match std::alloc::Layout::array::<ll::Limb>(size) {
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
		Ok(Self {
			neg: false,
			len: 0,
			cap: new_buf.len() / std::mem::size_of::<ll::Limb>(),
			limbs: new_buf.as_ptr() as *mut ll::Limb,
			ownership: BufferOwneership::Owned,
			R,
		})
	}

	fn swap(&mut self) -> Result<(), Error> {
		match self.ownership {
			BufferOwneership::Owned => {
				// TODO
				Ok(())
			},
			BufferOwneership::Borrowed => {
				// We used R's buffer, nothing to do

				// TODO
				Ok(())
			},
			BufferOwneership::Inline if self.len <= 1 => {
				let value = unsafe { self.limbs.read() };
				self.R.__set_inline(value, self.neg);
				Ok(())
			},
			BufferOwneership::Inline if self.len > 1 => {
				let layout = //.
					match std::alloc::Layout::array::<ll::Limb>(INLINE_BUF_SIZE) {
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
				unsafe {
					std::ptr::copy_nonoverlapping(
						self.limbs,
						new_buf.as_ptr() as *mut ll::Limb,
						INLINE_BUF_SIZE,
					);
				}
				// TODO - the allocated buffer needs to have a header with the length
				R.__set_allocated(new_buf, INLINE_BUF_SIZE, self.neg);
				Ok(())
			},
		}
		// TODO - set R.len and R.neg
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

	fn __limbs_ptr_small(&self) -> *const ll::Limb {
		&self.magn as *const ll::Limb
	}

	fn __limbs_ptr_large(&self) -> *const ll::Limb {
		debug_assert!(!self.is_small());
		let neg = self.is_negative();
		unsafe { self.vec.offset(-(neg as isize)).as_ptr() as *const ll::Limb }
	}

	fn __limbs_ptr(&self) -> *const ll::Limb {
		if self.is_small() {
			self.__limbs_ptr_small() //
		} else {
			self.__limbs_ptr_large()
		}
	}
	fn __limbs_len_small(&self) -> usize {
		(self.magn.value != 0) as usize
	}
	fn __limbs_len_large(&self) -> usize {
		self.magn.value
	}

	fn __limbs_len(&self) -> usize {
		if self.is_small() {
			self.__limbs_len_small() //
		} else {
			self.__limbs_len_large()
		}
	}

	fn __limbs_cap_small(&self) -> usize {
		1
	}

	fn __limbs_cap_large(&self) -> usize {
		unsafe { self.__limbs_ptr().offset(-1).read().value }
	}

	fn __limbs_cap(&self) -> usize {
		if self.is_small() {
			self.__limbs_cap_small() //
		} else {
			self.__limbs_cap_large()
		}
	}

	fn __set_inline(&mut self, value: ll::Limb, neg: bool) {
		let tag = 2 | (neg as usize);
		self.vec =
			unsafe { NonNull::new_unchecked(std::ptr::without_provenance::<u8>(tag) as *mut u8) };
		self.magn = value;
	}

	fn __set_allocated(&mut self, ptr: NonNull<ll::Limb>, len: usize, neg: bool) {
		self.vec = unsafe { NonNull::new_unchecked(ptr.as_ptr() as *mut u8).offset(neg as isize) };
		self.magn = ll::Limb { value: len };
	}

	pub fn view<'a>(&'a self) -> IntView<'a> {
		let neg = self.is_negative();
		if self.is_small() {
			IntView {
				neg,
				limbs: unsafe {
					std::slice::from_raw_parts(
						self.__limbs_ptr_small(),
						self.__limbs_len_small(), //
					)
				},
			}
		} else {
			IntView {
				neg,
				limbs: unsafe {
					std::slice::from_raw_parts(
						self.__limbs_ptr_large(),
						self.__limbs_len_large(), //
					)
				},
			}
		}
	}

	pub fn bit_capacity(&self) -> usize {
		self.__limbs_cap() * ll::Limb::BITS
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
				let ptr = self.__limbs_ptr().offset(-1);
				let size_bytes = (self.__limbs_cap() + 1) * std::mem::size_of::<ll::Limb>();
				std::intrinsics::assume(size_bytes > 0);
				let align = std::mem::align_of::<ll::Limb>();
				Global.deallocate(
					NonNull::new_unchecked(ptr as *mut u8),
					std::alloc::Layout::from_size_align(size_bytes, align).unwrap_unchecked(),
				);
			}
		}
	}
}
