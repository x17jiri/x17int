/*
use crate::error::{Error, ErrorKind};
use crate::{ll, Int};
use std::intrinsics::{assume, likely, unlikely};
use std::num::NonZeroUsize;
use std::ptr::{copy_nonoverlapping, NonNull};

#[derive(PartialEq)]
pub enum BufferOwnership {
	Inline,
	Borrowed,
	Owned,
}

pub struct Buffer<'a> {
	pub neg: bool,
	pub len: usize,
	pub cap: NonZeroUsize,
	pub limbs: NonNull<ll::Limb>,
	ownership: BufferOwnership,
	committed: bool,
	R: &'a mut Int,
}

const INLINE_BUF_SIZE: usize = 3;
pub type InlineBuffer = [ll::Limb; INLINE_BUF_SIZE];

impl<'a> Buffer<'a> {
	pub fn new(R: &'a mut Int, size: usize, inline: &'a mut InlineBuffer) -> Result<Self, Error> {
		if let Some(buf_view) = R.__large_buf_view() {
			// R has a buffer allocated, try to use it
			if size <= buf_view.cap.get() {
				return Ok(Self {
					neg: false,
					len: 0,
					cap: buf_view.cap,
					limbs: buf_view.limbs,
					ownership: BufferOwnership::Borrowed,
					committed: false,
					R,
				});
			}
		} else {
			// R uses inline buffer, try to use OUR inline buffer instead.
			// R's inline buffer is 1 limb long and it is unlikely to be big enough
			if size <= INLINE_BUF_SIZE {
				return Ok(Self {
					neg: false,
					len: 0,
					cap: NonZeroUsize::new(INLINE_BUF_SIZE).unwrap(),
					limbs: NonNull::from(&inline[0]),
					ownership: BufferOwnership::Inline,
					committed: false,
					R,
				});
			}
		}

		let new_buf = Int::__alloc(size)?;
		Ok(Self {
			neg: false,
			len: 0,
			cap: unsafe { NonZeroUsize::new_unchecked(new_buf.len()) },
			limbs: new_buf.as_non_null_ptr(),
			ownership: BufferOwnership::Owned,
			committed: false,
			R,
		})
	}

	#[must_use]
	pub fn commit(&mut self, len: usize, neg: bool) -> Result<(), Error> {
		self.len = len;
		self.neg = neg;
		self.committed = true;

		match self.ownership {
			BufferOwnership::Inline => {
				// We used our inline buffer, check if the value fits into R's inline buffer
				// If not, allocate a new buffer and copy the value
				if len <= Int::ONE_LIMB {
					if len == 0 {
						let value = ll::Limb { value: 0 };
						self.R.__set_inline(value, false);
						return Ok(());
					} else {
						let value = unsafe { self.limbs.read() };
						self.R.__set_inline(value, neg);
					}
				} else {
					let new_buf = Int::__alloc(INLINE_BUF_SIZE)?;
					unsafe {
						std::ptr::copy_nonoverlapping(
							self.limbs.as_ptr(),
							new_buf.as_mut_ptr(),
							INLINE_BUF_SIZE,
						);
						self.R.__set_allocated(new_buf, len, neg);
					}
				}
			},
			BufferOwnership::Borrowed => {
				// We used R's buffer, just update the length and sign
				unsafe { self.R.__set_len_neg(len, neg) };
			},
			BufferOwnership::Owned => {
				if let Some(buf_view) = self.R.__large_buf_view() {
					unsafe { Int::__free(buf_view.limbs) };
				}
				unsafe {
					let new_buf = NonNull::slice_from_raw_parts(self.limbs, len);
					self.R.__set_allocated(new_buf, len, neg);
				}
			},
		}
		Ok(())
	}

	#[inline(never)]
	fn cleanup(&self) {
		if self.ownership == BufferOwnership::Owned {
			unsafe { Int::__free(self.limbs) };
		}
	}

	pub fn as_slice(&mut self) -> &mut [ll::Limb] {
		unsafe { std::slice::from_raw_parts_mut(self.limbs.as_ptr(), self.cap.get()) }
	}
}

impl Drop for Buffer<'_> {
	fn drop(&mut self) {
		if unlikely(!self.committed) {
			self.cleanup();
		}
	}
}
*/
