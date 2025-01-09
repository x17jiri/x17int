use std::alloc::{Allocator, Global, Layout};
use std::intrinsics::assume;
use std::intrinsics::cold_path;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::error::assert;
use crate::limb::Limb;
use crate::ll;
use crate::Error;

pub struct LimbBuf {
	ptr: NonNull<ll::Limb>,
}

impl LimbBuf {
	pub const MIN_ALLOC_SIZE: usize = 3;
	pub const MAX_LIMBS: usize = usize::MAX / Limb::BITS;
	pub const MAX_BITS: usize = Self::MAX_LIMBS * Limb::BITS;

	pub fn as_mut_slice(&mut self) -> &mut [ll::Limb] {
		unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.cap()) }
	}

	pub fn as_non_null_ptr(&self) -> NonNull<ll::Limb> {
		self.ptr
	}

	pub unsafe fn from_non_null_ptr(ptr: NonNull<ll::Limb>) -> Self {
		Self { ptr }
	}

	pub fn cap(&self) -> usize {
		unsafe {
			let cap = self.ptr.offset(-1).read().0;
			debug_assert!(cap > 0);
			assume(cap > 0);
			cap
		}
	}

	#[inline(never)]
	fn __new(n: usize) -> Result<Self, Error> {
		let n = Self::MIN_ALLOC_SIZE.max(n);

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
		// assume it in `new()`. This branch should really be dead code.
		if new_buf.len() < n {
			panic!("Allocator returned a buffer that is too small.");
		}

		let ptr = new_buf.as_non_null_ptr();
		let ptr = ptr.cast::<ll::Limb>();

		let cap = (new_buf.len() / std::mem::size_of::<ll::Limb>()) - 1;
		let cap = //.
			if cap <= Self::MAX_LIMBS {
				cap
			} else {
				cold_path();
				Self::MAX_LIMBS
			};
		let cap = unsafe { NonZeroUsize::new_unchecked(cap) };

		unsafe { ptr.write(Limb(cap.get())) };
		let ptr = unsafe { ptr.offset(1) };

		Ok(Self { ptr })
	}

	pub fn new(n: usize) -> Result<LimbBuf, Error> {
		let buf = Self::__new(n).map_err(|e| {
			cold_path();
			e
		})?;
		// SAFETY: We verify this in `__new()`.
		unsafe {
			assume(buf.cap() >= n);
			assume(buf.cap() >= Self::MIN_ALLOC_SIZE);
		}
		Ok(buf)
	}
}

impl Deref for LimbBuf {
	type Target = [ll::Limb];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.cap()) }
	}
}

impl DerefMut for LimbBuf {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.as_mut_slice()
	}
}

impl Drop for LimbBuf {
	fn drop(&mut self) {
		unsafe {
			let ptr = self.ptr.offset(-1);
			let cap = self.ptr.read().0;

			let size = (cap + 1) * std::mem::size_of::<ll::Limb>();
			let align = std::mem::align_of::<ll::Limb>();
			debug_assert!(size > 0);

			assume(size > 0);
			Global.deallocate(ptr.cast::<u8>(), Layout::from_size_align_unchecked(size, align));
		}
	}
}
