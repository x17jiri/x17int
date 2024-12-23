// Implementation of tagged pointer with 1 bit tag.
//
// It uses sentinel value to represent null pointer. So internally, we have NonNull and
// Option<TaggedPtr<T>> can be the same size as a pointer.

use std::num::NonZeroUsize;
use std::ptr::NonNull;

#[derive(Clone, Copy)]
pub struct TaggedPtr<T> {
	__ptr: NonNull<u8>,
	phantom: std::marker::PhantomData<T>,
}

impl<T> TaggedPtr<T> {
	#[inline]
	pub fn new_null(tag: bool) -> Self {
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
	pub fn new(ptr: NonNull<T>, tag: bool) -> Self {
		Self {
			// I use offset because map_addr() doesn't generate good assembly here.
			// What the current implementation of map_addr() does is:
			//     let self_addr = self.addr() as isize;
			//     let dest_addr = addr as isize;
			//     let offset = dest_addr.wrapping_sub(self_addr);
			//     self.wrapping_byte_offset(offset)
			//
			// So it computes offset and adjusts the pointer by that offset.
			// It seems LLVM cannot optimize this calculation away.
			__ptr: unsafe { ptr.cast::<u8>().offset(tag as isize).cast() },
			phantom: std::marker::PhantomData,
		}
	}

	#[inline]
	pub fn is_null(&self) -> bool {
		self.__ptr.addr().get() & !3 == 0
	}

	#[inline]
	pub fn are_both_null(a: Self, b: Self) -> bool {
		(a.__ptr.addr().get() | b.__ptr.addr().get()) & !3 == 0
	}

	#[inline]
	pub fn tag(&self) -> bool {
		self.__ptr.addr().get() & 1 != 0
	}

	#[inline]
	pub fn ptr(&self) -> Option<NonNull<T>> {
		NonNull::new(self.__ptr.as_ptr().map_addr(|addr| addr & !3).cast())
	}
}
