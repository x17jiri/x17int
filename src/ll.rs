use core::ptr::NonNull;
use std::num::NonZeroUsize;
use std::ptr::copy_nonoverlapping;

#[derive(Clone, Copy, Default)]
pub struct Limb {
	pub value: usize,
}

impl Limb {
	pub const BITS: usize = usize::BITS as usize;
}

fn has_no_overlap(a: NonNull<Limb>, a_len: usize, b: NonNull<Limb>, b_len: usize) -> bool {
	let a_begin = a.as_ptr();
	let b_begin = b.as_ptr();
	let a_end = unsafe { a_begin.add(a_len) };
	let b_end = unsafe { b_begin.add(b_len) };
	a_end <= b_begin || b_end <= a_begin
}

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - the highest limb (if any) must be non-zero
pub fn bit_width(a: NonNull<Limb>, n: usize) -> usize {
	if let Some(n) = NonZeroUsize::new(n) {
		bit_width_nonzero(a, n) //
	} else {
		0
	}
}

pub fn bit_width_nonzero(a: NonNull<Limb>, n: NonZeroUsize) -> usize {
	let n = n.get();
	let hi = unsafe { a.offset(n as isize - 1).read().value };
	debug_assert!(hi != 0);
	n * Limb::BITS - (hi | 1).leading_zeros() as usize
}

/// {rp, n} = {ap, n}
///
/// Preconditions:
/// - allowed overlap: none
#[inline]
pub unsafe fn numcpy(rp: NonNull<Limb>, ap: NonNull<Limb>, n: NonZeroUsize) {
	debug_assert!(has_no_overlap(rp, n.get(), ap, n.get()));

	// I think there is a reasonable chance that 'n' will be
	// just a few limbs. So try to have the code for this case inlined.
	unsafe {
		let n = n.get();
		if n > 0 && n <= 4 {
			let a: usize = 0;
			let b: usize = n >> 2;
			let c: usize = n >> 1;
			let d: usize = n - 1;

			// n || a | b | c | d
			// ------------------
			// 1 || 0 | 0 | 0 | 0
			// 2 || 0 | 0 | 1 | 1
			// 3 || 0 | 0 | 1 | 2
			// 4 || 0 | 1 | 2 | 3

			rp.offset(a as isize).write(ap.offset(a as isize).read());
			rp.offset(b as isize).write(ap.offset(b as isize).read());
			rp.offset(c as isize).write(ap.offset(c as isize).read());
			rp.offset(d as isize).write(ap.offset(d as isize).read());
		} else {
			copy_nonoverlapping(ap.as_ptr(), rp.as_ptr(), n);
		}
	}
}
