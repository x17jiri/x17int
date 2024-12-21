// General notes about functions in this module:
//
// - These functions are safe
//
// - They provide "strong exception safety". If an error is returned, the output buffer
//   is left unchanged.
//
// - Some of them may allocate scratch space on the heap. If it happens, it is documented.
//   If there is allocation failure, an error is returned and the output buffer is left unchanged.
//
// - They don't allocate memory for the result. The caller provides a fixed-size buffer
//   and the result either fits into it or an error is returned.
//
// - There is always a function for estimating the size of the result. For example, for `add()`,
//   there is `add_est()`. These estimation functions are designed to be fast and may overestimate
//
// - They return length of the result. If `r` is the output buffer and `est_len` is the
//   estimated buffer size, then:
//     - r[0 ..< len] contains the result with any leading zeros trimmed
//	   - r[len ..< est_len] may be overwritten
//	   - r[est_len .. ] is left unchanged
//
// - If the inputs don't have any leading zeros, the outputs won't have them either

use core::ptr::NonNull;
use std::intrinsics::{assume, cold_path, likely, unlikely};
use std::num::NonZeroUsize;

pub use crate::blocks::Limb;
use crate::blocks::{self, trim_unchecked};
use crate::error::{assert, Error, ErrorKind};

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - the highest limb (if any) must be non-zero
/// TODO - what could we do if the highest limb is zero? What would be the proper interface?
pub fn bit_width(a: &[Limb]) -> usize {
	unsafe { blocks::bit_width_unchecked(a.as_ptr(), a.len()) }
}

#[inline]
pub fn numcpy_est(a: &[Limb]) -> usize {
	a.len()
}

#[inline(never)]
#[must_use]
pub fn numcpy(r: &mut [Limb], a: &[Limb]) -> Result<usize, Error> {
	assert(r.len() >= a.len(), || Error::new_buffer_too_small("ll::numcpy()"))?;
	unsafe {
		blocks::numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 0, a.len());
	}
	Ok(a.len())
}

#[inline]
pub fn add_est(a: &[Limb], b: &[Limb]) -> usize {
	a.len().max(b.len()) + 1
}

/// `r.len` must be at least `max(a.len, b.len) + 1` even if the final result is shorter.
#[inline(never)]
#[must_use]
pub fn add(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<usize, Error> {
	// Ensure that `a` is not shorter than `b`
	let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };

	assert(r.len() > a.len(), || Error::new_buffer_too_small("ll::add()"))?;

	let rp = r.as_mut_ptr();
	let ap = a.as_ptr();
	let an = a.len();
	let bp = b.as_ptr();
	let bn = b.len();
	unsafe {
		let carry = blocks::add_n_unchecked(rp, ap, bp, 0, bn);
		let carry = blocks::add_carry_unchecked(rp, ap, carry, bn, an);
		r.get_unchecked_mut(an).value = carry as Limb::Value;
		Ok(an + (carry as usize))
	}
}

/// This function does two things:
/// 1. It trims common prefix from A and B
///     - For example when subtracting `1173 - 1152`, the prefix `11` doesn't affect the result
///       and can be removed. `1173 - 1152 == 73 - 52`.
/// 2. It compares A and B and possibly swaps them to make sure that A >= B. The actual digits
///    are compared, not just lengths.
///     - With the numbers swapped, the subtraction `A - B` can never be negative.
///
/// The result is one of the following:
/// - (false, [], []) if A == B
/// - (false, A, B) if A > B
/// - (true, B, A) if A < B
fn __prep_sub<'a>(a: &'a [Limb], b: &'a [Limb]) -> (bool, &'a [Limb], &'a [Limb]) {
	let mut len = a.len();

	if a.len() != b.len() {
		let swapped = a.len() < b.len();
		let (a, b) = if swapped { (b, a) } else { (a, b) };

		unsafe {
			if a.get_unchecked(a.len() - 1).value != 0 {
				// The highest limb of A is non-zero. We're done.
				return (swapped, a, b);
			} else {
				// The highest limb of A is zero. We need to trim it.
				cold_path();

				assume(a.len() > b.len());
				let a_len = blocks::trim_unchecked(a.as_ptr(), b.len(), a.len() - 1);
				if a_len > b.len() {
					return (swapped, a.get_unchecked(..a_len), b);
				}

				// We trimmed A enough so that it's the same length as B.
				// Continue to the second part of this function and find which number is bigger.
				len = a_len;
			}
		}
	}

	// A and B have the same length
	unsafe {
		// Scan A and B from the highest limb to find the first difference
		while len > 0 && *a.get_unchecked(len - 1) == *b.get_unchecked(len - 1) {
			len -= 1;
		}

		// We found a difference. Swap the numbers if A < B
		let swapped = len > 0 && *a.get_unchecked(len - 1) < *b.get_unchecked(len - 1);
		let (a, b) = if swapped { (b, a) } else { (a, b) };

		(swapped, a.get_unchecked(..len), b.get_unchecked(..len))
	}
}

#[inline]
pub fn sub_est(a: &[Limb], b: &[Limb]) -> usize {
	a.len().max(b.len())
}

/// `r.len` must be at least `max(a.len, b.len)` even if the final result is shorter.
///
/// Writes to `r` and returns `(neg, n)` where:
///      r[0..<n] = abs(a - b)
///      neg = a < b
#[inline(never)]
#[must_use]
pub fn sub(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<(bool, usize), Error> {
	// Ensure that `a` is not smaller than `b`
	let (swapped, a, b) = __prep_sub(a, b);

	assert(r.len() >= a.len(), || Error::new_buffer_too_small("ll::sub()"))?;

	let rp = r.as_mut_ptr();
	let ap = a.as_ptr();
	let an = a.len();
	let bp = b.as_ptr();
	let bn = b.len();
	unsafe {
		let borrow = blocks::sub_n_unchecked(rp, ap, bp, 0, bn);
		let borrow = blocks::sub_borrow_unchecked(rp, ap, borrow, bn, an);
		debug_assert!(!borrow);
		let rn = trim_unchecked(rp, 0, an);
		Ok((swapped, rn))
	}
}

#[macro_use]
#[cfg(test)]
mod tests {
	use super::*;
	use crate::testvec;

	#[test]
	fn test_bit_width() {
		let a = testvec![];
		assert_eq!(bit_width(a.as_slice()), 0);

		let a = testvec![0];
		assert_eq!(bit_width(a.as_slice()), 1);

		let a = testvec![0, 0, 7];
		assert_eq!(bit_width(a.as_slice()), Limb::BITS * 2 + 3);
	}

	#[test]
	fn test_numcpy() {
		let a = testvec![];
		let mut r = testvec![];
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), Ok(0));
		assert_eq!(r, testvec![]);

		let a = testvec![1, 2, 3];
		let mut r = testvec![0, 0, 0];
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), Ok(3));
		assert_eq!(r, testvec![1, 2, 3]);

		let a = testvec![1, 2, 3];
		let mut r = testvec![0, 0, 0, 0];
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), Ok(3));
		assert_eq!(r, testvec![1, 2, 3, 0]);

		let a = testvec![1, 2, 3];
		let mut r = testvec![0, 0];
		let err = numcpy(r.as_mut_slice(), a.as_slice());
		assert_eq!(err.is_err(), true);
		assert_eq!(err.err().unwrap().kind, ErrorKind::BufferTooSmall);
		assert_eq!(r, testvec![0, 0]);
	}

	#[test]
	fn test_add() {
		let MAX = Limb::MAX;

		let a = testvec![];
		let b = testvec![];
		let mut r = testvec![100, 101, 102];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), Ok(0));
		assert_eq!(r, testvec![0, 101, 102]);

		let a = testvec![1, 2, 3];
		let b = testvec![4, 5, 6];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), Ok(3));
		assert_eq!(r, testvec![5, 7, 9, 0]);

		let a = testvec![1, 2, 3];
		let b = testvec![4, 5];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), Ok(3));
		assert_eq!(r, testvec![5, 7, 3, 0]);

		let a = testvec![1, 2];
		let b = testvec![4, 5, 6];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), Ok(3));
		assert_eq!(r, testvec![5, 7, 6, 0]);

		let a = testvec![MAX, MAX, MAX];
		let b = testvec![MAX];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), Ok(4));
		assert_eq!(r, testvec![MAX - 1, 0, 0, 1]);

		let a = testvec![MAX, MAX, MAX];
		let b = testvec![MAX];
		let mut r = testvec![100, 101, 102];
		let err = add(r.as_mut_slice(), a.as_slice(), b.as_slice());
		assert_eq!(err.is_err(), true);
		assert_eq!(err.err().unwrap().kind, ErrorKind::BufferTooSmall);
		assert_eq!(r, testvec![100, 101, 102]);
	}
}
