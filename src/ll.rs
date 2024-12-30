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
// - They don't allocate memory for the result. The caller provides a fixed-size buffer.
//
// - If the buffer is too small, the result will be truncated and only the lower part will be written
//
// - There is always a function for estimating the size of the result. For example, for `add()`,
//   there is `add_est()`. These estimation functions are designed to be fast and may overestimate.
//   When the buffer is at least as big as the estimated size, the result will always fit.
//
// - They return length of the result. If `r` is the output buffer and `est_len` is the
//   estimated buffer size, then:
//     - r[0 ..< len] contains the result with any leading zeros trimmed
//	   - r[len ..< est_len] may be overwritten
//	   - r[est_len .. ] is left unchanged
//
// - Returned values don't have leading zeros

use core::ptr::NonNull;
use std::intrinsics::{assume, cold_path, likely, unlikely};
use std::num::NonZeroUsize;

use crate::base_conv::BaseConv;
use crate::blocks;
pub use crate::blocks::Limb;
use crate::error::{assert, Error, ErrorKind};

//--------------------------------------------------------------------------------------------------
// bit_width

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - the highest limb (if any) must be non-zero
/// TODO - what could we do if the highest limb is zero? What would be the proper interface?
pub fn bit_width(a: &[Limb]) -> usize {
	unsafe { blocks::bit_width_unchecked(a.as_ptr(), a.len()) }
}

//--------------------------------------------------------------------------------------------------
// numcpy

#[inline]
pub fn numcpy_est(a: &[Limb]) -> usize {
	a.len()
}

#[inline]
pub fn numcpy(r: &mut [Limb], a: &[Limb]) -> usize {
	let rp = r.as_mut_ptr();
	let rn = r.len();

	let ap = a.as_ptr();
	let an = a.len();

	let n = an.min(rn);
	if n == 0 {
		cold_path();
		return 0;
	}

	unsafe {
		blocks::numcpy_unchecked_i_n(rp, ap, 0, n);
		blocks::cold_trim_unchecked(rp, 0, n)
	}
}

//--------------------------------------------------------------------------------------------------
// add

#[inline]
pub fn add_est(a: &[Limb], b: &[Limb]) -> usize {
	unsafe { a.len().max(b.len()).unchecked_add(1) }
}

#[inline]
pub fn add(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<usize, Error> {
	if r.len() <= a.len().max(b.len()) {
		cold_path();
		return Err(Error::new_buffer_too_small("add"));
	}
	let len = __add_trunc(r, a, b);
	unsafe { assume(len <= r.len()) };
	Ok(len)
}

#[inline]
pub fn add_trunc(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> usize {
	let len = __add_trunc(r, a, b);
	unsafe { assume(len <= r.len()) };
	len
}

#[inline(never)]
fn __add_trunc(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> usize {
	// Ensure that `a` is not shorter than `b`
	let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };

	unsafe {
		let rp = r.as_mut_ptr();
		let ap = a.as_ptr();
		let bp = b.as_ptr();

		let rn = r.len();
		let (an, bn) = //.
			if rn >= a.len() {
				(a.len(), b.len())
			} else {
				cold_path();
				(rn, b.len().min(rn))
			};

		if an == 0 {
			// We're adding `0 + 0`
			cold_path();
			return 0;
		}

		let carry = blocks::add_n_unchecked(rp, ap, bp, 0, bn);
		let carry = blocks::add_carry_unchecked(rp, ap, carry, bn, an);

		let len;
		if carry {
			if rn > an {
				rp.add(an).write(Limb::one());
				return an.unchecked_add(1);
			} else {
				cold_path();
				len = an;
			}
		} else {
			len = an;
		};
		blocks::cold_trim_unchecked(rp, 0, len)
	}
}

//--------------------------------------------------------------------------------------------------
// sub

#[inline]
pub fn sub_est(a: &[Limb], b: &[Limb]) -> usize {
	a.len().max(b.len())
}

#[inline]
pub fn sub(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<(bool, usize), Error> {
	if r.len() < a.len().max(b.len()) {
		cold_path();
		return Err(Error::new_buffer_too_small("sub"));
	}
	let (neg, len) = __sub_trunc(r, a, b);
	unsafe { assume(len <= r.len()) };
	Ok((neg, len))
}

/// Subtracts `A - B` and returns length of the result and whether the result is negative.
#[inline]
pub fn sub_trunc(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> (bool, usize) {
	let (neg, len) = __sub_trunc(r, a, b);
	unsafe { assume(len <= r.len()) };
	(neg, len)
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
			assume(a.len() > b.len());
			assume(a.len() > 0);

			if a.get_unchecked(a.len() - 1).is_not_zero() {
				// The highest limb of A is non-zero. We're done.
				return (swapped, a, b);
			} else {
				// The highest limb of A is zero. We need to trim it.
				cold_path();

				let a_len = blocks::trim_unchecked(a.as_ptr(), b.len(), a.len() - 1);
				if a_len != b.len() {
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

#[inline(never)]
fn __sub_trunc(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> (bool, usize) {
	// Ensure that `a` is not smaller than `b`
	let (swapped, a, b) = __prep_sub(a, b);

	unsafe {
		let rp = r.as_mut_ptr();
		let ap = a.as_ptr();
		let bp = b.as_ptr();

		let rn = r.len();
		let (an, bn) = //.
			if rn >= a.len() {
				(a.len(), b.len())
			} else {
				cold_path();
				(rn, b.len().min(rn))
			};

		if an == 0 {
			// We're subtracting `0 - 0`
			cold_path();
			return (false, 0);
		}

		let borrow = blocks::sub_n_unchecked(rp, ap, bp, 0, bn);
		let _borrow = blocks::sub_borrow_unchecked(rp, ap, borrow, bn, an);

		(swapped, blocks::trim_unchecked(rp, 0, an))
	}
}

//--------------------------------------------------------------------------------------------------
// mul

pub fn mul_est(a: &[Limb], b: &[Limb]) -> usize {
	if a.is_empty() || b.is_empty() {
		0 //
	} else {
		unsafe { a.len().unchecked_add(b.len()) }
	}
}

pub fn mul(
	r: &mut [Limb], a: &[Limb], b: &[Limb], scratch_alloc: &dyn std::alloc::Allocator,
) -> Result<usize, Error> {
	if a.is_empty() || b.is_empty() {
		return Ok(0);
	}
	if r.len() < unsafe { a.len().unchecked_add(b.len()) } {
		cold_path();
		return Err(Error::new_buffer_too_small("mul"));
	}
	unsafe {
		let len = __mul(r, a, b, scratch_alloc);
		assume(len <= r.len());
		Ok(len)
	}
}

#[inline(never)]
unsafe fn __mul(
	r: &mut [Limb], a: &[Limb], b: &[Limb], _scratch_alloc: &dyn std::alloc::Allocator,
) -> usize {
	debug_assert!(!r.is_empty());
	debug_assert!(!a.is_empty());
	debug_assert!(!b.is_empty());
	debug_assert!(a.len() >= b.len());
	unsafe {
		// Ensure that `a` is not smaller than `b`
		let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };

		let mut rp = r.as_mut_ptr();
		let ap = a.as_ptr();
		let mut bp = b.as_ptr();

		let mut re = rp.add(a.len());
		let be = bp.add(b.len());

		let mut t = blocks::mul_1_unchecked(rp, re, ap, bp.read(), Limb::zero());
		re.write(t);

		rp = rp.add(1);
		re = re.add(1);
		bp = bp.add(1);

		while bp != be {
			t = blocks::addmul_1_unchecked(rp, re, ap, bp.read());
			re.write(t);

			rp = rp.add(1);
			re = re.add(1);
			bp = bp.add(1);
		}

		let len = a.len().unchecked_add(b.len()).unchecked_sub(t.is_zero() as usize);

		blocks::cold_trim_unchecked(rp, 0, len)
	}
}

//--------------------------------------------------------------------------------------------------

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
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), 0);
		assert_eq!(r, testvec![]);

		let a = testvec![1, 2, 3];
		let mut r = testvec![0, 0, 0];
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), 3);
		assert_eq!(r, testvec![1, 2, 3]);

		let a = testvec![1, 2, 3];
		let mut r = testvec![0, 0, 0, 0];
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), 3);
		assert_eq!(r, testvec![1, 2, 3, 0]);

		let a = testvec![1, 2, 3];
		let mut r = testvec![0, 0];
		assert_eq!(numcpy(r.as_mut_slice(), a.as_slice()), 2);
		assert_eq!(r, testvec![1, 2]);
	}

	#[test]
	fn test_add() {
		let MAX = Limb::MAX;

		let a = testvec![];
		let b = testvec![];
		let mut r = testvec![100, 101, 102];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), 0);
		assert_eq!(r, testvec![100, 101, 102]);

		let a = testvec![1, 2, 3];
		let b = testvec![4, 5, 6];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), 3);
		assert_eq!(r, testvec![5, 7, 9, 103]);

		let a = testvec![1, 2, 3];
		let b = testvec![4, 5];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), 3);
		assert_eq!(r, testvec![5, 7, 3, 103]);

		let a = testvec![1, 2];
		let b = testvec![4, 5, 6];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), 3);
		assert_eq!(r, testvec![5, 7, 6, 103]);

		let a = testvec![MAX, MAX, MAX];
		let b = testvec![MAX];
		let mut r = testvec![100, 101, 102, 103];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), 4);
		assert_eq!(r, testvec![MAX - 1, 0, 0, 1]);

		let a = testvec![MAX, MAX, MAX];
		let b = testvec![MAX];
		let mut r = testvec![100, 101, 102];
		assert_eq!(add(r.as_mut_slice(), a.as_slice(), b.as_slice()), 1);
		assert_eq!(r, testvec![MAX - 1, 0, 0]);
	}
}
