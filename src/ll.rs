use core::ptr::NonNull;
use std::intrinsics::{assume, cold_path, likely, unlikely};
use std::num::NonZeroUsize;

use crate::blocks;
pub use crate::blocks::Limb;
use crate::error::{assert, Error, ErrorKind};

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - the highest limb (if any) must be non-zero
/// TODO - what could we do if the highest limb is zero? What would be the proper interface?
pub fn bit_width(a: &[Limb]) -> usize {
	if a.is_empty() {
		0 //
	} else {
		unsafe { blocks::bit_width_unchecked(a.as_ptr(), a.len()) }
	}
}

#[inline]
#[must_use]
pub fn numcpy(r: &mut [Limb], a: &[Limb]) -> Result<usize, Error> {
	if a.is_empty() {
		return Ok(0);
	}

	assert(r.len() >= a.len(), || Error::new_buffer_too_small("ll::numcpy()"))?;

	unsafe { blocks::numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), a.len()) };
	Ok(a.len())
}

/// `r.len` must be at least `max(a.len, b.len) + 1`
#[inline(never)]
#[must_use]
pub fn add(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<usize, Error> {
	// Ensure that `a` is the longer of the two numbers
	let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };

	assert(r.len() > a.len(), || Error::new_buffer_too_small("ll::add()"))?;

	let rp = r.as_mut_ptr();
	let ap = a.as_ptr();
	let bp = b.as_ptr();
	let mut carry;
	unsafe {
		carry = blocks::add_n_unchecked(rp, ap, bp, 0, b.len());
		carry = blocks::add_carry_unchecked(rp, ap, carry, b.len(), a.len());
	}
	r[a.len()].value = carry as Limb::Value;
	let r_len = a.len() + (carry as usize);
	unsafe { Ok(blocks::cold_trim_unchecked(r.as_mut_ptr(), 0, r_len)) }
}

pub fn __prep_sub<'a>(a: &'a [Limb], b: &'a [Limb]) -> (bool, &'a [Limb], &'a [Limb]) {
	let mut len = a.len();

	if a.len() != b.len() {
		let swapped = a.len() < b.len();
		let (a, b) = if swapped { (b, a) } else { (a, b) };

		unsafe {
			assume(a.len() > b.len());
			if a[a.len() - 1].value != 0 {
				// The highest limb of A is non-zero. We're done.
				return (swapped, a, b);
			} else {
				// The highest limb of A is zero. We need to trim it.
				cold_path();
				let a_len = blocks::trim_unchecked(a.as_ptr(), b.len(), a.len() - 1);
				if a_len > b.len() {
					return (swapped, a.get_unchecked(..a_len), b);
				}
				len = a_len; // a_len == b.len()
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

/// `r.len` must be at least `max(a.len, b.len)`
#[inline(never)]
#[must_use]
pub fn sub(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<(usize, bool), Error> {
	let (swapped, a, b) = __prep_sub(a, b);

	Ok((0, false)) // TODO
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
