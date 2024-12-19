use core::ptr::NonNull;
use std::intrinsics::{assume, cold_path, likely, unlikely};
use std::num::NonZeroUsize;

pub use crate::blocks::Limb;
use crate::{Error, blocks};

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - the highest limb (if any) must be non-zero
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

	if r.len() < a.len() {
		cold_path();
		return Err(Error {
			message: "ll::numcpy(): buffer too small",
		});
	}

	unsafe { blocks::numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), a.len()) };
	Ok(a.len())
}

#[inline(never)]
#[must_use]
pub fn add(r: &mut [Limb], a: &[Limb], b: &[Limb]) -> Result<usize, Error> {
	// Ensure that `a` is the longer of the two
	let (a, b) = if a.len() >= b.len() { (a, b) } else { (b, a) };

	if r.len() < a.len() {
		cold_path();
		return Err(Error { message: "ll::add(): buffer too small" });
	}

	let rp = r.as_mut_ptr();
	let ap = a.as_ptr();
	let bp = b.as_ptr();
	let mut carry;
	unsafe {
		carry = blocks::add_n_unchecked(rp, ap, bp, 0, b.len());
		carry = blocks::add_carry_unchecked(rp, ap, carry, b.len(), a.len());
	}
	if carry {
		if let Some(top) = r.get_mut(a.len()) {
			top.value = 1;
			Ok(a.len() + 1)
		} else {
			cold_path();
			Err(Error {
				message: "ll::add(): buffer too small for carry",
			})
		}
	} else {
		Ok(a.len())
	}
}
