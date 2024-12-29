use crate::blocks::Limb;
use crate::Error;
use crate::{base_info_gen, blocks};
use core::num::NonZeroU8;
use std::intrinsics::{assume, cold_path};

#[derive(Copy, Clone, Debug)]
pub struct BaseInfo {
	pub base: u8,
	pub bits_per_digit_ceil: usize,  // ceil(log2(base) * 65536)
	pub bits_per_digit_floor: usize, // floor(log2(base) * 65536)
	pub multiples: &'static [Limb],
}

impl BaseInfo {
	/// min_base() ..= max_base() are valid base values.
	pub const fn min_base() -> usize {
		2
	}

	/// min_base() ..= max_base() are valid base values.
	pub const fn max_base() -> usize {
		base_info_gen::BASE_INFO.len() + 1
	}

	/// Returns the base info for the given base.
	///
	/// min_base() ..= max_base() are valid base values.
	#[inline]
	pub fn get(base: usize) -> Option<&'static BaseInfo> {
		if base < 2 {
			return None;
		}
		let value = base_info_gen::BASE_INFO.get(base - 2);

		if let Some(value) = value {
			debug_assert!(value.base as usize == base);
		}

		value
	}

	/// Calculates how many limbs are needed to store given number of digits.
	///
	/// The result may be overestimated, but it is guaranteed to be enough.
	#[inline(never)]
	pub fn digits_to_limbs(&self, digits: usize) -> usize {
		const LIMB_BITS: Limb::DoubleValue = Limb::BITS as Limb::DoubleValue;
		let digits = digits as Limb::DoubleValue;
		let bits_per_digit_ceil = self.bits_per_digit_ceil as Limb::DoubleValue;
		((
			// number of bits as a fixed point number with 16 fractional bits
			digits * bits_per_digit_ceil

			// round up to whole bits
			+ 65535

			// round up to whole limbs
			+ 65536 * (LIMB_BITS - 1)
		) / (65536 * LIMB_BITS)) as usize
	}

	#[inline(never)]
	pub fn parse_segment(
		&self, input: &[u8], i: usize, mapping: &[i8; 256],
	) -> (Limb, usize, usize) {
		let base = self.base;
		let digits_per_limb = self.multiples.len();
		debug_assert!(digits_per_limb > 0);

		let mut val: Limb::Value = 0;
		let mut cnt = 0;

		let mut i = i;
		while i < input.len() {
			let digit = unsafe { *mapping.get_unchecked(*input.get_unchecked(i) as usize) };
			i += 1;
			if (digit as u8) < base {
				val = val * (base as Limb::Value) + (digit as u8 as Limb::Value);
				cnt += 1;
				if cnt >= digits_per_limb {
					break;
				}
			} else {
				if digit < 0 {
					i -= 1;
					cold_path();
					break;
				} else {
					continue;
				}
			}
		}

		(Limb { val }, i, cnt)
	}

	#[inline(never)]
	pub fn parse_digits(
		&self, r: &mut [Limb], digits: &[u8], mapping: &[i8; 256],
	) -> Result<usize, (usize, Error)> {
		let mut len = 0;

		let (limb, mut i, cnt) = self.parse_segment(digits, 0, mapping);
		if limb.is_not_zero() {
			if let Some(r) = r.get_mut(0) {
				*r = limb;
				len += 1;
			} else {
				cold_path();
				return Err((0, Error::new_buffer_too_small("BaseInfo::parse_digits")));
			}
		}

		if i >= digits.len() {
			// We've parsed the entire input.
			return Ok(len);
		}

		if cnt < self.multiples.len() {
			// If we didn't parse the entire input and we didn't get a whole limb,
			// there was a parsing error.
			cold_path();
			return Err((i, Error::new_parse_error("BaseInfo::parse_digits")));
		}

		// Repeat while we have some input left.
		while i < digits.len() {
			let (limb, j, cnt) = self.parse_segment(digits, 0, mapping);
			if cnt < self.multiples.len() {
				// We didn't get a whole limb.

				if j < digits.len() {
					// If we are not at the end of the input, there was a parsing error.
					cold_path();
					return Err((i, Error::new_parse_error("BaseInfo::parse_digits")));
				}

				if cnt == 0 {
					// We didn't parse any digits.
					// We need to cover this case because we will access self.multiples[cnt - 1].
					cold_path();
					break;
				}
			}

			if cnt > self.multiples.len() {
				// This would be a bug in the code.
				// We need to cover it in order to safely access self.multiples[cnt - 1].
				cold_path();
				return Err((i, Error::new_internal_error("BaseInfo::parse_digits")));
			}

			i = j;
			let h = //.
				unsafe {
					let rp = r.as_mut_ptr();
					let re = rp.add(len);
					assume(cnt > 0);
					assume(cnt <= self.multiples.len());
					blocks::mul_1_unchecked_(rp, re, self.multiples[cnt - 1], limb)
				};
			if h.is_not_zero() {
				if let Some(r) = r.get_mut(len) {
					*r = h;
					len += 1;
				} else {
					cold_path();
					return Err((i, Error::new_buffer_too_small("BaseInfo::parse_digits")));
				}
			}
		}

		Ok(len)
	}
}
