use crate::blocks::Limb;
use crate::Error;
use crate::{base_info_gen, blocks};
use std::intrinsics::{assume, cold_path};

#[derive(Copy, Clone, Debug)]
pub struct BaseInfo {
	pub base: u8,
	pub bits_per_digit_ceil: usize,  // ceil(log2(base) * 65536)
	pub bits_per_digit_floor: usize, // floor(log2(base) * 65536)
	pub digits_per_limb: u8,
	pub digits_per_limb_inv: usize,
	pub big_base: Limb, // base ** digits_per_limb
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
	pub fn parse_segment(&self, segment: &[u8]) -> Limb {
		let base = self.base as Limb::Value;
		let mut val: Limb::Value = 0;
		for digit in segment {
			val = val * base + (*digit as Limb::Value);
		}
		Limb { val }
	}

	#[inline(never)]
	pub fn parse_digits(&self, r: &mut [Limb], digits: &[u8]) -> usize {
		if digits.is_empty() || r.is_empty() {
			return 0;
		}

		let big_base = self.big_base;
		let digits_per_limb = self.digits_per_limb as usize;
		unsafe { assume(digits_per_limb != 0) };
		//-- Parse the top limb.

		let segments =
			(((digits.len() as u128) * (self.digits_per_limb_inv as u128)) >> usize::BITS) as usize;
		let top_len = digits.len().wrapping_sub(unsafe { segments.unchecked_mul(digits_per_limb) });

		let top_len =
			if top_len < digits_per_limb { top_len } else { digits.len() % digits_per_limb };
		let top = self.parse_segment(unsafe { digits.get_unchecked(..top_len) });
		r[0] = top;
		let mut len = top.is_not_zero() as usize;

		//-- Parse the rest

		let rp = r.as_mut_ptr();
		for i in (top_len..digits.len()).step_by(digits_per_limb) {
			let new_bottom =
				self.parse_segment(unsafe { digits.get_unchecked(i..i + digits_per_limb) });
			let new_top = unsafe { blocks::mul_1_unchecked_(rp, rp.add(len), big_base) };
			r[0].val += new_bottom.val;
			if new_top.is_not_zero() && len < r.len() {
				r[len] = new_top;
				len += 1;
			}
		}

		len
	}
}
