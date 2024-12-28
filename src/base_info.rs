use crate::base_info_gen;
use crate::blocks::Limb;

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
	#[inline]
	pub fn get(base: usize) -> Option<&'static BaseInfo> {
		if base < 2 {
			return None;
		}
		let value = base_info_gen::BASE_INFO.get(base - 2);

		if let Some(value) = value {
			debug_assert!(
				base == match value {
					BaseInfo::Pow2(BaseInfoPow2 { base, .. }) => *base as usize,
					BaseInfo::Other(BaseInfoOther { base, .. }) => *base as usize,
				}
			);
		}

		value
	}
}

impl BaseInfoPow2 {
	#[inline]
	pub fn digits_to_limbs(&self, digits: usize) -> usize {
		const LIMB_BITS: Limb::DoubleValue = Limb::BITS as Limb::DoubleValue;
		let digits = digits as Limb::DoubleValue;
		let bits_per_digit = self.bits_per_digit as Limb::DoubleValue;
		((
			// number of bits
			digits * bits_per_digit

			// round up to whole limbs
			+ (LIMB_BITS - 1)
		) / LIMB_BITS) as usize
	}

	pub fn parse_digits(&self, digits: &[u8]) -> Limb {
		debug_assert!(digits.len() <= self.digits_per_limb as usize);
		let mut val = 0;
		for &digit in digits {
			val <<= self.bits_per_digit as usize;
			val |= digit as Limb::Value;
		}
		Limb { val }
	}
}

impl BaseInfoOther {
	#[inline]
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

	pub fn parse_digits(&self, digits: &[u8]) -> Limb {
		debug_assert!(digits.len() <= self.digits_per_limb as usize);
		let mut val = 0;
		for &digit in digits {
			val *= self.base as Limb::Value;
			val += digit as Limb::Value;
		}
		Limb { val }
	}
}
