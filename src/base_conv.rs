use crate::base_conv_gen::BASE_CONV;
use crate::blocks::Limb;
use crate::Error;
use crate::{base_conv_gen, blocks};
use core::num::NonZeroU8;
use smallvec::{Array, SmallVec};
use std::fmt::DebugList;
use std::intrinsics::{assume, cold_path};

#[derive(Copy, Clone, Debug)]
pub struct BaseConv {
	pub base: u8,
	pub bits_per_digit_ceil: usize,  // ceil(log2(base) * 65536)
	pub bits_per_digit_floor: usize, // floor(log2(base) * 65536)
	pub last_multiple: Limb,
	pub multiples: &'static [Limb],
	pub parse_first_segment: fn(input: &[u8]) -> (Limb, usize),
	pub parse_next_segment: fn(segment: &[u8]) -> Limb,
}

impl BaseConv {
	/// min_base() ..= max_base() are valid base values.
	pub const fn min_base() -> usize {
		2
	}

	/// min_base() ..= max_base() are valid base values.
	pub const fn max_base() -> usize {
		base_conv_gen::BASE_CONV.len() + 1
	}

	/// Returns the base info for the given base.
	///
	/// min_base() ..= max_base() are valid base values.
	#[inline]
	pub fn get(base: usize) -> Option<&'static BaseConv> {
		if let Some(value) = base_conv_gen::BASE_CONV.get(base.wrapping_sub(2)) {
			debug_assert!(value.base as usize == base);
			Some(value)
		} else {
			cold_path();
			None
		}
	}

	/// Safety assumptions:
	/// - digits[0..<str.len()] is valid slice and can be written to.
	#[inline(never)]
	unsafe fn __str_to_digits(&self, str: &str, digits: *mut u8) -> Result<(bool, usize), Error> {
		let mut bytes = str.as_bytes();
		if bytes.is_empty() {
			cold_path();
			return Err(Error::new_parse_error("Int::from_str: empty string"));
		}

		let mut neg = false;
		match bytes[0] {
			b'+' => {
				bytes = &bytes[1..];
			},
			b'-' => {
				neg = true;
				bytes = &bytes[1..];
			},
			_ => {},
		}

		let mapping = //.
			if self.base <= 36 {
				&base_conv_gen::SHORT_MAPPING
			} else {
				&base_conv_gen::BASE64_MAPPING
			};

		let mut p = digits;
		for c in bytes {
			let digit = mapping[*c as usize];
			if (digit as u8) < self.base {
				unsafe {
					p.write(digit as u8);
					p = p.add(1);
				}
			} else {
				cold_path();
				if digit < 0 {
					continue;
				} else {
					return Err(Error::new_parse_error("Int::from_str: invalid digit"));
				}
			}
		}

		if p == digits {
			cold_path();
			return Err(Error::new_parse_error("Int::from_str: no digits found"));
		}

		Ok((neg, unsafe { p.offset_from(digits) as usize }))
	}

	#[inline(always)]
	pub fn str_to_digits(&self, str: &str, digits: &mut [u8]) -> Result<(bool, usize), Error> {
		if digits.len() < str.len() {
			cold_path();
			return Err(Error::new_buffer_too_small("str_to_digits"));
		}
		unsafe {
			let (neg, len) = self.__str_to_digits(str, digits.as_mut_ptr()).map_err(|e| {
				cold_path();
				e
			})?;

			assume(len > 0);
			assume(len <= digits.len());

			Ok((neg, len))
		}
	}

	/// Calculates how many limbs are needed to store given number of digits.
	///
	/// The result may be overestimated.
	#[inline]
	pub fn digits_to_int_est(&self, ndigits: usize) -> usize {
		type Big = u128;
		const _: () = assert!(std::mem::size_of::<Big>() >= 2 * std::mem::size_of::<usize>());

		const LIMB_BITS: Big = Limb::BITS as Big;
		let ndigits = ndigits as Big;
		let bits_per_digit_ceil = self.bits_per_digit_ceil as Big;
		let limbs = ((
			// number of bits as a fixed point number with 16 fractional bits
			ndigits * bits_per_digit_ceil

			// round up to whole bits
			+ 65535

			// round up to whole limbs
			+ 65536 * (LIMB_BITS - 1)
		) / (65536 * LIMB_BITS)) as usize;
		limbs
	}

	/// This is unsafe because it has the following assumptions:
	/// - `input` is NOT empty
	/// - `len <= r.len()`
	#[inline(never)]
	unsafe fn digits_to_int_big(
		&self, r: &mut [Limb], mut len: usize, input: &[u8],
	) -> Result<usize, ()> {
		let digits_per_limb = self.multiples.len();
		let big_base = self.last_multiple;

		// SAFETY: This function has a precondition that `input` is not empty
		debug_assert!(!input.is_empty());
		unsafe { assume(input.len() > 0) };

		let mut i = 0;
		while i < input.len() {
			let limb = (self.parse_next_segment)(&input[i..]);

			// SAFETY: `i < input.len() <= isize::MAX`. `digits_per_limb` is small.
			// `isize::MAX + small number` should not overflow `usize::MAX`
			debug_assert!(i.wrapping_add(digits_per_limb) > i);
			i = unsafe { i.unchecked_add(digits_per_limb) };

			// SAFETY: This function has a precondition that `len <= r.len()`
			// and we only update `len` if it's safe to do so
			debug_assert!(len <= r.len());
			unsafe { assume(len <= r.len()) };

			let h = blocks::mul_1_(&mut r[..len], big_base, limb);
			if h.is_not_zero() {
				if let Some(p) = r.get_mut(len) {
					*p = h;
					len += 1;
				} else {
					cold_path();
					return Err(());
				}
			}
		}

		Ok(len)
	}

	pub fn digits_to_int(&self, r: &mut [Limb], input: &[u8]) -> Result<usize, Error> {
		debug_assert!(!self.base.is_power_of_two()); // TODO - support power of two bases

		let (limb, pos) = (self.parse_first_segment)(input);
		let len;
		if r.len() > 0 {
			r[0] = limb;
			len = limb.is_not_zero() as usize;
		} else {
			cold_path();
			if limb.is_zero() {
				len = 0;
			} else {
				return Err(Error::new_buffer_too_small("digits_to_int"));
			}
		}

		if pos < input.len() {
			unsafe { self.digits_to_int_big(r, len, &input[pos..]) }.map_err(|_| {
				cold_path();
				Error::new_buffer_too_small("digits_to_int")
			})
		} else {
			Ok(len)
		}
	}
}

#[inline(never)]
pub fn parse_first_segment<const BASE: usize>(input: &[u8]) -> (Limb, usize) {
	debug_assert!(!BASE.is_power_of_two());

	let mut m = BASE as Limb::Value;
	let mut digits_per_limb = 1;
	while let Some(n) = m.checked_mul(BASE as Limb::Value) {
		m = n;
		digits_per_limb += 1;
	}
	debug_assert!(digits_per_limb == BASE_CONV[BASE].multiples.len());

	let mut val: Limb::Value = 0;
	let e = if input.len() > digits_per_limb { input.len() % digits_per_limb } else { input.len() };
	for i in 0..e {
		let digit = unsafe { *input.get_unchecked(i) };
		val = val * (BASE as Limb::Value) + (digit as Limb::Value);
	}
	(Limb { val }, e)
}

#[inline(never)]
pub fn parse_first_segment_pow2<const BASE: usize>(input: &[u8]) -> (Limb, usize) {
	debug_assert!(BASE.is_power_of_two());

	let BASE_BITS = BASE.trailing_zeros() as usize;
	let digits_per_limb = Limb::BITS / BASE_BITS;
	debug_assert!(digits_per_limb == BASE_CONV[BASE].multiples.len());

	let mut val: Limb::Value = 0;
	let e = if input.len() > digits_per_limb { input.len() % digits_per_limb } else { input.len() };
	for i in 0..e {
		let digit = unsafe { *input.get_unchecked(i) };
		val = val << BASE_BITS | (digit as Limb::Value);
	}
	(Limb { val }, e)
}

#[inline(never)]
pub fn parse_next_segment<const BASE: usize>(segment: &[u8]) -> Limb {
	debug_assert!(!BASE.is_power_of_two());

	let mut m = BASE as Limb::Value;
	let mut digits_per_limb = 1;
	while let Some(n) = m.checked_mul(BASE as Limb::Value) {
		m = n;
		digits_per_limb += 1;
	}
	debug_assert!(digits_per_limb == BASE_CONV[BASE].multiples.len());

	if let Some(segment) = segment.get(0..digits_per_limb) {
		let mut val: Limb::Value = 0;
		for digit in segment {
			val = val * (BASE as Limb::Value) + (*digit as Limb::Value);
		}
		Limb { val }
	} else {
		// This would be a bug in the code.
		cold_path();
		Limb::zero()
	}
}

#[inline(never)]
pub fn parse_next_segment_pow2<const BASE: usize>(segment: &[u8]) -> Limb {
	debug_assert!(BASE.is_power_of_two());

	let BASE_BITS = BASE.trailing_zeros() as usize;
	let digits_per_limb = Limb::BITS / BASE_BITS;
	debug_assert!(digits_per_limb == BASE_CONV[BASE].multiples.len());

	if let Some(segment) = segment.get(0..digits_per_limb) {
		let mut val: Limb::Value = 0;
		for digit in segment {
			val = val << BASE_BITS | (*digit as Limb::Value);
		}
		Limb { val }
	} else {
		// This would be a bug in the code.
		cold_path();
		Limb::zero()
	}
}
