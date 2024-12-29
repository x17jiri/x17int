use Vec;

type Limb = usize;
type DoubleLimb = u128;

const LIMB_BITS: usize = Limb::BITS as usize;
const LIMB_MAX: usize = Limb::MAX;
const MIN_BASE: usize = 2;
const MAX_BASE: usize = 64;

/*
pub struct BaseInfo {
	pub base: u8,
	pub bits_per_digit_ceil: usize,  // ceil(log2(base) * 65536)
	pub bits_per_digit_floor: usize, // floor(log2(base) * 65536)
	pub digits_per_limb: u8,
	pub digits_per_limb_inv: usize,
	pub big_base: Limb, // base ** digits_per_limb
}
*/

pub fn inv(val: usize) -> (usize, usize) {
	const _OK: () = assert!(std::mem::size_of::<u128>() >= 2 * std::mem::size_of::<usize>());

	let inv = ((1 as u128) << usize::BITS) / (val as u128) + 1;

	let extra = (inv * val as u128) - ((1 as u128) << usize::BITS);
	let cnt;
	if extra <= 1 {
		cnt = 0;
	} else {
		cnt = (((1 as u128) << usize::BITS) / extra)
			+ (((1 as u128) << usize::BITS) % extra != 0) as u128;
	}

	(inv as usize, cnt as usize)
}

fn base_info() -> String {
	let mut base_info = String::new();
	base_info.push_str("use crate::base_info::{BaseInfo, parse_segment, parse_segment_pow2};\n");
	base_info.push_str("use crate::blocks::Limb;\n");
	base_info.push_str("use core::num::NonZeroU8;\n");
	base_info.push_str("\n");
	base_info
		.push_str(&format!("pub const BASE_INFO: [BaseInfo; {}] = [\n", MAX_BASE - MIN_BASE + 1));
	for base in MIN_BASE..=MAX_BASE as usize {
		let mut multiples = Vec::new();

		let mut m = base as u128;
		while m <= (1_u128 << LIMB_BITS) {
			multiples.push(m);
			m *= base as u128;
		}

		if base.is_power_of_two() {
			let bits_per_digit = base.trailing_zeros() as usize;
			let digits_per_limb = LIMB_BITS / bits_per_digit;
			let (inv, cnt) = inv(digits_per_limb);
			base_info.push_str("\tBaseInfo {\n");
			base_info.push_str(&format!("\t\tbase: {},\n", base));
			base_info.push_str(&format!(
				"\t\tbits_per_digit_ceil: {}, // {} << 16\n",
				bits_per_digit * 65536,
				bits_per_digit
			));
			base_info.push_str(&format!(
				"\t\tbits_per_digit_floor: {}, // {} << 16\n",
				bits_per_digit * 65536,
				bits_per_digit
			));
			base_info.push_str(&format!(
				"\t\t// digits_per_limb: NonZeroU8::new({}).unwrap(),\n",
				digits_per_limb
			));
			if cnt == 0 {
				base_info.push_str("\t\t// This multiplicative inverse has no inprecision");
			} else {
				base_info.push_str(&format!("\t\t// This multiplicative inverse has an inprecision and will add an extra limb for every {} digits\n", cnt));
			}
			base_info.push_str(&format!("\t\t// digits_per_limb_inv: {},\n", inv));

			base_info.push_str(&format!("\t\tparse_segment: parse_segment_pow2::<{}>,\n", base));
		} else {
			let bits_per_digit_ceil = ((base as f64).log2() * 65536.0).ceil() as usize;
			let bits_per_digit_floor = ((base as f64).log2() * 65536.0).floor() as usize;

			let mut digits_per_limb = 1;
			let mut big_base = base as Limb;
			while big_base < LIMB_MAX / base as Limb {
				digits_per_limb += 1;
				big_base *= base as Limb;
			}

			let digits_per_limb2 = (LIMB_BITS << 16) / bits_per_digit_ceil;
			assert!(digits_per_limb == digits_per_limb2);

			let (inv, cnt) = inv(digits_per_limb);

			base_info.push_str("\tBaseInfo {\n");
			base_info.push_str(&format!("\t\tbase: {},\n", base));
			base_info.push_str(&format!(
				"\t\tbits_per_digit_ceil: {}, // {} << 16\n",
				bits_per_digit_ceil,
				bits_per_digit_ceil as f64 / 65536.0
			));
			base_info.push_str(&format!(
				"\t\tbits_per_digit_floor: {}, // {} << 16\n",
				bits_per_digit_floor,
				bits_per_digit_floor as f64 / 65536.0
			));
			base_info.push_str(&format!(
				"\t\t// digits_per_limb: NonZeroU8::new({}).unwrap(),\n",
				digits_per_limb
			));
			if cnt == 0 {
				base_info.push_str("\t\t// This multiplicative inverse has no inprecision");
			} else {
				base_info.push_str(&format!("\t\t// This multiplicative inverse has an inprecision and will add an extra limb for every {} digits\n", cnt));
			}
			base_info.push_str(&format!("\t\t// digits_per_limb_inv: {},\n", inv));
			base_info.push_str(&format!("\t\tparse_segment: parse_segment::<{}>,\n", base));
		}
		base_info.push_str("\t\tmultiples: &[\n");
		for m in multiples.iter() {
			if *m > LIMB_MAX as u128 {
				base_info.push_str("\t\t\t// value overflowed\n");
			}
			base_info.push_str(&format!("\t\t\tLimb {{ val: {} }},\n", *m as usize));
		}
		base_info.push_str("\t\t],\n");
		base_info.push_str("\t},\n");
	}
	base_info.push_str("];\n");
	base_info
}

fn main() {
	let base_info = base_info();

	// If `base_info` is different from the content of `src/base_info_gen.rs`, write it to the file.
	let path = std::path::Path::new("src/base_info_gen.rs");
	if let Ok(content) = std::fs::read_to_string(path) {
		if content != base_info {
			std::fs::write(path, base_info).unwrap();
		}
	} else {
		std::fs::write(path, base_info).unwrap();
	}
}
