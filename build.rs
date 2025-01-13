use Vec;

type Limb = usize;

const LIMB_BITS: usize = Limb::BITS as usize;
const LIMB_MAX: usize = Limb::MAX;
const MIN_BASE: usize = 2;
const MAX_BASE: usize = 64;

/*
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

fn base_conv() -> String {
	let mut base_conv = String::new();
	base_conv.push_str("use crate::base_conv::{BaseConv, /*parse_first_segment, parse_first_segment_pow2, parse_next_segment, parse_next_segment_pow2*/};\n");
	base_conv.push_str("use crate::limb::Limb;\n");
	base_conv.push_str("use crate::fixed_size::Invert2By1;\n");
	base_conv.push_str("use crate::fixed_size::Uint;\n");
	base_conv.push_str("use core::num::NonZeroU8;\n");
	base_conv.push_str("\n");
	base_conv
		.push_str(&format!("pub const BASE_CONV: [BaseConv; {}] = [\n", MAX_BASE - MIN_BASE + 1));
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
			base_conv.push_str("\tBaseConv {\n");
			base_conv.push_str(&format!("\t\tbase: {},\n", base));
			base_conv.push_str(&format!(
				"\t\tbits_per_digit_ceil: {}, // {} << 16\n",
				bits_per_digit * 65536,
				bits_per_digit
			));
			base_conv.push_str(&format!(
				"\t\tbits_per_digit_floor: {}, // {} << 16\n",
				bits_per_digit * 65536,
				bits_per_digit
			));
			base_conv.push_str("\t\t// digits_per_limb is equal to multiples.len()\n");
			base_conv.push_str(&format!(
				"\t\t// digits_per_limb: NonZeroU8::new({}).unwrap(),\n",
				digits_per_limb
			));
			if cnt == 0 {
				base_conv.push_str("\t\t// This multiplicative inverse has no inprecision");
			} else {
				base_conv.push_str(&format!("\t\t// This multiplicative inverse has an inprecision and will add an extra limb for every {} digits\n", cnt));
			}
			base_conv.push_str(&format!("\t\tdigits_per_limb_inv: {},\n", inv));

		//			base_conv.push_str(&format!(
		//				"\t\tparse_first_segment: parse_first_segment_pow2::<{}>,\n",
		//				base
		//			));
		//			base_conv.push_str(&format!(
		//				"\t\tparse_next_segment: parse_next_segment_pow2::<{}>,\n",
		//				base
		//			));
		} else {
			let bits_per_digit_ceil = ((base as f64).log2() * 65536.0).ceil() as usize;
			let bits_per_digit_floor = ((base as f64).log2() * 65536.0).floor() as usize;

			let mut digits_per_limb = 1;
			let mut last_multiple = base as Limb;
			while last_multiple < LIMB_MAX / base as Limb {
				digits_per_limb += 1;
				last_multiple *= base as Limb;
			}

			let digits_per_limb2 = (LIMB_BITS << 16) / bits_per_digit_ceil;
			assert!(digits_per_limb == digits_per_limb2);

			let (inv, cnt) = inv(digits_per_limb);

			base_conv.push_str("\tBaseConv {\n");
			base_conv.push_str(&format!("\t\tbase: {},\n", base));
			base_conv.push_str(&format!(
				"\t\tbits_per_digit_ceil: {}, // {} << 16\n",
				bits_per_digit_ceil,
				bits_per_digit_ceil as f64 / 65536.0
			));
			base_conv.push_str(&format!(
				"\t\tbits_per_digit_floor: {}, // {} << 16\n",
				bits_per_digit_floor,
				bits_per_digit_floor as f64 / 65536.0
			));
			base_conv.push_str("\t\t// digits_per_limb is equal to multiples.len()\n");
			base_conv.push_str(&format!(
				"\t\t// digits_per_limb: NonZeroU8::new({}).unwrap(),\n",
				digits_per_limb
			));
			if cnt == 0 {
				base_conv.push_str("\t\t// This multiplicative inverse has no inprecision");
			} else {
				base_conv.push_str(&format!("\t\t// This multiplicative inverse has an inprecision and will add an extra limb for every {} digits\n", cnt));
			}
			base_conv.push_str(&format!("\t\tdigits_per_limb_inv: {},\n", inv));
			//			base_conv
			//				.push_str(&format!("\t\tparse_first_segment: parse_first_segment::<{}>,\n", base));
			//			base_conv
			//				.push_str(&format!("\t\tparse_next_segment: parse_next_segment::<{}>,\n", base));
		}
		let last_multiple = *multiples.last().unwrap();
		if last_multiple > LIMB_MAX as u128 {
			base_conv.push_str("\t\t// value overflowed\n");
		}
		base_conv.push_str(&format!("\t\tlast_multiple: Limb({}),\n", last_multiple as usize));
		base_conv.push_str(&format!(
			"\t\tlast_multiple_inv: Invert2By1::new(Uint::new([Limb({})])),\n",
			last_multiple as usize
		));
		base_conv.push_str("\t\tmultiples: &[\n");
		for m in multiples.iter() {
			if *m > LIMB_MAX as u128 {
				base_conv.push_str("\t\t\t// value overflowed\n");
			}
			base_conv.push_str(&format!("\t\t\tLimb({}),\n", *m as usize));
		}
		base_conv.push_str("\t\t],\n");
		base_conv.push_str("\t},\n");
	}
	base_conv.push_str("];\n");
	base_conv.push_str("\n");
	base_conv.push_str("// SHORT_MAPPING is for bases 2 ..< 36\n");
	base_conv.push_str("pub const SHORT_MAPPING: [i8; 256] = [\n");
	let mut mapping = [127_i8; 256];
	for i in b'0'..=b'9' {
		mapping[i as usize] = (i - b'0') as i8;
	}
	for i in b'A'..=b'Z' {
		mapping[i as usize] = (i - b'A' + 10) as i8;
	}
	for i in b'a'..=b'z' {
		mapping[i as usize] = (i - b'a' + 36) as i8;
	}
	mapping[b'_' as usize] = -1;
	for i in 0..16 {
		let mut line = String::new();
		for j in 0..16 {
			line.push_str(&format!("{:4},", mapping[i * 16 + j]));
		}
		base_conv.push_str(&format!("\t{}\n", line));
	}
	base_conv.push_str("];\n");
	base_conv.push_str("// BASE64_MAPPING can also be used for bases 2 ..< 36\n");
	base_conv.push_str("// but it is mainly intended for bases 36 ..= 64\n");
	let mut mapping = [127_i8; 256];
	for i in b'A'..=b'Z' {
		mapping[i as usize] = (i - b'A') as i8;
	}
	for i in b'a'..=b'z' {
		mapping[i as usize] = (i - b'a' + 26) as i8;
	}
	for i in b'0'..=b'9' {
		mapping[i as usize] = (i - b'0' + 52) as i8;
	}
	mapping[b'+' as usize] = 62;
	mapping[b'/' as usize] = 63;
	mapping[b'-' as usize] = 62;
	mapping[b'_' as usize] = 63;
	base_conv.push_str("pub const BASE64_MAPPING: [i8; 256] = [\n");
	for i in 0..16 {
		let mut line = String::new();
		for j in 0..16 {
			line.push_str(&format!("{:4},", mapping[i * 16 + j]));
		}
		base_conv.push_str(&format!("\t{}\n", line));
	}
	base_conv.push_str("];\n");
	base_conv
}

fn main() {
	let base_conv = base_conv();

	// If `base_conv` is different from the content of `src/base_conv_gen.rs`, write it to the file.
	let path = std::path::Path::new("src/base_conv_gen.rs");
	if let Ok(content) = std::fs::read_to_string(path) {
		if content != base_conv {
			std::fs::write(path, base_conv).unwrap();
		}
	} else {
		std::fs::write(path, base_conv).unwrap();
	}
}
