use crate::limb::Limb;
use std::intrinsics::{assume, cold_path};

#[rustfmt::skip] pub const fn const_max(a: usize, b: usize) -> usize {
	if a >= b { a } else { b }
}

#[rustfmt::skip] pub const fn const_min(a: usize, b: usize) -> usize {
	if a <= b { a } else { b }
}

#[rustfmt::skip] const fn __const_select_unpredictable<T: Copy>(b: bool, t: T, f: T) -> T {
	if b { t } else { f }
}

fn __nonconst_select_unpredictable<T: Copy>(b: bool, t: T, f: T) -> T {
	std::intrinsics::select_unpredictable(b, t, f)
}

const fn select_unpredictable<T: Copy>(b: bool, t: T, f: T) -> T {
	std::intrinsics::const_eval_select(
		(b, t, f),
		__const_select_unpredictable,
		__nonconst_select_unpredictable,
	)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Uint<const NLIMBS: usize>
where
	[Limb; NLIMBS]: Default,
{
	pub limbs: [Limb; NLIMBS],
}

impl<const NLIMBS: usize> Uint<NLIMBS>
where
	[Limb; NLIMBS]: Default,
{
	pub const fn zero() -> Self {
		Self { limbs: [Limb::ZERO; NLIMBS] }
	}

	pub const fn max() -> Self {
		Self { limbs: [Limb::MAX; NLIMBS] }
	}

	pub const fn new(limbs: [Limb; NLIMBS]) -> Self {
		Self { limbs }
	}

	pub const fn is_zero(self) -> bool {
		let mut i = 0;
		while i < NLIMBS {
			if self.limbs[i].is_not_zero() {
				return false;
			}
			i += 1;
		}
		true
	}

	pub const fn is_not_zero(self) -> bool {
		!self.is_zero()
	}

	pub const fn add(self, b: Self) -> (Self, bool) {
		self.addc(b, false)
	}

	pub const fn wrapping_add(self, b: Self) -> Self {
		self.add(b).0
	}

	pub const fn add_extend(self, b: Self) -> Uint<{ NLIMBS + 1 }>
	where
		[Limb; NLIMBS + 1]: Default,
	{
		let (val, carry) = self.addc(b, false);
		let mut val = val.resize();
		val.limbs[NLIMBS] = Limb::from_bool(carry);
		val
	}

	pub const fn addc(self, b: Self, carry: bool) -> (Self, bool) {
		let mut r = [Limb::ZERO; NLIMBS];
		let mut carry = carry;

		let mut i = 0;
		while i < NLIMBS {
			(r[i], carry) = Limb::addc(self.limbs[i], b.limbs[i], carry);
			i += 1;
		}

		(Self { limbs: r }, carry)
	}

	pub const fn sub(self, b: Self) -> (Self, bool) {
		self.subb(b, false)
	}

	pub const fn wrapping_sub(self, b: Self) -> Self {
		self.sub(b).0
	}

	pub const fn subb(self, b: Self, borrow: bool) -> (Self, bool) {
		let mut r = [Limb::ZERO; NLIMBS];
		let mut borrow = borrow;

		let mut i = 0;
		while i < NLIMBS {
			(r[i], borrow) = Limb::subb(self.limbs[i], b.limbs[i], borrow);
			i += 1;
		}

		(Self { limbs: r }, borrow)
	}

	pub const fn not(self) -> Self {
		let mut r = [Limb::ZERO; NLIMBS];

		let mut i = 0;
		while i < NLIMBS {
			r[i] = self.limbs[i].const_bitnot();
			i += 1;
		}

		Self { limbs: r }
	}

	pub const fn bitand(self, b: Self) -> Self {
		let mut r = [Limb::ZERO; NLIMBS];

		let mut i = 0;
		while i < NLIMBS {
			r[i] = self.limbs[i].const_bitand(b.limbs[i]);
			i += 1;
		}

		Self { limbs: r }
	}

	pub const fn mask(self, mask: Limb) -> Self {
		let mut r = [Limb::ZERO; NLIMBS];

		let mut i = 0;
		while i < NLIMBS {
			r[i] = self.limbs[i].const_bitand(mask);
			i += 1;
		}

		Self { limbs: r }
	}

	/// If `condition`, return 2's complement of `a`. Otherwise, return `a` unchanged.
	/// I.e., if condition is true, negates all bits of `a` and adds 1.
	pub const fn neg_if(self, condition: bool) -> Self {
		let mut r = self.limbs;
		if NLIMBS > 0 && condition {
			let mut borrow;
			(r[0], borrow) = r[0].overflowing_neg();

			let mut i = 1;
			while i < NLIMBS {
				(r[i], borrow) = r[i].const_bitnot().overflowing_sub(Limb::from_bool(borrow));
				i += 1;
			}
		}
		Self { limbs: r }
	}

	/// Calculates:
	///     negative = a < b
	///     value = abs(a - b)
	///     return (value, negative)
	pub const fn sub_abs(self, b: Self) -> (Self, bool) {
		let (r, borrow) = self.sub(b);
		let r = r.neg_if(borrow);
		(r, borrow)
	}

	pub const fn mul<const B: usize>(self, b: Uint<B>) -> Uint<{ NLIMBS + B }>
	where
		[Limb; B]: Default,
		[Limb; NLIMBS + B]: Default,
	{
		let mut r = Uint::zero();

		let mut j = 0;
		while j < B {
			let mut high = Limb::ZERO;
			let mut i = 0;
			while i < NLIMBS {
				[r.limbs[i + j], high] = Limb::mul(self.limbs[i], b.limbs[j], high, r.limbs[i + j]);
				i += 1;
			}
			r.limbs[NLIMBS + j] = high;
			j += 1;
		}

		r
	}

	pub const fn mul_1(self, b: Limb) -> Uint<{ NLIMBS + 1 }>
	where
		[Limb; NLIMBS + 1]: Default,
	{
		self.mul(Uint { limbs: [b; 1] })
	}

	/// Shifts `a` left by `shift % Limb::BITS` bits.
	/// I.e., the shift is always less than Limb::BITS.
	pub const fn small_shl(self, shift: usize) -> Self {
		let shift = shift & (Limb::BITS - 1);
		let mut r = [Limb::ZERO; NLIMBS];
		match NLIMBS {
			0 => {},
			1 => {
				r[0] = self.limbs[0].const_shl(shift);
			},
			_ => {
				let t = Limb::make_double(self.limbs[0], self.limbs[1]) << shift;
				r[0] = Limb::from_low_half(t);
				r[1] = Limb::from_high_half(t);

				let mut i = 2;
				while i < NLIMBS {
					let t = Limb::make_double(self.limbs[i - 1], self.limbs[i]) << shift;
					r[i] = Limb::from_high_half(t);

					i += 1;
				}
			},
		}
		Self { limbs: r }
	}

	pub const fn small_shl_extend(self, shift: usize) -> Uint<{ NLIMBS + 1 }>
	where
		[Limb; NLIMBS + 1]: Default,
	{
		let shift = shift & (Limb::BITS - 1);
		match NLIMBS {
			0 => {
				return Uint::zero();
			},
			1 => {
				let mut r = [Limb::ZERO; NLIMBS + 1];
				let t = Limb::make_double(self.limbs[0], Limb::ZERO) << shift;
				r[0] = Limb::from_low_half(t);
				r[1] = Limb::from_high_half(t);
				return Uint { limbs: r };
			},
			_ => {
				let mut r = self.small_shl(shift).resize();
				let t = Limb::make_double(self.limbs[NLIMBS - 1], Limb::ZERO) << shift;
				r.limbs[NLIMBS] = Limb::from_high_half(t);
				return r;
			},
		}
	}

	/// Shifts `a` right by `shift % Limb::BITS` bits.
	/// I.e., the shift is always less than Limb::BITS.
	pub const fn small_shr(self, shift: usize) -> Self {
		let shift = shift & (Limb::BITS - 1);
		let mut r = [Limb::ZERO; NLIMBS];
		match NLIMBS {
			0 => {},
			1 => {
				r[0] = self.limbs[0].const_shr(shift);
			},
			_ => {
				let mut i = 0;
				while i < NLIMBS - 2 {
					let t = Limb::make_double(self.limbs[i], self.limbs[i + 1]) >> shift;
					r[i] = Limb::from_low_half(t);

					i += 1;
				}
				let t = Limb::make_double(self.limbs[NLIMBS - 2], self.limbs[NLIMBS - 1]) >> shift;
				r[NLIMBS - 2] = Limb::from_low_half(t);
				r[NLIMBS - 1] = Limb::from_high_half(t);
			},
		}
		Self { limbs: r }
	}

	pub const fn resize<const TO: usize>(self) -> Uint<TO>
	where
		[Limb; TO]: Default,
	{
		let mut r = Uint::zero();
		let N = const_min(NLIMBS, TO);

		let mut i = 0;
		while i < N {
			r.limbs[i] = self.limbs[i];
			i += 1;
		}

		r
	}

	pub const fn split<const LOW: usize>(self) -> (Uint<LOW>, Uint<{ NLIMBS - LOW }>)
	where
		[Limb; LOW]: Default,
		[Limb; NLIMBS - LOW]: Default,
	{
		let mut low = Uint::zero();
		let mut high = Uint::zero();

		let mut i = 0;
		while i < LOW {
			low.limbs[i] = self.limbs[i];
			i += 1;
		}
		while i < NLIMBS {
			high.limbs[i - LOW] = self.limbs[i];
			i += 1;
		}

		(low, high)
	}

	pub const fn __cmp(self, b: Self) -> (bool, bool) {
		let mut r = [Limb::ZERO; NLIMBS];
		let mut borrow = false;
		let mut zero = 0;

		let mut i = 0;
		while i < NLIMBS {
			(r[i], borrow) = Limb::subb(self.limbs[i], b.limbs[i], borrow);
			zero |= r[i].0;
			i += 1;
		}

		(borrow, zero == 0)
	}

	pub const fn eq(self, b: Self) -> bool {
		let (_, zero) = self.__cmp(b);
		zero
	}

	pub const fn ne(self, b: Self) -> bool {
		!self.eq(b)
	}

	pub const fn lt(self, b: Self) -> bool {
		match NLIMBS {
			0 => false,
			1 => self.limbs[0].0 < b.limbs[0].0,
			_ => {
				let aa = Limb::make_double(self.limbs[0], self.limbs[1]);
				let bb = Limb::make_double(b.limbs[0], b.limbs[1]);
				let mut borrow = aa < bb;

				let mut i = 2;
				while i < NLIMBS {
					(_, borrow) = Limb::subb(self.limbs[i], b.limbs[i], borrow);
					i += 1;
				}

				borrow
			},
		}
	}

	pub const fn le(self, b: Self) -> bool {
		!b.lt(self)
	}

	pub const fn gt(self, b: Self) -> bool {
		b.lt(self)
	}

	pub const fn ge(self, b: Self) -> bool {
		!self.lt(b)
	}
}

#[derive(Clone, Copy, Default, Debug)]
pub struct InvertNPlus1<const N: usize>
where
	[Limb; N]: Default,
{
	divisor: Uint<N>,
	shift: u16,
	inv: Limb,
}

impl<const N: usize> InvertNPlus1<N>
where
	[Limb; N]: Default,
	[Limb; N + 1]: Default,
	[Limb; (N + 1) + 2]: Default,
	[Limb; N + 1 - N]: Default,
	[Limb; (N + 1) + 1]: Default,
	[Limb; ((N + 1) + 1) + 1]: Default,
	[Limb; (N + 1) - 1]: Default,
	[Limb; (N + 1) + 1 - (N + 1)]: Default,
{
	pub const fn new(divisor: Uint<N>) -> Self {
		if divisor.limbs[N - 1].is_zero() {
			cold_path();
			return Self { divisor, shift: 0, inv: Limb(0) };
		}

		// For 64 bit limb, the double inverse is calculated as:
		//
		//     inv = floor((2^128 - 1) / divisor)
		//
		// If the divisor is normalized (i.e. the highest bit is set), the inverse will be 65 bits
		// wide. We only store the low 64 bits. We don't need to store bit 65 because it is always 1.

		// Left-shift the divisor to remove all leading zeros.
		// We will shift the dividend by the same amount, so the result will be unaffected.
		let shift = divisor.limbs[N - 1].leading_zeros();
		let normalized_divisor = divisor.small_shl(shift);

		const b32: usize = Limb::BITS / 2;
		const b33: usize = b32 + 1;
		const b31: usize = b32 - 1;

		// `nd33` is the normalized divisor shifted right by 31 bits.
		// So it has 31 leading zeros and its bith width is 33 bits.
		let nd33 = normalized_divisor.limbs[N - 1].const_shr(b31);

		// The dividend is `2**128 - 1`.
		let divident = Uint::<{ N + 1 }>::max();

		//==== Calculate `q1` - the highest 32 bits of the quotient ====

		// Make an estimate of `q1` based on the highest 33 bits of the normalized divisor.
		// By using more than 32 bits, we know the estimate will be at most one too large.
		unsafe { assume(nd33.is_not_zero()) };
		let nd33_inv = nd33.invert();
		let q1 = nd33_inv.mul_max();

		// correction - decrement `q1` if needed
		//
		// I.e., if `q1 * (normalized_divisor << 33) > divident`.
		// Since the divident is maximal 128-bit number, `multiplication > divident` is true
		// if the multiplication has 1 more bit.
		let m = normalized_divisor.mul_1(q1).small_shl_extend(b33);
		let (m, fix_needed) = m.split::<{ N + 1 }>();
		let q1 = q1.const_sub(fix_needed.limbs[0]);
		let m_fixed = m.wrapping_sub(normalized_divisor.small_shl_extend(b33));
		let m = select_unpredictable(fix_needed.is_not_zero(), m_fixed, m);
		let rem97 = m.not(); // this is equivalent to: `rem97 = divident - m`

		// Verify
		let nd_shl_33 = normalized_divisor.small_shl_extend(b33);
		debug_assert!(rem97.lt(nd_shl_33));
		// (normalized_divisor << 33) * q1 + rem97 == divident
		let check = nd_shl_33.mul_1(q1).add_extend(rem97.resize());
		debug_assert!(check.eq(divident.resize()));

		//==== Calculate `q2` - the next 32 bits of the quotient ====

		// We want to calculate an estimate of q2 based on 65 bits of rem97 and 33 bits of
		// normalized_divisor. First remove the 65-th bit so that the division is 64 by 33 bits.
		let (num, bit65) = rem97.small_shr(b32).split::<N>();
		let num_fixed = num.wrapping_sub(normalized_divisor);
		let num = select_unpredictable(bit65.is_not_zero(), num_fixed, num);
		let q2 = nd33_inv.mul(num.limbs[N - 1]).const_bitor(bit65.limbs[0].const_shl(b31));

		// correction - decrement `q2` if needed

		let mut nd_shl_1 = normalized_divisor.small_shl_extend(1);
		nd_shl_1.limbs[N].0 = 1; // help the optimizer figure out that the high limb is 1

		let m = nd_shl_1.mul_1(q2);
		let (rem65, fix2_needed) = rem97.sub(m.resize());
		let q2 = q2.wrapping_sub(Limb::from_bool(fix2_needed));
		let rem65_fixed = rem65.wrapping_add(nd_shl_1);
		let rem65 = select_unpredictable(fix2_needed, rem65_fixed, rem65);

		// Verify
		debug_assert!(rem65.lt(nd_shl_1));
		// (normalized_divisor << 1) * q2 + rem65 == rem97
		let check = nd_shl_1.mul_1(q2).add_extend(rem65.resize());
		debug_assert!(check.eq(rem97.resize()));

		//==== Calculate bit 0 ====

		let bit0 = rem65.ge(normalized_divisor.resize());

		//==== Put everything together ====

		let inv = q1.const_shl(b33).const_bitor(q2.const_shl(1)).const_bitor(Limb::from_bool(bit0));

		// Verify
		let check = normalized_divisor
			.mul_1(inv)
			.add_extend(normalized_divisor.small_shl_extend(b32).small_shl(b32));
		debug_assert!(check.le(divident.resize()));
		let rem = divident.wrapping_sub(check.resize());
		debug_assert!(rem.lt(normalized_divisor.resize()));

		Self { divisor, shift: shift as u16, inv }
	}

	pub const fn is_valid(&self) -> bool {
		self.divisor.is_not_zero()
	}

	#[inline(never)]
	pub const fn mul(self, a: Uint<{ N + 1 }>) -> (Uint<1>, Uint<N>) {
		// The result needs to fit into a single limb, so `a` needs to have at least
		// as many leading zeros as the divisor. So this shift should not overflow.
		let a_shifted = a.small_shl_extend(self.shift as usize);
		debug_assert!(a_shifted.limbs[N + 1].is_zero());
		let a_shifted = a_shifted.resize::<{ N + 1 }>();

		let inv = Uint::new([self.inv, Limb(1)]);

		// Multiply `a` by the inverted divisor.
		// `a` is 2 limbs and the inversion is also 2 limbs. Low limb of the inversion is `inv`.
		// High limb of the inversion is always 1 so we don't need to store it.
		//
		// In theory, the result of the multiplication will be 4 limbs `[m0, m1, m2, m3]`,
		// where the two highest limbs are the quotient: `q = [m2, m3]`.
		//
		// However, one of our preconditions is that the quotient fits in a single limb,
		// so `m3` should always be zero.
		let m = a_shifted.mul(inv);

		debug_assert!(m.limbs[N + 2].is_zero());
		let q = m.limbs[N + 1];

		// The calculated quotient may be one less than the actual quotient.
		// Calculate the remainder and if it is greater than the divisor, make a correction.

		let check = self.divisor.mul_1(q);
		let (rem, neg) = a.sub(check);
		debug_assert!(!neg);

		// make corrections if needed
		let (fixed_rem, need_fix) = rem.sub(self.divisor.resize());
		let rem = if need_fix { fixed_rem } else { rem };
		let q = q.wrapping_add(Limb::from_bool(need_fix));

		(Uint::new([q]), rem.resize())
	}
}

pub type Invert2By1 = InvertNPlus1<1>;

//--------------------------------------------------------------------------------------------------

#[inline(never)]
pub fn sub2(a: Uint<2>, b: Uint<2>) -> (Uint<2>, bool) {
	a.sub_abs(b)
}

#[inline(never)]
pub fn mul3(a: Uint<3>, b: Uint<3>) -> Uint<3> {
	a.mul(b).resize()
}

#[inline(never)]
pub fn shl3(a: Uint<3>, shift: usize) -> Uint<3> {
	a.small_shl(shift)
}

#[inline(never)]
pub fn shr3(a: Uint<3>, shift: usize) -> Uint<3> {
	a.small_shr(shift)
}

#[inline(never)]
pub fn test333(q2: Limb, normalized_divisor: Limb) -> [Limb; 2] {
	let q2_ = Uint { limbs: [q2, Limb::ZERO] };
	let nd_shl_1_ = Uint {
		limbs: [normalized_divisor.const_shl(1), Limb(1)],
	};
	let m_ = q2_.mul(nd_shl_1_);
	[m_.limbs[0], m_.limbs[1]]
}

#[inline(never)]
pub fn test_inv_new(divisor: Uint<1>) -> Invert2By1 {
	Invert2By1::new(divisor)
}

#[inline(never)]
pub fn test_inv_mul(inv: Invert2By1, a: Uint<2>) -> (Uint<1>, Uint<1>) {
	inv.mul(a)
}

#[inline(never)]
pub fn test_cmp1(a: Uint<2>, b: Uint<2>) -> bool {
	a.lt(b)
}

#[inline(never)]
pub fn test_cmp2(a: Uint<2>, b: Uint<2>) -> bool {
	let a = Limb::make_double(a.limbs[0], a.limbs[1]);
	let b = Limb::make_double(b.limbs[0], b.limbs[1]);
	a < b
}

mod tests {
	use super::*;

	use crate::testvec;

	#[test]
	fn test1() {
		let inv1 = InvertNPlus1::new(Uint::new([Limb(12157665459056928801)]));
		let inv2 = InvertNPlus1::new(Uint::new([Limb(2177953337809371136)]));
		let inv3 = InvertNPlus1::new(Uint::new([Limb(2862423051509815793)]));
		let inv4 = InvertNPlus1::new(Uint::new([Limb(1_000_000_000_000_000_000)]));
		let a = Limb::MAX;
		let b = Limb(700_000);
		let c = Limb::mul(a, b, Limb(0), Limb(0));
		let (q, r) = inv4.mul(c); //[Limb::new(a), Limb::new(b)]);
		println!("q = {:?}, r = {:?}", q, r);
	}
}
