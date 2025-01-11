use crate::fixed_size;
use crate::fixed_size::Uint;
use std::intrinsics::{assume, cold_path};

pub type Value = u64;
pub type Double = u128;

#[derive(Clone, Copy, Default, PartialEq, Debug, Eq, Ord, PartialOrd)]
pub struct Limb(pub Value);

impl Limb {
	pub const BITS: usize = usize::BITS as usize;

	pub const ZERO: Limb = Self(0);
	pub const MAX: Limb = Self(Value::MAX);

	#[inline]
	pub const fn make_double(low: Limb, high: Limb) -> Double {
		assert!(std::mem::size_of::<Double>() >= 2 * std::mem::size_of::<Value>());
		(low.0 as Double) | ((high.0 as Double) << Limb::BITS)
	}

	#[inline]
	pub const fn as_double(self) -> Double {
		self.0 as Double
	}

	#[inline]
	pub const fn from_low_half(value: Double) -> Limb {
		Limb(value as Value)
	}

	#[inline]
	pub const fn from_high_half(value: Double) -> Limb {
		Limb((value >> Limb::BITS) as Value)
	}

	#[inline]
	pub const fn from_bool(value: bool) -> Limb {
		Limb(value as Value)
	}

	#[inline]
	pub const fn leading_zeros(self) -> usize {
		self.0.leading_zeros() as usize
	}

	/// Returns number of bits needed to store the value.
	/// If the value is zero, it returns 0.
	#[inline]
	pub const fn bit_width(self) -> usize {
		unsafe { Self::BITS.unchecked_sub(self.0.leading_zeros() as usize) }
	}

	/// Returns number of bits needed to store the value.
	/// If the value is zero, it may return 0 or 1 depending on what's faster to compute.
	#[inline]
	pub const fn bit_width_nonzero(self) -> usize {
		unsafe { Self::BITS.unchecked_sub((self.0 | 1).leading_zeros() as usize) }
	}

	#[inline]
	pub const fn wrapping_sub(self, other: Limb) -> Limb {
		Limb(self.0.wrapping_sub(other.0))
	}

	#[inline]
	pub const fn wrapping_neg(self) -> Limb {
		Limb(self.0.wrapping_neg())
	}

	#[inline]
	pub const fn overflowing_neg(self) -> (Limb, bool) {
		let (value, overflow) = self.0.overflowing_neg();
		(Limb(value), overflow)
	}

	#[inline]
	pub const fn overflowing_add(self, other: Limb) -> (Limb, bool) {
		let (value, overflow) = self.0.overflowing_add(other.0);
		(Limb(value), overflow)
	}

	#[inline]
	pub const fn overflowing_sub(self, other: Limb) -> (Limb, bool) {
		let (value, overflow) = self.0.overflowing_sub(other.0);
		(Limb(value), overflow)
	}

	#[inline]
	const fn __const_addc(a: Limb, b: Limb, carry: bool) -> (Limb, bool) {
		let (sum, overflow1) = a.0.overflowing_add(b.0);
		let (sum, overflow2) = sum.overflowing_add(carry as Value);
		(Limb(sum), overflow1 | overflow2)
	}

	#[inline]
	fn __nonconst_addc(a: Limb, b: Limb, carry: bool) -> (Limb, bool) {
		#[cfg(target_arch = "x86_64")]
		unsafe {
			let c_in = carry as u8;
			let a = a.0 as u64;
			let b = b.0 as u64;
			let mut out: u64 = 0;
			let c_out = std::arch::x86_64::_addcarry_u64(c_in, a, b, &mut out);
			return (Limb(out as Value), c_out != 0);
		}

		#[cfg(not(target_arch = "x86_64"))]
		{
			Self::__const_addc(a, b, carry)
		}
	}

	/// Returns:
	///     (value, carry)
	/// Where:
	///     value = (a + b + carry) % 2**BITS
	///     carry = (a + b + carry) > MAX
	#[inline]
	pub const fn addc(a: Limb, b: Limb, carry: bool) -> (Limb, bool) {
		//return Self::__const_addc(a, b, carry);
		std::intrinsics::const_eval_select(
			(a, b, carry), //
			Self::__const_addc,
			Self::__nonconst_addc,
		)
	}

	#[inline]
	const fn __const_subb(a: Limb, b: Limb, borrow: bool) -> (Limb, bool) {
		let (diff, borrow1) = a.0.overflowing_sub(b.0);
		let (diff, borrow2) = diff.overflowing_sub(borrow as Value);
		(Limb(diff), borrow1 | borrow2)
	}

	#[inline]
	fn __nonconst_subb(a: Limb, b: Limb, borrow: bool) -> (Limb, bool) {
		#[cfg(target_arch = "x86_64")]
		unsafe {
			let c_in = borrow as u8;
			let a = a.0 as u64;
			let b = b.0 as u64;
			let mut out: u64 = 0;
			let c_out = std::arch::x86_64::_subborrow_u64(c_in, a, b, &mut out);
			return (Limb(out as Value), c_out != 0);
		}

		#[cfg(not(target_arch = "x86_64"))]
		{
			Self::__const_subb(a, b, borrow)
		}
	}

	/// Returns:
	///     (value, borrow)
	/// Where:
	///     value = (a - b - borrow) % 2**BITS
	///     borrow = (a - b - borrow) < 0
	#[inline]
	pub const fn subb(a: Limb, b: Limb, borrow: bool) -> (Limb, bool) {
		//return Self::__const_subb(a, b, borrow);
		std::intrinsics::const_eval_select(
			(a, b, borrow),
			Self::__const_subb,
			Self::__nonconst_subb,
		)
	}

	/// Returns:
	///     MAX / self
	/// Preconditions:
	///     self != 0
	#[inline]
	pub const fn invert(self) -> LimbInversion {
		LimbInversion::new(self)
	}

	/// Returns:
	///     [low, high]
	/// Where:
	///     big_value = a * b + c + d
	///     low = big_value % 2**BITS
	///     high = big_value / 2**BITS
	#[inline]
	pub const fn mul(a: Limb, b: Limb, c: Limb, d: Limb) -> [Limb; 2] {
		let t = (a.0 as Double) * (b.0 as Double) + (c.0 as Double) + (d.0 as Double);
		[Limb(t as Value), Limb((t >> Limb::BITS) as Value)]
	}

	// following functions should be removed when rust supports const traits

	#[inline]
	pub const fn is_zero(self) -> bool {
		self.0 == 0
	}

	#[inline]
	pub const fn is_not_zero(self) -> bool {
		self.0 != 0
	}

	#[inline]
	pub const fn const_shl(self, rhs: usize) -> Limb {
		Limb(self.0 << rhs)
	}

	#[inline]
	pub const fn const_shr(self, rhs: usize) -> Limb {
		Limb(self.0 >> rhs)
	}

	#[inline]
	pub const fn const_mul(self, rhs: Limb) -> Double {
		(self.0 as Double) * (rhs.0 as Double)
	}

	#[inline]
	pub const fn const_sub(self, rhs: Limb) -> Limb {
		Limb(self.0 - rhs.0)
	}

	#[inline]
	pub const fn const_bitor(self, rhs: Limb) -> Limb {
		Limb(self.0 | rhs.0)
	}

	#[inline]
	pub const fn const_eq(self, rhs: Limb) -> bool {
		self.0 == rhs.0
	}

	#[inline]
	pub const fn const_inc(self) -> Limb {
		Limb(self.0 + 1)
	}

	#[inline]
	pub const fn const_bitnot(self) -> Limb {
		Limb(!self.0)
	}

	#[inline]
	pub const fn const_bitand(self, rhs: Limb) -> Limb {
		Limb(self.0 & rhs.0)
	}
}

impl std::ops::Not for Limb {
	type Output = Self;

	#[inline]
	fn not(self) -> Self {
		Self(!self.0)
	}
}

impl std::ops::BitXor for Limb {
	type Output = Self;

	#[inline]
	fn bitxor(self, rhs: Self) -> Self {
		Self(self.0 ^ rhs.0)
	}
}

impl std::ops::BitAnd for Limb {
	type Output = Self;

	#[inline]
	fn bitand(self, rhs: Self) -> Self {
		Self(self.0 & rhs.0)
	}
}

impl std::ops::BitOr for Limb {
	type Output = Self;

	#[inline]
	fn bitor(self, rhs: Self) -> Self {
		Self(self.0 | rhs.0)
	}
}

impl std::ops::Shl<usize> for Limb {
	type Output = Self;

	#[inline]
	fn shl(self, rhs: usize) -> Self {
		Self(self.0 << rhs)
	}
}

impl std::ops::Shr<usize> for Limb {
	type Output = Self;

	#[inline]
	fn shr(self, rhs: usize) -> Self {
		Self(self.0 >> rhs)
	}
}

impl const std::ops::Add for Limb {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self {
		Self(self.0 + rhs.0)
	}
}

impl std::ops::Sub for Limb {
	type Output = Self;

	#[inline]
	fn sub(self, rhs: Self) -> Self {
		Self(self.0 - rhs.0)
	}
}

impl std::ops::Mul for Limb {
	type Output = Double;

	#[inline]
	fn mul(self, rhs: Self) -> Double {
		(self.0 as Double) * (rhs.0 as Double)
	}
}

impl std::cmp::PartialEq<Value> for Limb {
	#[inline]
	fn eq(&self, other: &Value) -> bool {
		self.0 == *other
	}
}

#[derive(Clone, Copy, Default, PartialEq, Debug, Eq, Ord, PartialOrd)]
pub struct LimbInversion {
	divisor: Value,
	inversion: Value,
}

impl LimbInversion {
	#[inline]
	pub const fn new(divisor: Limb) -> LimbInversion {
		LimbInversion {
			divisor: divisor.0,
			inversion: Limb::MAX.0.div_floor(divisor.0),
		}
	}

	#[inline]
	pub const fn divisor(self) -> Limb {
		Limb(self.divisor)
	}

	/// Multiplies `a` by the inverse of the divisor.
	/// This is equivalent to dividing `a` by the divisor.
	///
	///     a / self.divisor
	#[inline]
	pub const fn mul(self, a: Limb) -> Limb {
		let quot = (((a.0 as Double) * (self.inversion as Double)) >> Limb::BITS) as Value;
		let mul = quot * self.divisor;
		let rem = a.0 - mul;

		let quot = if rem >= self.divisor { quot + 1 } else { quot };

		debug_assert!(quot == a.0.div_floor(self.divisor));
		Limb(quot)
	}

	/// Calculates:
	///
	///     (a / self.divisor, a % self.divisor)
	#[inline]
	pub const fn qr(self, a: Limb) -> (Limb, Limb) {
		let quot = (((a.0 as Double) * (self.inversion as Double)) >> Limb::BITS) as Value;
		let mul = quot * self.divisor;
		let rem = a.0 - mul;

		let quot = if rem >= self.divisor { quot + 1 } else { quot };
		let rem = if rem >= self.divisor { rem - self.divisor } else { rem };

		debug_assert!(quot == a.0.div_floor(self.divisor));
		debug_assert!(quot * self.divisor + rem == a.0);
		(Limb(quot), Limb(rem))
	}

	/// Multiplies `Limb::MAX` by the inverse of the divisor.
	/// This is equivalent to dividing `Limb::MAX` by the divisor.
	///
	///     Limb::MAX / self.divisor
	///
	/// Since the inverse is based on dividing `Limb::MAX / divisor`,
	/// this operation is faster than `self.mul(Limb::MAX)`.
	#[inline]
	pub const fn mul_max(self) -> Limb {
		Limb(self.inversion)
	}
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Invert2By1 {
	divisor: Limb,
	shift: u16,
	inv: Limb,
}

impl Invert2By1 {
	pub const fn const_new(divisor: Limb) -> Self {
		let check = Self::new(divisor);

		if divisor.is_zero() {
			cold_path();
			return Self { divisor, shift: 0, inv: Limb(0) };
		}

		let shift = divisor.leading_zeros();
		let normalized_divisor = divisor.const_shl(shift);
		unsafe { assume(normalized_divisor.is_not_zero()) };

		let divident = Limb::make_double(Limb::MAX, Limb::MAX);
		let inverse = divident.div_floor(normalized_divisor.as_double());

		debug_assert!(divisor.const_eq(check.divisor));
		debug_assert!(shift == check.shift as usize);
		debug_assert!(Limb::from_low_half(inverse).const_eq(check.inv));

		Self {
			divisor,
			shift: shift as u16,
			inv: Limb::from_low_half(inverse),
		}
	}

	pub const fn new(divisor: Limb) -> Self {
		const N: usize = 1;
		let divisor = Uint::new([divisor]);

		if divisor.limbs[N - 1].is_zero() {
			cold_path();
			return Self {
				divisor: divisor.limbs[0], // TODO - return whole array
				shift: 0,
				inv: Limb(0),
			};
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

		// `nd33` is the normalized divisor shifted right by 31 bits.
		// So it has 31 leading zeros and its bith width is 33 bits.
		let nd33 = normalized_divisor.limbs[N - 1].const_shr(31);

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
		let m: Uint<{ N + 1 }> = normalized_divisor.mul(Uint::new([q1]));
		let fix_needed = m.limbs[N].const_shr(31); // 1 if we need a fix, 0 otherwise
		let q1 = q1.const_sub(fix_needed);
		let (m, _) = m.sub(normalized_divisor.mask(fix_needed.wrapping_neg()).resize());
		let rem97 = m.small_shl(33).not(); // this is equivalent to: `rem97 = divident - (m << 33)`

		// Verify
		let check1 = normalized_divisor.small_shl_extend(33);
		debug_assert!(rem97.lt(check1));
		// (normalized_divisor << 33) * q1 + rem97 == divident
		let check2 = check1.mul(Uint::new([q1])).add_extend(rem97.resize());
		debug_assert!(check2.eq(divident.resize()));

		//==== Calculate `q2` - the next 32 bits of the quotient ====

		// We want to calculate an estimate of q2 based on 65 bits of rem97 and 33 bits of
		// normalized_divisor. First remove the 65-th bit so that the division is 64 by 33 bits.
		let num = rem97.small_shr(32);
		let bit65 = num.limbs[N];
		let (num, _) = num.resize::<N>().sub(normalized_divisor.mask(bit65.wrapping_neg()));
		let q2 = nd33_inv.mul(num.limbs[N - 1]).const_bitor(bit65.const_shl(31));

		// correction - decrement `q2` if needed

		let mut nd_shl_1 = normalized_divisor.small_shl_extend(1);
		nd_shl_1.limbs[N].0 = 1;

		let m = nd_shl_1.mul(Uint::new([q2]));
		let (rem65, fix2_needed) = rem97.sub(m.resize());
		let q2 = q2.wrapping_sub(Limb::from_bool(fix2_needed));
		let (rem65, _) = rem65.add(if fix2_needed { nd_shl_1 } else { Uint::zero() });

		// Verify
		/*		let expected_q2 = rem97 / (normalized_divisor.as_double() << 1);
		let expected_rem65 = rem97 % (normalized_divisor.as_double() << 1);
		debug_assert!(q2.as_double() == expected_q2);
		debug_assert!(rem65 == expected_rem65);*/

		//==== Calculate bit 0 ====

		let bit0 = rem65.ge(normalized_divisor.resize());

		//==== Put everything together ====

		let inv = q1.const_shl(33).const_bitor(q2.const_shl(1)).const_bitor(Limb::from_bool(bit0));

		// Verify
		/*		let expected_inv = divident / normalized_divisor.as_double();
		debug_assert!(expected_inv >> Limb::BITS == 1);
		debug_assert!(Limb::from_low_half(expected_inv).const_eq(inv));*/

		Self {
			divisor: divisor.limbs[0],
			shift: shift as u16,
			inv,
		}
	}

	pub const fn is_valid(&self) -> bool {
		self.divisor.is_not_zero()
	}

	#[inline(never)]
	pub const fn mul(self, a: [Limb; 2]) -> (Limb, Limb) {
		let a = Limb::make_double(a[0], a[1]);

		// Help the compiler figure out that `shift < Limb::BITS`
		let shift = (self.shift as usize) & (Limb::BITS - 1);

		// The result needs to fit into a single limb, so `a` needs to have at least
		// as many leading zeros as the divisor. So this shift should not overflow.
		let a_shifted = a << shift;
		debug_assert!(a >> (2 * Limb::BITS - shift) == 0);

		let x0 = Limb::from_low_half(a_shifted);
		let x1 = Limb::from_high_half(a_shifted);

		let inv = self.inv;
		let zero = Limb(0);

		// Multiply `a` by the inverted divisor.
		// `a` is 2 limbs and the inversion is also 2 limbs. Low limb of the inversion is `inv`.
		// High limb of the inversion is always 1 so we don't need to store it.
		//
		// In theory, the result of the multiplication will be 4 limbs `[m0, m1, m2, m3]`,
		// where the two highest limbs are the quotient: `q = [m2, m3]`.
		//
		// However, one of our preconditions is that the quotient fits in a single limb,
		// so `m3` should always be zero.

		let [_m0, c1] = Limb::mul(x0, inv, zero, zero);
		let [_m1, c2] = Limb::mul(x1, inv, x0, c1);
		let (m2, m3) = Limb::addc(x1, c2, false);

		debug_assert!(!m3);
		let q = m2;

		// The calculated quotient may be one less than the actual quotient.
		// Calculate the remainder and if it is greater than the divisor, make a correction.

		let check = q.const_mul(self.divisor);
		let rem = a - check;
		let need_correction = rem >= self.divisor.as_double();

		// make corrections if needed
		let rem = if need_correction { rem - self.divisor.as_double() } else { rem };
		let q = if need_correction { q.const_inc() } else { q };

		(q, Limb::from_low_half(rem))
	}
}

mod tests {
	use super::*;

	use crate::testvec;

	#[test]
	fn test1() {
		let inv = Invert2By1::const_new(Limb(12157665459056928801));
		let inv = Invert2By1::const_new(Limb(2177953337809371136));
		let inv = Invert2By1::const_new(Limb(2862423051509815793));
		let inv = Invert2By1::const_new(Limb(1_000_000_000_000_000_000));
		let a = Limb::MAX;
		let b = Limb(700_000);
		let c = Limb::mul(a, b, Limb(0), Limb(0));
		let (q, r) = inv.mul(c); //[Limb::new(a), Limb::new(b)]);
		println!("q = {:?}, r = {:?}", q, r);
	}
}
