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
	pub const fn wrapping_add(self, other: Limb) -> Limb {
		Limb(self.0.wrapping_add(other.0))
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
