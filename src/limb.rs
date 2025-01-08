pub type Value = usize;
pub type Double = u128;
const OK: () = assert!(std::mem::size_of::<Double>() >= 2 * std::mem::size_of::<Value>());

#[derive(Clone, Copy, Default, PartialEq, Debug, Eq, Ord, PartialOrd)]
pub struct Limb(pub Value);

impl Limb {
	pub const BITS: usize = usize::BITS as usize;

	pub const ZERO: Limb = Self(0);
	pub const ONE: Limb = Self(1);
	pub const MAX: Limb = Self(usize::MAX);

	#[inline]
	pub const fn is_zero(self) -> bool {
		self.0 == 0
	}

	#[inline]
	pub const fn is_not_zero(self) -> bool {
		self.0 != 0
	}

	/// Returns number of bits needed to store the value.
	/// If the value is zero, it returns 1.
	#[inline]
	pub const fn bit_width(self) -> usize {
		unsafe { Self::BITS.unchecked_sub((self.0 | 1).leading_zeros() as usize) }
	}

	/// Returns:
	///     (value, carry)
	/// Where:
	///     value = (a + b + carry) % 2**BITS
	///     carry = (a + b + carry) > MAX
	#[inline]
	pub const fn addc(a: Limb, b: Limb, carry: bool) -> (Limb, bool) {
		let (sum, overflow1) = a.0.overflowing_add(b.0);
		let (sum, overflow2) = sum.overflowing_add(carry as usize);
		(Limb(sum), overflow1 | overflow2)
	}

	/// Returns:
	///     (value, borrow)
	/// Where:
	///     value = (a - b - borrow) % 2**BITS
	///     borrow = (a - b - borrow) < 0
	#[inline]
	pub const fn subb(a: Limb, b: Limb, borrow: bool) -> (Limb, bool) {
		let (diff, borrow1) = a.0.overflowing_sub(b.0);
		let (diff, borrow2) = diff.overflowing_sub(borrow as usize);
		(Limb(diff), borrow1 | borrow2)
	}

	/// Returns:
	///     MAX / self
	/// Preconditions:
	///     self != 0
	#[inline]
	pub const fn invert(self) -> LimbInversion {
		LimbInversion {
			divisor: self.0,
			inversion: Self::MAX.0.div_floor(self.0),
		}
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

#[derive(Clone, Copy, Default, PartialEq, Debug, Eq, Ord, PartialOrd)]
pub struct LimbInversion {
	divisor: Value,
	inversion: Value,
}

impl LimbInversion {
	#[inline]
	pub fn q(self, a: Limb) -> Limb {
		let quot = (((a.0 as Double) * (self.inversion as Double)) >> Limb::BITS) as Value;
		let mul = quot * self.divisor;
		let rem = a.0 - mul;

		let quot = if rem >= self.divisor { quot + 1 } else { quot };
		Limb(quot)
	}

	#[inline]
	pub fn qr(self, a: Limb) -> (Limb, Limb) {
		let quot = (((a.0 as Double) * (self.inversion as Double)) >> Limb::BITS) as Value;
		let mul = quot * self.divisor;
		let rem = a.0 - mul;

		let quot = if rem >= self.divisor { quot + 1 } else { quot };
		let rem = if rem >= self.divisor { rem - self.divisor } else { rem };
		(Limb(quot), Limb(rem))
	}
}
