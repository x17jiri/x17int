use crate::limb::Limb;
use std::intrinsics::{assume, cold_path};

#[rustfmt::skip] pub const fn const_max(a: usize, b: usize) -> usize {
	if a >= b { a } else { b }
}

#[rustfmt::skip] pub const fn const_min(a: usize, b: usize) -> usize {
	if a <= b { a } else { b }
}

#[derive(Clone, Copy)]
pub struct Uint<const NLIMBS: usize> {
	pub limbs: [Limb; NLIMBS],
}

impl<const NLIMBS: usize> Uint<NLIMBS> {
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

	pub const fn add_extend(self, b: Self) -> Uint<{ NLIMBS + 1 }> {
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

	pub const fn mul<const B: usize>(self, b: Uint<B>) -> Uint<{ NLIMBS + B }> {
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

	/// Shifts `a` left by `shift % Limb::BITS` bits.
	/// I.e., the shift is always less than Limb::BITS.
	pub const fn small_shl(self, shift: usize) -> Self {
		if NLIMBS == 0 {
			return Self::zero();
		}

		let shift = shift & (Limb::BITS - 1);
		let mut r = [Limb::ZERO; NLIMBS];
		r[0] = self.limbs[0].const_shl(shift);

		let mut i = 1;
		while i < NLIMBS {
			let t = Limb::make_double(self.limbs[i - 1], self.limbs[i]) << shift;
			r[i] = Limb::from_high_half(t);

			i += 1;
		}

		Self { limbs: r }
	}

	pub const fn small_shl_extend(self, shift: usize) -> Uint<{ NLIMBS + 1 }> {
		if NLIMBS == 0 {
			return Uint::zero();
		}

		let mut r = self.small_shl(shift).resize();
		let t = Limb::make_double(self.limbs[NLIMBS - 1], Limb::ZERO) << shift;
		r.limbs[NLIMBS] = Limb::from_high_half(t);

		r
	}

	/// Shifts `a` right by `shift % Limb::BITS` bits.
	/// I.e., the shift is always less than Limb::BITS.
	pub const fn small_shr(self, shift: usize) -> Self {
		if NLIMBS == 0 {
			return Self::zero();
		}

		let shift = shift & (Limb::BITS - 1);
		let mut r = [Limb::ZERO; NLIMBS];

		let mut i = 1;
		while i < NLIMBS {
			let t = Limb::make_double(self.limbs[i - 1], self.limbs[i]) >> shift;
			r[i - 1] = Limb::from_low_half(t);

			i += 1;
		}
		r[NLIMBS - 1] = self.limbs[NLIMBS - 1].const_shr(shift);

		Self { limbs: r }
	}

	pub const fn resize<const TO: usize>(self) -> Uint<TO> {
		let mut r = Uint::zero();
		let N = const_min(NLIMBS, TO);

		let mut i = 0;
		while i < N {
			r.limbs[i] = self.limbs[i];
			i += 1;
		}

		r
	}

	pub const fn eq(self, b: Self) -> bool {
		let mut i = 0;
		while i < NLIMBS {
			if self.limbs[i].0 != b.limbs[i].0 {
				return false;
			}
			i += 1;
		}
		true
	}

	pub const fn ne(self, b: Self) -> bool {
		!self.eq(b)
	}

	pub const fn lt(self, b: Self) -> bool {
		let mut i = NLIMBS;
		while i > 0 {
			i -= 1;
			if self.limbs[i].0 < b.limbs[i].0 {
				return true;
			}
			if self.limbs[i].0 > b.limbs[i].0 {
				return false;
			}
		}
		false
	}

	pub const fn le(self, b: Self) -> bool {
		let mut i = NLIMBS;
		while i > 0 {
			i -= 1;
			if self.limbs[i].0 < b.limbs[i].0 {
				return true;
			}
			if self.limbs[i].0 > b.limbs[i].0 {
				return false;
			}
		}
		true
	}

	pub const fn gt(self, b: Self) -> bool {
		!self.le(b)
	}

	pub const fn ge(self, b: Self) -> bool {
		!self.lt(b)
	}
}

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
