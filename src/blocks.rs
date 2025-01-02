use std::intrinsics::{assume, cold_path, select_unpredictable};

#[derive(Clone, Copy, Default, PartialEq, Debug, Eq, Ord, PartialOrd)]
pub struct Limb {
	pub val: usize,
}

impl Limb {
	pub type Value = usize;
	pub type DoubleValue = u128;

	const OK: () =
		assert!(std::mem::size_of::<Self::DoubleValue>() >= 2 * std::mem::size_of::<Self::Value>());

	pub const BITS: usize = usize::BITS as usize;

	pub const ZERO: Limb = Self { val: 0 };
	pub const ONE: Limb = Self { val: 1 };
	pub const MAX: Limb = Self { val: Limb::Value::MAX };

	#[inline]
	pub const fn new(val: usize) -> Self {
		let _ = Self::OK;
		Self { val }
	}

	#[inline]
	pub const fn is_zero(&self) -> bool {
		self.val == 0
	}

	#[inline]
	pub const fn is_not_zero(&self) -> bool {
		self.val != 0
	}

	#[inline]
	pub const fn bit_width(self) -> usize {
		unsafe { Self::BITS.unchecked_sub((self.val | 1).leading_zeros() as usize) }
	}

	#[inline]
	pub const fn bit_neg(self) -> Self {
		Self { val: !self.val }
	}

	#[inline]
	pub const fn addc(a: Limb, b: Limb, c: bool) -> (Limb, bool) {
		let (sum, overflow1) = a.val.overflowing_add(b.val);
		let (sum, overflow2) = sum.overflowing_add(c as usize);
		(Limb { val: sum }, overflow1 | overflow2)
	}

	#[inline]
	pub fn sum<const N: usize>(a: [Limb; N]) -> [Limb; 2] {
		type V = Limb::Value;
		type D = Limb::DoubleValue;
		let mut t = a[0].val as D;
		for i in 1..N {
			t += a[i].val as D;
		}
		[Limb { val: t as V }, Limb { val: (t >> Limb::BITS) as V }]
	}

	#[inline]
	pub const fn subb(a: Limb, b: Limb, borrow: bool) -> (Limb, bool) {
		let (diff, borrow1) = a.val.overflowing_sub(b.val);
		let (diff, borrow2) = diff.overflowing_sub(borrow as usize);
		(Limb { val: diff }, borrow1 | borrow2)
	}

	// result = a * b + c + d
	#[inline]
	pub const fn mul(a: Limb, b: Limb, c: Limb, d: Limb) -> [Limb; 2] {
		type V = Limb::Value;
		type D = Limb::DoubleValue;
		let t = (a.val as D) * (b.val as D) + (c.val as D) + (d.val as D);
		[Limb { val: t as V }, Limb { val: (t >> Limb::BITS) as V }]
	}
}

pub struct LimbInv {
	divisor: Limb,
	inv: [Limb; 2],
}

impl LimbInv {
	/*
	/// Preconditions:
	/// ```rust
	///     divisor != 0
	/// ```
		*/
	#[inline(never)]
	pub const fn new(divisor: Limb) -> Self {
		if divisor.is_zero() {
			cold_path();
			return Self { divisor, inv: [Limb::ZERO, Limb::ZERO] };
		}

		// Let's assume we have 64-bit limbs.
		//
		// We don't want to do any corrections after the division, so the inverse needs to be
		// at least 128 bits wide.
		//
		// So, we will be computing `inverse = ceil(2^128 / limb)`.
		//
		// However, `2^128` is not representable in 128 bits and Rust doesn't have larger numbers.
		// So instead we will compute `inverse = floor((2^128 - 1) / limb) + 1`.
		//
		// This should work for all values > 1.
		//   - when `limb` is NOT an exact divisor of `2^128`:
		//     - `floor(2^128 / limb) == floor((2^128 - 1) / limb)`
		//     - `2^128 / limb` always has a fractional part, so `ceil == floor + 1`
		//   - when `limb` IS an exact divisor of `2^128`:
		//     - `floor(2^128 / limb) == floor((2^128 - 1) / limb) + 1`
		//     - `2^128 / limb` has no fractional part, so `ceil == floor`

		type V = Limb::Value;
		type D = Limb::DoubleValue;
		const FF: D = Limb::MAX.val as D;
		const FF_FF: D = (FF << Limb::BITS) | FF;

		let inv = FF_FF / (divisor.val as D);

		Self {
			divisor,
			inv: [Limb { val: inv as V }, Limb { val: (inv >> Limb::BITS) as V }],
		}
	}

	#[inline(never)]
	pub const fn mul(self, a: [Limb; 2]) -> ([Limb; 2], Limb) {
		type V = Limb::Value;
		type D = Limb::DoubleValue;

		let inv = self.inv;
		let zero = Limb::ZERO;

		// Multiply `a` by `inv`. The result will be 4 limbs `[n0, n1, n2, n3]`.
		//
		// The two highest limbs are the quotient: `q = [n2, n3]`.

		let [m0, m1] = Limb::mul(a[0], inv[0], zero, zero);
		let [m1, m2] = Limb::mul(a[0], inv[1], m1, zero);

		let _n0 = m0;
		let [_n1, n2] = Limb::mul(a[1], inv[0], zero, m1);
		let [n2, n3] = Limb::mul(a[1], inv[1], n2, m2);

		let q = ((n3.val as D) << Limb::BITS) | (n2.val as D);

		// The calculated quotient may be one less than the actual quotient.
		// Calculate the remainder and if it is greater than the divisor, make a correction.

		let a = ((a[1].val as D) << Limb::BITS) | (a[0].val as D);
		let check = q.wrapping_mul(self.divisor.val as D);
		let rem = a.wrapping_sub(check);

		let correction = rem >= (self.divisor.val as D);

		// correct the remainder if needed
		let original_rem = rem as V;
		let corrected_rem = original_rem.wrapping_sub(self.divisor.val);
		let rem = if correction { corrected_rem } else { original_rem };

		// correct the quotient if needed
		let (q0, carry) = Limb::addc(n2, zero, correction);
		let (q1, _) = Limb::addc(n3, zero, carry);

		([q0, q1], Limb { val: rem })
	}
}

/*impl std::ops::Mul<LimbInv> for Limb {
	type Output = Limb;

	#[inline]
	fn mul(self, rhs: LimbInv) -> Limb {
		rhs.mul(self)
	}
}*/

#[inline]
fn has_overlap(a: *const Limb, a_len: usize, b: *const Limb, b_len: usize) -> bool {
	let a_end = unsafe { a.add(a_len) };
	let b_end = unsafe { b.add(b_len) };
	a_end > b && b_end > a
}

#[inline]
fn has_no_overlap(a: *const Limb, a_len: usize, b: *const Limb, b_len: usize) -> bool {
	!has_overlap(a, a_len, b, b_len)
}

//--------------------------------------------------------------------------------------------------
// bit_width

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - a[0..<n] is a valid slice
/// - if n > 0, then a[n - 1] is non-zero
///
/// If the highest limb is zero, the result is as if the limb was 1. This is
/// technically incorrect, but the behavior is defined.
#[inline]
pub unsafe fn bit_width_unchecked(a: *const Limb, n: usize) -> usize {
	if n == 0 {
		0
	} else {
		unsafe {
			n.unchecked_mul(Limb::BITS)
				.unchecked_add(a.add(n - 1).read().bit_width())
				.unchecked_sub(Limb::BITS)
		}
	}
}

//--------------------------------------------------------------------------------------------------
// numcpy

/// rp[i..<n] = ap[i..<n]
///
/// Preconditions:
/// - i <= n
/// - rp[0..<n] and ap[0..<n] are valid slices
#[inline]
pub unsafe fn numcpy_unchecked(mut rp: *mut Limb, re: *mut Limb, mut ap: *const Limb) {
	unsafe {
		if rp == re {
			return;
		}

		let d = re.sub_ptr(rp);
		if d < 4 {
			rp.write(ap.add(0).read());
			rp = rp.add(1);
			ap = ap.add(1);
			if rp == re {
				return;
			}

			rp.write(ap.add(0).read());
			rp = rp.add(1);
			ap = ap.add(1);
			if rp == re {
				return;
			}

			rp.write(ap.add(0).read());
		} else {
			std::ptr::copy_nonoverlapping(ap, rp, d);
		}
	}
}

#[inline]
pub unsafe fn numcpy_unchecked_i_n(mut rp: *mut Limb, mut ap: *const Limb, i: usize, n: usize) {
	unsafe {
		let re = rp.add(n);
		rp = rp.add(i);
		ap = ap.add(i);
		numcpy_unchecked(rp, re, ap);
	}
}

#[inline]
pub unsafe fn negcpy_unchecked(mut rp: *mut Limb, re: *mut Limb, mut ap: *const Limb) {
	unsafe {
		while rp != re {
			rp.write(ap.read().bit_neg());
			rp = rp.add(1);
			ap = ap.add(1);
		}
	}
}

//--------------------------------------------------------------------------------------------------
// trim

#[inline]
pub unsafe fn trim_unchecked(p: *const Limb, i: usize, mut n: usize) -> usize {
	debug_assert!(i <= n);
	unsafe {
		if i != n {
			while p.add(n.unchecked_sub(1)).read().is_zero() {
				n = n.unchecked_sub(1);
				if n == i {
					break;
				}
			}
		}
		n
	}
}

/// This is like `trim_unchecked()`, but it assumes that the highest limb is probably non-zero and
/// so it is unlikely that any actual trimming will be needed.
#[inline]
pub unsafe fn cold_trim_unchecked(p: *const Limb, i: usize, n: usize) -> usize {
	debug_assert!(i <= n);
	unsafe {
		if i != n {
			if p.add(n.unchecked_sub(1)).read().is_not_zero() {
				n
			} else {
				cold_path();
				trim_unchecked(p, i, n.unchecked_sub(1))
			}
		} else {
			n
		}
	}
}

//--------------------------------------------------------------------------------------------------
// add

pub unsafe fn add_n_unchecked(
	rp: *mut Limb, ap: *const Limb, bp: *const Limb, i: usize, n: usize,
) -> bool {
	unsafe {
		let re = rp.add(n);
		let mut rp = rp.add(i);
		let mut ap = ap.add(i);
		let mut bp = bp.add(i);

		let mut carry = false;
		while rp != re {
			let (sum, overflow) = Limb::addc(ap.read(), bp.read(), carry);
			rp.write(sum);
			carry = overflow;

			rp = rp.add(1);
			ap = ap.add(1);
			bp = bp.add(1);
		}

		carry
	}
}

pub unsafe fn add_carry_unchecked(
	rp: *mut Limb, ap: *const Limb, carry: bool, i: usize, n: usize,
) -> bool {
	debug_assert!(has_no_overlap(rp, n, ap, n));
	unsafe {
		let re = rp.add(n);
		let mut rp = rp.add(i);
		let mut ap = ap.add(i);

		let mut carry = carry;
		while carry {
			if rp == re {
				return true;
			}

			let (sum, overflow) = Limb::addc(ap.read(), Limb::ZERO, carry);
			rp.write(sum);
			carry = overflow;

			rp = rp.add(1);
			ap = ap.add(1);
		}

		numcpy_unchecked(rp, re, ap);
		return false;
	}
}

pub unsafe fn neg_add_carry_unchecked(
	rp: *mut Limb, ap: *const Limb, carry: bool, i: usize, n: usize,
) -> bool {
	debug_assert!(has_no_overlap(rp, n, ap, n));
	unsafe {
		let re = rp.add(n);
		let mut rp = rp.add(i);
		let mut ap = ap.add(i);

		let mut carry = carry;
		while carry {
			if rp == re {
				return true;
			}

			let (sum, overflow) = Limb::addc(ap.add(i).read().bit_neg(), Limb::ZERO, carry);
			rp.add(i).write(sum);
			carry = overflow;

			rp = rp.add(1);
			ap = ap.add(1);
		}

		negcpy_unchecked(rp, re, ap);
		false
	}
}

pub unsafe fn add_1_unchecked(rp: *mut Limb, ap: *const Limb, b: Limb, i: usize, n: usize) -> bool {
	debug_assert!(i < n);
	debug_assert!(has_no_overlap(rp, n, ap, n));
	unsafe {
		let a = ap.add(i).read();
		let (sum, carry) = Limb::addc(a, b, false);
		rp.add(i).write(sum);
		add_carry_unchecked(rp, ap, carry, i.unchecked_add(1), n)
	}
}

pub unsafe fn add_carry_unchecked_(rp: *mut Limb, re: *mut Limb, carry: bool) -> bool {
	unsafe {
		let mut rp = rp;
		let mut carry = carry;
		while carry {
			if rp == re {
				return true;
			}

			let (sum, overflow) = Limb::addc(rp.read(), Limb::ZERO, carry);
			rp.write(sum);
			carry = overflow;

			rp = rp.add(1);
		}

		false
	}
}

/// In-place version of `add_1_unchecked()`.
pub unsafe fn add_1_unchecked_(rp: *mut Limb, re: *mut Limb, b: Limb) -> bool {
	debug_assert!(rp < re);
	unsafe {
		let (mut sum, mut carry) = Limb::addc(rp.read(), b, false);
		add_carry_unchecked_(rp.add(1), re, carry)
	}
}

//--------------------------------------------------------------------------------------------------
// sub

pub unsafe fn sub_n_unchecked(
	rp: *mut Limb, ap: *const Limb, bp: *const Limb, i: usize, n: usize,
) -> bool {
	debug_assert!(has_no_overlap(rp, n, ap, n));
	debug_assert!(has_no_overlap(rp, n, bp, n));
	unsafe {
		let re = rp.add(n);
		let mut rp = rp.add(i);
		let mut ap = ap.add(i);
		let mut bp = bp.add(i);

		let mut borrow = false;
		while rp != re {
			let (diff, underflow) = Limb::subb(ap.read(), bp.read(), borrow);
			rp.write(diff);
			borrow = underflow;

			rp = rp.add(1);
			ap = ap.add(1);
			bp = bp.add(1);
		}

		borrow
	}
}

pub unsafe fn sub_borrow_unchecked(
	rp: *mut Limb, ap: *const Limb, borrow: bool, i: usize, n: usize,
) -> bool {
	debug_assert!(has_no_overlap(rp, n, ap, n));
	unsafe {
		let re = rp.add(n);
		let mut rp = rp.add(i);
		let mut ap = ap.add(i);

		let mut borrow = borrow;
		while rp != re {
			if !borrow {
				numcpy_unchecked(rp, re, ap);
				break;
			}

			let (diff, underflow) = Limb::subb(ap.read(), Limb::ZERO, borrow);
			rp.write(diff);
			borrow = underflow;

			rp = rp.add(1);
			ap = ap.add(1);
		}

		borrow
	}
}

pub unsafe fn sub_1_unchecked(rp: *mut Limb, ap: *const Limb, b: Limb, i: usize, n: usize) -> bool {
	debug_assert!(i < n);
	debug_assert!(has_no_overlap(rp, n, ap, n));
	unsafe {
		let a = ap.add(i).read();
		let (d, borrow) = Limb::subb(a, b, false);
		rp.add(i).write(d);
		sub_borrow_unchecked(rp, ap, borrow, i.unchecked_add(1), n)
	}
}

//--------------------------------------------------------------------------------------------------
// mul

/// Preconditions:
/// - n > 0
/// - a.len() > 0
#[inline(never)]
pub unsafe fn mul_1_unchecked(
	rp: *mut Limb, re: *mut Limb, ap: *const Limb, b: Limb, c: Limb,
) -> Limb {
	unsafe {
		let mut rp = rp;
		let mut ap = ap;

		let mut carry = c;
		while rp != re {
			let [lo, hi] = Limb::mul(ap.read(), b, carry, Limb::ZERO);
			rp.write(lo);
			carry = hi;

			rp = rp.add(1);
			ap = ap.add(1);
		}
		carry
	}
}

/// In-place version of `mul_1_unchecked()`.
#[inline(never)]
pub unsafe fn mul_1_unchecked_(rp: *mut Limb, re: *mut Limb, b: Limb, c: Limb) -> Limb {
	unsafe {
		let mut rp = rp;

		let mut carry = c;
		while rp != re {
			let [lo, hi] = Limb::mul(rp.read(), b, carry, Limb::ZERO);
			rp.write(lo);
			carry = hi;

			rp = rp.add(1);
		}
		carry
	}
}

#[inline]
pub fn mul_1_(r: &mut [Limb], b: Limb, c: Limb) -> Limb {
	unsafe { mul_1_unchecked_(r.as_mut_ptr(), r.as_mut_ptr().add(r.len()), b, c) }
}

#[inline(never)]
pub unsafe fn addmul_1_unchecked(rp: *mut Limb, re: *mut Limb, ap: *const Limb, b: Limb) -> Limb {
	unsafe {
		let mut rp = rp;
		let mut ap = ap;

		let mut carry = Limb::ZERO;
		while rp != re {
			let [lo, hi] = Limb::mul(ap.read(), b, carry, rp.read());
			rp.write(lo);
			carry = hi;

			rp = rp.add(1);
			ap = ap.add(1);
		}
		carry
	}
}

//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
	use super::*;

	use crate::testvec;

	/*
	#[test]
	fn test_bit_width() {
		unsafe {
			let a = testvec![0x12345678];
			assert_eq!(bit_width_unchecked(a.as_ptr(), a.len()), 29);

			let a = testvec![0];
			assert_eq!(bit_width_unchecked(a.as_ptr(), a.len()), 1);

			let a = testvec![0, 0];
			assert_eq!(bit_width_unchecked(a.as_ptr(), a.len()), Limb::BITS + 1);

			let a = testvec![0, Limb::MAX];
			assert_eq!(bit_width_unchecked(a.as_ptr(), a.len()), 2 * Limb::BITS);

			let a = testvec![111, Limb::MAX, 0x12345678];
			assert_eq!(bit_width_unchecked(a.as_ptr(), a.len()), 2 * Limb::BITS + 29);
		}
	}

	#[test]
	fn test_numcpy() {
		unsafe {
			let a = testvec![15, 17, 19, 21, 23, 25, 27, 29, 31, 33];
			let mut r = testvec![0, 0, 0];
			numcpy_unchecked_i_n(r.as_mut_ptr(), a.as_ptr(), 1, 2);
			assert_eq!(r, testvec![0, 17, 0]);

			let mut r = testvec![1, 2, 3];
			numcpy_unchecked_i_n(r.as_mut_ptr(), a.as_ptr(), 0, 2);
			assert_eq!(r, testvec![15, 17, 3]);

			let mut r = testvec![1, 2, 3, 4, 5, 6, 7, 8];
			numcpy_unchecked_i_n(r.as_mut_ptr(), a.as_ptr(), 0, 3);
			assert_eq!(r, testvec![15, 17, 19, 4, 5, 6, 7, 8]);

			let mut r = testvec![1, 2, 3, 4, 5, 6, 7, 18];
			numcpy_unchecked_i_n(r.as_mut_ptr(), a.as_ptr(), 0, 4);
			assert_eq!(r, testvec![15, 17, 19, 21, 5, 6, 7, 18]);

			let mut r = testvec![1, 2, 3, 4, 5, 6, 227, 18];
			numcpy_unchecked_i_n(r.as_mut_ptr(), a.as_ptr(), 0, 5);
			assert_eq!(r, testvec![15, 17, 19, 21, 23, 6, 227, 18]);

			let a = testvec![115, 17, 119, 21, 123, 25, 127, 29, 131, 33];
			let mut r = testvec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
			numcpy_unchecked_i_n(r.as_mut_ptr(), a.as_ptr(), 2, 10);
			assert_eq!(r, testvec![1, 2, 119, 21, 123, 25, 127, 29, 131, 33, 11, 12, 13]);
		}
	}

	#[test]
	#[rustfmt::skip]
	fn test_add3() {
		let ZERO = Limb { value: 0 };
		let ONE = Limb { value: 1 };
		let THIRD = Limb { value: Limb::MAX / 3 };
		let TWO_THIRDS = Limb { value: Limb::MAX - THIRD.value };
		let MAX = Limb { value: Limb::MAX };

		assert_eq!(add3(ZERO, ZERO, false), (ZERO, false));
		assert_eq!(add3(ZERO, ZERO, true), (ONE, false));
		assert_eq!(add3(ZERO, TWO_THIRDS, false), (TWO_THIRDS, false));
		assert_eq!(add3(ZERO, TWO_THIRDS, true), (Limb { value: 2 * (Limb::MAX / 3) + 1 }, false));
		assert_eq!(add3(ZERO, MAX, false), (MAX, false));
		assert_eq!(add3(ZERO, MAX, true), (ZERO, true));

		assert_eq!(add3(ONE, ZERO, false), (ONE, false));
		assert_eq!(add3(ONE, ZERO, true), (Limb { value: 2 }, false));
		assert_eq!(add3(ONE, TWO_THIRDS, false), (Limb { value: 2 * (Limb::MAX / 3) + 1 }, false));
		assert_eq!(add3(ONE, TWO_THIRDS, true), (Limb { value: 2 * (Limb::MAX / 3) + 2 }, false));
		assert_eq!(add3(ONE, MAX, false), (ZERO, true));
		assert_eq!(add3(ONE, MAX, true), (ONE, true));

		assert_eq!(add3(THIRD, TWO_THIRDS, false), (MAX, false));
		assert_eq!(add3(THIRD, TWO_THIRDS, true), (Limb { value: 0 }, true));
		assert_eq!(add3(THIRD, MAX, false), (Limb { value: THIRD.value - 1 }, true));
		assert_eq!(add3(THIRD, MAX, true), (THIRD, true));

		assert_eq!(add3(MAX, ZERO, false), (MAX, false));
		assert_eq!(add3(MAX, ZERO, true), (ZERO, true));
		assert_eq!(add3(MAX, TWO_THIRDS, false), (Limb { value: TWO_THIRDS.value - 1 }, true));
		assert_eq!(add3(MAX, TWO_THIRDS, true), (TWO_THIRDS, true));
		assert_eq!(add3(MAX, MAX, false), (Limb { value: Limb::MAX - 1 }, true));
		assert_eq!(add3(MAX, MAX, true), (MAX, true));
	}

	#[test]
	fn test_add_n() {
		unsafe {
			let THIRD: Limb::Value = Limb::MAX / 3;
			let HALF: Limb::Value = Limb::MAX / 2;
			let TWO_THIRDS: Limb::Value = Limb::MAX - THIRD;
			let MAX: Limb::Value = Limb::MAX;

			let a = testvec![HALF, 1, 2, 3, TWO_THIRDS, MAX, MAX];
			let b = testvec![TWO_THIRDS, 1, 2, 3, HALF, MAX, 0];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_n_unchecked(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 0, 7);
			assert_eq!(
				r,
				testvec![
					TWO_THIRDS - (Limb::MAX - HALF + 1),
					3,
					4,
					6,
					TWO_THIRDS - (Limb::MAX - HALF + 1),
					MAX,
					0,
					7,
					8,
					9,
					10
				]
			);
			assert_eq!(carry, true);

			let a = testvec![HALF, 1, 2, 3, TWO_THIRDS, MAX, TWO_THIRDS];
			let b = testvec![TWO_THIRDS, 1, 2, 3, HALF, MAX, 0];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_n_unchecked(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 0, 7);
			assert_eq!(
				r,
				testvec![
					TWO_THIRDS - (Limb::MAX - HALF + 1),
					3,
					4,
					6,
					TWO_THIRDS - (Limb::MAX - HALF + 1),
					MAX,
					TWO_THIRDS + 1,
					7,
					8,
					9,
					10
				]
			);
			assert_eq!(carry, false);
		}
	}

	#[test]
	fn test_add_carry() {
		unsafe {
			let THIRD: Limb::Value = Limb::MAX / 3;
			let HALF: Limb::Value = Limb::MAX / 2;
			let TWO_THIRDS: Limb::Value = Limb::MAX - THIRD;
			let MAX: Limb::Value = Limb::MAX;

			let a = testvec![HALF, 1, 2, 3, TWO_THIRDS, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), false, 0, 7);
			assert_eq!(r, testvec![HALF, 1, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), true, 0, 7);
			assert_eq!(r, testvec![HALF + 1, 1, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let a = testvec![MAX, 1, 2, 3, TWO_THIRDS, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), false, 0, 7);
			assert_eq!(r, testvec![MAX, 1, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), true, 0, 7);
			assert_eq!(r, testvec![0, 2, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let a = testvec![MAX, MAX, 2, 3, TWO_THIRDS, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), false, 0, 7);
			assert_eq!(r, testvec![MAX, MAX, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), true, 0, 7);
			assert_eq!(r, testvec![0, 0, 3, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let a = testvec![MAX, MAX, MAX, MAX, MAX, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), false, 0, 7);
			assert_eq!(r, testvec![MAX, MAX, MAX, MAX, MAX, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let carry = add_carry_unchecked(r.as_mut_ptr(), a.as_ptr(), true, 0, 7);
			assert_eq!(r, testvec![0, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10]);
			assert_eq!(carry, true);
		}
	}

	#[test]
	fn test_add_1() {
		unsafe {
			let HALF: Limb::Value = Limb::MAX / 2;
			let TWO_THIRDS: Limb::Value = Limb::MAX / 3 * 2;
			let MAX: Limb::Value = Limb::MAX;

			let a = testvec![HALF, 1, 2, 3, TWO_THIRDS, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_1_unchecked(r.as_mut_ptr(), a.as_ptr(), Limb { value: HALF }, 0, 7);
			assert_eq!(r, testvec![HALF + HALF, 1, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let a = testvec![TWO_THIRDS, 1, 2, 3, TWO_THIRDS, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_1_unchecked(r.as_mut_ptr(), a.as_ptr(), Limb { value: HALF }, 0, 7);
			assert_eq!(
				r,
				testvec![TWO_THIRDS - (MAX - HALF) - 1, 2, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]
			);
			assert_eq!(carry, false);

			let a = testvec![TWO_THIRDS, MAX, MAX, MAX, MAX, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_1_unchecked(r.as_mut_ptr(), a.as_ptr(), Limb { value: HALF }, 0, 7);
			assert_eq!(r, testvec![TWO_THIRDS - (MAX - HALF) - 1, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10]);
			assert_eq!(carry, true);
		}
	}
	*/

	#[test]
	fn test1() {
		let inv = LimbInv::new(Limb::new(10_000_000_000_000_000_000));
		let a = Limb::MAX.val;
		let b = 700_000;
		let c = Limb::mul(Limb::new(a), Limb::new(b), Limb::ZERO, Limb::ZERO);
		let (q, r) = inv.mul(c);
		println!("q = {:?}, r = {:?}", q, r);
	}
}
