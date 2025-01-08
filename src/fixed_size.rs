use crate::blocks::Limb;

#[inline]
pub fn overflowing_add<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let mut r: [Limb; N] = [Limb::default(); N];
	let mut carry = false;
	for i in 0..N {
		(r[i], carry) = Limb::addc(a[i], b[i], carry);
	}
	(r, carry)
}

#[inline]
pub fn overflowing_sub<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let mut r: [Limb; N] = [Limb::default(); N];
	let mut borrow = false;
	for i in 0..N {
		(r[i], borrow) = Limb::subb(a[i], b[i], borrow);
	}
	(r, borrow)
}

#[inline]
pub fn neg_if<const N: usize>(a: &[Limb; N], condition: bool) -> ([Limb; N], bool) {
	let mut r: [Limb; N] = [Limb::default(); N];
	let mask = if condition { Limb::MAX } else { Limb::ZERO };
	let mut carry = condition;
	for i in 0..N {
		(r[i], carry) = Limb::addc(a[i] ^ mask, Limb::ZERO, carry);
	}
	(r, carry)
}

/// Calculates:
///     negative = a < b
///     value = abs(a - b)
///     return (value, negative)
#[inline]
pub fn sub_abs<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let (r, borrow) = overflowing_sub(a, b);
	let (r, _) = neg_if(&r, borrow);
	(r, borrow)
}

#[inline]
pub fn mul<const A: usize, const B: usize>(a: &[Limb; A], b: &[Limb; B]) -> [Limb; A + B] {
	let mut r: [Limb; A + B] = [Limb::ZERO; A + B];
	if B == 0 {
		return r;
	}
	let mut high = Limb::ZERO;
	for i in 0..A {
		[r[i], high] = Limb::mul(a[i], b[0], high, Limb::ZERO);
	}
	for j in 1..B {
		let mut carry = Limb::ZERO;
		for i in 0..A {
			// TODO
		}
	}
	r
}

//--------------------------------------------------------------------------------------------------

#[inline(never)]
pub fn sub4(a: &[Limb; 4], b: &[Limb; 4]) -> ([Limb; 4], bool) {
	sub_abs(a, b)
}
