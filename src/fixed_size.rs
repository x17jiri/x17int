use crate::limb::Limb;
use std::intrinsics::{assume, cold_path};

pub fn overflowing_add<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let mut r: [Limb; N] = [Limb::default(); N];
	let mut carry = false;
	for i in 0..N {
		(r[i], carry) = Limb::addc(a[i], b[i], carry);
	}
	(r, carry)
}

pub fn overflowing_sub<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let mut r: [Limb; N] = [Limb::default(); N];
	let mut borrow = false;
	for i in 0..N {
		(r[i], borrow) = Limb::subb(a[i], b[i], borrow);
	}
	(r, borrow)
}

/// If `condition`, returns 2's complement of `a`. Otherwise, returns `a` unchanged.
/// I.e., if condition is true, negates all bits of a and adds 1.
pub fn neg_if<const N: usize>(a: &[Limb; N], condition: bool) -> [Limb; N] {
	let mut r: [Limb; N] = [Limb::default(); N];
	let mask = if condition { Limb::MAX } else { Limb(0) };
	let mut carry = condition;
	for i in 0..N {
		(r[i], carry) = Limb::addc(a[i] ^ mask, Limb(0), carry);
	}
	r
}

/// Calculates:
///     negative = a < b
///     value = abs(a - b)
///     return (value, negative)
pub fn sub_abs<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let (r, borrow) = overflowing_sub(a, b);
	let r = neg_if(&r, borrow);
	(r, borrow)
}

pub fn mul<const A: usize, const B: usize>(a: &[Limb; A], b: &[Limb; B]) -> [Limb; A + B] {
	let mut r: [Limb; A + B] = [Limb(0); A + B];
	if B == 0 {
		return r;
	}
	let mut high = Limb(0);
	for i in 0..A {
		[r[i], high] = Limb::mul(a[i], b[0], high, Limb(0));
	}
	for j in 1..B {
		high = Limb(0);
		for i in 0..A {
			[r[i + j], high] = Limb::mul(a[i], b[j], high, r[i + j]);
		}
	}
	r[A + B - 1] = high;
	r
}

//--------------------------------------------------------------------------------------------------

#[inline(never)]
pub fn sub4(a: &[Limb; 4], b: &[Limb; 4]) -> ([Limb; 4], bool) {
	sub_abs(a, b)
}

#[inline(never)]
pub fn mul3(a: &[Limb; 3], b: &[Limb; 3]) -> [Limb; 6] {
	mul(a, b)
}
