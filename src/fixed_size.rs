use crate::limb::Limb;
use std::intrinsics::{assume, cold_path};

pub const fn const_max(a: usize, b: usize) -> usize {
	if a >= b {
		a
	} else {
		b
	}
}

pub const fn const_min(a: usize, b: usize) -> usize {
	if a <= b {
		a
	} else {
		b
	}
}

pub const fn add<const A: usize, const B: usize>(
	a: &[Limb; A], b: &[Limb; B],
) -> ([Limb; const_max(A, B)], bool) {
	let mut r = [Limb(0); const_max(A, B)];
	let mut carry = false;

	let mut i = 0;
	while i < const_min(A, B) {
		(r[i], carry) = Limb::addc(a[i], b[i], carry);
		i += 1;
	}
	if A >= B {
		while i < A {
			(r[i], carry) = Limb::addc(a[i], Limb(0), carry);
			i += 1;
		}
	} else {
		while i < B {
			(r[i], carry) = Limb::addc(Limb(0), b[i], carry);
			i += 1;
		}
	}

	(r, carry)
}

pub const fn sub<const A: usize, const B: usize>(
	a: &[Limb; A], b: &[Limb; B],
) -> ([Limb; const_max(A, B)], bool) {
	let mut r = [Limb(0); const_max(A, B)];
	let mut borrow = false;

	let mut i = 0;
	while i < const_min(A, B) {
		(r[i], borrow) = Limb::subb(a[i], b[i], borrow);
		i += 1;
	}
	if A >= B {
		while i < A {
			(r[i], borrow) = Limb::subb(a[i], Limb(0), borrow);
			i += 1;
		}
	} else {
		while i < B {
			(r[i], borrow) = Limb::subb(Limb(0), b[i], borrow);
			i += 1;
		}
	}

	(r, borrow)
}

/// If `condition`, return 2's complement of `a`. Otherwise, return `a` unchanged.
/// I.e., if condition is true, negates all bits of `a` and adds 1.
pub const fn neg_if<const N: usize>(a: &[Limb; N], condition: bool) -> [Limb; N] {
	let mut r = *a;
	if condition {
		let mut borrow;
		(r[0], borrow) = a[0].overflowing_neg();
		let mut i = 1;
		while i < N {
			(r[i], borrow) = Limb::subb(r[i].const_bitnot(), Limb(0), borrow);
			i += 1;
		}
	}
	r
}

/// Calculates:
///     negative = a < b
///     value = abs(a - b)
///     return (value, negative)
pub const fn sub_abs<const A: usize, const B: usize>(
	a: &[Limb; A], b: &[Limb; B],
) -> ([Limb; const_max(A, B)], bool)
where
	[Limb; const_max(B, A)]:,
{
	let (r, borrow) = sub(a, b);
	let r = neg_if(&r, borrow);
	(r, borrow)
}

pub const fn mul<const A: usize, const B: usize>(a: &[Limb; A], b: &[Limb; B]) -> [Limb; A + B] {
	let mut r: [Limb; A + B] = [Limb(0); A + B];
	if B == 0 {
		return r;
	}
	let mut high = Limb(0);
	let mut i = 0;
	while i < A {
		[r[i], high] = Limb::mul(a[i], b[0], high, Limb(0));
		i += 1;
	}
	r[A] = high;

	let mut j = 1;
	while j < B {
		high = Limb(0);
		let mut i = 0;
		while i < A {
			[r[i + j], high] = Limb::mul(a[i], b[j], high, r[i + j]);
			i += 1;
		}
		r[A + j] = high;
		j += 1;
	}
	r
}

/// Shifts `a` left by `shift % Limb::BITS` bits.
/// I.e., the shift is always less than Limb::BITS.
pub const fn small_shl<const N: usize>(a: [Limb; N], shift: usize) -> [Limb; N] {
	if N == 0 {
		return [Limb(0); N];
	}

	let shift = shift & (Limb::BITS - 1);

	let mut r = [Limb(0); N];
	r[0] = a[0].const_shl(shift);

	let mut i = 1;
	while i < N {
		// TODO - replace this `while` with a `for` loop when Rust supports it

		let t = Limb::make_double(a[i - 1], a[i]) << shift;
		r[i] = Limb::from_high_half(t);

		i += 1;
	}
	r
}

/// Shifts `a` right by `shift % Limb::BITS` bits.
/// I.e., the shift is always less than Limb::BITS.
pub const fn small_shr<const N: usize>(a: [Limb; N], shift: usize) -> [Limb; N] {
	if N == 0 {
		return [Limb(0); N];
	}

	let shift = shift & (Limb::BITS - 1);

	let mut r = [Limb(0); N];

	let mut i = 1;
	while i < N {
		// TODO - replace this `while` with a `for` loop when Rust supports it

		let t = Limb::make_double(a[i - 1], a[i]) >> shift;
		r[i - 1] = Limb::from_low_half(t);

		i += 1;
	}
	r[N - 1] = a[N - 1].const_shr(shift);
	r
}

//--------------------------------------------------------------------------------------------------

#[inline(never)]
pub fn sub2(a: [Limb; 2], b: [Limb; 2]) -> ([Limb; 2], bool) {
	sub_abs(&a, &b)
}

#[inline(never)]
pub fn mul3(a: &[Limb; 3], b: &[Limb; 3]) -> [Limb; 6] {
	mul(a, b)
}

#[inline(never)]
pub fn shl3(a: [Limb; 3], shift: usize) -> [Limb; 3] {
	small_shl(a, shift)
}

#[inline(never)]
pub fn shr3(a: [Limb; 3], shift: usize) -> [Limb; 3] {
	small_shr(a, shift)
}
