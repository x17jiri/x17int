use std::intrinsics::{assume, cold_path};

#[derive(Clone, Copy, Default, PartialEq, Debug, PartialOrd)]
pub struct Limb {
	pub value: usize,
}

impl Limb {
	pub type Value = usize;
	pub const BITS: usize = usize::BITS as usize;
	pub const MAX: Self::Value = usize::MAX;
}

#[inline]
fn has_no_overlap(a: *const Limb, a_len: usize, b: *const Limb, b_len: usize) -> bool {
	let a_end = unsafe { a.add(a_len) };
	let b_end = unsafe { b.add(b_len) };
	a_end <= b || b_end <= a
}

/// Returns the number of bits needed to store the number.
///
/// Preconditions:
/// - n > 0
/// - a[0..<n] is a valid slice
/// - a[n - 1] is non-zero
///
/// If the highest limb is zero, the result is as if the limb was 1. This is
/// technically incorrect, but the behavior is defined.
#[inline]
pub unsafe fn bit_width_unchecked(a: *const Limb, n: usize) -> usize {
	debug_assert!(n > 0);
	let hi = unsafe { a.offset(n as isize - 1).read().value };
	n * Limb::BITS - (hi | 1).leading_zeros() as usize
}

/// {rp, n} = {ap, n}
///
/// Preconditions:
/// - if n > 0, then rp[0..<n] and ap[0..<n] are valid slices
/// - allowed overlap: none
#[inline]
pub unsafe fn numcpy_unchecked(rp: *mut Limb, ap: *const Limb, n: usize) {
	debug_assert!(n > 0);
	debug_assert!(has_no_overlap(rp, n, ap, n));

	// I think there is a reasonable chance that 'n' will be
	// just a few limbs. So try to have the code for this case inlined.
	unsafe {
		if n <= 4 {
			let a: usize = 0;
			let b: usize = n >> 2;
			let c: usize = n >> 1;
			let d: usize = n - 1;

			// n || a | b | c | d
			// ------------------
			// 1 || 0 | 0 | 0 | 0
			// 2 || 0 | 0 | 1 | 1
			// 3 || 0 | 0 | 1 | 2
			// 4 || 0 | 1 | 2 | 3

			rp.offset(a as isize).write(ap.offset(a as isize).read());
			rp.offset(b as isize).write(ap.offset(b as isize).read());
			rp.offset(c as isize).write(ap.offset(c as isize).read());
			rp.offset(d as isize).write(ap.offset(d as isize).read());
		} else {
			std::ptr::copy_nonoverlapping(ap, rp, n);
		}
	}
}

pub unsafe fn trim_unchecked(p: *const Limb, i: usize, mut n: usize) -> usize {
	while n > i && p.add(n - 1).read().value == 0 {
		n -= 1;
	}
	n
}

#[inline(always)]
pub unsafe fn cold_trim_unchecked(p: *const Limb, i: usize, mut n: usize) -> usize {
	if i >= n {
		return n;
	}

	if p.add(n - 1).read().value == 0 {
		cold_path();
		n -= 1;
		while i < n && p.add(n - 1).read().value == 0 {
			n -= 1;
		}
	}
	n
}

#[inline]
pub unsafe fn add3(a: Limb, b: Limb, carry: bool) -> (Limb, bool) {
	let (sum, overflow1) = a.value.overflowing_add(b.value);
	let (sum, overflow2) = sum.overflowing_add(carry as usize);
	(Limb { value: sum }, overflow1 | overflow2)
}

pub unsafe fn add_n_unchecked(
	rp: *mut Limb, ap: *const Limb, bp: *const Limb, i: usize, n: usize,
) -> bool {
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));
	debug_assert!(rp as *const Limb == bp || has_no_overlap(rp, n, bp, n));

	let mut carry = false;
	for i in i..n {
		let a = ap.add(i).read();
		let b = bp.add(i).read();
		let (s, c) = add3(a, b, carry);
		rp.add(i).write(s);
		carry = c;
	}
	carry
}

pub unsafe fn add_carry_unchecked(
	rp: *mut Limb, ap: *const Limb, mut carry: bool, i: usize, n: usize,
) -> bool {
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));

	let mut i = i;
	while carry {
		if i == n {
			return true;
		}

		let a = ap.add(i).read();
		let s = a.value.wrapping_add(1);
		rp.add(i).write(Limb { value: s });
		carry = s == 0;

		i += 1;
	}

	if i != n && rp as *const Limb != ap {
		numcpy_unchecked(rp.add(i), ap.add(i), n - i);
	}
	false
}

pub unsafe fn add_1_unchecked(rp: *mut Limb, ap: *const Limb, i: usize, n: usize, b: Limb) -> bool {
	debug_assert!(i < n);
	// TODO - when checking overlap, we should consider `i` as well
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));

	let a = ap.add(i).read();
	let (s, carry) = add3(a, b, false);
	rp.add(i).write(s);
	add_carry_unchecked(rp, ap, carry, i + 1, n)
}

#[inline]
pub unsafe fn sub3(a: Limb, b: Limb, borrow: bool) -> (Limb, bool) {
	let (diff, borrow1) = a.value.overflowing_sub(b.value);
	let (diff, borrow2) = diff.overflowing_sub(borrow as usize);
	(Limb { value: diff }, borrow1 | borrow2)
}

pub unsafe fn sub_n_unchecked(
	rp: *mut Limb, ap: *const Limb, bp: *const Limb, i: usize, n: usize,
) -> bool {
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));
	debug_assert!(rp as *const Limb == bp || has_no_overlap(rp, n, bp, n));

	let mut borrow = false;
	for i in i..n {
		let a = ap.add(i).read();
		let b = bp.add(i).read();
		let (d, b) = sub3(a, b, borrow);
		rp.add(i).write(d);
		borrow = b;
	}
	borrow
}

pub unsafe fn sub_borrow_unchecked(
	rp: *mut Limb, ap: *const Limb, mut borrow: bool, i: usize, n: usize,
) -> bool {
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));

	let mut i = i;
	while borrow {
		if i == n {
			return true;
		}

		let a = ap.add(i).read();
		borrow = a.value == 0;
		rp.add(i).write(Limb { value: a.value.wrapping_sub(1) });

		i += 1;
	}

	if i != n && rp as *const Limb != ap {
		numcpy_unchecked(rp.add(i), ap.add(i), n - i);
	}
	false
}

/// This is functionally equivalent to:
/// ```rust
///     let borrow = sub_n_unchecked(rp, ap, bp, i, n);
///     let len = trim_unchecked(rp, 0, n);
///     return len;
/// ```
/// but it leads to better assembly because the trim is only called when needed.
pub unsafe fn sub_borrow_trim_unchecked(
	rp: *mut Limb, ap: *const Limb, mut borrow: bool, i: usize, n: usize,
) -> usize {
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));

	let mut i = i;
	while borrow {
		if i == n {
			return n;
		}

		let a = ap.add(i).read();
		borrow = a.value == 0;
		rp.add(i).write(Limb { value: a.value.wrapping_sub(1) });

		i += 1;
	}

	if i != n {
		if rp as *const Limb != ap {
			numcpy_unchecked(rp.add(i), ap.add(i), n - i);
		}
	}
	trim_unchecked(rp, 0, n)
}

pub unsafe fn sub_1_unchecked(rp: *mut Limb, ap: *const Limb, i: usize, n: usize, b: Limb) -> bool {
	debug_assert!(i < n);
	// TODO - when checking overlap, we should consider `i` as well
	debug_assert!(rp as *const Limb == ap || has_no_overlap(rp, n, ap, n));

	let a = ap.add(i).read();
	let (d, borrow) = sub3(a, b, false);
	rp.add(i).write(d);
	sub_borrow_unchecked(rp, ap, borrow, i + 1, n)
}

#[cfg(test)]
mod tests {
	use super::*;

	use crate::testvec;

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
			let mut r = testvec![0, 0];
			numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 1);
			assert_eq!(r, testvec![15, 0]);

			let mut r = testvec![1, 2, 3];
			numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 2);
			assert_eq!(r, testvec![15, 17, 3]);

			let mut r = testvec![1, 2, 3, 4, 5, 6, 7, 8];
			numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 3);
			assert_eq!(r, testvec![15, 17, 19, 4, 5, 6, 7, 8]);

			let mut r = testvec![1, 2, 3, 4, 5, 6, 7, 18];
			numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 4);
			assert_eq!(r, testvec![15, 17, 19, 21, 5, 6, 7, 18]);

			let mut r = testvec![1, 2, 3, 4, 5, 6, 227, 18];
			numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 5);
			assert_eq!(r, testvec![15, 17, 19, 21, 23, 6, 227, 18]);

			let a = testvec![115, 17, 119, 21, 123, 25, 127, 29, 131, 33];
			let mut r = testvec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
			numcpy_unchecked(r.as_mut_ptr(), a.as_ptr(), 10);
			assert_eq!(r, testvec![115, 17, 119, 21, 123, 25, 127, 29, 131, 33, 11, 12, 13]);
		}
	}

	#[test]
	#[rustfmt::skip]
	fn test_add3() {
		unsafe {
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
			let carry = add_1_unchecked(r.as_mut_ptr(), a.as_ptr(), 0, 7, Limb { value: HALF });
			assert_eq!(r, testvec![HALF + HALF, 1, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]);
			assert_eq!(carry, false);

			let a = testvec![TWO_THIRDS, 1, 2, 3, TWO_THIRDS, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_1_unchecked(r.as_mut_ptr(), a.as_ptr(), 0, 7, Limb { value: HALF });
			assert_eq!(
				r,
				testvec![TWO_THIRDS - (MAX - HALF) - 1, 2, 2, 3, TWO_THIRDS, MAX, MAX, 7, 8, 9, 10]
			);
			assert_eq!(carry, false);

			let a = testvec![TWO_THIRDS, MAX, MAX, MAX, MAX, MAX, MAX];
			let mut r = testvec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
			let carry = add_1_unchecked(r.as_mut_ptr(), a.as_ptr(), 0, 7, Limb { value: HALF });
			assert_eq!(r, testvec![TWO_THIRDS - (MAX - HALF) - 1, 0, 0, 0, 0, 0, 0, 7, 8, 9, 10]);
			assert_eq!(carry, true);
		}
	}
}
