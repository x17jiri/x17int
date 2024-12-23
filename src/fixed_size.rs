use crate::blocks;
use crate::Limb;

#[inline(always)]
pub fn add<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let mut r: [Limb; N] = [Limb { value: 0 }; N];
	let mut carry = false;
	for i in 0..N {
		(r[i], carry) = blocks::add3(a[i], b[i], carry);
	}
	(r, carry)
}

#[inline(always)]
pub fn sub<const N: usize>(a: &[Limb; N], b: &[Limb; N]) -> ([Limb; N], bool) {
	let mut swapped = false;
	for i in (0..N).rev() {
		swapped = a[i] < b[i];
		if a[i] != b[i] {
			break;
		}
	}
	let (a, b) = if swapped { (b, a) } else { (a, b) };

	let mut r: [Limb; N] = [Limb { value: 0 }; N];
	let mut borrow = false;
	for i in 0..N {
		(r[i], borrow) = blocks::sub3(a[i], b[i], borrow);
	}
	debug_assert!(!borrow);
	(r, swapped)
}
