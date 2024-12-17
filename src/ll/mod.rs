pub struct Limb {
	pub value: usize,
}

impl Limb {
	pub const BITS: usize = usize::BITS as usize;
}

// Returns the number of bits needed to store the number.
//
// Preconditions:
// 	- n > 0, i.e., zero length inputs are not allowed
// 	- the highest limb must be non-zero
pub fn bit_width(a: *const Limb, n: usize) -> usize {
	debug_assert!(n > 0);
	let hi = unsafe { a.offset(n as isize - 1).read().value };
	debug_assert!(hi != 0);
	n * Limb::BITS - (hi | 1).leading_zeros() as usize
}
