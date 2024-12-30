use std::intrinsics::cold_path;

#[derive(PartialEq)]
pub struct Error {
	pub kind: ErrorKind,
	pub message: &'static str,
}

#[derive(PartialEq, Debug)]
pub enum ErrorKind {
	AllocationFailed,
	BufferTooSmall,
	ParseError,
	InvalidBase,
	InternalError,
}

impl std::fmt::Debug for Error {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("Error").field("kind", &self.kind).field("message", &self.message).finish()
	}
}

impl Error {
	pub fn new(kind: ErrorKind, msg: &'static str) -> Self {
		Self { kind, message: msg }
	}

	pub fn new_alloc_failed(msg: &'static str) -> Self {
		Self::new(ErrorKind::AllocationFailed, msg)
	}

	pub fn new_buffer_too_small(msg: &'static str) -> Self {
		Self::new(ErrorKind::BufferTooSmall, msg)
	}

	pub fn new_invalid_base(msg: &'static str) -> Self {
		Self::new(ErrorKind::InvalidBase, msg)
	}

	pub fn new_parse_error(msg: &'static str) -> Self {
		Self::new(ErrorKind::ParseError, msg)
	}

	pub fn new_internal_error(msg: &'static str) -> Self {
		Self::new(ErrorKind::InternalError, msg)
	}
}

#[inline(always)]
#[must_use]
pub fn assert(what: bool, err: fn() -> Error) -> Result<(), Error> {
	if what {
		Ok(())
	} else {
		cold_path();
		Err(err())
	}
}
