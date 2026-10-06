//! Errors returned by the validated import APIs.
use std::{fmt, io};

#[derive(Debug)]
/// Failure to read, validate, or represent a model or grouping configuration.
pub enum LoadError {
    /// An underlying reader or file operation failed.
    Io(io::Error),
    /// The model is truncated, inconsistent, or syntactically invalid.
    MalformedModel(String),
    /// The grouping configuration violates its schema or feature contract.
    MalformedConfig(String),
    /// Valid or recognizable metadata requests unsupported model semantics.
    Unsupported(String),
    /// An import budget or compact representation limit was exceeded.
    Limit(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::MalformedModel(s) => write!(f, "malformed model: {s}"),
            Self::MalformedConfig(s) => write!(f, "malformed grouping configuration: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported model capability: {s}"),
            Self::Limit(s) => write!(f, "import limit exceeded: {s}"),
        }
    }
}
impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl LoadError {
    pub(crate) fn at(self, context: &str) -> Self {
        match self {
            Self::MalformedModel(s) => Self::MalformedModel(format!("{context}: {s}")),
            Self::Unsupported(s) => Self::Unsupported(format!("{context}: {s}")),
            Self::Limit(s) => Self::Limit(format!("{context}: {s}")),
            other => other,
        }
    }
}
