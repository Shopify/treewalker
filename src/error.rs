//! Errors returned by the import APIs.
use std::io;

/// Failure to read, validate, or represent a model or grouping configuration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LoadError {
    /// An underlying reader or file operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// The model is truncated, inconsistent, or syntactically invalid.
    #[error("malformed model: {0}")]
    MalformedModel(String),
    /// The grouping configuration violates its schema or feature contract.
    #[error("malformed grouping configuration: {0}")]
    MalformedConfig(String),
    /// Valid or recognizable metadata requests unsupported model semantics.
    #[error("unsupported model capability: {0}")]
    Unsupported(String),
    /// An import budget or compact representation limit was exceeded.
    #[error("import limit exceeded: {0}")]
    Limit(String),
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
