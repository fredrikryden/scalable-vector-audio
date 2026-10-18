// Concern: declares the directory-level located Refusal, wrapping a Diag with the file it landed in | Non-concern: raising one (graph.rs), the code registry (diag.rs) | IO: none

use crate::diag::{ByteSpan, DiagCode};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Location {
    pub path: String,
    pub span: ByteSpan,
    /// The line and column, from 1, where a check that read the file's text knows them.
    pub start: Option<(usize, usize)>,
}

/// `{ at, code, reason }`: never a panic, never a poison node — a hole in the graph
/// is data beside it, not inside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub at: Location,
    pub code: DiagCode,
    pub reason: String,
}

impl Refusal {
    pub fn new(
        path: impl Into<String>,
        span: ByteSpan,
        code: DiagCode,
        reason: impl Into<String>,
    ) -> Refusal {
        Refusal {
            at: Location {
                path: path.into(),
                span,
                start: None,
            },
            code,
            reason: reason.into(),
        }
    }

    pub fn placed(mut self, start: (usize, usize)) -> Refusal {
        self.at.start = Some(start);
        self
    }
}
