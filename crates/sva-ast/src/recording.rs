// Concern: a decoded recording, and each `sample("<path>")` resolved to one | Non-concern: decoding a file, what it means at a rate (sva-engine) | IO: (&Expr, &dyn Source) -> Expr or Unheld

use std::sync::Arc;

use crate::expr::{Arg, Binds, Expr, Literal, map_children};
use crate::source::Source;

/// The path names it for a reader and is no part of what it is.
#[derive(Clone, Debug)]
pub struct Recording {
    pub path: String,
    pub rate: u32,
    pub planes: Arc<Vec<Vec<f32>>>,
}

impl Recording {
    pub fn frames(&self) -> usize {
        self.planes.iter().map(Vec::len).min().unwrap_or(0)
    }
}

impl PartialEq for Recording {
    fn eq(&self, other: &Recording) -> bool {
        self.rate == other.rate
            && (Arc::ptr_eq(&self.planes, &other.planes)
                || self.planes.len() == other.planes.len()
                    && self.planes.iter().zip(other.planes.iter()).all(|(a, b)| {
                        a.len() == b.len()
                            && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                    }))
    }
}

pub fn resolve_recordings(expr: &Expr, source: &dyn Source) -> Result<Expr, Unheld> {
    if let Expr::Call { name, args, span } = expr
        && name == "sample"
        && let [Arg::Pos(Expr::Lit(Literal::Path(path)))] = args.as_slice()
    {
        let held = source.recording(path).map_err(|why| Unheld {
            path: path.clone(),
            why: Some(why),
        })?;
        let Some(recording) = held else {
            return Err(Unheld {
                path: path.clone(),
                why: None,
            });
        };
        return Ok(Expr::Call {
            name: name.clone(),
            args: vec![Arg::Pos(Expr::Lit(Literal::Recorded(recording)))],
            span: *span,
        });
    }
    map_children(expr, Binds::Substitute, |child| {
        resolve_recordings(child, source)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unheld {
    pub path: String,
    pub why: Option<String>,
}

impl Unheld {
    pub fn message(&self) -> String {
        match &self.why {
            Some(why) => format!("`sample(\"{}\")` could not be read: {why}", self.path),
            None => format!(
                "`sample(\"{}\")` names no recording: no file is there, relative to the \
                 directory sva-cli runs in or absolute, and a page passed none under that path in a \
                 solve's `recordings`",
                self.path
            ),
        }
    }
}
