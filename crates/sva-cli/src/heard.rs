// Concern: a composition's node texts beside the recordings a target names | Non-concern: decoding a recording (recording.rs) | IO: (path) -> text, a recording

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use sva_ast::{Dir, Listing, Recording, Source};
use sva_core::CliError;

/// A path is relative to the directory sva-cli runs in, never the composition's.
pub struct Heard {
    under: Dir,
    cwd: PathBuf,
    decoded: RefCell<HashMap<String, Arc<Recording>>>,
}

impl Heard {
    pub fn at(dir: impl Into<PathBuf>) -> Result<Heard, CliError> {
        Ok(Heard {
            under: Dir::at(dir),
            cwd: sva_core::cwd()?,
            decoded: RefCell::default(),
        })
    }
}

impl Source for Heard {
    fn paths(&self) -> Result<Listing, String> {
        self.under.paths()
    }

    fn get(&self, path: &str) -> Result<Option<Cow<'_, str>>, String> {
        self.under.get(path)
    }

    fn generation(&self, path: &str) -> Option<u64> {
        self.under.generation(path)
    }

    fn name(&self) -> String {
        self.under.name()
    }

    fn recording(&self, path: &str) -> Result<Option<Arc<Recording>>, String> {
        if let Some(held) = self.decoded.borrow().get(path) {
            return Ok(Some(Arc::clone(held)));
        }
        let file = self.cwd.join(path);
        let (planes, rate) = match crate::recording::read_recording(&file) {
            Ok(read) => read,
            Err(CliError::NotFound(_)) => return Ok(None),
            Err(e) => return Err(e.message()),
        };
        let recording = Arc::new(Recording {
            path: path.to_string(),
            rate,
            planes: Arc::new(planes),
        });
        let held = Arc::clone(&recording);
        self.decoded.borrow_mut().insert(path.to_string(), held);
        Ok(Some(recording))
    }
}
