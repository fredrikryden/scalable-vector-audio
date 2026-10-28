// Concern: fits a recording as sources by minimum description length, writes them as a composition, reports each pass | Non-concern: argv (sva-cli), JS (sva-wasm) | IO: (samples, rate) -> Fit, JSON

pub mod analysis;
pub mod bits;
pub mod fit;
pub mod grid;
mod optimize;
pub mod report;
pub mod sources;

pub use optimize::{End, TOLERANCE};
