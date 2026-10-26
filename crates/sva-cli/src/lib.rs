// Concern: wires this CLI's subcommand modules atop sva-core's pipeline | Non-concern: the pipeline itself (sva-core), JS bindings (sva-wasm) | IO: none

mod args;
mod arity;
mod bare;
mod calibrations;
mod composition;
mod destination;
mod device;
mod directory;
mod heard;
mod help;
mod lint;
mod new;
mod output;
mod panic;
mod rates;
mod recording;
mod render;
mod solve;
mod store;
mod terminal;
mod trace;
mod variables;
mod wav;

pub use args::{AnalyzeArgs, CacheAt, Command, Format, RenderArgs, SolveArgs, parse_args};
pub use composition::{composition, located};
pub use destination::{Framing, refuse_inside, write as write_destination};
pub use help::help_text;

/// The composition at `dir`, its `sample("<path>")` reads decoded off the files they name.
pub fn heard(dir: std::path::PathBuf) -> Result<impl sva_ast::Source, CliError> {
    heard::Heard::at(dir)
}
pub use lint::{BODY_CHARS, Finding, LintReport, lint, lint_bounded};
pub use new::{FILES, NEXT, Scaffolded, scaffold};
pub use output::{help_data, lint_data, new_data, trace_data, version_data};
pub use panic::{Stopped, caught, stopped};
pub use recording::{READS, read_recording};
pub use render::{analyze, render};
pub use solve::solve;
pub use sva_core::{
    Builtins, Callable, CliError, builtins, builtins_data, cwd, error_envelope, success_envelope,
};
pub use terminal::{colored, diagnostics_text};
pub use trace::{Traceable, trace};
pub use wav::{SampleEncoding, write_channels, write_wav};

pub const NAME: &str = "sva-cli";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
