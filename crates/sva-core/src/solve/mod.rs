// Concern: the seed refusal while the starter is rebuilt, the solver's modules, `fits/`, a cost leaf's hint outside a cost | Non-concern: argv (sva-cli), JS options | IO: () -> CliError

pub mod add;
pub mod atoms;
pub mod bits;
pub mod fit;
pub mod grid;
pub mod optimize;
pub mod report;

use sva_ast::Source;

use crate::cli_error::CliError;

pub const LOAD_JSON: &str = "load_json";

/// No node lives in it.
pub const FITS: &str = "fits";

pub fn seed_unbuilt() -> CliError {
    CliError::usage(
        "seed.unbuilt",
        "the solver is being rebuilt from scratch, so no starter of a recording is written in \
         this build",
    )
}

pub const SOLVE_ONLY: &str = "`repr.<reading>(...)` and `load_json(...)` were leaves of a \
     retired `sva-cli solve` cost, and nothing takes them now: `sva-cli solve <recording>` \
     fits a recording, and `render --representation` reads a node";

fn loads_at(text: &str, byte: usize) -> bool {
    let before = text[..byte.min(text.len())].trim_end();
    let before = before.strip_suffix('(').map(str::trim_end);
    before.is_some_and(|b| b.ends_with(LOAD_JSON))
}

pub(crate) fn hinted(refused: CliError, source: &dyn Source, target: Option<&str>) -> CliError {
    let solve_only = match &refused {
        CliError::Refusals(rs) => rs.iter().any(|r| {
            r.code == sva_ast::DiagCode::MisplacedPath
                && source
                    .get(&r.at.path)
                    .ok()
                    .flatten()
                    .is_some_and(|text| loads_at(&text, r.at.span.start))
        }),
        CliError::BadProbe(_) => {
            target.is_some_and(|t| t.match_indices('"').any(|(at, _)| loads_at(t, at)))
        }
        _ => false,
    };
    match solve_only {
        true => CliError::Hinted {
            error: Box::new(refused),
            help: SOLVE_ONLY,
        },
        false => refused,
    }
}
