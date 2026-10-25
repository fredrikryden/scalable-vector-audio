// Concern: the solve and seed refusals while the solver is rebuilt, `fits/`, the hint a cost's leaf earns outside a cost | Non-concern: argv (sva-cli), JS options (sva-wasm) | IO: () -> CliError

use sva_ast::Source;

use crate::cli_error::CliError;

pub const LOAD_JSON: &str = "load_json";

/// No node lives in it.
pub const FITS: &str = "fits";

pub fn unbuilt() -> CliError {
    CliError::usage(
        "solve.unbuilt",
        "the solver is being rebuilt from scratch, so `solve` fits nothing in this build",
    )
}

pub fn seed_unbuilt() -> CliError {
    CliError::usage(
        "seed.unbuilt",
        "the solver is being rebuilt from scratch, so no starter of a recording is written in \
         this build",
    )
}

pub const SOLVE_ONLY: &str = "`repr.<reading>(...)` and `load_json(...)` are leaves of a \
     `sva-cli solve` cost, and only a cost takes them: write the cost on solve's command line \
     or in `fits/<name>`, as `sva-cli solve @fits/<name> --vary ...`; a node renders, and \
     `render --representation` reads it";

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
