// Concern: parses a solve's argv: recording, loudness, threshold, the fit resumed, where it writes, whether over files | Non-concern: running the fit (solve.rs) | IO: (argv tail) -> Command or CliError

use std::path::PathBuf;

use super::{Command, value};
use sva_core::CliError;
use sva_core::solve::fit::{SPL_DB, TOLERANCE};

/// Where a solve writes its fit unless told.
pub const OUT: &str = "fit";

#[derive(Debug, PartialEq)]
pub struct SolveArgs {
    pub path: PathBuf,
    pub spl_db: f64,
    /// `None` the default, a resumed fit's 0.
    pub tolerance: Option<f64>,
    pub from: Option<PathBuf>,
    pub out: PathBuf,
    /// A directory that holds files may be written over.
    pub confirm: bool,
}

pub(super) fn solve_args(rest: &[String]) -> Result<Command, CliError> {
    let mut it = rest.iter();
    let path = it.next().filter(|a| !a.starts_with("--")).ok_or_else(|| {
        CliError::usage(
            "cli.missing_argument",
            "solve needs a recording to fit, as `sva-cli solve clip.wav`",
        )
    })?;
    let mut args = SolveArgs {
        path: PathBuf::from(path),
        spl_db: SPL_DB,
        tolerance: None,
        from: None,
        out: PathBuf::from(OUT),
        confirm: false,
    };
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--spl" => {
                let raw = value(&mut it, "--spl")?;
                args.spl_db = raw
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| {
                        CliError::usage(
                            "cli.bad_value",
                            format!("--spl takes a full-scale sine's dB SPL, not `{raw}`"),
                        )
                    })?;
            }
            "--tolerance" => {
                let raw = value(&mut it, "--tolerance")?;
                args.tolerance = Some(
                    raw.parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite() && *v >= 0.0)
                        .ok_or_else(|| {
                            CliError::usage(
                                "cli.bad_value",
                                format!(
                                    "--tolerance takes a pass's least drop as a share of the \
                                     description length, 0 or more (default {TOLERANCE}), not \
                                     `{raw}`"
                                ),
                            )
                        })?,
                );
            }
            "--from" => args.from = Some(PathBuf::from(value(&mut it, "--from")?)),
            "--out" => args.out = PathBuf::from(value(&mut it, "--out")?),
            "--confirm" => args.confirm = true,
            other => {
                return Err(CliError::usage(
                    "cli.unknown_argument",
                    format!("unknown argument `{other}`"),
                ));
            }
        }
    }
    Ok(Command::Solve(Box::new(args)))
}
