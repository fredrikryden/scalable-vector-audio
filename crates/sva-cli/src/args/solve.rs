// Concern: parses a solve's argv: the recording, its levels, its loudness, where its levels are written and N0 kept | Non-concern: running the pyramid (solve.rs) | IO: (argv tail) -> Command or CliError

use std::path::PathBuf;

use super::reading::count;
use super::{CacheAt, Command, value};
use sva_core::CliError;
use sva_core::solve::pyramid::{LEVELS, SPL_DB};

/// Where a solve writes its levels unless told.
pub const OUT: &str = "fit";

/// Levels run unless told.
pub const COARSEST: usize = 1;

#[derive(Debug, PartialEq)]
pub struct SolveArgs {
    pub path: PathBuf,
    pub levels: usize,
    /// The coarsest levels run, at most `levels`.
    pub coarsest: usize,
    pub spl_db: f64,
    pub out: PathBuf,
    /// A directory that holds files may be written over.
    pub confirm: bool,
    /// Where N0's calibrations are kept between runs.
    pub cache: CacheAt,
    /// Every level grows in rounds.
    pub every_level: bool,
}

pub(super) fn solve_args(rest: &[String]) -> Result<Command, CliError> {
    let mut it = rest.iter();
    let mut coarsest = None;
    let path = it.next().filter(|a| !a.starts_with("--")).ok_or_else(|| {
        CliError::usage(
            "cli.missing_argument",
            "solve needs a recording to fit, as `sva-cli solve clip.wav`",
        )
    })?;
    let mut args = SolveArgs {
        path: PathBuf::from(path),
        levels: LEVELS,
        coarsest: 0,
        spl_db: SPL_DB,
        out: PathBuf::from(OUT),
        confirm: false,
        cache: CacheAt::Platform,
        every_level: false,
    };
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--levels" => {
                args.levels = count(&value(&mut it, "--levels")?, "--levels", "levels")?.get();
            }
            "--coarsest" => {
                coarsest =
                    Some(count(&value(&mut it, "--coarsest")?, "--coarsest", "levels")?.get());
            }
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
            "--out" => args.out = PathBuf::from(value(&mut it, "--out")?),
            "--confirm" => args.confirm = true,
            "--every-level" => args.every_level = true,
            "--cache" => {
                args.cache = match value(&mut it, "--cache")?.as_str() {
                    "none" => CacheAt::Off,
                    path => CacheAt::Path(PathBuf::from(path)),
                };
            }
            other => {
                return Err(CliError::usage(
                    "cli.unknown_argument",
                    format!("unknown argument `{other}`"),
                ));
            }
        }
    }
    args.coarsest = coarsest.unwrap_or(COARSEST);
    if args.coarsest > args.levels {
        return Err(CliError::usage(
            "cli.bad_value",
            format!(
                "--coarsest {} runs more than the {} levels; pass at most --levels",
                args.coarsest, args.levels
            ),
        ));
    }
    Ok(Command::Solve(Box::new(args)))
}
