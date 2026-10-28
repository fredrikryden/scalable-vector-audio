// Concern: parses a solve's argv: recording, loudness, threshold, windows, the fit resumed, its out directory | Non-concern: running the fit (solve.rs) | IO: (argv tail) -> Command or CliError

use std::path::PathBuf;

use super::{CacheAt, Command, value};
use sva_core::CliError;
use sva_solve::TOLERANCE;
use sva_solve::fit::{SPL_DB, Window};

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
    /// The store every render reads through.
    pub cache: CacheAt,
    pub windows: Option<Vec<Window>>,
}

/// `windows` as `--windows` spells them: `<secs>:<below_hz>` per band, comma-separated.
pub fn spelled(windows: &[Window]) -> String {
    let each: Vec<String> = windows
        .iter()
        .map(|w| match w.below_hz {
            Some(hz) => format!("{}:{hz}", w.secs),
            None => w.secs.to_string(),
        })
        .collect();
    each.join(",")
}

fn windows(raw: &str) -> Result<Vec<Window>, CliError> {
    let band = |each: &str| {
        let (secs, below) = match each.split_once(':') {
            Some((secs, hz)) => (secs, Some(hz.parse::<f64>().ok()?)),
            None => (each, None),
        };
        Some(Window {
            below_hz: below,
            secs: secs.parse().ok()?,
        })
    };
    raw.split(',')
        .map(band)
        .collect::<Option<_>>()
        .ok_or_else(|| {
            CliError::usage(
                "cli.bad_value",
                format!(
                    "--windows takes each band's `<secs>:<below_hz>` low to high, comma-separated, \
                 the last's edge left out, as `0.032:500,0.008`, not `{raw}`"
                ),
            )
        })
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
        cache: CacheAt::Platform,
        windows: None,
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
            "--windows" => args.windows = Some(windows(&value(&mut it, "--windows")?)?),
            "--out" => args.out = PathBuf::from(value(&mut it, "--out")?),
            "--confirm" => args.confirm = true,
            "--cache" => {
                args.cache = CacheAt::named(&value(&mut it, "--cache")?);
            }
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
