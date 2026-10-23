// Concern: parses a solve's cost, its variables, stops and flags | Non-concern: running it (solve.rs), the cost's grammar (sva-core) | IO: (argv tail) -> Command or CliError

use std::path::PathBuf;

use super::reading::{count, hertz};
use super::{CacheAt, Command, value};
use sva_core::CliError;
use sva_core::continuation::Scheme;
use sva_core::solve::prox::Settings;
use sva_core::solve::{DEFAULT_COST_TOL, DEFAULT_MAX_RENDERS, DEFAULT_STEP_TOL, Stops, Vary, vary};

/// How a solve searches: compass steps over rendered readings, or the priced proximal fit, alone
/// or through a continuation's stages.
#[derive(Debug, PartialEq)]
pub enum Method {
    Compass,
    Prox(Settings, Option<Scheme>),
}

#[derive(Debug, PartialEq)]
pub struct SolveArgs {
    /// The cost, or `@fits/<name>`: read once the composition is known.
    pub cost: String,
    pub vary: Vec<Vary>,
    pub stops: Stops,
    pub sets: Vec<(String, String)>,
    pub rate: Option<u32>,
    pub threads: Option<std::num::NonZeroUsize>,
    pub cache: CacheAt,
    pub write: bool,
    pub confirm: bool,
    pub method: Method,
}

fn number(raw: &str, flag: &str, fits: impl Fn(f64) -> bool, says: &str) -> Result<f64, CliError> {
    match raw.parse::<f64>() {
        Ok(v) if v.is_finite() && fits(v) => Ok(v),
        _ => Err(CliError::usage(
            "cli.bad_value",
            format!("{flag} takes {says}, not `{raw}`"),
        )),
    }
}

pub(super) fn solve_args(rest: &[String]) -> Result<Command, CliError> {
    let mut it = rest.iter();
    let cost = it.next().filter(|a| !a.starts_with("--")).ok_or_else(|| {
        CliError::usage(
            "cli.missing_argument",
            "solve needs a cost, as `sva-cli solve 'norm(repr.bands(@master) - \
             repr.bands(sample(\"../ref.wav\")))' --vary gain=0..2`, or a cost file, as \
             `sva-cli solve @fits/lead --vary gain=0..2`",
        )
    })?;
    let mut stops = Stops {
        max_renders: DEFAULT_MAX_RENDERS,
        step_tol: DEFAULT_STEP_TOL,
        cost_tol: DEFAULT_COST_TOL,
    };
    let (mut vary_all, mut sets) = (Vec::new(), Vec::new());
    let (mut rate, mut threads, mut cache) = (None, None, CacheAt::Platform);
    let (mut write, mut confirm) = (false, false);
    let (mut prox, mut settings, mut prox_flag) = (false, Settings::default(), None);
    let (mut continuation, mut rounds) = (None, None);
    let mut compass_flag = None;
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--method" => match value(&mut it, "--method")?.as_str() {
                "compass" => prox = false,
                "prox" => prox = true,
                other => {
                    return Err(CliError::usage(
                        "cli.bad_value",
                        format!("--method takes `compass` or `prox`, not `{other}`"),
                    ));
                }
            },
            "--continuation" => {
                continuation = Some(value(&mut it, "--continuation")?);
                prox_flag = Some("--continuation");
            }
            "--rounds" => {
                rounds = Some(count(&value(&mut it, "--rounds")?, "--rounds", "rounds")?.get());
                prox_flag = Some("--rounds");
            }
            "--steps" => {
                settings.steps = count(&value(&mut it, "--steps")?, "--steps", "steps")?.get();
                prox_flag = Some("--steps");
            }
            "--draws" => {
                settings.draws = count(&value(&mut it, "--draws")?, "--draws", "draws")?.get();
                prox_flag = Some("--draws");
            }
            "--vary" => vary_all.push(vary(&value(&mut it, "--vary")?)?),
            "--set" => sets.push(sva_core::set(&value(&mut it, "--set")?)?),
            "--max-renders" => {
                compass_flag = Some("--max-renders");
                let raw = value(&mut it, "--max-renders")?;
                stops.max_renders = count(&raw, "--max-renders", "renders")?.get();
            }
            "--step-tol" => {
                compass_flag = Some("--step-tol");
                let raw = value(&mut it, "--step-tol")?;
                let says = "a fraction of each variable's first step above 0 and at most 1";
                stops.step_tol = number(&raw, "--step-tol", |v| v > 0.0 && v <= 1.0, says)?;
            }
            "--cost-tol" => {
                compass_flag = Some("--cost-tol");
                let raw = value(&mut it, "--cost-tol")?;
                let says = "a cost at or above 0";
                stops.cost_tol = number(&raw, "--cost-tol", |v| v >= 0.0, says)?;
            }
            "--rate" => rate = Some(hertz(&value(&mut it, "--rate")?)?),
            "--threads" => {
                threads = Some(count(
                    &value(&mut it, "--threads")?,
                    "--threads",
                    "threads",
                )?);
            }
            "--cache" => {
                cache = match value(&mut it, "--cache")?.as_str() {
                    "none" => CacheAt::Off,
                    path => CacheAt::Path(PathBuf::from(path)),
                };
            }
            "--write" => write = true,
            "--confirm" => confirm = true,
            other => {
                return Err(CliError::usage(
                    "cli.unknown_argument",
                    format!("unknown argument `{other}`"),
                ));
            }
        }
    }
    if vary_all.is_empty() {
        return Err(CliError::usage(
            "solve.no_vary",
            "solve needs at least one `--vary <name>=<lo>..<hi>`, a variable under \
             `variables/` it fits",
        ));
    }
    if let (true, Some(flag)) = (prox, compass_flag) {
        return Err(CliError::usage(
            "cli.compass_flag",
            format!(
                "`{flag}` stops the compass search, which `--method prox` does not run; it is \
                 set by `--steps`, `--draws` and `--continuation`"
            ),
        ));
    }
    if let (false, Some(flag)) = (prox, prox_flag) {
        return Err(CliError::usage(
            "cli.prox_flag",
            format!("`{flag}` sets the priced fit, which only `--method prox` takes"),
        ));
    }
    if let (None, Some(_)) = (&continuation, rounds) {
        return Err(CliError::usage(
            "cli.prox_flag",
            "`--rounds` sets the k-medians rounds of a continuation's clusters and splits, which \
             only `--continuation` runs",
        ));
    }
    let scheme = continuation
        .map(|text| Scheme::parse(&text, rounds))
        .transpose()?;
    if confirm && !write {
        return Err(CliError::usage(
            "cli.confirm_without_write",
            "`--confirm` lets `--write` replace a variable's file; write both, or neither",
        ));
    }
    Ok(Command::Solve(Box::new(SolveArgs {
        cost: cost.clone(),
        vary: vary_all,
        stops,
        sets,
        rate,
        threads,
        cache,
        write,
        confirm,
        method: match prox {
            true => Method::Prox(settings, scheme),
            false => Method::Compass,
        },
    })))
}
