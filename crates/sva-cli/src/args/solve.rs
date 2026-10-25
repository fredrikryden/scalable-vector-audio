// Concern: parses a solve's argv, each flag's shape and the flags it may not mix, while the solver is rebuilt | Non-concern: what a cost or a variable means | IO: (argv tail) -> Command or CliError

use super::reading::{count, hertz};
use super::{Command, value};
use sva_core::CliError;

fn number(raw: &str, flag: &str, fits: impl Fn(f64) -> bool, says: &str) -> Result<(), CliError> {
    match raw.parse::<f64>() {
        Ok(v) if v.is_finite() && fits(v) => Ok(()),
        _ => Err(CliError::usage(
            "cli.bad_value",
            format!("{flag} takes {says}, not `{raw}`"),
        )),
    }
}

pub(super) fn solve_args(rest: &[String]) -> Result<Command, CliError> {
    let mut it = rest.iter();
    it.next().filter(|a| !a.starts_with("--")).ok_or_else(|| {
        CliError::usage(
            "cli.missing_argument",
            "solve needs a cost, as `sva-cli solve 'norm(repr.bands(@master) - \
             repr.bands(sample(\"../ref.wav\")))' --vary gain=0..2`, or a cost file, as \
             `sva-cli solve @fits/lead --vary gain=0..2`",
        )
    })?;
    let (mut varies, mut write, mut confirm) = (false, false, false);
    let (mut prox, mut prox_flag, mut compass_flag) = (false, None, None);
    let (mut continuation, mut rounds) = (false, false);
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
                value(&mut it, "--continuation")?;
                continuation = true;
                prox_flag = Some("--continuation");
            }
            "--rounds" => {
                count(&value(&mut it, "--rounds")?, "--rounds", "rounds")?;
                rounds = true;
                prox_flag = Some("--rounds");
            }
            "--steps" => {
                let raw = value(&mut it, "--steps")?;
                raw.parse::<usize>().map_err(|_| {
                    CliError::usage(
                        "cli.bad_value",
                        format!("--steps needs a whole number of steps, got `{raw}`"),
                    )
                })?;
                prox_flag = Some("--steps");
            }
            "--draws" => {
                count(&value(&mut it, "--draws")?, "--draws", "draws")?;
                prox_flag = Some("--draws");
            }
            "--heartbeat" => {
                let raw = value(&mut it, "--heartbeat")?;
                number(&raw, "--heartbeat", |v| v >= 0.0, "seconds at or above 0")?;
                prox_flag = Some("--heartbeat");
            }
            "--vary" => {
                value(&mut it, "--vary")?;
                varies = true;
            }
            "--set" => {
                sva_core::set(&value(&mut it, "--set")?)?;
            }
            "--max-renders" => {
                compass_flag = Some("--max-renders");
                count(
                    &value(&mut it, "--max-renders")?,
                    "--max-renders",
                    "renders",
                )?;
            }
            "--step-tol" => {
                compass_flag = Some("--step-tol");
                let raw = value(&mut it, "--step-tol")?;
                let says = "a fraction of each variable's first step above 0 and at most 1";
                number(&raw, "--step-tol", |v| v > 0.0 && v <= 1.0, says)?;
            }
            "--cost-tol" => {
                compass_flag = Some("--cost-tol");
                let raw = value(&mut it, "--cost-tol")?;
                number(&raw, "--cost-tol", |v| v >= 0.0, "a cost at or above 0")?;
            }
            "--rate" => {
                hertz(&value(&mut it, "--rate")?)?;
            }
            "--threads" => {
                count(&value(&mut it, "--threads")?, "--threads", "threads")?;
            }
            "--cache" => {
                value(&mut it, "--cache")?;
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
    if !varies {
        return Err(CliError::usage(
            "solve.no_vary",
            "solve needs at least one `--vary <name>=<lo>..<hi>`, a variable under \
             `variables/` it fits",
        ));
    }
    if let (true, Some(flag)) = (prox, compass_flag) {
        return Err(CliError::usage(
            "cli.compass_flag",
            format!("`{flag}` stops the compass search, which `--method prox` does not run"),
        ));
    }
    if let (false, Some(flag)) = (prox, prox_flag) {
        return Err(CliError::usage(
            "cli.prox_flag",
            format!("`{flag}` sets the priced fit, which only `--method prox` takes"),
        ));
    }
    if rounds && !continuation {
        return Err(CliError::usage(
            "cli.prox_flag",
            "`--rounds` sets a continuation's rounds, which only `--continuation` runs",
        ));
    }
    if confirm && !write {
        return Err(CliError::usage(
            "cli.confirm_without_write",
            "`--confirm` lets `--write` replace a variable's file; write both, or neither",
        ));
    }
    Ok(Command::Solve)
}
