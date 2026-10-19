// Concern: one `--vary` variable: bounds, step and scale in one unit, its start off its file, its value as text | Non-concern: searching over it | IO: (text) -> Vary; (file) -> a start

use sva_ast::{LogUnit, Source, TokenKind, VARIABLES, tokenize, variable};

use crate::cli_error::CliError;

const UNITS: [&str; 8] = ["", "hz", "khz", "s", "ms", "db", "ct", "st"];

#[derive(Clone, Debug, PartialEq)]
pub struct Vary {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
    /// Linear, the step in `unit`; on a log scale, the ratio each step multiplies by.
    pub step: Option<f64>,
    pub log: bool,
    pub unit: &'static str,
}

fn bad(code: &'static str, message: String) -> CliError {
    CliError::usage(code, message)
}

/// `name=lo..hi[/step][,log]`, as `cutoff=200hz..8khz/1.25,log` or `gain=-12db..0db/1db`.
pub fn vary(written: &str) -> Result<Vary, CliError> {
    let shape = || {
        bad(
            "solve.bad_vary",
            format!(
                "`--vary {written}` is not `<name>=<lo>..<hi>[/<step>][,log]`; write one \
                 variable's bounds in one unit, as `--vary gain=0..2/0.25` or \
                 `--vary cutoff=200hz..8000hz/1.25,log`"
            ),
        )
    };
    let (name, rest) = written.split_once('=').ok_or_else(shape)?;
    let name = name.trim();
    if name.is_empty() {
        return Err(shape());
    }
    let (rest, log) = match rest.trim().strip_suffix(",log") {
        Some(rest) => (rest, true),
        None => (rest.trim(), false),
    };
    let (range, step) = match rest.split_once('/') {
        Some((range, step)) => (range, Some(step.trim())),
        None => (rest, None),
    };
    let (lo, hi) = range.split_once("..").ok_or_else(shape)?;
    let (lo, lo_unit) = amount(lo.trim(), written)?;
    let (hi, hi_unit) = amount(hi.trim(), written)?;
    let unit = match (lo_unit, hi_unit) {
        (a, b) if a == b => a,
        (a, b) => {
            return Err(bad(
                "solve.bad_vary",
                format!(
                    "`--vary {written}` writes its bounds in `{a}` and `{b}`; write both in one \
                     unit, as `200hz..8000hz`"
                ),
            ));
        }
    };
    if lo >= hi {
        return Err(bad(
            "solve.bad_vary",
            format!("`--vary {written}` has no room: its low bound is not under its high one"),
        ));
    }
    if log && lo <= 0.0 {
        return Err(bad(
            "solve.bad_vary",
            format!(
                "`--vary {written}` is on a log scale, which needs both bounds above zero; drop \
                 `,log` or raise the low bound"
            ),
        ));
    }
    let step = match step {
        None => None,
        Some(text) => Some(step_of(text, unit, log, written)?),
    };
    Ok(Vary {
        name: name.to_string(),
        lo,
        hi,
        step,
        log,
        unit,
    })
}

fn step_of(text: &str, unit: &'static str, log: bool, written: &str) -> Result<f64, CliError> {
    let (step, step_unit) = amount(text, written)?;
    match log {
        true if step_unit.is_empty() && step > 1.0 => Ok(step),
        true => Err(bad(
            "solve.bad_vary",
            format!(
                "`--vary {written}` is on a log scale, where a step is the ratio one step \
                 multiplies by, a bare number above 1, as `/1.25`"
            ),
        )),
        false if step > 0.0 && (step_unit == unit || step_unit.is_empty()) => Ok(step),
        false => Err(bad(
            "solve.bad_vary",
            format!(
                "`--vary {written}` steps by `{text}`; a step is above zero and written in the \
                 bounds' own unit, or with none"
            ),
        )),
    }
}

/// A number and the suffix it was written with.
fn amount(text: &str, written: &str) -> Result<(f64, &'static str), CliError> {
    let digits = text
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(text.len());
    let (number, suffix) = text.split_at(digits);
    let unit = UNITS.iter().find(|u| **u == suffix).copied();
    match (number.parse::<f64>(), unit) {
        (Ok(n), Some(unit)) if n.is_finite() => Ok((n, unit)),
        _ => Err(bad(
            "solve.bad_vary",
            format!(
                "`{text}` in `--vary {written}` is no number with a unit; write a decimal and \
                 one of `{}` or none, as `900hz` or `-6db`",
                UNITS[1..].join("`, `")
            ),
        )),
    }
}

impl Vary {
    /// `value` as the variable's file would hold it, rounded where its grid runs out of digits.
    pub fn text(&self, value: f64, precision: Precision) -> String {
        let number = match precision {
            Precision::Decimals(d) => trimmed(&format!("{value:.d$}")),
            Precision::Significant(s) => {
                let d = (s as i32 - 1 - value.abs().log10().floor() as i32).max(0) as usize;
                trimmed(&format!("{value:.d$}"))
            }
            Precision::Shortest => format!("{value}"),
        };
        let number = match number.as_str() {
            "-0" => "0".to_string(),
            _ => number,
        };
        format!("{number}{}", self.unit)
    }

    /// The number `text` writes, as the render reads it in this unit.
    pub fn number(&self, text: &str) -> f64 {
        text.strip_suffix(self.unit)
            .and_then(|n| n.parse().ok())
            .expect("a value this variable wrote")
    }

    /// The variable's current value off its file, in this variable's unit.
    pub fn start(&self, source: &dyn Source) -> Result<f64, CliError> {
        let path = variable(&self.name);
        let text = source.get(&path).map_err(CliError::Io)?;
        let Some(text) = text else {
            let listed = source.paths().map_err(CliError::Io)?;
            let held: Vec<String> = listed
                .found
                .iter()
                .filter_map(|p| p.strip_prefix(&format!("{VARIABLES}/")))
                .map(|p| format!("`{p}`"))
                .collect();
            return Err(bad(
                "solve.no_variable",
                format!(
                    "`--vary {}` names no file `{path}`; a variable is a file under \
                     `{VARIABLES}/` holding one number, and here there are {}",
                    self.name,
                    match held.is_empty() {
                        true => "none".to_string(),
                        false => held.join(", "),
                    }
                ),
            ));
        };
        let body = sva_ast::strip_line_comment(&text[sva_ast::body_at(&text)..]).trim();
        let held = held_number(body).ok_or_else(|| self.no_number(body))?;
        self.in_unit(held).ok_or_else(|| self.no_number(body))
    }

    fn no_number(&self, body: &str) -> CliError {
        bad(
            "solve.reaches_no_number",
            format!(
                "`{}/{}` holds `{body}`, which is no number in `{}`; a varied variable holds one \
                 number, as `0.5`, `900hz` or `-6db`; set the file to one, or vary another",
                VARIABLES,
                self.name,
                match self.unit {
                    "" => "plain numbers",
                    unit => unit,
                }
            ),
        )
    }

    fn in_unit(&self, held: Held) -> Option<f64> {
        let ratio = |v: f64| (v > 0.0).then_some(v);
        Some(match (held, self.unit) {
            (Held::Plain(v) | Held::Note(v), "" | "hz") => v,
            (Held::Plain(v) | Held::Note(v), "khz") => v / 1000.0,
            (Held::Plain(v), "db") => 20.0 * ratio(v)?.log10(),
            (Held::Plain(v), "ct") => 1200.0 * ratio(v)?.log2(),
            (Held::Plain(v), "st") => 12.0 * ratio(v)?.log2(),
            (Held::Log(v, LogUnit::Decibels), "db")
            | (Held::Log(v, LogUnit::Cents), "ct")
            | (Held::Log(v, LogUnit::Semitones), "st") => v,
            (Held::Log(v, unit), "") => unit.resolve(v),
            (Held::Secs(v), "s") => v,
            (Held::Secs(v), "ms") => v * 1000.0,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Decimals(usize),
    Significant(usize),
    Shortest,
}

#[derive(Clone, Copy)]
enum Held {
    Plain(f64),
    Note(f64),
    Log(f64, LogUnit),
    Secs(f64),
}

/// One literal, a minus before it allowed, or a note name, which is its hertz.
fn held_number(body: &str) -> Option<Held> {
    let tokens = tokenize(body).ok()?;
    let (sign, rest) = match tokens.as_slice() {
        [first, rest @ ..] if first.kind == TokenKind::Minus => (-1.0, rest),
        all => (1.0, all),
    };
    match rest {
        [one] => match &one.kind {
            TokenKind::Num(n) => Some(Held::Plain(sign * n)),
            TokenKind::Log(n, unit) => Some(Held::Log(sign * n, *unit)),
            TokenKind::Time(n, sva_ast::SpanUnit::Seconds) => Some(Held::Secs(sign * n)),
            TokenKind::Ident(name) if sign > 0.0 => sva_ast::note_hz(name).map(Held::Note),
            _ => None,
        },
        _ => None,
    }
}

fn trimmed(text: &str) -> String {
    match text.contains('.') {
        true => text.trim_end_matches('0').trim_end_matches('.').to_string(),
        false => text.to_string(),
    }
}
