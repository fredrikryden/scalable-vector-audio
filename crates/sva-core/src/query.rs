// Concern: the observations both front ends ask for, each written as a call with its own arguments | Non-concern: argv (sva-cli's args/), taking the reading (sva-engine) | IO: (text) -> Asked

use std::path::{Path, PathBuf};

use sva_ast::SpanUnit;
use sva_ast::score::{Field, Kind, Reading as Folding, Refused, evaluate};
use sva_engine::harmony::voice_leading::Norm;
use sva_engine::{
    Arg, DEFAULT_BUFFER_SECS, Given, MAX_PINNED_FRAME, READINGS, Reading, Representation,
    pinned_frame,
};

use crate::cli_error::CliError;

pub const RETIRED: [(&str, &str); 2] = [
    ("exact-envelope", "envelope, symbolic on a closed form"),
    ("exact-derivative", "derivative, symbolic on a closed form"),
];

/// One reading written as a call, `spectrum(peaks=8, frame=50ms)=out.json`: its name, its
/// arguments as written, and where it goes.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub name: String,
    pub args: Vec<(String, String)>,
    /// No destination is stdout.
    pub dest: Option<PathBuf>,
}

/// Where a reading goes instead of stdout: a path, its extension picking the format, or a
/// live device named after `alsa:`.
#[derive(Clone, Debug, PartialEq)]
pub enum Destination {
    File(PathBuf),
    Alsa {
        /// Any ALSA PCM name: `default`, `hw:1`, `plughw:1,0`.
        device: String,
        /// `samples(buffer=50ms)`: how much the device holds ahead of what it plays.
        buffer_secs: f64,
    },
}

/// A call resolved to the reading it names and the arguments only a front end reads. Its
/// times stay as written until a render says what a bar lasts.
#[derive(Clone, Debug, PartialEq)]
pub struct Asked {
    pub name: String,
    given: Given,
    times: Vec<(Arg, Time)>,
    pub dest: Option<Destination>,
    /// `bindings(node=@voice)`: the instance whose bindings are read.
    pub node: Option<String>,
    /// `ledger(brief=1)`: only the rows that clipped.
    pub brief: bool,
    /// `ledger(skim=1)`: each row's level alone.
    pub skim: bool,
}

pub fn is_wav(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"))
}

pub fn retired(name: &str) -> Option<&'static str> {
    RETIRED
        .iter()
        .find(|(gone, _)| *gone == name)
        .map(|(_, write)| *write)
}

/// A comma list of calls; a comma inside a call's parentheses separates its arguments, and
/// one inside an `alsa:` device name, `hw:1,0`, stays in it where no reading follows it.
pub fn calls(text: &str) -> Result<Vec<Call>, CliError> {
    let mut joined: Vec<String> = Vec::new();
    for one in split_outside(text, ',')? {
        match joined.last_mut() {
            Some(last) if names_a_device(last) && !names_a_reading(one) => {
                last.push(',');
                last.push_str(one);
            }
            _ => joined.push(one.to_string()),
        }
    }
    joined.iter().map(|one| call(one.trim())).collect()
}

fn names_a_device(call: &str) -> bool {
    let close = call.rfind(')').map_or(0, |at| at + 1);
    call[close..]
        .split_once('=')
        .is_some_and(|(_, dest)| dest.trim_start().starts_with("alsa:"))
}

fn names_a_reading(text: &str) -> bool {
    let head = text[..text.find(['(', '=']).unwrap_or(text.len())].trim();
    READINGS.iter().any(|row| row.name == head) || retired(head).is_some()
}

/// `name`, `name(k=v, ...)`, either followed by `=path`.
pub fn call(text: &str) -> Result<Call, CliError> {
    let head_end = text.find(['(', '=']).unwrap_or(text.len());
    let name = text[..head_end].trim().to_string();
    if name.is_empty() {
        return Err(unwritten(text, "it names no reading"));
    }
    let mut rest = &text[head_end..];
    let mut args = Vec::new();
    if rest.starts_with('(') {
        let close = closing(rest).ok_or_else(|| unwritten(text, "its `(` is never closed"))?;
        let inner = rest[1..close].trim();
        if !inner.is_empty() {
            for arg in split_outside(inner, ',')? {
                let Some((key, value)) = arg.split_once('=') else {
                    return Err(unwritten(
                        text,
                        &format!("`{}` is no named argument; write `key=value`", arg.trim()),
                    ));
                };
                args.push((key.trim().to_string(), value.trim().to_string()));
            }
        }
        rest = &rest[close + 1..];
    }
    let dest = match rest.trim_start().strip_prefix('=') {
        Some("") => {
            return Err(unwritten(
                text,
                "`=` names no destination; drop it to read the reading under \
                 `data.representations`",
            ));
        }
        Some(path) => Some(PathBuf::from(path)),
        None if rest.trim().is_empty() => None,
        None => {
            return Err(unwritten(
                text,
                &format!("`{}` follows the call", rest.trim()),
            ));
        }
    };
    Ok(Call { name, args, dest })
}

/// Every call resolved. An answer keys each reading by its name, so a name asked twice
/// would print one key twice and a JSON reader would keep only the last; it refuses.
pub fn asked_all(calls: &[Call]) -> Result<Vec<Asked>, CliError> {
    let mut out: Vec<Asked> = Vec::with_capacity(calls.len());
    for call in calls {
        let one = asked(call)?;
        if out.iter().any(|held| held.name == one.name) {
            return Err(CliError::usage(
                "query.duplicate",
                format!(
                    "`{}` is asked more than once; ask each reading once, every argument in \
                 its one call",
                    one.name
                ),
            ));
        }
        out.push(one);
    }
    Ok(out)
}

/// A time argument as written: seconds, or bars a tempo counts.
#[derive(Clone, Debug, PartialEq)]
struct Time {
    amount: f64,
    unit: SpanUnit,
    written: String,
}

impl Asked {
    fn reading(&self) -> &'static Reading {
        READINGS
            .iter()
            .find(|row| row.name == self.name)
            .expect("an asked reading names a row")
    }

    /// Whether the reading answers off samples alone, as `Representation::off_samples`.
    pub fn off_samples(&self, sampled: bool) -> bool {
        self.reading().reads.off_samples(sampled)
    }

    /// The reading at `rate`, every time argument in seconds: a bar is `seconds_per_bar`,
    /// and one written where no tempo is known refuses.
    pub fn representation(
        &self,
        rate: u32,
        seconds_per_bar: Option<f64>,
    ) -> Result<Representation, CliError> {
        let mut given = self.given;
        for (arg, time) in &self.times {
            let secs = match (time.unit, seconds_per_bar) {
                (SpanUnit::Seconds, _) => time.amount,
                (SpanUnit::Bars, Some(per)) => time.amount * per,
                (SpanUnit::Bars, None) => {
                    return Err(CliError::BadTempo(format!(
                        "`{}({}={})` is written in bars, and this reading has no tempo to \
                         count them in: a render reads it off `variables/bpm` and \
                         `variables/meter`, and a file holds none; write seconds",
                        self.name,
                        arg.key(),
                        time.written
                    )));
                }
            };
            if *arg == Arg::Frame && (secs * f64::from(rate)).round() < 1.0 {
                return Err(CliError::usage(
                    "query.bad_value",
                    format!(
                        "`{}(frame={})` is {secs}s, under half a sample at {rate} Hz, and a \
                         frame steps by whole samples, rounded; ask at least {}s, half a sample",
                        self.name,
                        time.written,
                        0.5 / f64::from(rate)
                    ),
                ));
            }
            match arg {
                Arg::Frame => given.frame = Some(secs),
                Arg::Window => given.window = Some(secs),
                Arg::Cell => given.cell_secs = secs,
                Arg::Smooth => given.smooth_secs = secs,
                _ => unreachable!("only a time argument is held as a time"),
            }
        }
        let representation = self.reading().representation(&given);
        pinned(representation, rate)?;
        Ok(representation)
    }

    pub fn path(&self) -> Option<&Path> {
        match &self.dest {
            Some(Destination::File(path)) => Some(path),
            Some(Destination::Alsa { .. }) | None => None,
        }
    }
}

/// The reading a call names, each argument read as that reading takes it.
pub fn asked(call: &Call) -> Result<Asked, CliError> {
    let name = call.name.as_str();
    if let Some(write) = retired(name) {
        return Err(CliError::usage(
            "query.retired",
            format!("`{name}` left the language; write `{write}`"),
        ));
    }
    let Some(row) = READINGS.iter().find(|row| row.name == name) else {
        let names: Vec<&str> = READINGS.iter().map(|row| row.name).collect();
        return Err(CliError::usage(
            "query.unknown_representation",
            format!(
                "unknown representation `{name}`; the representations are {}",
                names.join(", ")
            ),
        ));
    };
    let (mut given, mut node, mut brief, mut skim) = (Given::default(), None, false, false);
    let (mut buffer, mut times) = (None, Vec::new());
    for (key, raw) in &call.args {
        let Some(arg) = row.takes.iter().find(|arg| arg.key() == key) else {
            let keys: Vec<&str> = row.takes.iter().map(|arg| arg.key()).collect();
            return Err(CliError::usage(
                "query.unknown_argument",
                match keys[..] {
                    [] => format!("`{name}` takes no arguments, not `{key}`"),
                    _ => format!("`{name}` takes {}, not `{key}`", keys.join(", ")),
                },
            ));
        };
        match arg {
            Arg::Peaks => given.peaks = count(raw, key)?,
            Arg::Depth => given.depth = count(raw, key)?,
            Arg::Frame | Arg::Window | Arg::Cell | Arg::Smooth => {
                times.push((*arg, time(raw, key)?));
            }
            Arg::Oversample => given.oversample = factor(raw)?,
            Arg::Brief => brief = switch(raw, key)?,
            Arg::Skim => skim = switch(raw, key)?,
            Arg::Node => node = Some(self::node(raw)?),
            Arg::Buffer => buffer = Some(microseconds(raw, key)?),
            Arg::Floor => given.floor_db = decibels(raw, key)?,
            Arg::Norm => given.norm = norm(raw)?,
            Arg::Pairs => given.max_pairs = count(raw, key)?,
            Arg::Curve => given.curve = switch(raw, key)?,
            Arg::Bpm => given.bpm = Some(tempo(raw, key)?),
        }
    }
    if row.takes.contains(&Arg::Node) && node.is_none() {
        return Err(CliError::usage(
            "query.missing_argument",
            format!("`{name}` reads one instance's {name}; name it, as `{name}(node=@voice)`"),
        ));
    }
    Ok(Asked {
        name: call.name.clone(),
        given,
        times,
        dest: destination(name, call.dest.as_deref(), buffer)?,
        node,
        brief,
        skim,
    })
}

/// `alsa:` opens a device name, and only a device has a buffer to size.
fn destination(
    name: &str,
    dest: Option<&Path>,
    buffer: Option<f64>,
) -> Result<Option<Destination>, CliError> {
    let device = dest.and_then(|d| d.to_str()?.strip_prefix("alsa:"));
    match (dest, device, buffer) {
        (_, Some(""), _) => Err(CliError::usage(
            "destination.no_device",
            format!(
                "`{name}=alsa:` names no device; write an ALSA device after it, as \
                 `alsa:default` or `alsa:hw:1`"
            ),
        )),
        (_, Some(device), _) => Ok(Some(Destination::Alsa {
            device: device.to_string(),
            buffer_secs: buffer.unwrap_or(DEFAULT_BUFFER_SECS),
        })),
        (_, None, Some(_)) => Err(CliError::usage(
            "query.buffer_without_device",
            format!(
                "`buffer` sizes a live device's buffer, and `{name}` names none; write \
                 `{name}(buffer=...)=alsa:<device>`, or drop `buffer`"
            ),
        )),
        (dest, None, None) => Ok(dest.map(|d| Destination::File(d.to_path_buf()))),
    }
}

/// `raw` split at each `sep` no parenthesis encloses.
fn split_outside(raw: &str, sep: char) -> Result<Vec<&str>, CliError> {
    let (mut depth, mut from, mut out) = (0i32, 0, Vec::new());
    for (at, c) in raw.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            c if c == sep && depth == 0 => {
                out.push(&raw[from..at]);
                from = at + c.len_utf8();
            }
            _ => {}
        }
        if depth < 0 {
            return Err(unwritten(raw, "a `)` closes nothing"));
        }
    }
    out.push(&raw[from..]);
    Ok(out)
}

/// The byte of the `)` that closes the `(` `text` opens with.
fn closing(text: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (at, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(at);
                }
            }
            _ => {}
        }
    }
    None
}

fn unwritten(text: &str, why: &str) -> CliError {
    CliError::usage(
        "query.malformed",
        format!(
            "`{text}` is no reading: {why}. write a call, as `spectrum(peaks=8, frame=50ms)`, \
             `=path` after it to write it to a file"
        ),
    )
}

pub fn wav_path(raw: &str) -> Result<PathBuf, CliError> {
    match is_wav(Path::new(raw)) {
        true => Ok(PathBuf::from(raw)),
        false => Err(CliError::usage(
            "analyze.not_wav",
            format!("`{raw}` is no `.wav` file; mp3 and every other format are out of scope"),
        )),
    }
}

fn count(raw: &str, key: &str) -> Result<usize, CliError> {
    raw.parse::<usize>().map_err(|_| {
        CliError::usage(
            "query.bad_value",
            format!("`{key}` needs a whole count, got `{raw}`"),
        )
    })
}

fn factor(raw: &str) -> Result<u32, CliError> {
    match raw.parse::<u32>() {
        Ok(k) if (2..=16).contains(&k) && k.is_power_of_two() => Ok(k),
        _ => Err(CliError::usage(
            "query.bad_value",
            format!("`oversample` needs a power of two from 2 to 16, got `{raw}`"),
        )),
    }
}

/// A positive time as a score folds one, `50ms`, `1b/8` or `2b + 1b/3`; a bare number is
/// seconds. Bars stay bars until a tempo is known.
fn time(raw: &str, key: &str) -> Result<Time, CliError> {
    let bad = |why: String| {
        CliError::usage(
            "query.bad_value",
            format!(
                "`{key}` needs a positive time, in seconds as `50ms` or `0.05`, or in bars as \
                 `1b/8`: {why}"
            ),
        )
    };
    let folding = Folding::<f64> {
        field: Field::Time,
        mark: &|_| None,
        degree: &|_| Err("a reading's time counts no degrees".to_string()),
        own: None,
    };
    let value = evaluate(raw, &folding).map_err(|refused| match refused {
        Refused::At(_, why) => bad(why),
        Refused::Mark(name, _) => bad(format!(
            "`mark({name})` is a score's, and a reading has none"
        )),
    })?;
    let unit = match value.kind {
        Kind::Bars => SpanUnit::Bars,
        Kind::Seconds | Kind::Number => SpanUnit::Seconds,
        other => return Err(bad(format!("`{raw}` is a {}", other.name()))),
    };
    match value.amount > 0.0 && value.amount.is_finite() {
        true => Ok(Time {
            amount: value.amount,
            unit,
            written: raw.to_string(),
        }),
        false => Err(bad(format!("`{raw}` is not over zero"))),
    }
}

/// A transform sized to fit is not the grid two windows asked to share.
fn pinned(representation: Representation, rate: u32) -> Result<(), CliError> {
    let Representation::Spectrum {
        frame_secs: Some(secs),
        ..
    } = representation
    else {
        return Ok(());
    };
    let sr = f64::from(rate);
    let frame = pinned_frame(secs, sr);
    match frame > MAX_PINNED_FRAME {
        true => Err(CliError::usage(
            "query.bad_value",
            format!(
                "`spectrum(frame={secs})` sizes its transform to {frame} samples at {sr} Hz, \
                 past the {MAX_PINNED_FRAME}-sample bound"
            ),
        )),
        false => Ok(()),
    }
}

/// A positive time the language spells, `50ms` or `1.5s`; a bare number is seconds.
pub fn seconds(raw: &str, key: &str) -> Result<f64, CliError> {
    let held = match sva_ast::tokenize(raw).as_deref() {
        Ok([one]) => match one.kind {
            sva_ast::TokenKind::Num(v)
            | sva_ast::TokenKind::Time(v, sva_ast::SpanUnit::Seconds) => Some(v),
            _ => None,
        },
        _ => None,
    };
    match held {
        Some(v) if v > 0.0 && v.is_finite() => Ok(v),
        _ => Err(CliError::usage(
            "query.bad_value",
            format!("`{key}` needs a positive time, as `50ms` or `0.05`, got `{raw}`"),
        )),
    }
}

/// A bare positive number of beats per minute.
fn tempo(raw: &str, key: &str) -> Result<f64, CliError> {
    let held = match sva_ast::tokenize(raw).as_deref() {
        Ok([one]) => match one.kind {
            sva_ast::TokenKind::Num(v) => Some(v),
            _ => None,
        },
        _ => None,
    };
    match held {
        Some(v) if v > 0.0 && v.is_finite() => Ok(v),
        _ => Err(CliError::usage(
            "query.bad_value",
            format!("`{key}` needs a positive tempo in beats per minute, as `140`, got `{raw}`"),
        )),
    }
}

/// A device takes its buffer in whole microseconds, as a C `int`.
fn microseconds(raw: &str, key: &str) -> Result<f64, CliError> {
    let secs = seconds(raw, key)?;
    match (1.0..=f64::from(i32::MAX)).contains(&(secs * 1e6).round()) {
        true => Ok(secs),
        false => Err(CliError::usage(
            "query.bad_value",
            format!(
                "`{key}` is a whole number of microseconds from 1us to {}s, got `{raw}`",
                i32::MAX / 1_000_000
            ),
        )),
    }
}

/// A level the language spells in decibels, `-40db`; a bare number names no unit.
fn decibels(raw: &str, key: &str) -> Result<f64, CliError> {
    use sva_ast::{LogUnit::Decibels, TokenKind};
    let kinds: Option<Vec<TokenKind>> = sva_ast::tokenize(raw)
        .ok()
        .map(|tokens| tokens.into_iter().map(|t| t.kind).collect());
    let held = match kinds.as_deref() {
        Some([TokenKind::Log(v, Decibels)]) => Some(*v),
        Some([TokenKind::Minus, TokenKind::Log(v, Decibels)]) => Some(-v),
        _ => None,
    };
    match held {
        Some(v) if v.is_finite() => Ok(v),
        _ => Err(CliError::usage(
            "query.bad_value",
            format!("`{key}` needs a level in decibels, as `-40db`, got `{raw}`"),
        )),
    }
}

fn norm(raw: &str) -> Result<Norm, CliError> {
    [Norm::L1, Norm::L2]
        .into_iter()
        .find(|n| n.key() == raw)
        .ok_or_else(|| {
            CliError::usage(
                "query.bad_value",
                format!("`norm` is `l1` or `l2`, got `{raw}`"),
            )
        })
}

fn switch(raw: &str, key: &str) -> Result<bool, CliError> {
    match raw {
        "1" => Ok(true),
        "0" => Ok(false),
        _ => Err(CliError::usage(
            "query.bad_value",
            format!("`{key}` is `1` or `0`, got `{raw}`"),
        )),
    }
}

/// A ref as the language writes one, `@voice`.
fn node(raw: &str) -> Result<String, CliError> {
    match raw.strip_prefix('@') {
        Some(path) if !path.is_empty() => Ok(path.to_string()),
        _ => Err(CliError::usage(
            "query.bad_value",
            format!("`node` names an instance as a ref, as `@voice`, not `{raw}`"),
        )),
    }
}
