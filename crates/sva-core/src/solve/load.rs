// Concern: reads each load of a cost once, before any render, checking a reading against its leaf | Non-concern: one reading's JSON (decode.rs), the cost | IO: (Cost, files) -> Vec<Held>

use std::collections::BTreeMap;

use serde_json::Value;
use sva_ast::Source;
use sva_engine::fitted_frame;

use super::cost::{Cost, From, Held, Leaf, Load, Step};
use crate::cli_error::CliError;
use crate::decode::{reading, shape};
use crate::query::{Call, Spelled, Taken, asked};
use crate::{Job, planned};

/// Where `load_json` finds a file's text: in the CLI, relative to the directory it runs in.
pub trait Texts {
    /// `Ok(None)` is no file there; `Err` is one that could not be read as text.
    fn text(&self, path: &str) -> Result<Option<String>, String>;
}

/// No file at all: a front end with no filesystem hands each value over instead.
pub struct NoFiles;

impl Texts for NoFiles {
    fn text(&self, _: &str) -> Result<Option<String>, String> {
        Ok(None)
    }
}

/// What the render side of a reading decides, which a loaded reading must match.
pub struct Side<'a> {
    pub source: &'a dyn Source,
    pub rate: Option<u32>,
}

fn refused(code: &'static str, message: String) -> CliError {
    CliError::usage(code, message)
}

/// Each load's value, every file read and parsed once; a reading decoded by its leaf and
/// checked against what that leaf renders.
pub fn loaded(cost: &Cost, texts: &dyn Texts, side: &Side) -> Result<Vec<Held>, CliError> {
    let mut files: BTreeMap<&str, Value> = BTreeMap::new();
    let paired = cost.paired();
    let mut out = Vec::with_capacity(cost.loads.len());
    for (load, pair) in cost.loads.iter().zip(paired) {
        let value = match &load.from {
            From::Value(value) => value,
            From::File { file, path, .. } => {
                if !files.contains_key(file.as_str()) {
                    files.insert(file, parsed(file, texts)?);
                }
                walked(&files[file.as_str()], load, path)?
            }
        };
        out.push(held(load, value, pair.map(|at| &cost.leaves[at]), side)?);
    }
    Ok(out)
}

fn parsed(file: &str, texts: &dyn Texts) -> Result<Value, CliError> {
    let text = texts
        .text(file)
        .map_err(|why| {
            refused(
                "solve.bad_json",
                format!("`{file}` could not be read: {why}"),
            )
        })?
        .ok_or_else(|| {
            CliError::NotFound(format!(
                "`load_json(\"{file}\", ...)` names no file: a path is relative to the directory \
                 sva-cli runs in, or absolute"
            ))
        })?;
    serde_json::from_str(&text).map_err(|e| {
        refused(
            "solve.bad_json",
            format!("`{file}` is no JSON: {e}; `load_json` reads one JSON document"),
        )
    })
}

fn where_in(load: &Load) -> String {
    match &load.from {
        From::File {
            file, path_text, ..
        } => format!("`{file}` at `{path_text}`"),
        From::Value(_) => format!("`{}`", load.written),
    }
}

/// The value `path` names; a step that finds nothing names what is there instead.
fn walked<'v>(root: &'v Value, load: &Load, path: &[Step]) -> Result<&'v Value, CliError> {
    let From::File { file, .. } = &load.from else {
        unreachable!("only a file is walked");
    };
    let mut at = root;
    let mut reached = String::new();
    for step in path {
        let next = match (step, at) {
            (Step::Key(key), Value::Object(map)) => map.get(key),
            (Step::Index(k), Value::Array(list)) => list.get(*k),
            _ => None,
        };
        let Some(next) = next else {
            let held = match at {
                Value::Object(map) => {
                    let keys: Vec<String> = map.keys().take(24).map(|k| written_key(k)).collect();
                    format!("holds the keys {}", keys.join(", "))
                }
                Value::Array(list) => format!("is a list of {} items, from `[0]`", list.len()),
                other => format!("is {}", shape(other)),
            };
            let step = match step {
                Step::Key(key) => written_key(key),
                Step::Index(k) => format!("[{k}]"),
            };
            let here = if reached.is_empty() { "." } else { &reached };
            return Err(refused(
                "solve.no_such_key",
                format!("`{file}` has no `{step}` under `{here}`: `{here}` {held}"),
            ));
        };
        reached.push_str(&match step {
            Step::Key(key) => written_key(key),
            Step::Index(k) => format!("[{k}]"),
        });
        at = next;
    }
    Ok(at)
}

/// A key as a path step spells it: `.name`, else `['any key']`.
fn written_key(key: &str) -> String {
    let plain = key.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    match plain {
        true => format!(".{key}"),
        false => format!("['{key}']"),
    }
}

fn held(load: &Load, value: &Value, leaf: Option<&Leaf>, side: &Side) -> Result<Held, CliError> {
    let here = where_in(load);
    match value {
        Value::Number(n) => Ok(Held::Number(n.as_f64().unwrap_or(f64::NAN))),
        Value::Array(list) => list
            .iter()
            .map(Value::as_f64)
            .collect::<Option<Vec<f64>>>()
            .map(Held::Vector)
            .ok_or_else(|| {
                refused(
                    "solve.bad_load",
                    format!(
                        "{here} is a list that holds more than numbers; a list loads as a \
                         vector of numbers"
                    ),
                )
            }),
        Value::Object(_) => match leaf {
            Some(leaf) => decoded(value, leaf, &here, side).map(Held::Reading),
            None => Err(refused(
                "solve.unpaired_load",
                format!(
                    "{here} holds an object, which loads only as the reading a `repr.` leaf \
                     subtracts, as `norm(repr.bands(@a) - {})`",
                    load.written
                ),
            )),
        },
        other => Err(refused(
            "solve.bad_load",
            format!(
                "{here} is {}; a load holds a number, a list of numbers, or a reading as \
                 `analyze` prints its `data`",
                shape(other)
            ),
        )),
    }
}

/// The first `has_more: true` under `value`, where a list was cut short.
fn cut_short(value: &Value, path: &str) -> Option<String> {
    match value {
        Value::Object(map) => {
            if map.get("has_more") == Some(&Value::Bool(true)) {
                return Some(path.to_string());
            }
            map.iter()
                .find_map(|(k, v)| cut_short(v, &format!("{path}{}", written_key(k))))
        }
        Value::Array(list) => list
            .iter()
            .enumerate()
            .find_map(|(k, v)| cut_short(v, &format!("{path}[{k}]"))),
        _ => None,
    }
}

fn mismatched(here: &str, leaf: &Leaf, what: String) -> CliError {
    refused(
        "solve.mismatched_load",
        format!("{here} and `{}` differ: {what}", leaf.written),
    )
}

/// `value` is analyze's `data`: its interval, rate and profile beside `representations`.
fn decoded(
    value: &Value,
    leaf: &Leaf,
    here: &str,
    side: &Side,
) -> Result<sva_engine::Output, CliError> {
    let name = &leaf.asked.name;
    let Some(readings) = value.get("representations").and_then(Value::as_object) else {
        let keys: Vec<String> = value
            .as_object()
            .map(|map| map.keys().take(24).map(|k| written_key(k)).collect())
            .unwrap_or_default();
        return Err(refused(
            "solve.bad_load",
            format!(
                "{here} holds {} and no `representations`; a reading loads from `analyze`'s \
                 `data`, the object holding `interval` and `representations`: from a bench \
                 download `.readings.{name}`, from a reading `analyze` wrote to a file `.data`",
                keys.join(", ")
            ),
        ));
    };
    if let Some(at) = cut_short(value, "") {
        return Err(refused(
            "solve.cut_short",
            format!(
                "{here} was cut short at `{at}` (`has_more: true`); write the reading whole, \
                 to a file, as `--representation {name}=ref.json`, or pass `limit: null`"
            ),
        ));
    }
    let Some(answer) = readings.get(name.as_str()) else {
        let held: Vec<&str> = readings.keys().map(String::as_str).collect();
        return Err(mismatched(
            here,
            leaf,
            format!("it holds {} and no `{name}`", held.join(", ")),
        ));
    };
    let field = |key: &str| answer.get(key).or_else(|| value.get(key));
    let Some((theirs_name, theirs_args)) = answer
        .get("representation")
        .and_then(Value::as_str)
        .zip(answer.get("args").and_then(Value::as_object))
    else {
        return Err(refused(
            "solve.load_without_args",
            format!(
                "{here} holds `{name}` with no `representation` and `args`, what it was read \
                 with, so its arguments cannot be checked; it was printed before readings \
                 carried them: read it again"
            ),
        ));
    };
    if theirs_name != name {
        return Err(mismatched(
            here,
            leaf,
            format!("it holds a `{theirs_name}` reading under `{name}`"),
        ));
    }
    let job = Job {
        rate: side.rate,
        ..Job::over(side.source, &leaf.target)
    };
    let (render, _, per_bar) = planned(&job)?;
    let rate = render.config.rate;
    let profile = render.config.profile.name;
    if field("source").and_then(Value::as_str) != Some("measured") {
        return Err(mismatched(
            here,
            leaf,
            "it was not measured off samples (`source` is not `measured`), and a cost reads \
             every leaf off samples; read it as `sample(<target>)`"
                .to_string(),
        ));
    }
    for (key, at) in [("sample_rate", value), ("rate", answer)] {
        let theirs = at.get(key).and_then(Value::as_u64);
        if theirs != Some(u64::from(rate)) {
            let theirs = theirs.map_or("no rate".to_string(), |r| format!("{r} Hz"));
            return Err(mismatched(
                here,
                leaf,
                format!(
                    "it was read at {theirs}, and the solve renders at {rate} Hz; solve at its \
                     rate, `--rate` (a page's `rate` option)"
                ),
            ));
        }
    }
    for at in [value, answer] {
        let theirs = at.get("profile").and_then(Value::as_str);
        if theirs != Some(profile) {
            return Err(mismatched(
                here,
                leaf,
                format!(
                    "it was read under the profile `{}`, and the solve renders under `{profile}`",
                    theirs.unwrap_or("none")
                ),
            ));
        }
    }
    let span = |key: &str| {
        value
            .get("interval")
            .and_then(|i| i.get(key))
            .and_then(Value::as_f64)
    };
    let theirs = match (span("start_secs"), span("end_secs")) {
        (Some(start), Some(end)) => Some(((end - start) * f64::from(rate)).round()),
        _ => None,
    };
    let ours = render.range.map(|r| r.len() as f64);
    if theirs.is_none() || theirs != ours {
        let samples = |n: Option<f64>| {
            n.map_or("no interval".to_string(), |n| {
                format!("{n} samples ({} s)", n / f64::from(rate))
            })
        };
        return Err(mismatched(
            here,
            leaf,
            format!(
                "it reads {}, and the leaf's target renders {}; frames past the shorter side \
                 would count as silence, so crop the target to the same length",
                samples(theirs),
                samples(ours)
            ),
        ));
    }
    let taken = Taken::Measured {
        fitted_frame: render.range.map(|r| fitted_frame(r.len())),
    };
    let mut written = Vec::with_capacity(theirs_args.len());
    for (key, value) in theirs_args {
        match value {
            Value::String(text) => written.push((key.clone(), text.clone())),
            Value::Number(n) => written.push((key.clone(), n.to_string())),
            Value::Null => {}
            other => {
                return Err(refused(
                    "solve.bad_reading",
                    format!(
                        "{here} holds `{name}` with `args.{key}` {}; an argument is a number or a \
                         string as `--representation` spells it, or null",
                        shape(other)
                    ),
                ));
            }
        }
    }
    let theirs = asked(&Call {
        name: name.clone(),
        args: written,
        dest: None,
    })
    .and_then(|one| one.args(rate, None, taken))
    .map_err(|e| {
        refused(
            "solve.bad_reading",
            format!("{here} holds `{name}` with `args` its reading refuses: {e}"),
        )
    })?;
    let ours = leaf.asked.args(rate, per_bar, taken)?;
    let spelled = |v: &Option<Spelled>| {
        v.as_ref()
            .map_or("none".to_string(), |v| format!("`{}`", v.text()))
    };
    for ((key, theirs), (_, ours)) in theirs.iter().zip(&ours) {
        if theirs != ours {
            return Err(mismatched(
                here,
                leaf,
                format!(
                    "its `{key}` is {}, and the leaf's is {}",
                    spelled(theirs),
                    spelled(ours)
                ),
            ));
        }
    }
    let Some(inner) = answer.get("value") else {
        return Err(refused(
            "solve.bad_reading",
            format!("{here} holds `{name}` with no `value`"),
        ));
    };
    let representation = leaf.asked.representation(rate, per_bar)?;
    let path = format!("representations.{name}.value");
    reading(representation, inner, &path)
        .map_err(|why| refused("solve.bad_reading", format!("{here}: {why}")))
}
