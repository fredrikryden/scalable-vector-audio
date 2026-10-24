// Concern: reads a score's JSON lines into kinds and items, resolving its marks, times and their order | Non-concern: folding a value (value.rs), the song around it | IO: (&str) -> Score or a Diag

use std::collections::HashMap;
use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;

use crate::diag::{ByteSpan, Diag};
use crate::expr::{Expr, Literal};
use crate::filename::SpanUnit;
use crate::score::value::{Field, Kind, Reading, Refused, Value, evaluate};
use crate::score::{
    Bind, EXTENSION, Event, Item, Line, Mark, Move, Named, Ramp, Ratio, Score, Shape, Time, To,
    located,
};
use crate::series::folded;

const FORMAT_KEYS: [&str; 9] = [
    "annotation",
    "at",
    "node",
    "deg",
    "mark",
    "comment",
    "lane",
    "to",
    "shape",
];
const MOVE_KEYS: [&str; 7] = ["at", "node", "lane", "to", "len", "shape", "comment"];
const MARK_KEYS: [&str; 3] = ["at", "mark", "comment"];
const EVENT_KEYS: &str = "`at`, `node`, `deg`, `len`, `comment` and the node's parameters";

/// A file with no extension whose first line, past blank and `;` lines, opens with `{`.
pub(crate) fn is_old_score(content: &str) -> bool {
    content
        .lines()
        .find(|line| !line.trim().is_empty() && !line.trim_start().starts_with(';'))
        .is_some_and(|line| line.trim_start().starts_with('{'))
}

#[derive(Clone, Copy, PartialEq)]
enum LineKind {
    Event,
    Mark,
    Comment,
    Move,
}

/// One field's key and its value's raw text, each beside the byte it starts at.
struct Keyed<'a> {
    key: String,
    key_at: usize,
    raw: &'a str,
    at: usize,
}

/// One value of one item: a list's item or the field's single value.
#[derive(Clone, Copy)]
struct Raw<'a> {
    text: &'a str,
    at: usize,
}

impl Raw<'_> {
    fn is_string(&self) -> bool {
        self.text.starts_with('"')
    }
}

/// A line read up to its times, which wait for every mark in the file.
struct Pending<'a> {
    number: usize,
    span: ByteSpan,
    kind: LineKind,
    count: usize,
    fields: Vec<(Keyed<'a>, Vec<Raw<'a>>)>,
}

impl<'a> Pending<'a> {
    fn raw(&self, key: &str, i: usize) -> Option<Raw<'a>> {
        let (_, items) = self.fields.iter().find(|(f, _)| f.key == key)?;
        Some(items[if items.len() == 1 { 0 } else { i }])
    }
}

pub fn parse_score(text: &str) -> Result<Score, Diag> {
    let refuse = |at: usize, message: String| located(text, at, &message);
    let rows: Vec<&str> = text.split_terminator('\n').collect();
    let mut annotation = None;
    let mut pending: Vec<Pending> = Vec::new();
    let mut start = 0usize;
    for (index, row) in rows.iter().enumerate() {
        let span = ByteSpan::new(start, start + row.len());
        start += row.len() + 1;
        let number = index + 1;
        if row.trim().is_empty() {
            return Err(refuse(
                span.start,
                format!(
                    "line {number} is empty, and an empty line is no JSON; a score is one JSON \
                     object per line, and only the file's last line may end it"
                ),
            ));
        }
        if row.trim_start().starts_with(';') {
            return Err(refuse(
                span.start,
                "a `;` line is no JSON; a score documents itself in its line-1 `annotation`, \
                 and a note at a time is a timed comment, {\"at\": \"4b\", \"comment\": \"...\"}"
                    .to_string(),
            ));
        }
        let fields = fields(row, span.start).map_err(|(at, why)| refuse(at, why))?;
        if let Some(f) = fields.iter().find(|f| f.key == "annotation") {
            if number != 1 || fields.len() != 1 {
                return Err(refuse(
                    f.key_at,
                    "`annotation` is line 1 only, alone: {\"annotation\": \"Models: ... | \
                     Neglects: ... | IO: ... | Tags: ...\"}"
                        .to_string(),
                ));
            }
            annotation = Some(
                string(
                    Raw {
                        text: f.raw,
                        at: f.at,
                    },
                    "annotation",
                )
                .map_err(|(at, why)| refuse(at, why))?,
            );
            continue;
        }
        pending.push(read_line(number, span, fields).map_err(|(at, why)| refuse(at, why))?);
    }
    let marks = marks(&pending).map_err(|(at, why)| refuse(at, why))?;
    let mut unit: Option<(SpanUnit, usize)> = None;
    let mut lines = Vec::with_capacity(pending.len());
    let mut before: Option<(Ratio, usize)> = None;
    for line in &pending {
        let mut items = Vec::with_capacity(line.count);
        for i in 0..line.count {
            let item = item(line, i, &marks, &mut unit).map_err(|(at, why)| refuse(at, why))?;
            let at = item.at();
            let earlier = match items.last().map(Item::at) {
                Some(last) => Some((last.amount, line.number)),
                None => before,
            };
            if let Some((was, on)) = earlier
                && at.amount < was
            {
                let unit = unit.expect("a time was read").0;
                let where_ = match on == line.number {
                    true => "earlier on this line".to_string(),
                    false => format!("on line {on}"),
                };
                return Err(refuse(
                    at.at,
                    format!(
                        "`at` is {} here, before {} {where_}; lines are sorted by `at`, and a \
                         list's items never go backwards",
                        crate::score::spelled(at.amount.to_f64(), unit),
                        crate::score::spelled(was.to_f64(), unit)
                    ),
                ));
            }
            items.push(item);
        }
        before = items.first().map(|i| (i.at().amount, line.number));
        lines.push(Line {
            number: line.number,
            span: line.span,
            items,
        });
    }
    Ok(Score {
        annotation,
        unit: unit.map_or(SpanUnit::Bars, |(u, _)| u),
        lines,
    })
}

type Inside = (usize, String);

fn fields(row: &str, start: usize) -> Result<Vec<Keyed<'_>>, Inside> {
    let Fields(raw) = serde_json::from_str(row).map_err(|e| {
        let message = e.to_string();
        let message = message
            .rsplit_once(" at line ")
            .map_or(&*message, |(m, _)| m);
        let twice = message
            .strip_prefix('`')
            .and_then(|m| m.split_once("` is written twice"))
            .and_then(|(key, _)| row.rfind(&format!("\"{key}\"")));
        let at = twice.unwrap_or(e.column().max(1) - 1);
        (start + at, message.to_string())
    })?;
    Ok(raw
        .into_iter()
        .map(|(key, value)| {
            let from = value.get().as_ptr() as usize - row.as_ptr() as usize;
            let key_at = row[..from].rfind(&format!("\"{key}\"")).unwrap_or(from);
            Keyed {
                key,
                key_at: start + key_at,
                raw: value.get(),
                at: start + from,
            }
        })
        .collect())
}

fn read_line<'a>(
    number: usize,
    span: ByteSpan,
    fields: Vec<Keyed<'a>>,
) -> Result<Pending<'a>, Inside> {
    if let Some(f) = fields.iter().find(|f| f.key == "sound") {
        return Err((
            f.key_at,
            "`sound` is now `node`: a line plays any node, a sound or a score; write \"node\": \
             \"@../instruments/bass\""
                .to_string(),
        ));
    }
    let has = |key: &str| fields.iter().any(|f| f.key == key);
    let kind = match () {
        _ if has("lane") => LineKind::Move,
        _ if has("mark") => LineKind::Mark,
        _ if has("node") => LineKind::Event,
        _ if has("at") && has("comment") && fields.len() == 2 => LineKind::Comment,
        _ => {
            return Err((
                span.start,
                "a line plays a node, marks, moves a lane, or comments: it holds `node`, `mark` \
                 or `lane`, and a timed comment is exactly `at` and `comment`"
                    .to_string(),
            ));
        }
    };
    let taken = |key: &str| match kind {
        LineKind::Move => MOVE_KEYS.contains(&key),
        LineKind::Mark => MARK_KEYS.contains(&key),
        LineKind::Comment => key == "at" || key == "comment",
        LineKind::Event => {
            !FORMAT_KEYS.contains(&key) || ["at", "node", "deg", "comment"].contains(&key)
        }
    };
    if let Some(f) = fields.iter().find(|f| !taken(&f.key)) {
        let keys = match kind {
            LineKind::Move => "`at`, `node`, `lane`, `to`, `len`, `shape` and `comment`",
            LineKind::Mark => "`at`, `mark` and `comment`",
            LineKind::Comment => "`at` and `comment`",
            LineKind::Event => EVENT_KEYS,
        };
        let name = match kind {
            LineKind::Move => "a move",
            LineKind::Mark => "a mark",
            LineKind::Comment => "a timed comment",
            LineKind::Event => "an event",
        };
        return Err((
            f.key_at,
            format!("{name} takes no `{}`; it takes {keys}", f.key),
        ));
    }
    let end = span.end;
    if !has("at") {
        return Err((
            end,
            "`at` is missing; every line but the annotation has one".to_string(),
        ));
    }
    if kind == LineKind::Move && !has("to") {
        return Err((
            end,
            "a move needs `to`, the value its lane goes to".to_string(),
        ));
    }
    let mut listed: Option<(String, usize)> = None;
    let mut out = Vec::with_capacity(fields.len());
    for f in fields {
        let items = items(&f)?;
        if f.raw.starts_with('[') {
            match &listed {
                Some((first, n)) if *n != items.len() => {
                    return Err((
                        f.at,
                        format!(
                            "`{first}` lists {}, `{}` lists {}; lists on one line hold one item \
                             per event, move or mark, so they are one length",
                            counted(*n),
                            f.key,
                            counted(items.len())
                        ),
                    ));
                }
                Some(_) => {}
                None => listed = Some((f.key.clone(), items.len())),
            }
        }
        out.push((f, items));
    }
    Ok(Pending {
        number,
        span,
        kind,
        count: listed.map_or(1, |(_, n)| n),
        fields: out,
    })
}

fn counted(n: usize) -> String {
    format!("{n} item{}", if n == 1 { "" } else { "s" })
}

fn json_type(raw: &str) -> Option<&'static str> {
    match raw.as_bytes().first() {
        Some(b't' | b'f') => Some("a JSON boolean"),
        Some(b'n') => Some("JSON null"),
        Some(b'{') => Some("a JSON object"),
        _ => None,
    }
}

fn items<'a>(f: &Keyed<'a>) -> Result<Vec<Raw<'a>>, Inside> {
    let typed = |raw: Raw<'a>| match json_type(raw.text) {
        Some(kind) => Err((
            raw.at,
            format!("`{}` holds {kind}; a value is a number or a string", f.key),
        )),
        None => Ok(raw),
    };
    if !f.raw.starts_with('[') {
        return typed(Raw {
            text: f.raw,
            at: f.at,
        })
        .map(|raw| vec![raw]);
    }
    if f.key == "comment" || f.key == "annotation" {
        return Err((
            f.at,
            format!(
                "`{}` is a list; a comment is never broadcast, so write one string",
                f.key
            ),
        ));
    }
    let list: Vec<&RawValue> =
        serde_json::from_str(f.raw).map_err(|e| (f.at, format!("`{}` {e}", f.key)))?;
    if list.is_empty() {
        return Err((
            f.at,
            format!(
                "`{}` lists no item; a list holds one item per event, as [1, 3, 5]",
                f.key
            ),
        ));
    }
    list.iter()
        .map(|item| {
            let at = f.at + (item.get().as_ptr() as usize - f.raw.as_ptr() as usize);
            match item.get().starts_with('[') {
                true => Err((
                    at,
                    format!("`{}` holds a list in a list; each item is one value", f.key),
                )),
                false => typed(Raw {
                    text: item.get(),
                    at,
                }),
            }
        })
        .collect()
}

/// A JSON string's contents, which a score value never needs to escape.
fn string(raw: Raw, key: &str) -> Result<String, Inside> {
    if !raw.is_string() {
        return Err((raw.at, format!("`{key}` is {}, not a string", raw.text)));
    }
    serde_json::from_str::<&str>(raw.text)
        .map(str::to_string)
        .map_err(|_| {
            (
                raw.at,
                format!("`{key}` holds a `\\` escape, and no score value needs one"),
            )
        })
}

fn mark_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes.next().is_some_and(|b| b.is_ascii_lowercase())
        && bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// Every mark's position from the file's zero, each `at` read once the marks it names are.
fn marks(pending: &[Pending]) -> Result<HashMap<String, Value<Ratio>>, Inside> {
    let mut written: Vec<(String, Raw, usize)> = Vec::new();
    for line in pending.iter().filter(|l| l.kind == LineKind::Mark) {
        for i in 0..line.count {
            let raw = line.raw("mark", i).expect("a mark line holds `mark`");
            let name = string(raw, "mark")?;
            if !mark_name(&name) {
                return Err((
                    raw.at,
                    format!(
                        "mark `{name}` is no mark name: one matches [a-z][a-z0-9_]*, so it never \
                         reads as arithmetic, as `pre_drop`"
                    ),
                ));
            }
            if let Some((_, _, on)) = written.iter().find(|(held, ..)| *held == name) {
                return Err((
                    raw.at,
                    format!("mark `{name}` is defined twice, here and on line {on}"),
                ));
            }
            written.push((name, line.raw("at", i).expect("checked"), line.number));
        }
    }
    let mut resolved: HashMap<String, Value<Ratio>> = HashMap::new();
    loop {
        let before = resolved.len();
        for (name, at, _) in &written {
            if resolved.contains_key(name) {
                continue;
            }
            match time(*at, &resolved, "at") {
                Ok(value) => {
                    resolved.insert(
                        name.clone(),
                        Value {
                            position: true,
                            ..value
                        },
                    );
                }
                Err(Refused::Mark(named, by)) => {
                    if !written.iter().any(|(held, ..)| *held == named) {
                        return Err((
                            at.at + by,
                            format!("mark({named}) names no mark in this file"),
                        ));
                    }
                }
                Err(Refused::At(by, why)) => return Err((inside(*at) + by, why)),
            }
        }
        if resolved.len() == written.len() {
            return Ok(resolved);
        }
        if resolved.len() == before {
            let left: Vec<&str> = written
                .iter()
                .filter(|(name, ..)| !resolved.contains_key(name))
                .map(|(name, ..)| name.as_str())
                .collect();
            let (_, at, _) = written
                .iter()
                .find(|(name, ..)| !resolved.contains_key(name))
                .expect("one is left");
            return Err((
                at.at,
                format!(
                    "the marks `{}` are defined in a cycle: each one's `at` reads another",
                    left.join("`, `")
                ),
            ));
        }
    }
}

/// The byte a string's contents start at.
fn inside(raw: Raw) -> usize {
    raw.at + usize::from(raw.is_string())
}

fn time(
    raw: Raw,
    marks: &HashMap<String, Value<Ratio>>,
    key: &str,
) -> Result<Value<Ratio>, Refused> {
    if !raw.is_string() {
        return Err(Refused::At(
            0,
            format!(
                "`{key}` is {}, a bare number; a time names its unit, as \"{}b\" or \"{}s\"",
                raw.text, raw.text, raw.text
            ),
        ));
    }
    let s = serde_json::from_str::<&str>(raw.text).map_err(|_| {
        Refused::At(
            0,
            format!("`{key}` holds a `\\` escape, and no time needs one"),
        )
    })?;
    let mark = |name: &str| marks.get(name).copied();
    let reading = Reading {
        field: Field::Time,
        mark: &mark,
        degree: &|_| Err(String::new()),
        own: None,
    };
    let value = evaluate(s, &reading)?;
    if !value.kind.is_time() {
        return Err(Refused::At(
            0,
            format!(
                "`{s}` is a {}, no time; a time names its unit, as \"0.25b\" or \"0.5s\", and a \
                 bare number has none",
                value.kind.name()
            ),
        ));
    }
    Ok(value)
}

/// A time field of one item, read in the file's one unit.
fn timed(
    raw: Raw,
    key: &str,
    marks: &HashMap<String, Value<Ratio>>,
    unit: &mut Option<(SpanUnit, usize)>,
    line: usize,
) -> Result<Value<Ratio>, Inside> {
    let value = time(raw, marks, key).map_err(|r| match r {
        Refused::At(by, why) => (inside(raw) + by, why),
        Refused::Mark(name, by) => (
            inside(raw) + by,
            format!("mark({name}) names no mark in this file"),
        ),
    })?;
    let this = match value.kind {
        Kind::Bars => SpanUnit::Bars,
        _ => SpanUnit::Seconds,
    };
    match unit {
        Some((held, on)) if *held != this => {
            let name = |u: SpanUnit| match u {
                SpanUnit::Bars => "bars",
                SpanUnit::Seconds => "seconds",
            };
            Err((
                raw.at,
                format!(
                    "this `{key}` counts in {} and line {on} in {}; one file counts all its times \
                     in bars or all in seconds",
                    name(this),
                    name(*held)
                ),
            ))
        }
        Some(_) => Ok(value),
        None => {
            *unit = Some((this, line));
            Ok(value)
        }
    }
}

fn item(
    line: &Pending,
    i: usize,
    marks: &HashMap<String, Value<Ratio>>,
    unit: &mut Option<(SpanUnit, usize)>,
) -> Result<Item, Inside> {
    let raw = |key: &str| line.raw(key, i);
    let at_raw = raw("at").expect("checked");
    let at = timed(at_raw, "at", marks, unit, line.number)?;
    let at = Time {
        amount: at.amount,
        at: at_raw.at,
    };
    let len = match raw("len") {
        None => None,
        Some(len_raw) => {
            let len = timed(len_raw, "len", marks, unit, line.number)?;
            if len.position {
                return Err((
                    len_raw.at,
                    "`len` holds a position: a mark is a position, and `mark(a) - mark(b)` is the \
                     duration between two"
                        .to_string(),
                ));
            }
            if !len.amount.is_positive() {
                return Err((
                    len_raw.at,
                    format!(
                        "`len` is {}; a window has a length, so it is greater than 0",
                        len_raw.text
                    ),
                ));
            }
            Some(Time {
                amount: len.amount,
                at: len_raw.at,
            })
        }
    };
    Ok(match line.kind {
        LineKind::Comment => Item::Comment(at),
        LineKind::Mark => Item::Mark(Mark {
            at,
            name: string(raw("mark").expect("a mark"), "mark")?,
        }),
        LineKind::Move => Item::Move(moved(line, i, at, len)?),
        LineKind::Event => Item::Event(event(line, i, at, len)?),
    })
}

fn node(raw: Raw, key: &str) -> Result<Named, Inside> {
    let refuse = || {
        (
            raw.at,
            format!(
                "`{key}` is an `@ref` string such as \"@../instruments/bass\", not {}",
                raw.text
            ),
        )
    };
    let s = string(raw, key).map_err(|_| refuse())?;
    let Some(path) = s.strip_prefix('@') else {
        return Err(refuse());
    };
    if let Some(bare) = path.strip_suffix(EXTENSION) {
        return Err((
            raw.at,
            format!("`{s}` names a file's extension; a ref drops `{EXTENSION}`, as `@{bare}`"),
        ));
    }
    match !path.is_empty() && crate::lexer::whole_ref_path(path) {
        true => Ok(Named {
            name: path.to_string(),
            at: raw.at,
        }),
        false => Err(refuse()),
    }
}

fn event(line: &Pending, i: usize, at: Time, len: Option<Time>) -> Result<Event, Inside> {
    let deg = match line.raw("deg", i) {
        None => None,
        Some(raw) => match serde_json::from_str::<i64>(raw.text) {
            Ok(d) => Some((d, raw.at)),
            Err(_) => {
                return Err((
                    raw.at,
                    format!(
                        "`deg` is {}; a degree is a whole number counted from 1, the tonic, and \
                         a chromatic note binds `f0`, as \"f0\": \"deg(7)*1st\"",
                        raw.text
                    ),
                ));
            }
        },
    };
    if let (Some((_, d_at)), Some(_)) = (deg, line.raw("f0", i)) {
        return Err((
            d_at,
            "`deg` and `f0` both set the pitch; one pitch per line".to_string(),
        ));
    }
    let mut binds = Vec::new();
    for (f, _) in &line.fields {
        if FORMAT_KEYS.contains(&f.key.as_str()) || f.key == "len" {
            continue;
        }
        let raw = line.raw(&f.key, i).expect("a field");
        binds.push(Bind {
            name: f.key.clone(),
            value: value(raw)?,
            key: f.key_at,
            at: raw.at,
        });
    }
    Ok(Event {
        at,
        node: node(line.raw("node", i).expect("an event"), "node")?,
        deg,
        len,
        binds,
    })
}

/// A number, or a constant in a string: literals, arithmetic over them and `deg(n)`. Numbers
/// alone fold here, so `"0.8*0.9"` binds the number `0.72` does.
fn value(raw: Raw) -> Result<Expr, Inside> {
    if let Ok(n) = serde_json::from_str::<f64>(raw.text) {
        return Ok(Expr::Lit(Literal::Num(n)));
    }
    let s = string(raw, "value")?;
    let from = inside(raw);
    let reading = Reading::<f64> {
        field: Field::Value,
        mark: &|_| None,
        degree: &|_| Ok(1.0),
        own: None,
    };
    evaluate(&s, &reading).map_err(|r| match r {
        Refused::At(by, why) => (from + by, why),
        Refused::Mark(_, by) => (
            from + by,
            "reads `mark(...)` outside a time field".to_string(),
        ),
    })?;
    let expr = crate::parser::parse(&s).map_err(|d| (from + d.span.start, d.message))?;
    match folded(&expr, &|_| None) {
        Some(n) => Ok(Expr::Lit(Literal::Num(n))),
        None => Ok(crate::ingest::relocate(&expr, from)),
    }
}

fn moved(line: &Pending, i: usize, at: Time, len: Option<Time>) -> Result<Move, Inside> {
    let lane_raw = line.raw("lane", i).expect("a move");
    let lane = string(lane_raw, "lane")?;
    let identifier = lane.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && lane.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !identifier {
        return Err((
            lane_raw.at,
            format!("`lane` is `{lane}`; a lane names a parameter or a variable, as `cutoff`"),
        ));
    }
    let node = line
        .raw("node", i)
        .map(|raw| node(raw, "node"))
        .transpose()?;
    let to_raw = line.raw("to", i).expect("checked");
    let to = To {
        text: match to_raw.is_string() {
            true => string(to_raw, "to")?,
            false => to_raw.text.to_string(),
        },
        at: to_raw.at,
        inner: inside(to_raw),
    };
    if node.is_some() || lane != "scale" {
        let home = Value {
            amount: 1.0,
            kind: Kind::Home,
            position: false,
        };
        let reading = Reading {
            field: Field::To,
            mark: &|_| None,
            degree: &|_| Ok(1.0),
            own: Some((&lane, home)),
        };
        evaluate(&to.text, &reading).map_err(|r| match r {
            Refused::At(by, why) => (to.inner + by, why),
            Refused::Mark(_, by) => (
                to.inner + by,
                "reads `mark(...)` outside a time field".to_string(),
            ),
        })?;
    }
    let shape = match line.raw("shape", i) {
        None => None,
        Some(raw) => Some(match string(raw, "shape")?.as_str() {
            "lin" => (Shape::Lin, raw.at),
            "exp" => (Shape::Exp, raw.at),
            other => {
                return Err((
                    raw.at,
                    format!("`shape` is `{other}`; a ramp's shape is `lin` or `exp`"),
                ));
            }
        }),
    };
    let ramp = match (len, shape) {
        (Some(len), Some((shape, _))) => Some(Ramp { len, shape }),
        (None, None) => None,
        (None, Some((_, at))) => {
            return Err((
                at,
                "`shape` names a ramp's curve, and a move with no `len` is a set; a ramp names \
                 `len` and `shape`, `lin` or `exp`"
                    .to_string(),
            ));
        }
        (Some(len), None) => {
            return Err((
                len.at,
                "a ramp names its `shape`, `lin` or `exp`, beside its `len`".to_string(),
            ));
        }
    };
    Ok(Move {
        at,
        node,
        lane: Named {
            name: lane,
            at: lane_raw.at,
        },
        to,
        ramp,
    })
}

/// A line's keys in written order, each beside its value's own bytes in the line.
struct Fields<'a>(Vec<(String, &'a RawValue)>);

impl<'de> Deserialize<'de> for Fields<'de> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Fields<'de>, D::Error> {
        d.deserialize_map(FieldsVisitor)
    }
}

struct FieldsVisitor;

impl<'de> Visitor<'de> for FieldsVisitor {
    type Value = Fields<'de>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("one JSON object, such as {\"at\": \"0b\", \"node\": \"@../instruments/bass\"}")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Fields<'de>, A::Error> {
        let mut fields: Vec<(String, &'de RawValue)> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if fields.iter().any(|(held, _)| *held == key) {
                return Err(de::Error::custom(format!(
                    "`{key}` is written twice on one line"
                )));
            }
            let value = map.next_value::<&'de RawValue>()?;
            fields.push((key, value));
        }
        Ok(Fields(fields))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SONG: &str = "{\"annotation\": \"Models: a | Neglects: b | IO: (t) -> amplitude | Tags: c\"}\n\
        {\"at\": \"mark(verse) - 2b\", \"node\": \"@crash\", \"len\": \"2b\"}\n\
        {\"at\": \"8b\", \"mark\": \"verse\"}\n\
        {\"at\": \"mark(verse)\", \"lane\": \"energy\", \"to\": 0.45, \"comment\": \"up\"}\n\
        {\"at\": [\"mark(verse)\", \"9b\"], \"node\": [\"@a\", \"@b\"], \"deg\": 1, \"vel\": [0.5, 0.6]}\n\
        {\"at\": \"10b\", \"comment\": \"air\"}\n";

    #[test]
    fn each_line_is_the_kind_its_keys_tell_its_marks_resolved_in_any_order() {
        let score = parse_score(SONG).expect("a score");
        assert!(
            score
                .annotation
                .as_deref()
                .is_some_and(|a| a.starts_with("Models"))
        );
        let kinds: Vec<&str> = score.items().map(|(_, i)| i.kind()).collect();
        assert_eq!(
            kinds,
            ["event", "mark", "move", "event", "event", "comment"]
        );
        let ats: Vec<f64> = score.items().map(|(_, i)| i.at().amount.to_f64()).collect();
        assert_eq!(ats, [6.0, 8.0, 8.0, 8.0, 9.0, 10.0]);
        let Item::Event(second) = &score.lines[3].items[1] else {
            panic!("an event")
        };
        assert_eq!(second.node.name, "b");
        assert_eq!(second.deg.map(|(d, _)| d), Some(1));
        assert_eq!(second.binds[0].value, Expr::Lit(Literal::Num(0.6)));
        let Item::Move(moved) = &score.lines[2].items[0] else {
            panic!("a move")
        };
        assert_eq!(
            (moved.lane.name.as_str(), moved.to.text.as_str()),
            ("energy", "0.45")
        );
        assert_eq!(moved.ramp, None);
    }
}
