// Concern: reads a score's JSON Lines and lowers it to the sum of each event's node shifted to its time | Non-concern: checking a line against its node (tuning.rs) | IO: (&str) -> Score -> Expr

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;

use crate::diag::{ByteSpan, Diag, DiagCode};
use crate::expr::{Address, Arg, BinOp, DEGREE, Expr, Literal};
use crate::filename::SpanUnit;
use crate::lexer::{Token, TokenKind, tokenize};
use crate::series::folded;
use crate::vocabulary::is_language_value;

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub events: Vec<Event>,
}

/// One node placed by a line: at `at`, `node` plays with `args` bound, `len` among them
/// where written. A line whose fields list N items places N events, the i-th taking item i.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub line: usize,
    pub span: ByteSpan,
    pub at: Time,
    pub node: String,
    pub deg: Option<i64>,
    /// Where this event's degree is written, which a refusal about it names.
    pub deg_at: ByteSpan,
    pub args: Vec<(String, Expr)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Time {
    pub amount: f64,
    pub unit: SpanUnit,
}

impl Time {
    /// As a score writes it, `0.25b` or `0.5s`.
    pub fn spelled(self) -> String {
        match self.unit {
            SpanUnit::Bars => format!("{}b", self.amount),
            SpanUnit::Seconds => format!("{}s", self.amount),
        }
    }

    fn literal(self) -> Expr {
        Expr::Lit(match self.unit {
            SpanUnit::Bars => Literal::Bars(self.amount),
            SpanUnit::Seconds => Literal::Num(self.amount),
        })
    }
}

/// A `;` line in a score is a comment, as in every other file.
fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with(crate::lexer::COMMENT as char)
}

/// No equation opens with `{`, so the first line a score writes past its comments says what
/// the file is.
pub fn is_score(content: &str) -> bool {
    content
        .lines()
        .find(|line| !line.trim().is_empty() && !is_comment(line))
        .is_some_and(|line| line.trim_start().starts_with('{'))
}

pub fn parse_score(content: &str) -> Result<Score, Diag> {
    let mut events: Vec<Event> = Vec::new();
    let mut start = 0usize;
    for (index, line) in content.split('\n').enumerate() {
        let span = ByteSpan::new(start, start + line.len());
        start += line.len() + 1;
        if line.trim().is_empty() || is_comment(line) {
            continue;
        }
        let number = index + 1;
        let Fields(fields) = serde_json::from_str(line).map_err(|e| {
            let column = e.column().max(1);
            let message = e.to_string();
            let message = message
                .rsplit_once(" at line ")
                .map_or(&*message, |(m, _)| m);
            refused(span.start + column - 1, (number, column), message)
        })?;
        let refuse =
            |at: usize, message: String| refused(span.start + at, (number, at + 1), &message);
        let written = Line::read(line, span.start, number, fields, &refuse)?;
        for i in 0..written.count {
            let (at, by) = *item(&written.at, i);
            if let Some(before) = events.last() {
                sorted(before, at, (number, by - span.start + 1), by)?;
            }
            let (deg, deg_at) = match written.deg.is_empty() {
                true => (None, span),
                false => {
                    let (d, at) = *item(&written.deg, i);
                    (Some(d), at)
                }
            };
            let args = written.args.iter();
            events.push(Event {
                line: number,
                span,
                at,
                node: item(&written.node, i).clone(),
                deg,
                deg_at,
                args: args.map(|(k, v)| (k.clone(), item(v, i).clone())).collect(),
            });
        }
    }
    Ok(Score { events })
}

/// Where a refusal on a score line lands: its byte, and the line and column it names.
pub fn refused(at: usize, (line, column): (usize, usize), message: &str) -> Diag {
    Diag::new(
        DiagCode::BadScore,
        ByteSpan::new(at, at),
        format!("line {line}, column {column}: {message}"),
    )
}

fn sorted(
    before: &Event,
    next: Time,
    (line, column): (usize, usize),
    at: usize,
) -> Result<(), Diag> {
    let earlier = match before.line == line {
        true => "earlier on this line".to_string(),
        false => format!("on line {}", before.line),
    };
    if before.at.unit != next.unit {
        return Err(refused(
            at,
            (line, column),
            &format!(
                "`at` is {} here and {} {earlier}; one score counts its time in bars or in \
                 seconds, not both",
                next.spelled(),
                before.at.spelled(),
            ),
        ));
    }
    if next.amount < before.at.amount {
        return Err(refused(
            at,
            (line, column),
            &format!(
                "`at` is {} here, before {} {earlier}; a score's events are sorted by `at`, \
                 a list's items too",
                next.spelled(),
                before.at.spelled()
            ),
        ));
    }
    Ok(())
}

impl Score {
    /// Σ nodeᵢ(t − atᵢ; argsᵢ), summed as a balanced tree so its depth is log₂ of its events.
    /// A degree stays a `deg` call, which the tuning pass reads off the key's scale.
    pub fn lowered(&self) -> Expr {
        balanced(self.events.iter().map(Event::lowered).collect())
    }
}

impl Event {
    fn lowered(&self) -> Expr {
        let mut binds = self.args.clone();
        if let Some(deg) = self.deg {
            let degree = Expr::Call {
                name: DEGREE.to_string(),
                args: vec![Arg::Pos(Expr::Lit(Literal::Num(deg as f64)))],
                span: self.deg_at,
            };
            binds.push(("f0".to_string(), degree));
        }
        Expr::Ref {
            path: self.node.clone(),
            arg: Box::new(shifted(self.at)),
            binds,
            address: Address::Time,
            span: self.span,
        }
    }
}

fn shifted(at: Time) -> Expr {
    let t = Box::new(Expr::Var("t".to_string()));
    let by = |amount: f64| Box::new(Time { amount, ..at }.literal());
    match at.amount {
        0.0 => *t,
        a if a < 0.0 => Expr::Bin(BinOp::Add, t, by(-a)),
        a => Expr::Bin(BinOp::Sub, t, by(a)),
    }
}

fn balanced(mut terms: Vec<Expr>) -> Expr {
    match terms.len() {
        0 => Expr::Lit(Literal::Num(0.0)),
        1 => terms.pop().expect("one term"),
        n => {
            let right = terms.split_off(n / 2);
            Expr::Bin(
                BinOp::Add,
                Box::new(balanced(terms)),
                Box::new(balanced(right)),
            )
        }
    }
}

/// A line's keys in written order, each beside its value's own bytes in the line, so a refusal
/// inside a value names the column it is written at.
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
        f.write_str(
            "one event, an object such as {\"at\": \"0b\", \"node\": \"@../instruments/bass\"}",
        )
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Fields<'de>, A::Error> {
        let mut fields: Vec<(String, &'de RawValue)> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            if fields.iter().any(|(held, _)| *held == key) {
                return Err(de::Error::custom(format!("`{key}` is written twice")));
            }
            let value = map.next_value::<&'de RawValue>()?;
            fields.push((key, value));
        }
        Ok(Fields(fields))
    }
}

/// What a refusal inside one value says, at a byte of that value as the line writes it.
type Inside = (usize, String);

/// A score binds constants; what moves over time is an equation's job.
const MOVING: &str = "a score binds constants, and a moving argument belongs in an equation that \
                      reads an automation score (see SCORES in --help)";

/// The fields a line names, each value read where it is written: one item, or a list's,
/// item i of each list going to its event i and a single value to every event.
struct Line {
    count: usize,
    /// Each `at` beside its byte in the score, where a refusal of its order lands.
    at: Vec<(Time, usize)>,
    node: Vec<String>,
    deg: Vec<(i64, ByteSpan)>,
    args: Vec<(String, Vec<Expr>)>,
}

/// Item `i` of a field: its own where it lists one per event, else its one value.
fn item<T>(items: &[T], i: usize) -> &T {
    &items[if items.len() == 1 { 0 } else { i }]
}

impl Line {
    fn read(
        line: &str,
        start: usize,
        number: usize,
        fields: Vec<(String, &RawValue)>,
        refuse: &dyn Fn(usize, String) -> Diag,
    ) -> Result<Line, Diag> {
        let (mut at, mut node) = (Vec::new(), Vec::new());
        let mut deg: Vec<(i64, ByteSpan)> = Vec::new();
        let mut args: Vec<(String, Vec<Expr>)> = Vec::new();
        let mut listed: Option<(&str, usize)> = None;
        for (key, raw) in &fields {
            let from = raw.get().as_ptr() as usize - line.as_ptr() as usize;
            let located = |(by, message): Inside| refuse(from + by, message);
            if key == "sound" {
                let written = line[..from].rfind("\"sound\"").unwrap_or(from);
                return Err(refuse(
                    written,
                    "`sound` is now `node`, since what a line plays is any node: a sound, \
                     another score or a control curve; write \"node\": \"@../instruments/bass\""
                        .to_string(),
                ));
            }
            let items = items(key, raw.get()).map_err(located)?;
            if raw.get().starts_with('[') {
                match listed {
                    Some((first, n)) if n != items.len() => {
                        return Err(Diag::new(
                            DiagCode::BadScore,
                            ByteSpan::new(start + from, start + from),
                            format!(
                                "line {number}: `{first}` has {}, `{key}` has {}",
                                counted(n),
                                items.len()
                            ),
                        ));
                    }
                    Some(_) => {}
                    None => listed = Some((key, items.len())),
                }
            }
            for (by, written) in items {
                let at_byte = from + by;
                let located = |(by, message): Inside| refuse(at_byte + by, message);
                match key.as_str() {
                    "at" => at.push((time(written, true).map_err(located)?, start + at_byte)),
                    "node" => node.push(self::node(written).map_err(located)?),
                    "deg" => {
                        let d = degree(written).map_err(located)?;
                        let span = ByteSpan::new(start + at_byte, start + at_byte + written.len());
                        deg.push((d, span));
                    }
                    "len" => {
                        let len = time(written, false).map_err(located)?;
                        bound(&mut args, key).push(len.literal());
                    }
                    _ => {
                        let value = value(written).map_err(located)?;
                        let value = crate::ingest::relocate(&value, start + at_byte + 1);
                        bound(&mut args, key).push(value);
                    }
                }
            }
        }
        let end = line.trim_end().len().saturating_sub(1);
        if at.is_empty() {
            return Err(refuse(end, "missing field `at`".to_string()));
        }
        if node.is_empty() {
            return Err(refuse(end, "missing field `node`".to_string()));
        }
        if let Some((_, written)) = deg.first()
            && args.iter().any(|(key, _)| key == "f0")
        {
            return Err(refuse(
                written.start - start,
                "`f0` and `deg` both set the pitch; write one of them".to_string(),
            ));
        }
        Ok(Line {
            count: listed.map_or(1, |(_, n)| n),
            at,
            node,
            deg,
            args,
        })
    }
}

/// The items `key` binds where it is pushed, the slot opened on its first.
fn bound<'a>(args: &'a mut Vec<(String, Vec<Expr>)>, key: &str) -> &'a mut Vec<Expr> {
    if args.last().is_none_or(|(held, _)| held != key) {
        args.push((key.to_string(), Vec::new()));
    }
    &mut args.last_mut().expect("just pushed").1
}

/// `3 items`, or `1 item`.
fn counted(n: usize) -> String {
    format!("{n} item{}", if n == 1 { "" } else { "s" })
}

/// A field's value as one item, or a list's items, each beside its byte in `raw`.
fn items<'a>(key: &str, raw: &'a str) -> Result<Vec<(usize, &'a str)>, Inside> {
    if !raw.starts_with('[') {
        return Ok(vec![(0, raw)]);
    }
    let list: Vec<&RawValue> =
        serde_json::from_str(raw).map_err(|e| (0, format!("`{key}` {e}")))?;
    if list.is_empty() {
        return Err((
            0,
            format!("`{key}` lists no item; a list holds one item per event, as [1, 3, 5]"),
        ));
    }
    list.iter()
        .map(|item| {
            let by = item.get().as_ptr() as usize - raw.as_ptr() as usize;
            match item.get().starts_with('[') {
                true => Err((
                    by,
                    format!("`{key}` lists a list; each item is one value for one event"),
                )),
                false => Ok((by, item.get())),
            }
        })
        .collect()
}

/// A JSON string's contents, which an SVA value never needs to escape.
fn string(raw: &str) -> Option<Result<&str, Inside>> {
    raw.starts_with('"').then(|| {
        serde_json::from_str::<&str>(raw).map_err(|_| {
            (
                0,
                format!("{raw} holds a `\\` escape, and no score value needs one"),
            )
        })
    })
}

/// A time in a string, `"0.25b"`, or constant arithmetic over times, `"2b + 1b/3"`, folded
/// here in its one unit; `signed` admits one before 0.
fn time(raw: &str, signed: bool) -> Result<Time, Inside> {
    let no_unit = || {
        format!(
            "{raw} is no time: write a number with a unit, as \"0.25b\" or \"0.5s\", or \
             arithmetic over them, as \"2b + 1b/3\"; a bare number has no unit"
        )
    };
    let s = string(raw).unwrap_or_else(|| Err((0, no_unit())))?;
    let inside = |(by, message): Inside| (by + 1, message);
    let tokens = tokenize(s).map_err(|d| inside((d.span.start, d.message)))?;
    constant(s, &tokens, false).map_err(inside)?;
    let mut unit: Option<SpanUnit> = None;
    for token in &tokens {
        let this = match token.kind {
            TokenKind::Time(_, this) => this,
            TokenKind::Samples(_) => {
                return Err(inside((
                    token.span.start,
                    format!(
                        "`{s}` counts a render's own samples; a score's time is bars or seconds"
                    ),
                )));
            }
            _ => continue,
        };
        match unit {
            Some(held) if held != this => {
                return Err(inside((
                    token.span.start,
                    format!("`{s}` counts in bars and in seconds; one time is written in one"),
                )));
            }
            _ => unit = Some(this),
        }
    }
    let unit = unit.ok_or_else(|| (0, no_unit()))?;
    let mut folding = Folding {
        s,
        tokens: &tokens,
        next: 0,
    };
    let (amount, is_time) = folding.sum().map_err(inside)?;
    if let Some(left) = tokens.get(folding.next) {
        return Err(inside((
            left.span.start,
            format!("`{s}` is no time: an operator or the end was expected here"),
        )));
    }
    if !is_time {
        return Err((0, no_unit()));
    }
    if !amount.is_finite() {
        return Err((1, format!("`{s}` folds to no number")));
    }
    if !signed && amount < 0.0 {
        return Err((1, format!("`{s}` is a window, so it is no shorter than 0")));
    }
    Ok(Time { amount, unit })
}

/// A time's tokens folded with their dimension: a sum of times, a time scaled by a number or
/// divided by one, so a bare number never takes a unit it was not written with.
struct Folding<'a> {
    s: &'a str,
    tokens: &'a [Token],
    next: usize,
}

/// A value, and whether it is a time rather than a number.
type Folded = (f64, bool);

impl Folding<'_> {
    fn op(&self, ops: [TokenKind; 2]) -> Option<(bool, usize)> {
        let token = self.tokens.get(self.next)?;
        ops.contains(&token.kind)
            .then_some((token.kind == ops[0], token.span.start))
    }

    fn refused(&self, at: usize, why: &str) -> Inside {
        (at, format!("`{}` {why}; a bare number has no unit", self.s))
    }

    fn sum(&mut self) -> Result<Folded, Inside> {
        let mut left = self.product()?;
        while let Some((plus, at)) = self.op([TokenKind::Plus, TokenKind::Minus]) {
            self.next += 1;
            let right = self.product()?;
            if left.1 != right.1 {
                return Err(self.refused(at, "adds a number to a time"));
            }
            left.0 = if plus {
                left.0 + right.0
            } else {
                left.0 - right.0
            };
        }
        Ok(left)
    }

    fn product(&mut self) -> Result<Folded, Inside> {
        let mut left = self.unary()?;
        while let Some((times, at)) = self.op([TokenKind::Star, TokenKind::Slash]) {
            self.next += 1;
            let right = self.unary()?;
            left = match (times, left.1, right.1) {
                (true, true, true) => return Err(self.refused(at, "multiplies a time by a time")),
                (false, _, true) => return Err(self.refused(at, "divides by a time")),
                (true, l, r) => (left.0 * right.0, l || r),
                (false, l, _) => (left.0 / right.0, l),
            };
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Folded, Inside> {
        match self.op([TokenKind::Minus, TokenKind::Plus]) {
            Some((minus, _)) => {
                self.next += 1;
                let (v, is_time) = self.unary()?;
                Ok((if minus { -v } else { v }, is_time))
            }
            None => self.primary(),
        }
    }

    fn primary(&mut self) -> Result<Folded, Inside> {
        let token = self.tokens.get(self.next);
        let at = token.map_or(self.s.len(), |t| t.span.start);
        self.next += 1;
        match token.map(|t| &t.kind) {
            Some(TokenKind::Num(n)) => Ok((*n, false)),
            Some(TokenKind::Time(n, _)) => Ok((*n, true)),
            Some(TokenKind::LParen) => {
                let inner = self.sum()?;
                match self.tokens.get(self.next).map(|t| &t.kind) {
                    Some(TokenKind::RParen) => {
                        self.next += 1;
                        Ok(inner)
                    }
                    _ => Err((at, format!("`{}` opens a ( it never closes", self.s))),
                }
            }
            _ => Err((
                at,
                format!(
                    "`{}` is no time: a time is arithmetic over numbers and times in bars or \
                     seconds",
                    self.s
                ),
            )),
        }
    }
}

/// Refuses whatever a constant cannot hold: `t`, a ref, a name, or a call other than `deg`
/// where `degrees` admits it.
fn constant(s: &str, tokens: &[Token], degrees: bool) -> Result<(), Inside> {
    for (k, token) in tokens.iter().enumerate() {
        let at = token.span.start;
        let called = matches!(tokens.get(k + 1).map(|t| &t.kind), Some(TokenKind::LParen));
        let why = match &token.kind {
            TokenKind::Ref(path) => format!("`{s}` reads `@{path}`; {MOVING}"),
            TokenKind::Ident(name) if called && degrees && name == DEGREE => continue,
            TokenKind::Ident(name) if called => format!(
                "`{s}` calls `{name}`; a score value is arithmetic over literals{}, and a \
                 sound's own math belongs in its equation",
                if degrees { " and `deg(n)`" } else { "" }
            ),
            TokenKind::Ident(name) if name == "t" || name == "f" => {
                format!("`{s}` reads `{name}`; {MOVING}")
            }
            TokenKind::Ident(name) if is_language_value(name) && name != "i" => continue,
            TokenKind::Ident(name) => format!(
                "`{s}` names `{name}`, which a score cannot read: it binds constants, numbers, \
                 literals with units such as \"0.03s\" or \"-6db\", and note names such as \"A1\""
            ),
            TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::Lt
            | TokenKind::Le
            | TokenKind::Gt
            | TokenKind::Ge
            | TokenKind::Dot => format!("`{s}` is no constant: a score value is arithmetic"),
            _ => continue,
        };
        return Err((at, why));
    }
    Ok(())
}

fn node(raw: &str) -> Result<String, Inside> {
    let refuse = || {
        (
            0,
            format!("`node` is an `@ref` string such as \"@../instruments/bass\", not {raw}"),
        )
    };
    let s = string(raw).unwrap_or_else(|| Err(refuse()))?;
    match s.strip_prefix('@') {
        Some(path) if !path.is_empty() && crate::lexer::whole_ref_path(path) => {
            Ok(path.to_string())
        }
        _ => Err(refuse()),
    }
}

/// One whole scale degree.
fn degree(written: &str) -> Result<i64, Inside> {
    serde_json::from_str::<i64>(written).map_err(|_| {
        (
            0,
            format!(
                "`deg` is a whole scale degree or a chord of them, as 5 or [1, 3, 5], not \
                 {written}; a chromatic note binds `f0`, as \"f0\": \"deg(7)*1st\""
            ),
        )
    })
}

/// A number, or a constant in a string: one literal (`"0.03s"`, `"-6db"`, `"A1"`), or
/// arithmetic over literals and `deg(n)`, as `"0.8*0.9"` or `"deg(7)*1st"`. Arithmetic over
/// numbers alone folds here, so `"0.8*0.9"` binds the number `0.72` does.
fn value(raw: &str) -> Result<Expr, Inside> {
    if let Ok(n) = serde_json::from_str::<f64>(raw) {
        return Ok(Expr::Lit(Literal::Num(n)));
    }
    let s = string(raw).unwrap_or_else(|| {
        Err((
            0,
            format!(
                "{raw} is no value: a score binds a number, or a string such as \"0.03s\", \
                 \"-6db\", \"A1\", \"0.8*0.9\" or \"deg(7)*1st\""
            ),
        ))
    })?;
    let inside = |(by, message): Inside| (by + 1, message);
    let tokens = tokenize(s).map_err(|d| inside((d.span.start, d.message)))?;
    constant(s, &tokens, true).map_err(inside)?;
    let expr = crate::parser::parse(s).map_err(|d| inside((d.span.start, d.message)))?;
    match folded(&expr, &|_| None) {
        Some(n) if n.is_finite() => Ok(Expr::Lit(Literal::Num(n))),
        Some(_) => Err((1, format!("`{s}` folds to no number"))),
        None => Ok(expr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(text: &str) -> String {
        parse_score(text).unwrap_err().message
    }

    fn line(fields: &str) -> Score {
        parse_score(&format!("{{\"at\": \"0b\", \"node\": \"@b\", {fields}}}\n")).unwrap()
    }

    #[test]
    fn each_line_is_an_event_in_written_order() {
        let score = parse_score(
            "{\"at\": \"0b\", \"node\": \"@bass\", \"deg\": 1, \"vel\": 0.9}\n\n\
             {\"at\": \"0.5b\", \"node\": \"@bass\", \"len\": \"0.25b\", \"cutoff\": \"900hz\"}\n",
        )
        .unwrap();
        assert_eq!(score.events.len(), 2);
        assert_eq!(score.events[1].line, 3);
        assert_eq!(score.events[1].args[0].0, "len");
        assert_eq!(score.events[1].args[1].1, Expr::Lit(Literal::Num(900.0)));
    }

    #[test]
    fn a_refusal_names_the_line_and_column() {
        let text = "{\"at\": \"0b\", \"node\": \"@bass\"}\n{\"at\": 0.5, \"node\": \"@bass\"}\n";
        let refused = refusal(text);
        assert!(refused.starts_with("line 2, column 8:"), "{refused}");
        assert!(refused.contains("bare number"), "{refused}");
        assert!(refusal("{\"at\": \"0b\"}").contains("missing field `node`"));
        assert!(refusal("{\"at\": \"0b\", \"node\": \"@b\", \"at\": \"1b\"}").contains("twice"));
        assert!(
            refusal("{\"at\": \"0b\", \"node\": \"@b\", \"deg\": 1, \"f0\": 440}")
                .contains("`deg`")
        );
        assert!(
            refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": \"x+1\"}").contains("literal")
        );
    }

    #[test]
    fn a_refusal_inside_a_value_names_its_own_column() {
        let refused = refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": \"0.5*sin(t)\"}");
        assert!(refused.starts_with("line 1, column 40: "), "{refused}");
        assert!(refused.contains("calls `sin`"), "{refused}");
        let moving = refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": \"0.5 + t\"}");
        assert!(moving.starts_with("line 1, column 42: "), "{moving}");
        assert!(moving.contains("automation score"), "{moving}");
        let reads = refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": \"@../lfo\"}");
        assert!(reads.contains("reads `@../lfo`"), "{reads}");
        let mixed = refusal("{\"at\": \"1b + 0.5s\", \"node\": \"@b\"}");
        assert!(mixed.starts_with("line 1, column 14: "), "{mixed}");
    }

    #[test]
    fn a_value_is_constant_arithmetic_and_numbers_alone_fold() {
        let score = line("\"len\": \"1b/3\", \"vel\": \"0.8*0.9\", \"f0\": \"deg(7)*1st\"");
        let args = &score.events[0].args;
        assert_eq!(args[0].1, Expr::Lit(Literal::Bars(1.0 / 3.0)));
        assert_eq!(args[1].1, Expr::Lit(Literal::Num(0.8 * 0.9)));
        assert!(matches!(&args[2].1, Expr::Bin(BinOp::Mul, l, _)
            if matches!(&**l, Expr::Call { name, .. } if name == DEGREE)));
    }

    #[test]
    fn a_time_folds_in_bars_before_any_tempo() {
        let score = parse_score("{\"at\": \"2b + 1b/3\", \"node\": \"@b\"}\n").unwrap();
        assert_eq!(score.events[0].at.amount, 2.0 + 1.0 / 3.0);
        assert_eq!(score.events[0].at.unit, SpanUnit::Bars);
        assert!(refusal("{\"at\": \"1/3\", \"node\": \"@b\"}").contains("no time"));
        let at =
            |written: &str| parse_score(&format!("{{\"at\": \"{written}\", \"node\": \"@b\"}}"));
        assert_eq!(at("(2b + 1b)/3").unwrap().events[0].at.amount, 1.0);
        assert_eq!(at("-0.5s").unwrap().events[0].at.amount, -0.5);
        for (written, why) in [
            ("2b + 1", "adds a number to a time"),
            ("1b*1b", "multiplies a time by a time"),
            ("1b/1b", "divides by a time"),
            ("(1b", "never closes"),
            ("1b 2b", "no time"),
        ] {
            let refused = at(written).unwrap_err().message;
            assert!(refused.contains(why), "{written}: {refused}");
        }
        assert!(
            refusal("{\"at\": \"0b\", \"node\": \"@b\", \"len\": \"1b - 2b\"}").contains("window")
        );
    }

    #[test]
    fn a_chord_is_one_event_per_degree_each_located() {
        let score = line("\"deg\": [1, 3, 5]");
        let degrees: Vec<_> = score.events.iter().map(|e| e.deg).collect();
        assert_eq!(degrees, [Some(1), Some(3), Some(5)]);
        assert_eq!(score.events[2].deg_at, ByteSpan::new(41, 42));
        assert!(refusal("{\"at\": \"0b\", \"node\": \"@b\", \"deg\": []}").contains("no item"));
        let chromatic = refusal("{\"at\": \"0b\", \"node\": \"@b\", \"deg\": [1, 6.5]}");
        assert!(chromatic.starts_with("line 1, column 39: "), "{chromatic}");
        assert!(chromatic.contains("deg(7)*1st"), "{chromatic}");
    }

    #[test]
    fn a_list_on_any_field_broadcasts_item_by_item() {
        let score = line(
            "\"deg\": [1, 3, 5], \"vel\": [0.8, 0.5, 0.6], \"len\": [\"1b\", \"0.5b\", \"1b\"], \
             \"cutoff\": \"900hz\"",
        );
        assert_eq!(score.events.len(), 3);
        let second = &score.events[1];
        assert_eq!(second.deg, Some(3));
        assert_eq!(
            second.args[0],
            ("vel".to_string(), Expr::Lit(Literal::Num(0.5)))
        );
        assert_eq!(second.args[1].1, Expr::Lit(Literal::Bars(0.5)));
        assert_eq!(second.args[2].1, Expr::Lit(Literal::Num(900.0)));
        let run = parse_score(
            "{\"at\": [\"0b\", \"1b/3\"], \"node\": [\"@a\", \"@b\"], \"f0\": [\"deg(5)\", \"deg(7)*1st\"]}",
        )
        .unwrap();
        let placed: Vec<_> = run.events.iter().map(|e| (e.at.amount, &*e.node)).collect();
        assert_eq!(placed, [(0.0, "a"), (1.0 / 3.0, "b")]);
        assert!(matches!(
            &run.events[1].args[0].1,
            Expr::Bin(BinOp::Mul, ..)
        ));
    }

    #[test]
    fn lists_of_different_lengths_refuse_naming_both() {
        let text = "{\"at\": \"0b\", \"node\": \"@b\"}\n; a comment\n\n\
                    {\"at\": \"1b\", \"node\": \"@b\", \"deg\": [1, 3, 5], \"vel\": [0.8, 0.5]}\n";
        assert_eq!(refusal(text), "line 4: `deg` has 3 items, `vel` has 2");
        let one = refusal("{\"at\": [\"0b\"], \"node\": \"@b\", \"deg\": [1, 3]}");
        assert_eq!(one, "line 1: `at` has 1 item, `deg` has 2");
    }

    #[test]
    fn a_list_item_keeps_its_fields_rules_and_its_own_column() {
        let refused = refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": [0.5, \"t\"]}");
        assert!(refused.starts_with("line 1, column 42: "), "{refused}");
        assert!(refused.contains("automation score"), "{refused}");
        let late = refusal("{\"at\": [\"1b\", \"0.5b\"], \"node\": \"@b\"}");
        assert!(late.starts_with("line 1, column 15: "), "{late}");
        assert!(late.contains("earlier on this line"), "{late}");
        assert!(
            refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": [[1]]}").contains("lists a list")
        );
        assert!(refusal("{\"at\": \"0b\", \"node\": \"@b\", \"vel\": []}").contains("no item"));
    }

    #[test]
    fn sound_is_refused_as_the_old_name_of_node() {
        let refused = refusal("{\"at\": \"0b\", \"sound\": \"@b\"}");
        assert!(
            refused.starts_with("line 1, column 14: `sound` is now `node`"),
            "{refused}"
        );
    }

    #[test]
    fn comment_lines_are_skipped_and_a_comment_may_lead_a_score() {
        let text = "; Models: a bar | Neglects: the rest | IO: (t) -> amplitude | Tags: score\n\
                    {\"at\": \"0b\", \"node\": \"@b\"}\n  ; bar 2\n{\"at\": \"1b\", \"node\": \"@b\"}\n";
        assert!(is_score(text));
        let score = parse_score(text).unwrap();
        assert_eq!(score.events.len(), 2);
        assert_eq!(score.events[1].line, 4);
        assert!(!is_score("; a comment\nsin(t)\n"));
    }

    #[test]
    fn lines_out_of_order_or_mixing_units_refuse() {
        let late = "{\"at\": \"1b\", \"node\": \"@b\"}\n{\"at\": \"0.5b\", \"node\": \"@b\"}\n";
        assert!(refusal(late).contains("sorted"));
        let mixed = "{\"at\": \"0b\", \"node\": \"@b\"}\n{\"at\": \"1s\", \"node\": \"@b\"}\n";
        assert!(refusal(mixed).contains("not both"));
    }

    #[test]
    fn the_sum_is_balanced() {
        let line = "{\"at\": \"0.5s\", \"node\": \"@b\"}\n";
        let score = parse_score(&line.repeat(1024)).unwrap();
        fn depth(e: &Expr) -> usize {
            match e {
                Expr::Bin(BinOp::Add, l, r) => 1 + depth(l).max(depth(r)),
                _ => 0,
            }
        }
        assert_eq!(depth(&score.lowered()), 10);
    }
}
