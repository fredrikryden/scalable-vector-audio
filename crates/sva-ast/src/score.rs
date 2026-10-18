// Concern: reads a score's JSON Lines and lowers it to the sum of each line's sound shifted to its time | Non-concern: checking a line against its sound (tuning.rs) | IO: (&str) -> Score -> Expr

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, Visitor};

use crate::diag::{ByteSpan, Diag, DiagCode};
use crate::expr::{Address, BinOp, Expr, Literal};
use crate::filename::SpanUnit;
use crate::lexer::{TokenKind, tokenize};

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub events: Vec<Event>,
}

/// One line: at `at`, `sound` plays with `args` bound, `len` among them where written.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub line: usize,
    pub span: ByteSpan,
    pub at: Time,
    pub sound: String,
    pub deg: Option<i64>,
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

/// No equation opens with `{`, so the first line a score writes says what the file is.
pub fn is_score(content: &str) -> bool {
    content
        .lines()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim_start().starts_with('{'))
}

pub fn parse_score(content: &str) -> Result<Score, Diag> {
    let mut events: Vec<Event> = Vec::new();
    let mut start = 0usize;
    for (index, line) in content.split('\n').enumerate() {
        let span = ByteSpan::new(start, start + line.len());
        start += line.len() + 1;
        if line.trim().is_empty() {
            continue;
        }
        let number = index + 1;
        let written: Line = serde_json::from_str(line).map_err(|e| {
            let column = e.column().max(1);
            let message = e.to_string();
            let message = message
                .rsplit_once(" at line ")
                .map_or(&*message, |(m, _)| m);
            refused(span.start + column - 1, (number, column), message)
        })?;
        if let Some(before) = events.last() {
            sorted(before, &written, (number, span.start))?;
        }
        events.push(Event {
            line: number,
            span,
            at: written.at,
            sound: written.sound,
            deg: written.deg,
            args: written.args,
        });
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

fn sorted(before: &Event, next: &Line, (line, at): (usize, usize)) -> Result<(), Diag> {
    if before.at.unit != next.at.unit {
        return Err(refused(
            at,
            (line, 1),
            &format!(
                "`at` is {} here and {} on line {}; one score counts its time in bars or in \
                 seconds, not both",
                next.at.spelled(),
                before.at.spelled(),
                before.line
            ),
        ));
    }
    if next.at.amount < before.at.amount {
        return Err(refused(
            at,
            (line, 1),
            &format!(
                "`at` is {} here, before line {}'s {}; a score's lines are sorted by `at`",
                next.at.spelled(),
                before.line,
                before.at.spelled()
            ),
        ));
    }
    Ok(())
}

impl Score {
    /// Σ soundᵢ(t − atᵢ; argsᵢ), summed as a balanced tree so its depth is log₂ of its lines.
    pub fn lowered(&self, pitch: &dyn Fn(&Event, i64) -> Expr) -> Expr {
        let terms: Vec<Expr> = self.events.iter().map(|e| e.lowered(pitch)).collect();
        balanced(terms)
    }
}

impl Event {
    fn lowered(&self, pitch: &dyn Fn(&Event, i64) -> Expr) -> Expr {
        let mut binds = self.args.clone();
        if let Some(deg) = self.deg {
            binds.push(("f0".to_string(), pitch(self, deg)));
        }
        Expr::Ref {
            path: self.sound.clone(),
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

/// The fields a line names, read in written order so a key written twice is refused.
struct Line {
    at: Time,
    sound: String,
    deg: Option<i64>,
    args: Vec<(String, Expr)>,
}

impl<'de> Deserialize<'de> for Line {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Line, D::Error> {
        d.deserialize_map(LineVisitor)
    }
}

struct LineVisitor;

impl<'de> Visitor<'de> for LineVisitor {
    type Value = Line;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("one event, an object such as {\"at\": \"0b\", \"sound\": \"@../voice/note\"}")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Line, A::Error> {
        let (mut at, mut sound, mut deg) = (None, None, None);
        let mut args: Vec<(String, Expr)> = Vec::new();
        while let Some(key) = map.next_key::<String>()? {
            let twice = match key.as_str() {
                "at" => at.is_some(),
                "sound" => sound.is_some(),
                "deg" => deg.is_some(),
                other => args.iter().any(|(held, _)| held == other),
            };
            if twice {
                return Err(de::Error::custom(format!("`{key}` is written twice")));
            }
            match key.as_str() {
                "at" => at = Some(map.next_value::<When<true>>()?.0),
                "sound" => sound = Some(map.next_value::<Sound>()?.0),
                "deg" => deg = Some(map.next_value::<i64>()?),
                "len" => {
                    let len = map.next_value::<When<false>>()?.0;
                    args.push((key, len.literal()));
                }
                _ => {
                    let value = map.next_value::<Argument>()?.0;
                    args.push((key, value));
                }
            }
        }
        let at = at.ok_or_else(|| de::Error::missing_field("at"))?;
        let sound = sound.ok_or_else(|| de::Error::missing_field("sound"))?;
        if deg.is_some() && args.iter().any(|(key, _)| key == "f0") {
            return Err(de::Error::custom(
                "`f0` and `deg` both set the pitch; write one of them",
            ));
        }
        Ok(Line {
            at,
            sound,
            deg,
            args,
        })
    }
}

/// A time in a string with its unit, `"0.25b"` or `"0.5s"`; `SIGNED` admits a leading `-`.
struct When<const SIGNED: bool>(Time);

impl<'de, const SIGNED: bool> Deserialize<'de> for When<SIGNED> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_str(WhenVisitor::<SIGNED>)
    }
}

struct WhenVisitor<const SIGNED: bool>;

impl<'de, const SIGNED: bool> Visitor<'de> for WhenVisitor<SIGNED> {
    type Value = When<SIGNED>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(
            "a time literal string such as \"0.25b\" or \"0.5s\"; a bare number has no unit",
        )
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<When<SIGNED>, E> {
        let tokens = tokenize(s).map_err(|d| E::custom(d.message))?;
        let (sign, rest) = match tokens.as_slice() {
            [first, rest @ ..] if SIGNED && first.kind == TokenKind::Minus => (-1.0, rest),
            all => (1.0, all),
        };
        match rest {
            [only] => match only.kind {
                TokenKind::Time(amount, unit) => Ok(When(Time {
                    amount: sign * amount,
                    unit,
                })),
                _ => Err(E::custom(format!(
                    "`{s}` is no time literal: write a number with a unit, as \"0.25b\" or \
                     \"0.5s\""
                ))),
            },
            _ => Err(E::custom(format!(
                "`{s}` is no time literal: write one number with a unit, as \"0.25b\" or \
                 \"0.5s\""
            ))),
        }
    }
}

struct Sound(String);

impl<'de> Deserialize<'de> for Sound {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_str(SoundVisitor)
    }
}

struct SoundVisitor;

impl<'de> Visitor<'de> for SoundVisitor {
    type Value = Sound;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an `@ref` string such as \"@../voice/note\"")
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Sound, E> {
        match s.strip_prefix('@') {
            Some(path) if !path.is_empty() && crate::lexer::whole_ref_path(path) => {
                Ok(Sound(path.to_string()))
            }
            _ => Err(E::custom(format!(
                "`sound` is an `@ref` string such as \"@../voice/note\", not `{s}`"
            ))),
        }
    }
}

/// A number, or one literal in a string: `"0.03s"`, `"900hz"`, `"-6db"`, `"A1"`.
struct Argument(Expr);

impl<'de> Deserialize<'de> for Argument {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(ArgumentVisitor)
    }
}

struct ArgumentVisitor;

impl<'de> Visitor<'de> for ArgumentVisitor {
    type Value = Argument;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number, or a literal string such as \"0.03s\", \"-6db\" or \"A1\"")
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Argument, E> {
        Ok(Argument(Expr::Lit(Literal::Num(v as f64))))
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Argument, E> {
        Ok(Argument(Expr::Lit(Literal::Num(v as f64))))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Argument, E> {
        Ok(Argument(Expr::Lit(Literal::Num(v))))
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Argument, E> {
        let tokens = tokenize(s).map_err(|d| E::custom(d.message))?;
        let unsigned = match tokens.as_slice() {
            [first, rest @ ..] if first.kind == TokenKind::Minus => rest,
            all => all,
        };
        let literal = match unsigned {
            [only] => match &only.kind {
                TokenKind::Num(_)
                | TokenKind::Time(..)
                | TokenKind::Samples(_)
                | TokenKind::Log(..) => true,
                TokenKind::Ident(name) => {
                    unsigned.len() == tokens.len() && crate::vocabulary::note_midi(name).is_some()
                }
                _ => false,
            },
            _ => false,
        };
        if !literal {
            return Err(E::custom(format!(
                "`{s}` is not one literal; a score binds a number, a number with a unit such \
                 as \"0.03s\" or \"-6db\", or a note name such as \"A1\""
            )));
        }
        crate::parser::parse(s)
            .map(Argument)
            .map_err(|d| E::custom(d.message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(text: &str) -> String {
        parse_score(text).unwrap_err().message
    }

    #[test]
    fn each_line_is_an_event_in_written_order() {
        let score = parse_score(
            "{\"at\": \"0b\", \"sound\": \"@bass\", \"deg\": 1, \"vel\": 0.9}\n\n\
             {\"at\": \"0.5b\", \"sound\": \"@bass\", \"len\": \"0.25b\", \"cutoff\": \"900hz\"}\n",
        )
        .unwrap();
        assert_eq!(score.events.len(), 2);
        assert_eq!(score.events[1].line, 3);
        assert_eq!(score.events[1].args[0].0, "len");
        assert_eq!(score.events[1].args[1].1, Expr::Lit(Literal::Num(900.0)));
    }

    #[test]
    fn a_refusal_names_the_line_and_column() {
        let text = "{\"at\": \"0b\", \"sound\": \"@bass\"}\n{\"at\": 0.5, \"sound\": \"@bass\"}\n";
        let refused = refusal(text);
        assert!(refused.starts_with("line 2, column "), "{refused}");
        assert!(refused.contains("bare number"), "{refused}");
        assert!(refusal("{\"at\": \"0b\"}").contains("missing field `sound`"));
        assert!(refusal("{\"at\": \"0b\", \"sound\": \"@b\", \"at\": \"1b\"}").contains("twice"));
        assert!(
            refusal("{\"at\": \"0b\", \"sound\": \"@b\", \"deg\": 1, \"f0\": 440}")
                .contains("`deg`")
        );
        assert!(
            refusal("{\"at\": \"0b\", \"sound\": \"@b\", \"vel\": \"x+1\"}").contains("literal")
        );
    }

    #[test]
    fn lines_out_of_order_or_mixing_units_refuse() {
        let late = "{\"at\": \"1b\", \"sound\": \"@b\"}\n{\"at\": \"0.5b\", \"sound\": \"@b\"}\n";
        assert!(refusal(late).contains("sorted"));
        let mixed = "{\"at\": \"0b\", \"sound\": \"@b\"}\n{\"at\": \"1s\", \"sound\": \"@b\"}\n";
        assert!(refusal(mixed).contains("not both"));
    }

    #[test]
    fn the_sum_is_balanced() {
        let line = "{\"at\": \"0.5s\", \"sound\": \"@b\"}\n";
        let score = parse_score(&line.repeat(1024)).unwrap();
        fn depth(e: &Expr) -> usize {
            match e {
                Expr::Bin(BinOp::Add, l, r) => 1 + depth(l).max(depth(r)),
                _ => 0,
            }
        }
        assert_eq!(
            depth(&score.lowered(&|_, _| Expr::Lit(Literal::Num(1.0)))),
            10
        );
    }
}
