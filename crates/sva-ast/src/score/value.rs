// Concern: folds one score value's constant arithmetic to an amount and its kind, a mark a position | Non-concern: which field holds it (read.rs) | IO: (text, what it may read) -> Value or why not

use crate::expr::DEGREE;
use crate::filename::SpanUnit;
use crate::lexer::{Token, TokenKind, tokenize};
use crate::score::ratio::Ratio;
use crate::vocabulary::note_hz;

/// What a value measures. `Home` is a move's own lane named in its `to`, whose kind the
/// file alone cannot say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Bars,
    Seconds,
    Hz,
    Number,
    Home,
}

impl Kind {
    pub fn of_unit(unit: SpanUnit) -> Kind {
        match unit {
            SpanUnit::Bars => Kind::Bars,
            SpanUnit::Seconds => Kind::Seconds,
        }
    }

    pub fn is_time(self) -> bool {
        matches!(self, Kind::Bars | Kind::Seconds)
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Bars => "time in bars",
            Kind::Seconds => "time in seconds",
            Kind::Hz => "frequency",
            Kind::Number => "plain number",
            Kind::Home => "the lane's own value",
        }
    }
}

/// Exact for a time field's bars, a double for every other value.
pub trait Amount: Copy {
    fn of(x: f64) -> Option<Self>;
    fn add(self, o: Self) -> Option<Self>;
    fn sub(self, o: Self) -> Option<Self>;
    fn mul(self, o: Self) -> Option<Self>;
    fn div(self, o: Self) -> Option<Self>;
    fn neg(self) -> Self;
}

impl Amount for Ratio {
    fn of(x: f64) -> Option<Ratio> {
        Ratio::of_f64(x)
    }
    fn add(self, o: Ratio) -> Option<Ratio> {
        self.plus(o)
    }
    fn sub(self, o: Ratio) -> Option<Ratio> {
        self.minus(o)
    }
    fn mul(self, o: Ratio) -> Option<Ratio> {
        self.times(o)
    }
    fn div(self, o: Ratio) -> Option<Ratio> {
        self.over(o)
    }
    fn neg(self) -> Ratio {
        self.negated()
    }
}

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

impl Amount for f64 {
    fn of(x: f64) -> Option<f64> {
        finite(x)
    }
    fn add(self, o: f64) -> Option<f64> {
        finite(self + o)
    }
    fn sub(self, o: f64) -> Option<f64> {
        finite(self - o)
    }
    fn mul(self, o: f64) -> Option<f64> {
        finite(self * o)
    }
    fn div(self, o: f64) -> Option<f64> {
        finite(self / o)
    }
    fn neg(self) -> f64 {
        -self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value<N> {
    pub amount: N,
    pub kind: Kind,
    /// A mark, or arithmetic keeping one: a point, where every other time is a length.
    pub position: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Time,
    Value,
    To,
}

/// What a value may read besides literals.
pub struct Reading<'a, N> {
    pub field: Field,
    pub mark: &'a dyn Fn(&str) -> Option<Value<N>>,
    pub degree: &'a dyn Fn(i64) -> Result<N, String>,
    pub own: Option<(&'a str, Value<N>)>,
}

/// Why a value is no constant: at a byte of its text, or a mark not yet known by name.
#[derive(Debug)]
pub enum Refused {
    At(usize, String),
    Mark(String, usize),
}

/// A score binds constants; what moves over time is a lane.
pub const MOVING: &str = "a score binds constants; what moves is a lane, a move line naming \
                          `lane` and `to`";

pub fn evaluate<N: Amount>(s: &str, reading: &Reading<N>) -> Result<Value<N>, Refused> {
    let tokens = tokenize(s).map_err(|d| Refused::At(d.span.start, d.message))?;
    let mut walk = Walk {
        s,
        tokens: &tokens,
        next: 0,
        reading,
    };
    if tokens.is_empty() {
        return Err(Refused::At(
            0,
            format!("`{s}` is empty; a value is a constant"),
        ));
    }
    let value = walk.sum()?;
    if let Some(left) = tokens.get(walk.next) {
        return Err(Refused::At(
            left.span.start,
            format!("`{s}` is no constant: an operator or the end was expected here"),
        ));
    }
    Ok(value)
}

struct Walk<'a, 'r, N> {
    s: &'a str,
    tokens: &'a [Token],
    next: usize,
    reading: &'a Reading<'r, N>,
}

const POSITION: &str = "a mark is a position: write `mark(a) ± time`, or `mark(a) - mark(b)` \
                        for a duration";

impl<N: Amount> Walk<'_, '_, N> {
    fn refused(&self, at: usize, why: String) -> Refused {
        Refused::At(at, format!("`{}` {why}", self.s))
    }

    fn op(&self, ops: [TokenKind; 2]) -> Option<(bool, usize)> {
        let token = self.tokens.get(self.next)?;
        ops.contains(&token.kind)
            .then_some((token.kind == ops[0], token.span.start))
    }

    fn number(&self, at: usize, x: Option<N>) -> Result<N, Refused> {
        x.ok_or_else(|| self.refused(at, "folds to no number a score can hold".to_string()))
    }

    fn sum(&mut self) -> Result<Value<N>, Refused> {
        let mut left = self.product()?;
        while let Some((plus, at)) = self.op([TokenKind::Plus, TokenKind::Minus]) {
            self.next += 1;
            let right = self.product()?;
            let position = match (left.position, right.position, plus) {
                (true, true, false) => false,
                (true, false, _) => true,
                (false, true, true) => true,
                (false, false, _) => false,
                _ => return Err(Refused::At(at, POSITION.to_string())),
            };
            let kind = self.same_kind(left.kind, right.kind, at, plus)?;
            let amount = match plus {
                true => left.amount.add(right.amount),
                false => left.amount.sub(right.amount),
            };
            left = Value {
                amount: self.number(at, amount)?,
                kind,
                position,
            };
        }
        Ok(left)
    }

    fn same_kind(&self, a: Kind, b: Kind, at: usize, plus: bool) -> Result<Kind, Refused> {
        let verb = if plus { "adds" } else { "subtracts" };
        match (a, b) {
            (a, b) if a == b => Ok(a),
            (Kind::Home, other) | (other, Kind::Home) => Ok(other),
            (a, b) if a.is_time() && b.is_time() => Err(self.refused(
                at,
                "counts in bars and in seconds; one file writes its times in one".to_string(),
            )),
            (a, b) if a.is_time() || b.is_time() => Err(self.refused(
                at,
                format!(
                    "{verb} a {} and a {}; a time needs its unit on every term, as \"2b + 1b/3\"",
                    a.name(),
                    b.name()
                ),
            )),
            (a, b) => Err(self.refused(
                at,
                format!(
                    "{verb} a {} and a {}; both sides need one kind",
                    a.name(),
                    b.name()
                ),
            )),
        }
    }

    fn product(&mut self) -> Result<Value<N>, Refused> {
        let mut left = self.unary()?;
        while let Some((times, at)) = self.op([TokenKind::Star, TokenKind::Slash]) {
            self.next += 1;
            let right = self.unary()?;
            if left.position || right.position {
                return Err(Refused::At(at, POSITION.to_string()));
            }
            let kind = self.scaled(left.kind, right.kind, at, times)?;
            let amount = match times {
                true => left.amount.mul(right.amount),
                false => left.amount.div(right.amount),
            };
            left = Value {
                amount: self.number(at, amount)?,
                kind,
                position: false,
            };
        }
        Ok(left)
    }

    fn scaled(&self, a: Kind, b: Kind, at: usize, times: bool) -> Result<Kind, Refused> {
        use Kind::{Home, Number};
        match (times, a, b) {
            (true, Number, other) | (true, other, Number) => Ok(other),
            (true, Home, other) | (true, other, Home) => Ok(other),
            (true, a, b) if a.is_time() && b.is_time() => Err(self.refused(
                at,
                "multiplies a time by a time; scale a time by a plain number".to_string(),
            )),
            (false, a, b) if a.is_time() && b.is_time() => Err(self.refused(
                at,
                "divides a time by a time; divide a time by a plain number".to_string(),
            )),
            (false, other, Number) => Ok(other),
            (false, Home, _) | (false, _, Home) => Ok(Home),
            (false, a, b) if a == b => Ok(Number),
            (_, a, b) => Err(self.refused(
                at,
                format!(
                    "{} a {} by a {}, which has no meaning",
                    if times { "multiplies" } else { "divides" },
                    a.name(),
                    b.name()
                ),
            )),
        }
    }

    fn unary(&mut self) -> Result<Value<N>, Refused> {
        match self.op([TokenKind::Minus, TokenKind::Plus]) {
            Some((minus, at)) => {
                self.next += 1;
                let v = self.unary()?;
                if minus && v.position {
                    return Err(Refused::At(at, POSITION.to_string()));
                }
                Ok(Value {
                    amount: if minus { v.amount.neg() } else { v.amount },
                    ..v
                })
            }
            None => self.primary(),
        }
    }

    fn plain(&self, at: usize, x: f64, kind: Kind) -> Result<Value<N>, Refused> {
        Ok(Value {
            amount: self.number(at, N::of(x))?,
            kind,
            position: false,
        })
    }

    fn primary(&mut self) -> Result<Value<N>, Refused> {
        let Some(token) = self.tokens.get(self.next) else {
            return Err(self.refused(self.s.len(), "ends where a value was expected".to_string()));
        };
        let at = token.span.start;
        self.next += 1;
        let written = &self.s[token.span.start..token.span.end];
        let called = matches!(
            self.tokens.get(self.next).map(|t| &t.kind),
            Some(TokenKind::LParen)
        );
        match &token.kind {
            TokenKind::Num(n) if written.ends_with("hz") => self.plain(at, *n, Kind::Hz),
            TokenKind::Num(n) => self.plain(at, *n, Kind::Number),
            TokenKind::Time(n, unit) => self.plain(at, *n, Kind::of_unit(*unit)),
            TokenKind::Log(n, unit) => self.plain(at, unit.resolve(*n), Kind::Number),
            TokenKind::Samples(_) => Err(self.refused(
                at,
                "counts a render's own samples; a score's time is bars or seconds".to_string(),
            )),
            TokenKind::LParen => {
                let inner = self.sum()?;
                match self.tokens.get(self.next).map(|t| &t.kind) {
                    Some(TokenKind::RParen) => {
                        self.next += 1;
                        Ok(inner)
                    }
                    _ => Err(self.refused(at, "opens a ( it never closes".to_string())),
                }
            }
            TokenKind::Ident(name) if called && name == DEGREE => self.degree(at),
            TokenKind::Ident(name) if called && name == "mark" => self.mark(at),
            TokenKind::Ident(name) if called => Err(self.refused(
                at,
                format!(
                    "calls `{name}`; a score value is arithmetic over literals, `deg(n)` and \
                     `mark(name)`, and a sound's own math belongs in its equation"
                ),
            )),
            TokenKind::Ident(name) if self.reading.own.is_some_and(|(own, _)| own == name) => {
                let (_, home) = self.reading.own.expect("matched");
                Ok(home)
            }
            TokenKind::Ident(name) if name == "pi" => {
                self.plain(at, std::f64::consts::PI, Kind::Number)
            }
            TokenKind::Ident(name) if note_hz(name).is_some() => {
                self.plain(at, note_hz(name).expect("a note"), Kind::Hz)
            }
            TokenKind::Ident(name) if name == "t" || name == "f" => {
                Err(self.refused(at, format!("reads `{name}`; {MOVING}")))
            }
            TokenKind::Ref(path) => Err(self.refused(at, format!("reads `@{path}`; {MOVING}"))),
            TokenKind::Ident(name) => Err(self.refused(
                at,
                format!(
                    "names `{name}`, which a score cannot read: {MOVING}; a value is numbers, \
                     literals with units such as \"0.03s\", \"900hz\" or \"-6db\", and note \
                     names such as \"A1\""
                ),
            )),
            _ => Err(self.refused(
                at,
                "is no constant: a score value is arithmetic".to_string(),
            )),
        }
    }

    /// The one argument of a call, as its tokens: `(` was just read past the name.
    fn argument(&mut self, at: usize) -> Result<&str, Refused> {
        let open = self.next;
        let mut depth = 0usize;
        while let Some(token) = self.tokens.get(self.next) {
            self.next += 1;
            match token.kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen if depth == 1 => {
                    let from = self.tokens[open].span.end;
                    return Ok(&self.s[from..token.span.start]);
                }
                TokenKind::RParen => depth -= 1,
                _ => {}
            }
        }
        Err(self.refused(at, "opens a ( it never closes".to_string()))
    }

    fn degree(&mut self, at: usize) -> Result<Value<N>, Refused> {
        if self.reading.field == Field::Time {
            return Err(self.refused(
                at,
                "reads `deg(...)` in a time field; a degree stands in any value but `at` and \
                 `len`"
                    .to_string(),
            ));
        }
        let inner = self.argument(at)?;
        let plain = Reading::<f64> {
            field: Field::Value,
            mark: &|_| None,
            degree: &|_| Err(String::new()),
            own: None,
        };
        let whole = match evaluate(inner, &plain) {
            Ok(Value {
                amount,
                kind: Kind::Number,
                ..
            }) if amount.fract() == 0.0 && amount.abs() < 2f64.powi(53) => amount as i64,
            _ => {
                return Err(self.refused(
                    at,
                    "reads `deg` of no whole degree; write deg(5) or deg(-6), and a chromatic \
                     note as deg(7)*1st"
                        .to_string(),
                ));
            }
        };
        let hz = (self.reading.degree)(whole).map_err(|why| Refused::At(at, why))?;
        Ok(Value {
            amount: hz,
            kind: Kind::Hz,
            position: false,
        })
    }

    fn mark(&mut self, at: usize) -> Result<Value<N>, Refused> {
        if self.reading.field != Field::Time {
            return Err(self.refused(
                at,
                "reads `mark(...)` outside a time field; a mark stands only in `at` and `len`"
                    .to_string(),
            ));
        }
        let name = self.argument(at)?.trim().to_string();
        match (self.reading.mark)(&name) {
            Some(value) => Ok(Value {
                position: true,
                ..value
            }),
            None => Err(Refused::Mark(name, at)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(s: &str) -> Result<Value<Ratio>, String> {
        let mark = |name: &str| {
            (name == "drop").then(|| Value {
                amount: Ratio::int(8),
                kind: Kind::Bars,
                position: true,
            })
        };
        let reading = Reading {
            field: Field::Time,
            mark: &mark,
            degree: &|_| Err(String::new()),
            own: None,
        };
        evaluate(s, &reading).map_err(|r| match r {
            Refused::At(_, why) => why,
            Refused::Mark(name, _) => format!("mark {name}"),
        })
    }

    #[test]
    fn times_fold_exactly_with_their_unit() {
        let third = time("2b + 1b/3").unwrap();
        assert_eq!(third.kind, Kind::Bars);
        assert_eq!(
            third.amount,
            Ratio::int(7).over(Ratio::int(3)).expect("a third")
        );
        assert_eq!(time("40ms").unwrap().kind, Kind::Seconds);
        for (written, why) in [
            ("2b + 1", "adds a time in bars and a plain number"),
            ("1b*1b", "multiplies a time by a time"),
            ("1b/1b", "divides a time by a time"),
            ("1b + 1s", "counts in bars and in seconds"),
            ("2*mark(drop)", "a mark is a position"),
            ("mark(drop) + mark(drop)", "a mark is a position"),
            ("deg(5)", "time field"),
            ("t + 1b", "reads `t`"),
            ("@x", "reads `@x`"),
            ("sin(1)", "calls `sin`"),
            ("mark(verse)", "mark verse"),
        ] {
            let refused = time(written).unwrap_err();
            assert!(refused.contains(why), "{written}: {refused}");
        }
        let gap = time("mark(drop) - mark(drop)").unwrap();
        assert!(!gap.position);
        assert!(time("mark(drop) - 2b").unwrap().position);
    }
}
