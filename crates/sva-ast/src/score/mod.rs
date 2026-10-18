// Concern: a score file's lines as its own file reads them: kinds, items, times from its zero | Non-concern: where a song places it, lanes and lowering (song/) | IO: (&str) -> Score or a located Diag

mod ratio;
mod read;
mod value;

pub use ratio::Ratio;
pub(crate) use read::is_old_score;
pub use read::parse_score;
pub use value::{Amount, Field, Kind, MOVING, Reading, Refused, Value, evaluate};

use crate::diag::{ByteSpan, Diag, DiagCode};
use crate::expr::{Address, BinOp, DEGREE, Expr, Literal, children};
use crate::filename::SpanUnit;
use crate::graph::VARIABLES;

pub const EXTENSION: &str = ".jsonl";

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub annotation: Option<String>,
    pub unit: SpanUnit,
    pub lines: Vec<Line>,
}

impl Score {
    pub fn items(&self) -> impl Iterator<Item = (&Line, &Item)> {
        self.lines
            .iter()
            .flat_map(|line| line.items.iter().map(move |item| (line, item)))
    }

    /// Every file a line names: what loading a score follows, as any node's body. An event
    /// reads its node at `t`; a move names its node and plays it at no time of its own, so a
    /// song moving what places it is no cycle. A score sounds only as a song lowers it.
    pub(crate) fn reads(&self, path: &str) -> Expr {
        let root = "../".repeat(path.matches('/').count());
        let mut named: Vec<(String, usize, bool)> = Vec::new();
        for (_, item) in self.items() {
            match item {
                Item::Event(e) => named.push((e.node.name.clone(), e.node.at, true)),
                Item::Move(m) => match &m.node {
                    Some(node) => named.push((node.name.clone(), node.at, false)),
                    None => {
                        let variable = format!("{root}{VARIABLES}/{}", m.lane.name);
                        named.push((variable, m.lane.at, false));
                        if let Some(scale) = m.to.text.strip_prefix('@') {
                            named.push((scale.to_string(), m.to.inner, false));
                        }
                    }
                },
                Item::Mark(_) | Item::Comment(_) => {}
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        named.retain(|(name, _, played)| seen.insert((name.clone(), *played)));
        let read = |(path, at, played): (String, usize, bool)| Expr::Ref {
            path,
            arg: Box::new(match played {
                true => Expr::Var("t".to_string()),
                false => Expr::Lit(Literal::Num(0.0)),
            }),
            binds: Vec::new(),
            address: Address::Time,
            span: ByteSpan::new(at, at),
        };
        named
            .into_iter()
            .map(read)
            .reduce(|a, b| Expr::Bin(BinOp::Add, Box::new(a), Box::new(b)))
            .unwrap_or(Expr::Lit(Literal::Num(0.0)))
    }

    pub(crate) fn reads_degrees(&self) -> bool {
        fn calls(e: &Expr) -> bool {
            matches!(e, Expr::Call { name, .. } if name == DEGREE)
                || children(e, crate::expr::Binds::Substitute)
                    .into_iter()
                    .any(calls)
        }
        self.items().any(|(_, item)| match item {
            Item::Event(e) => e.deg.is_some() || e.binds.iter().any(|b| calls(&b.value)),
            Item::Move(m) => m.to.text.contains("deg("),
            _ => false,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub number: usize,
    pub span: ByteSpan,
    pub items: Vec<Item>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Event(Event),
    Mark(Mark),
    Comment(Time),
    Move(Move),
}

impl Item {
    pub fn at(&self) -> Time {
        match self {
            Item::Event(e) => e.at,
            Item::Mark(m) => m.at,
            Item::Comment(at) => *at,
            Item::Move(m) => m.at,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Item::Event(_) => "event",
            Item::Mark(_) => "mark",
            Item::Comment(_) => "comment",
            Item::Move(_) => "move",
        }
    }
}

/// From the score's zero in its file's unit, beside the byte its value is written at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Time {
    pub amount: Ratio,
    pub at: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Named {
    pub name: String,
    pub at: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub at: Time,
    pub node: Named,
    pub deg: Option<(i64, usize)>,
    pub len: Option<Time>,
    pub binds: Vec<Bind>,
}

/// A parameter a line binds: `key` is the byte its key is written at, `at` its value's.
#[derive(Clone, Debug, PartialEq)]
pub struct Bind {
    pub name: String,
    pub value: Expr,
    pub key: usize,
    pub at: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub at: Time,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Move {
    pub at: Time,
    pub node: Option<Named>,
    pub lane: Named,
    pub to: To,
    pub ramp: Option<Ramp>,
}

/// A move's `to` as written, read once the lane's home value is known.
#[derive(Clone, Debug, PartialEq)]
pub struct To {
    pub text: String,
    /// The byte the value starts at, and the byte its text starts at, past a string's quote.
    pub at: usize,
    pub inner: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp {
    pub len: Time,
    pub shape: Shape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Lin,
    Exp,
}

impl Shape {
    pub fn name(self) -> &'static str {
        match self {
            Shape::Lin => "lin",
            Shape::Exp => "exp",
        }
    }
}

pub fn spelled(amount: f64, unit: SpanUnit) -> String {
    match unit {
        SpanUnit::Bars => format!("{amount}b"),
        SpanUnit::Seconds => format!("{amount}s"),
    }
}

/// The line and column, from 1, a byte of `text` stands at; columns count bytes.
pub fn line_column(text: &str, at: usize) -> (usize, usize) {
    let before = &text[..at.min(text.len())];
    let line = before.matches('\n').count() + 1;
    let column = before.len() - before.rfind('\n').map_or(0, |n| n + 1) + 1;
    (line, column)
}

pub fn refused(
    file: String,
    text: &str,
    at: usize,
    code: DiagCode,
    message: &str,
) -> crate::Refusal {
    let place = line_column(text, at);
    let diag = located(text, at, message);
    crate::Refusal::new(file, diag.span, code, diag.message).placed(place)
}

pub fn located(text: &str, at: usize, message: &str) -> Diag {
    let (line, column) = line_column(text, at);
    Diag::new(
        DiagCode::BadScore,
        ByteSpan::at(at),
        format!("line {line}, column {column}: {message}"),
    )
}
