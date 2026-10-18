// Concern: sweeps each lane's moves in song order into its value over time, refusing ties, overlaps and values it cannot hold | Non-concern: lowering a lane (lower.rs) | IO: (moves) -> Lanes or Refusals

use std::cmp::Ordering;

use crate::diag::DiagCode;
use crate::graph::{Graph, VARIABLES};
use crate::refusal::Refusal;
use crate::scale::Scale;
use crate::score::{Field, Item, Kind, Move, Reading, Refused, Shape, Value, evaluate};
use crate::song::place::{Reached, at_line};
use crate::song::tempo::Curve;

/// A lane's name: a variable under `variables/`, or a node's parameter.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LaneId {
    Variable(String),
    Sound(String, String),
    /// Degree n's ratio in the scale lane's value: what an equation's own `deg(n)` reads
    /// where the scale moves.
    Degree(i64),
}

impl LaneId {
    /// `variables/energy` or `instruments/supersaw:cutoff`: the lane node's path, and the lane
    /// parameter's name, which no file can spell.
    pub fn name(&self) -> String {
        match self {
            LaneId::Variable(name) => format!("{VARIABLES}/{name}"),
            LaneId::Sound(node, param) => format!("{node}:{param}"),
            LaneId::Degree(n) => format!("{VARIABLES}/scale:{n}"),
        }
    }

    pub fn is_variable(&self, name: &str) -> bool {
        matches!(self, LaneId::Variable(held) if held == name)
    }

    fn own_name(&self) -> &str {
        match self {
            LaneId::Variable(name) | LaneId::Sound(_, name) => name,
            LaneId::Degree(_) => "scale",
        }
    }
}

/// What a lane holds until its first move.
#[derive(Clone, Debug, PartialEq)]
pub enum Home {
    Number(f64, Kind),
    Scale(Scale),
    /// A default that is no constant, as `4*f0`: each instance reads its own.
    Default(String),
    None,
}

impl Home {
    fn number(&self) -> Option<(f64, Kind)> {
        match self {
            Home::Number(v, k) => Some((*v, *k)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Held {
    Number(f64),
    Scale(Scale),
}

impl Held {
    pub fn number(&self) -> Option<f64> {
        match self {
            Held::Number(v) => Some(*v),
            Held::Scale(_) => None,
        }
    }
}

/// One move on its lane, in song seconds, with the value it goes from and to.
pub struct Swept {
    pub reached: Reached,
    pub start: f64,
    pub end: f64,
    pub shape: Option<Shape>,
    pub from: Option<Held>,
    pub to: Held,
}

pub struct Lane {
    pub id: LaneId,
    pub home: Home,
    pub moves: Vec<Swept>,
}

/// Degree n's ratio lane, set wherever the scale lane is.
pub(crate) fn degree_lane(scale: &Lane, n: i64) -> Option<Lane> {
    let ratio = |held: &Held| match held {
        Held::Scale(s) => s.ratio(n).map(Held::Number),
        Held::Number(_) => None,
    };
    let Home::Scale(home) = &scale.home else {
        return None;
    };
    let mut moves = Vec::with_capacity(scale.moves.len());
    for m in &scale.moves {
        moves.push(Swept {
            reached: m.reached.clone(),
            start: m.start,
            end: m.end,
            shape: None,
            from: m.from.as_ref().and_then(ratio),
            to: ratio(&m.to)?,
        });
    }
    Some(Lane {
        id: LaneId::Degree(n),
        home: Home::Number(home.ratio(n)?, Kind::Number),
        moves,
    })
}

/// A lane's value over a span: its home, or one value held.
#[derive(Clone, Debug, PartialEq)]
pub enum Hold {
    Home,
    Value(Held),
}

impl Lane {
    /// The value from second `s` on, sets at `s` included; `None` inside a ramp that moves.
    fn state(&self, s: f64) -> Option<Hold> {
        let mut held = Hold::Home;
        for m in &self.moves {
            if m.start > s {
                break;
            }
            if s < m.end && m.from.as_ref() != Some(&m.to) {
                return None;
            }
            held = Hold::Value(m.to.clone());
        }
        Some(held)
    }

    /// The value held at every instant from `a` to `b`, both read; `b` open where `None`.
    pub fn constant_over(&self, a: f64, b: Option<f64>) -> Option<Hold> {
        let held = self.state(a)?;
        let later = self
            .moves
            .iter()
            .filter(|m| m.start > a && b.is_none_or(|b| m.start <= b));
        for m in later {
            let moving = m.end > m.start && m.from.as_ref() != Some(&m.to);
            if moving || Hold::Value(m.to.clone()) != held {
                return None;
            }
        }
        Some(held)
    }

    /// Whether it changes after second `a`, or is moving through it.
    pub fn changes_after(&self, a: f64) -> bool {
        self.constant_over(a, None).is_none()
    }

    /// Its value at second `s`, a ramp's included; `None` before a first set over no constant.
    pub fn at(&self, s: f64) -> Option<Held> {
        let mut held = match &self.home {
            Home::Number(v, _) => Some(Held::Number(*v)),
            Home::Scale(scale) => Some(Held::Scale(scale.clone())),
            _ => None,
        };
        for m in &self.moves {
            if m.start > s {
                break;
            }
            if s < m.end {
                let (Some(Held::Number(a)), Held::Number(b)) = (&m.from, &m.to) else {
                    return held;
                };
                let u = (s - m.start) / (m.end - m.start);
                return Some(Held::Number(match m.shape {
                    Some(Shape::Exp) => a * (b / a).powf(u),
                    _ => a + (b - a) * u,
                }));
            }
            held = Some(m.to.clone());
        }
        held
    }

    /// Whether the home value is what `s` reads: before any move.
    pub fn at_home(&self, s: f64) -> bool {
        self.moves.first().is_none_or(|m| m.start > s)
    }
}

/// What a variable's file holds as a lane's home, or why it takes no moves.
fn variable_home(graph: &Graph, name: &str) -> Result<Home, String> {
    let path = format!("{VARIABLES}/{name}");
    let text = graph.text(&path).unwrap_or("");
    let body = text[crate::ingest::body_at(text)..].trim();
    if name == "meter" {
        return Err("`meter` takes no moves; a song's bars keep one meter".to_string());
    }
    if name == "scale" {
        let expr = graph.source_expr(&path).expect("a variable read");
        return graph
            .scale_at(&expr, &path, None)
            .map(Home::Scale)
            .map_err(|why| format!("`{path}` holds no scale a move can start from: {why}"));
    }
    match constant(body, None) {
        Some((v, kind)) => Ok(Home::Number(v, kind)),
        None => Err(format!(
            "`{path}` holds `{body}`, an equation; a variable that moves holds a constant, a \
             meter or a scale, so move a variable holding one"
        )),
    }
}

/// A constant expression's value and kind, a bar a home bar's seconds.
fn constant(text: &str, per_bar: Option<f64>) -> Option<(f64, Kind)> {
    let reading = Reading::<f64> {
        field: Field::Value,
        mark: &|_| None,
        degree: &|_| Err(String::new()),
        own: None,
    };
    let value = evaluate(text, &reading).ok()?;
    Some(as_seconds(value, per_bar))
}

fn as_seconds(value: Value<f64>, per_bar: Option<f64>) -> (f64, Kind) {
    match (value.kind, per_bar) {
        (Kind::Bars, Some(per_bar)) => (value.amount * per_bar, Kind::Seconds),
        (kind, _) => (value.amount, kind),
    }
}

fn kind_of(kind: Kind) -> &'static str {
    match kind {
        Kind::Bars | Kind::Seconds => "a time",
        Kind::Hz => "a frequency",
        _ => "a plain number",
    }
}

fn sound_home(graph: &Graph, node: &str, param: &str, per_bar: Option<f64>) -> Home {
    let text = graph.text(node).unwrap_or("");
    match crate::ingest::default_texts(text)
        .into_iter()
        .find(|(name, _)| name == param)
    {
        None => Home::None,
        Some((_, value)) => match constant(&value, per_bar) {
            Some((v, kind)) => Home::Number(v, kind),
            None => Home::Default(value),
        },
    }
}

pub(crate) struct Sweeping<'a> {
    pub graph: &'a Graph,
    pub curve: &'a Curve,
    pub per_bar: Option<f64>,
    /// The key at a second, in hertz, and the scale; a `to` reading `deg(n)` reads them.
    pub degree: &'a dyn Fn(f64, i64) -> Result<f64, String>,
}

fn the_move<'g>(graph: &'g Graph, r: &Reached) -> &'g Move {
    let score = graph.score(&r.score).expect("a placed score");
    match &score.lines[r.line].items[r.item] {
        Item::Move(m) => m,
        _ => unreachable!("a reached move is a move"),
    }
}

impl Sweeping<'_> {
    fn refuse(&self, r: &Reached, at: usize, message: String) -> Refusal {
        let curve = self.curve;
        at_line(
            self.graph,
            &r.score,
            at,
            DiagCode::BadLane,
            &format!("{message} ({})", r.chain.text(curve)),
        )
    }

    fn line_of(&self, r: &Reached) -> String {
        let number = self.graph.score(&r.score).expect("a score").lines[r.line].number;
        format!("{}{}:{number}", r.score, crate::score::EXTENSION)
    }

    /// Every lane's moves, swept; `reached` in any order.
    pub fn sweep(&self, reached: Vec<Reached>) -> Result<Vec<Lane>, Vec<Refusal>> {
        let mut by_lane: Vec<(crate::song::LaneId, Vec<Reached>)> = Vec::new();
        for r in reached {
            match by_lane.iter_mut().find(|(id, _)| *id == r.lane) {
                Some((_, held)) => held.push(r),
                None => by_lane.push((r.lane.clone(), vec![r])),
            }
        }
        by_lane.sort_by(|a, b| a.0.cmp(&b.0));
        let mut lanes = Vec::new();
        let mut refusals = Vec::new();
        for (id, moves) in by_lane {
            match self.lane(id, moves) {
                Ok(lane) => lanes.push(lane),
                Err(refused) => refusals.push(refused),
            }
        }
        match refusals.is_empty() {
            true => Ok(lanes),
            false => Err(refusals),
        }
    }

    fn lane(&self, id: LaneId, mut moves: Vec<Reached>) -> Result<Lane, Refusal> {
        let graph = self.graph;
        let curve = self.curve;
        let ramp = |r: &Reached| r.end.is_some();
        moves.sort_by(|a, b| a.start.cmp(b.start, curve).then(ramp(a).cmp(&ramp(b))));
        let home = match &id {
            LaneId::Variable(name) => variable_home(graph, name)
                .map_err(|why| self.refuse(&moves[0], the_move(graph, &moves[0]).lane.at, why))?,
            LaneId::Sound(node, param) => sound_home(graph, node, param, self.per_bar),
            LaneId::Degree(_) => unreachable!("a degree lane is derived, never moved"),
        };
        for (i, m) in moves.iter().enumerate() {
            let here = the_move(graph, m);
            if i > 0 && moves[i - 1].start.cmp(m.start, curve) == Ordering::Equal {
                let before = &moves[i - 1];
                if !(before.end.is_none() && m.end.is_some()) {
                    let same =
                        before.score == m.score && before.line == m.line && before.item == m.item;
                    let why = match same {
                        true => format!(
                            "two moves on `{}` at {}: one line placed twice at one position, \
                             {} and {}",
                            id.name(),
                            m.start.spelled(curve),
                            before.chain.text(curve),
                            m.chain.text(curve)
                        ),
                        false => format!(
                            "two moves on `{}` at {}: this line and {}; one position takes a \
                             set and a ramp from it, and nothing else",
                            id.name(),
                            m.start.spelled(curve),
                            self.line_of(before)
                        ),
                    };
                    return Err(self.refuse(m, here.at.at, why));
                }
            }
            if let Some(r) = moves[..i].iter().find(|r| {
                r.end.is_some_and(|end| {
                    r.start.cmp(m.start, curve) == Ordering::Less
                        && m.start.cmp(end, curve) == Ordering::Less
                })
            }) {
                return Err(self.refuse(
                    m,
                    here.at.at,
                    format!(
                        "this move on `{}` starts at {}, inside the ramp from {} to {} on {}; a \
                         move starts where a ramp ends or before it starts",
                        id.name(),
                        m.start.spelled(curve),
                        r.start.spelled(curve),
                        r.end.expect("a ramp").spelled(curve),
                        self.line_of(r)
                    ),
                ));
            }
        }
        let mut held: Option<Held> = match &home {
            Home::Number(v, _) => Some(Held::Number(*v)),
            Home::Scale(scale) => Some(Held::Scale(scale.clone())),
            _ => None,
        };
        let mut swept = Vec::with_capacity(moves.len());
        for r in moves {
            let m = the_move(graph, &r);
            let start = r.start.seconds(curve);
            let end = r.end.map_or(start, |e| e.seconds(curve));
            if id.is_variable("bpm") && r.start.bars(curve) < 0.0 {
                return Err(self.refuse(
                    &r,
                    m.at.at,
                    format!(
                        "a `bpm` move at bar {}, before bar 0, where the tempo curve starts; \
                         move the tempo from bar 0 on",
                        r.start.bars(curve)
                    ),
                ));
            }
            let to = match &home {
                Home::Scale(_) => {
                    if m.ramp.is_some() {
                        return Err(self.refuse(
                            &r,
                            m.to.at,
                            "`scale` takes sets only: a scale has no value between two, so a \
                             move on it has no `len` or `shape`"
                                .to_string(),
                        ));
                    }
                    let written = crate::parser::parse(&m.to.text)
                        .map_err(|d| self.refuse(&r, m.to.inner + d.span.start, d.message))?;
                    Held::Scale(
                        graph
                            .scale_at(&written, &r.score, None)
                            .map_err(|why| self.refuse(&r, m.to.at, why))?,
                    )
                }
                _ => Held::Number(self.to(&id, &home, &r, m, start)?),
            };
            let from = held.clone();
            if m.ramp.is_some() {
                let Some(from) = &from else {
                    return Err(self.refuse(
                        &r,
                        m.to.at,
                        format!(
                            "this ramp on `{}` starts from its home value, and the home is \
                             `{}`, no constant; set it first, as {{\"at\": \"...\", \"node\": \
                             \"...\", \"lane\": \"{}\", \"to\": ...}} at or before this ramp",
                            id.name(),
                            match &home {
                                Home::Default(text) => text.as_str(),
                                _ => "unset",
                            },
                            id.own_name()
                        ),
                    ));
                };
                if let (Some(Shape::Exp), Held::Number(a), Held::Number(b)) =
                    (m.ramp.map(|r| r.shape), from, &to)
                    && (*a == 0.0 || *b == 0.0 || (*a > 0.0) != (*b > 0.0))
                {
                    return Err(self.refuse(
                        &r,
                        m.to.at,
                        format!(
                            "an `exp` ramp from {a} to {b}: `exp` multiplies, so it never \
                             reaches or crosses 0; ramp `lin`, or between two values of one sign"
                        ),
                    ));
                }
            }
            held = Some(to.clone());
            swept.push(Swept {
                start,
                end,
                shape: m.ramp.map(|r| r.shape),
                from,
                to,
                reached: r,
            });
        }
        Ok(Lane {
            id,
            home,
            moves: swept,
        })
    }

    fn to(
        &self,
        id: &LaneId,
        home: &Home,
        r: &Reached,
        m: &Move,
        start: f64,
    ) -> Result<f64, Refusal> {
        let name = id.own_name();
        let own = home.number().map(|(amount, kind)| {
            (
                name,
                Value {
                    amount,
                    kind,
                    position: false,
                },
            )
        });
        let degree = |n: i64| (self.degree)(start, n);
        let reading = Reading {
            field: Field::To,
            mark: &|_| None,
            degree: &degree,
            own,
        };
        let value = evaluate(&m.to.text, &reading).map_err(|refused| match refused {
            Refused::At(by, _) if own.is_none() && m.to.text.contains(name) => self.refuse(
                r,
                m.to.inner + by,
                format!(
                    "`to` names `{name}`, the lane's home value, and `{}` has none that is a \
                     constant ({}); set it first, to a number",
                    id.name(),
                    match home {
                        Home::Default(text) => format!("its default is `{text}`"),
                        _ => "it has no default".to_string(),
                    }
                ),
            ),
            Refused::At(by, why) => self.refuse(r, m.to.inner + by, why),
            Refused::Mark(_, by) => self.refuse(
                r,
                m.to.inner + by,
                "reads `mark(...)` outside a time field".to_string(),
            ),
        })?;
        let (amount, kind) = as_seconds(value, self.per_bar);
        if let Some((_, want)) = home.number() {
            let time = |k: Kind| k.is_time();
            if kind != want && !(time(kind) && time(want)) {
                return Err(self.refuse(
                    r,
                    m.to.at,
                    format!(
                        "`to` is {}, and `{}`'s home value is {}; a lane holds one kind",
                        kind_of(kind),
                        id.name(),
                        kind_of(want)
                    ),
                ));
            }
        }
        Ok(amount)
    }
}
