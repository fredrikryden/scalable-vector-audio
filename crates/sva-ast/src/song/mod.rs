// Concern: derives a song from its root: placements on the tempo curve, lanes, the equations it lowers to | Non-concern: an instance's span, which the caller asks the engine | IO: (&Graph, root) -> Song

mod lanes;
mod lower;
mod place;
mod reach;
mod tempo;

use std::collections::BTreeSet;

pub use lanes::{Held, Hold, Home, Lane, LaneId, Swept};
pub use place::parameters;
pub use place::{Chain, Instance, PlacedMark, Pos, Reached};
pub use tempo::Curve;

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::Expr;
use crate::graph::{Defined, Graph, VARIABLES, variable_text};
use crate::refusal::Refusal;
use crate::scale::Scale;
use crate::score::{EXTENSION, Item, Ratio, Shape, located};

use lanes::Sweeping;
use place::{Container, Placing, at_line};
use reach::Reach;
use tempo::TempoMove;

/// One instance's span from its onset: the instants of its first and last samples, `None` an
/// end the engine leaves open.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    pub start: Option<f64>,
    pub end: Option<f64>,
}

/// A warning a song's files earn: it prints, and the render goes on.
#[derive(Clone, Debug, PartialEq)]
pub struct Warning {
    pub code: &'static str,
    pub file: String,
    pub span: ByteSpan,
    pub start: (usize, usize),
    pub message: String,
}

pub struct Song {
    root: String,
    curve: Curve,
    tree: Container,
    instances: Vec<Instance>,
    lanes: Vec<Lane>,
    marks: Vec<PlacedMark>,
    instruments: Vec<(String, Defined)>,
    asked: Vec<(usize, Expr)>,
    warnings: Vec<Warning>,
    key: Option<f64>,
    scale: Option<Scale>,
    passed: Vec<(usize, LaneId)>,
    containers: BTreeSet<String>,
}

/// What a song lowers to: each node's new body, the root's included.
pub struct Lowered {
    pub nodes: Vec<(String, Defined)>,
}

fn refused(refusals: Vec<Refusal>) -> Result<(), Vec<Refusal>> {
    match refusals.is_empty() {
        true => Ok(()),
        false => Err(refusals),
    }
}

impl Graph {
    /// Steps 3 to 7 short of the spans: every line `root` reaches placed on the tempo curve, every
    /// lane swept, and each node no score sits under lowered with the lanes reaching it.
    pub fn song(&self, root: &str) -> Result<Song, Vec<Refusal>> {
        let values = self.values(None)?;
        let (bpm, beats) = match &values.tempo {
            Some(tempo) => (tempo.bpm, tempo.beats()),
            None => (60.0, 1.0),
        };
        let steady = Curve::new(bpm, beats, &[]);
        let zero = Some(Pos::Bars(Ratio::ZERO));
        let tempo = {
            let mut first = Placing::new(self, &steady);
            first.place(root, zero, None, Chain::default());
            refused(std::mem::take(&mut first.refusals))?;
            let reached: Vec<Reached> = first
                .moves
                .into_iter()
                .filter(|m| m.lane.is_variable("bpm"))
                .collect();
            let mut refusals = Vec::new();
            for m in reached
                .iter()
                .filter(|m| matches!(m.start, Pos::Seconds(_)))
            {
                refusals.push(self.through_seconds(m, &steady, "a `bpm` move"));
            }
            refused(refusals)?;
            let sweeping = Sweeping {
                graph: self,
                curve: &steady,
                per_bar: values.tempo.as_ref().map(|t| t.seconds_per_bar()),
                degree: &|_, _| Err("a `bpm` move reads no degree".to_string()),
            };
            let lanes = sweeping.sweep(reached)?;
            lanes
                .first()
                .map(|lane| {
                    lane.moves
                        .iter()
                        .map(|m| {
                            let Pos::Bars(start) = m.reached.start else {
                                unreachable!("refused above")
                            };
                            let len = match m.reached.end {
                                Some(Pos::Bars(end)) => end.minus(start).expect("a bar span"),
                                _ => Ratio::ZERO,
                            };
                            TempoMove {
                                start,
                                len,
                                shape: m.shape.unwrap_or(Shape::Lin),
                                from: m.from.as_ref().and_then(Held::number).unwrap_or(bpm),
                                to: m.to.number().expect("bpm is a number"),
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let curve = Curve::new(bpm, beats, &tempo);
        let mut placing = Placing::new(self, &curve);
        let tree = placing.place(root, zero, None, Chain::default());
        let mut refusals = std::mem::take(&mut placing.refusals);
        if curve.moves() {
            for m in placing
                .moves
                .iter()
                .filter(|m| matches!(m.start, Pos::Seconds(_)))
            {
                refusals.push(self.through_seconds(m, &curve, "a move"));
            }
            for (score, line, item, lost) in &placing.bars_lost {
                let at = self.score(score).expect("a score").lines[*line].items[*item].at();
                refusals.push(at_line(
                    self,
                    score,
                    at.at,
                    DiagCode::BadLane,
                    &format!(
                        "this line counts in bars, and the tempo moves, so a bar is a place on \
                         the tempo curve; `{}` reads what places it at a time that is no \
                         constant shift of `t`, so it has no song time: place it at `t - d`",
                        lost.path
                    ),
                ));
            }
        }
        refused(refusals)?;
        let key = self.key_hz();
        let scale = values.scale.clone();
        let degree = |at: f64, n: i64| -> Result<f64, String> {
            let _ = at;
            let ratio = scale.as_ref().and_then(|s| s.ratio(n)).ok_or_else(|| {
                "`deg` counts in `variables/scale`, and there is none".to_string()
            })?;
            key.map(|k| k * ratio)
                .ok_or_else(|| "`deg` counts from `variables/key`, a constant note".to_string())
        };
        let sweeping = Sweeping {
            graph: self,
            curve: &curve,
            per_bar: values.tempo.as_ref().map(|t| t.seconds_per_bar()),
            degree: &degree,
        };
        let placed_moves = std::mem::take(&mut placing.moves);
        let lanes = sweeping.sweep(placed_moves)?;
        let marks = std::mem::take(&mut placing.marks);
        let containers = std::mem::take(&mut placing.placed);
        let instances_lost = std::mem::take(&mut placing.instances_lost);
        let mut instances = std::mem::take(&mut placing.instances);
        let moved: BTreeSet<LaneId> = lanes.iter().map(|l| l.id.clone()).collect();
        let mut reach = Reach::new(self, &moved, &values);
        for instance in &mut instances {
            let bound: BTreeSet<&str> = instance.bound.iter().map(String::as_str).collect();
            let mut held = reach.sound(&instance.node, &bound);
            let read = instance.node.strip_prefix(&format!("{VARIABLES}/"));
            if !read.is_some_and(|name| moved.contains(&LaneId::Variable(name.to_string()))) {
                held.extend(reach.of(&instance.node, &bound));
            }
            held.retain(|l| !l.is_variable("scale"));
            held.sort();
            held.dedup();
            instance.lanes = held;
        }
        let mut lanes = lanes;
        if let Some(scale) = lanes.iter().find(|l| l.id.is_variable("scale")) {
            let read: BTreeSet<i64> = instances
                .iter()
                .flat_map(|i| &i.lanes)
                .filter_map(|l| match l {
                    LaneId::Degree(n) => Some(*n),
                    _ => None,
                })
                .collect();
            let derived: Vec<Lane> = read
                .into_iter()
                .filter_map(|n| lanes::degree_lane(scale, n))
                .collect();
            lanes.extend(derived);
        }
        let mut song = Song {
            root: root.to_string(),
            curve,
            tree,
            instances,
            lanes,
            marks,
            instruments: Vec::new(),
            asked: Vec::new(),
            warnings: Vec::new(),
            key,
            scale: values.scale.clone(),
            passed: Vec::new(),
            containers,
        };
        let mut refusals = song.unbound(self);
        for (k, lost) in instances_lost {
            let instance = &song.instances[k];
            if instance.lanes.is_empty() {
                continue;
            }
            let lanes: Vec<String> = instance.lanes.iter().map(LaneId::name).collect();
            let (file, span) = lost.unwrap_or((song.root.clone(), ByteSpan::at(0)));
            refusals.push(Refusal::new(
                file.clone(),
                span,
                DiagCode::BadLane,
                format!(
                    "`@{}` has no song time: `{file}` reads it at a time that is no constant \
                     shift of `t`, and the lanes `{}` move it; a lane reaches an instance at its \
                     song position, so read it at `t - d`",
                    instance.node,
                    lanes.join("`, `")
                ),
            ));
        }
        refused(refusals)?;
        let mut lowered = BTreeSet::new();
        for instance in &song.instances {
            let mut open: Vec<String> = vec![instance.node.clone()];
            while let Some(node) = open.pop() {
                if reach.of(&node, &BTreeSet::new()).is_empty() || !lowered.insert(node.clone()) {
                    continue;
                }
                let mut sites = Vec::new();
                if let Some(defined) = self.source(&node) {
                    for e in std::iter::once(&defined.body)
                        .chain(defined.defaults.iter().map(|(_, v)| v))
                    {
                        place::sites(self, &node, e, &mut sites);
                    }
                }
                open.extend(
                    sites
                        .iter()
                        .filter_map(|e| reach::invoked(self, &node, e).map(|(n, ..)| n)),
                );
            }
        }
        let mut refusals = Vec::new();
        for node in lowered {
            match reach.lowered(&node) {
                Ok(defined) => song.instruments.push((node, defined)),
                Err(refusal) => refusals.push(refusal),
            }
        }
        refused(refusals)?;
        song.asked = song.asked_spans(self)?;
        song.warnings = song.warned(self);
        Ok(song)
    }

    fn through_seconds(&self, m: &Reached, curve: &Curve, what: &str) -> Refusal {
        let at = match &self.score(&m.score).expect("a score").lines[m.line].items[m.item] {
            Item::Move(mv) => mv.at.at,
            _ => 0,
        };
        at_line(
            self,
            &m.score,
            at,
            DiagCode::BadLane,
            &format!(
                "{what} on `{}` is reached through seconds ({}), and while the tempo moves a \
                 move sits on a bar, so a lane compares exact positions; place it from a file \
                 counting in bars",
                m.lane.name(),
                m.chain.text(curve)
            ),
        )
    }

    /// The tonic `variables/key` holds, in hertz, where it is a constant.
    fn key_hz(&self) -> Option<f64> {
        let text = variable_text(self, "key")?;
        let reading = crate::score::Reading::<f64> {
            field: crate::score::Field::Value,
            mark: &|_| None,
            degree: &|_| Err(String::new()),
            own: None,
        };
        crate::score::evaluate(text.trim(), &reading)
            .ok()
            .map(|v| v.amount)
    }
}

impl Song {
    pub fn curve(&self) -> &Curve {
        &self.curve
    }

    pub fn lanes(&self) -> &[Lane] {
        &self.lanes
    }

    pub fn marks(&self) -> &[PlacedMark] {
        &self.marks
    }

    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Each node no score sits under, lowered with the lanes reaching it as parameters.
    pub fn instruments(&self) -> &[(String, Defined)] {
        &self.instruments
    }

    /// Each instance whose span decides a fold, as the node a probe at the root reads:
    /// `@node(t, ...)` with every lane at its onset value.
    pub fn asked(&self) -> Vec<Expr> {
        self.asked.iter().map(|(_, e)| e.clone()).collect()
    }

    pub fn lane(&self, id: &LaneId) -> Option<&Lane> {
        self.lanes.iter().find(|l| &l.id == id)
    }

    /// The moved lanes of `node`'s own parameters: a container equation's are passed on as
    /// moving values, never folded, since no one span bounds what it places.
    pub(crate) fn own_lanes(&self, graph: &Graph, node: &str) -> Vec<LaneId> {
        parameters(graph, node)
            .into_iter()
            .map(|p| LaneId::Sound(node.to_string(), p))
            .filter(|l| self.lane(l).is_some())
            .collect()
    }

    /// The key at second `at`, in hertz.
    pub fn key_at(&self, at: f64) -> Option<f64> {
        match self.lane(&LaneId::Variable("key".to_string())) {
            Some(lane) => lane.at(at).and_then(|h| h.number()),
            None => self.key,
        }
    }

    pub fn scale_at(&self, at: f64) -> Option<Scale> {
        match self.lane(&LaneId::Variable("scale".to_string())) {
            Some(lane) => match lane.at(at) {
                Some(Held::Scale(scale)) => Some(scale),
                _ => None,
            },
            None => self.scale.clone(),
        }
    }

    /// Every parameter a score line's instance reads that nothing binds: not the line, not a
    /// lane set by its onset, not a default.
    fn unbound(&self, graph: &Graph) -> Vec<Refusal> {
        let mut out = Vec::new();
        let mut seen = BTreeSet::new();
        for instance in &self.instances {
            let Some((score, li, ii)) = &instance.line else {
                continue;
            };
            let Item::Event(event) = &graph.score(score).expect("a score").lines[*li].items[*ii]
            else {
                continue;
            };
            let params = parameters(graph, &instance.node);
            let read: Vec<&str> = params.iter().map(String::as_str).collect();
            for bound in &instance.bound {
                if read.contains(&bound.as_str()) {
                    continue;
                }
                let (key, at) = match bound.as_str() {
                    "f0" if event.deg.is_some() => ("deg", event.deg.expect("a deg").1),
                    "len" => ("len", event.len.expect("a len").at),
                    name => (
                        name,
                        event
                            .binds
                            .iter()
                            .find(|b| b.name == name)
                            .map_or(event.node.at, |b| b.key),
                    ),
                };
                if !seen.insert((score.clone(), *li, key.to_string())) {
                    continue;
                }
                let set = match key == bound {
                    true => format!("`{key}`"),
                    false => format!("`{key}`, which sets `{bound}`,"),
                };
                let chord = match key {
                    "chord" => "; a chord is a `deg` list, one event per voice",
                    _ => "",
                };
                out.push(at_line(
                    graph,
                    score,
                    at,
                    DiagCode::BadScore,
                    &format!(
                        "{set} is no parameter `@{}` reads; it reads `{}`{chord}",
                        event.node.name,
                        read.join("`, `")
                    ),
                ));
            }
            let defaults: Vec<&str> = graph
                .source(&instance.node)
                .map(|d| d.defaults.iter().map(|(k, _)| k.as_str()).collect())
                .unwrap_or_default();
            let at = instance.at.map(|p| p.seconds(&self.curve));
            for p in &params {
                if defaults.contains(&p.as_str()) || instance.bound.contains(p) {
                    continue;
                }
                let lane = LaneId::Sound(instance.node.clone(), p.clone());
                let set = self
                    .lane(&lane)
                    .is_some_and(|l| at.is_some_and(|at| l.at(at).is_some()));
                if set || !seen.insert((score.clone(), *li, format!("unbound {p}"))) {
                    continue;
                }
                out.push(at_line(
                    graph,
                    score,
                    event.node.at,
                    DiagCode::BadScore,
                    &format!(
                        "`@{}` reads `{p}`, which nothing binds here: bind it on this line, set \
                         the lane `{}` at or before this line, or give it a default, `{p} = ...`",
                        event.node.name,
                        lane.name()
                    ),
                ));
            }
        }
        out
    }

    /// The probe expressions for every instance a lane may move under.
    fn asked_spans(&self, graph: &Graph) -> Result<Vec<(usize, Expr)>, Vec<Refusal>> {
        let mut out = Vec::new();
        for (k, instance) in self.instances.iter().enumerate() {
            let Some(at) = instance.at else { continue };
            let at = at.seconds(&self.curve);
            let moving = instance
                .lanes
                .iter()
                .filter_map(|l| self.lane(l))
                .any(|l| l.changes_after(at));
            if moving {
                out.push((k, self.probe(graph, k, at)?));
            }
        }
        Ok(out)
    }

    fn warned(&self, graph: &Graph) -> Vec<Warning> {
        let mut out = Vec::new();
        let curve = &self.curve;
        for instance in &self.instances {
            let (Some((score, li, ii)), Some(at)) = (&instance.line, instance.at) else {
                continue;
            };
            let Item::Event(event) = &graph.score(score).expect("a score").lines[*li].items[*ii]
            else {
                continue;
            };
            let start = at.seconds(curve);
            let end = start + self.len_seconds(graph, score, at, event.len.map(|l| l.amount));
            for bound in &instance.bound {
                let lane = LaneId::Sound(instance.node.clone(), bound.clone());
                let Some(l) = self.lane(&lane) else { continue };
                let ramp = l.moves.iter().find(|m| {
                    m.end > m.start && m.end > start && (m.start < end || m.start <= start)
                });
                if let Some(ramp) = ramp {
                    let key_at = event
                        .binds
                        .iter()
                        .find(|b| &b.name == bound)
                        .map_or(event.node.at, |b| b.key);
                    out.push(warn(
                        graph,
                        score,
                        key_at,
                        "line-overrides-lane",
                        &format!(
                            "this line binds `{bound}`, and the lane `{}` ramps during its event \
                             ({}); the line wins over the lane",
                            lane.name(),
                            self.line_of(graph, &ramp.reached)
                        ),
                    ));
                }
            }
        }
        for lane in &self.lanes {
            let reached = lane.id.is_variable("bpm")
                || self
                    .instances
                    .iter()
                    .any(|i| i.lanes.contains(&lane.id) || i.node == lane.id.name())
                || matches!(&lane.id, LaneId::Sound(node, _) if self.containers.contains(node))
                || self.degrees_read(graph, &lane.id);
            if !reached {
                let first = &lane.moves[0].reached;
                let at = match &graph.score(&first.score).expect("a score").lines[first.line].items
                    [first.item]
                {
                    Item::Move(m) => m.lane.at,
                    _ => 0,
                };
                out.push(warn(
                    graph,
                    &first.score,
                    at,
                    "lane-reaches-nothing",
                    &format!(
                        "the lane `{}` has moves, and no instance in the song reads it: its \
                         node is never placed, or every invoker passes the parameter",
                        lane.id.name()
                    ),
                ));
            }
        }
        out
    }

    /// Whether a degree reads `key` or `scale` here: a score line's, or an equation's own.
    fn degrees_read(&self, graph: &Graph, id: &LaneId) -> bool {
        let own = |i: &Instance| i.lanes.iter().any(|l| matches!(l, LaneId::Degree(_)));
        if id.is_variable("scale") && self.instances.iter().any(own) {
            return true;
        }
        (id.is_variable("key") || id.is_variable("scale"))
            && self.instances.iter().any(|i| {
                i.line.as_ref().is_some_and(|(score, li, ii)| {
                    match &graph.score(score).expect("a score").lines[*li].items[*ii] {
                        Item::Event(e) => {
                            e.deg.is_some()
                                || e.binds
                                    .iter()
                                    .any(|b| crate::render_expr(&b.value).contains("deg("))
                        }
                        _ => false,
                    }
                })
            })
    }

    fn line_of(&self, graph: &Graph, r: &Reached) -> String {
        let number = graph.score(&r.score).expect("a score").lines[r.line].number;
        format!("{}{EXTENSION}:{number}", r.score)
    }

    /// The seconds a line's `len` lasts from `at`, on the curve where it counts in bars.
    fn len_seconds(&self, graph: &Graph, score: &str, at: Pos, len: Option<Ratio>) -> f64 {
        let Some(len) = len else { return 0.0 };
        match graph.score(score).expect("a score").unit {
            crate::filename::SpanUnit::Seconds => len.to_f64(),
            crate::filename::SpanUnit::Bars => match at {
                Pos::Bars(b) => self.curve.duration(b, len),
                Pos::Seconds(s) => self
                    .curve
                    .after(s, len)
                    .unwrap_or(len.to_f64() * self.curve.seconds_per_bar()),
            },
        }
    }

    /// Steps 7 and 8: each instance's lanes folded over its span or passed on, and every
    /// container lowered per placement; what is passed is kept for a refusal to name.
    pub fn lowered(&mut self, graph: &Graph, spans: &[Span]) -> Result<Lowered, Vec<Refusal>> {
        let (lowered, passed) = lower::Lowering::new(graph, self, spans).lowered()?;
        self.passed = passed;
        Ok(lowered)
    }

    /// Each instance and lane a lowering passed on as a moving value.
    pub fn passed(&self) -> &[(usize, LaneId)] {
        &self.passed
    }

    /// Where the first move of `lane` under instance `k` is written, its `to`: the file, the
    /// byte and the file's text.
    pub fn moved_under<'g>(
        &self,
        graph: &'g Graph,
        k: usize,
        lane: &LaneId,
    ) -> Option<(String, usize, &'g str)> {
        let at = self.instances[k].at?.seconds(&self.curve);
        let l = self.lane(lane)?;
        let m = l
            .moves
            .iter()
            .find(|m| m.end > at || m.start >= at)
            .or(l.moves.last())?;
        let r = &m.reached;
        let Item::Move(mv) = &graph.score(&r.score)?.lines[r.line].items[r.item] else {
            return None;
        };
        Some((
            format!("{}{EXTENSION}", r.score),
            mv.to.at,
            graph.text(&r.score)?,
        ))
    }
}

fn warn(graph: &Graph, score: &str, at: usize, code: &'static str, message: &str) -> Warning {
    let text = graph.text(score).unwrap_or("");
    let diag = located(text, at, message);
    Warning {
        code,
        file: format!("{score}{EXTENSION}"),
        span: diag.span,
        start: crate::score::line_column(text, at),
        message: diag.message,
    }
}

impl Graph {
    /// Each node's body a song lowered, kept beside its source.
    pub fn apply(&mut self, lowered: Lowered) {
        for (path, defined) in lowered.nodes {
            self.derive(&path, defined);
        }
    }
}
