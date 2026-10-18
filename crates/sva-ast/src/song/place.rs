// Concern: walks every path from a song's root, placing each score line and invocation in song time | Non-concern: what a lane holds (lanes.rs) | IO: (&Graph, root, &Curve) -> placements

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::{Address, Arg, BinOp, Expr, Literal};
use crate::filename::SpanUnit;
use crate::graph::{Graph, VARIABLES, resolve_ref_path};
use crate::refusal::Refusal;
use crate::score::{EXTENSION, Item, Ratio, Score, spelled};
use crate::series::folded;
use crate::song::LaneId;
use crate::song::tempo::Curve;
use crate::vocabulary::is_builtin;

/// Where a line sits in song time: exact bars where every placement above it is in bars,
/// else seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pos {
    Bars(Ratio),
    Seconds(f64),
}

impl Pos {
    pub fn seconds(self, curve: &Curve) -> f64 {
        match self {
            Pos::Bars(b) => curve.seconds(b),
            Pos::Seconds(s) => s,
        }
    }

    pub fn bars(self, curve: &Curve) -> f64 {
        match self {
            Pos::Bars(b) => b.to_f64(),
            Pos::Seconds(s) => curve.bars(s),
        }
    }

    /// Bars where both are exact, seconds otherwise.
    pub fn cmp(self, other: Pos, curve: &Curve) -> Ordering {
        match (self, other) {
            (Pos::Bars(a), Pos::Bars(b)) => a.cmp(&b),
            (a, b) => a.seconds(curve).total_cmp(&b.seconds(curve)),
        }
    }

    pub fn spelled(self, curve: &Curve) -> String {
        match self {
            Pos::Bars(b) => spelled(b.to_f64(), SpanUnit::Bars),
            Pos::Seconds(s) => format!("{}b ({s}s)", curve.bars(s)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Offset {
    Bars(Ratio),
    Seconds(f64),
}

pub(crate) fn shifted(at: Option<Pos>, by: Offset, curve: &Curve) -> Option<Pos> {
    Some(match (at?, by) {
        (Pos::Bars(b), Offset::Bars(a)) => Pos::Bars(b.plus(a)?),
        (Pos::Seconds(s), Offset::Bars(a)) => Pos::Seconds(
            s + curve
                .after(s, a)
                .unwrap_or(a.to_f64() * curve.seconds_per_bar()),
        ),
        (p, Offset::Seconds(0.0)) => p,
        (p, Offset::Seconds(d)) => Pos::Seconds(p.seconds(curve) + d),
    })
}

/// How a line came to be placed, innermost first.
#[derive(Clone, Debug, Default)]
pub struct Chain(Option<Rc<Link>>);

#[derive(Debug)]
struct Link {
    file: String,
    line: Option<usize>,
    item: usize,
    at: Option<Pos>,
    /// Seconds a view's read delays what it places: nonzero only on a read at `t - d` of a
    /// score.
    delay: f64,
    up: Chain,
}

impl Chain {
    /// Placed by line `number`'s item `item` of score `file` at `at`.
    fn by(&self, file: String, (number, item): (usize, usize), at: Option<Pos>) -> Chain {
        Chain(Some(Rc::new(Link {
            file,
            line: Some(number),
            item,
            at,
            delay: 0.0,
            up: self.clone(),
        })))
    }

    /// Read by equation `file` at `at`, a view delaying what it reads by `delay` seconds.
    fn read(&self, file: String, at: Option<Pos>, delay: f64) -> Chain {
        Chain(Some(Rc::new(Link {
            file,
            line: None,
            item: 0,
            at,
            delay,
            up: self.clone(),
        })))
    }

    /// Seconds the views above it delay its sound past its song time.
    pub fn delay(&self) -> f64 {
        let mut sum = 0.0;
        let mut at = &self.0;
        while let Some(link) = at {
            sum += link.delay;
            at = &link.up.0;
        }
        sum
    }

    /// `placed by score/drop.jsonl:3 at 40b, placed by score/song.jsonl:25 at 40b`.
    pub fn text(&self, curve: &Curve) -> String {
        self.links(curve, true)
    }

    /// The same, the innermost line's own position left out: a list's items share it.
    pub fn line_text(&self, curve: &Curve) -> String {
        self.links(curve, false)
    }

    fn links(&self, curve: &Curve, first: bool) -> String {
        let mut out = Vec::new();
        let mut at = &self.0;
        while let Some(link) = at {
            let line = link.line.map_or(String::new(), |l| format!(":{l}"));
            let pos = match (out.is_empty() && !first, link.at) {
                (true, _) => String::new(),
                (false, Some(p)) => format!(" at {}", p.spelled(curve)),
                (false, None) => " with no song time".to_string(),
            };
            let late = match link.delay {
                0.0 => String::new(),
                d => format!(" sounding {d}s later"),
            };
            out.push(format!("placed by {}{line}{pos}{late}", link.file));
            at = &link.up.0;
        }
        out.join(", ")
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// The score lines and items placing it, innermost first: which reads led there is left out.
    fn placement(&self) -> Vec<(&str, usize, usize, Option<Pos>)> {
        let mut out = Vec::new();
        let mut at = &self.0;
        while let Some(link) = at {
            if let Some(line) = link.line {
                out.push((link.file.as_str(), line, link.item, link.at));
            }
            at = &link.up.0;
        }
        out
    }
}

pub(crate) struct Container {
    pub path: String,
    pub at: Option<Pos>,
    pub body: Body,
}

pub(crate) enum Body {
    Score(Vec<Term>),
    Equation(Vec<Site>),
}

/// One placed event of a score: its line and item, and what it plays.
pub(crate) struct Term {
    pub line: usize,
    pub item: usize,
    pub target: Target,
}

pub(crate) enum Target {
    Container(Box<Container>),
    Instance(usize),
}

/// One invocation in an equation's body, in the order [`sites`] visits them.
pub(crate) enum Site {
    Read(Option<f64>, Target),
    Concat(Vec<(f64, f64, Target)>),
}

/// A node no lane lowering places line by line: a sound, or an equation reaching no score.
pub struct Instance {
    pub node: String,
    /// Its song time, where it reads the lanes, key and scale.
    pub at: Option<Pos>,
    pub bound: BTreeSet<String>,
    pub chain: Chain,
    /// The score line placing it, as (score, line index, item index).
    pub line: Option<(String, usize, usize)>,
    pub lanes: Vec<LaneId>,
}

impl Instance {
    /// When it sounds: its song time plus the delays of the views above it.
    pub fn sounds(&self, curve: &Curve) -> Option<Pos> {
        shifted(self.at, Offset::Seconds(self.chain.delay()), curve)
    }
}

#[derive(Clone)]
pub struct Reached {
    pub lane: LaneId,
    pub score: String,
    pub line: usize,
    pub item: usize,
    pub start: Pos,
    pub end: Option<Pos>,
    pub chain: Chain,
}

impl Reached {
    /// The same line placed by the same score lines at the same position, whatever reads led
    /// there.
    fn same(&self, other: &Reached) -> bool {
        (&self.score, self.line, self.item) == (&other.score, other.line, other.item)
            && self.lane == other.lane
            && self.start == other.start
            && self.end == other.end
            && self.chain.placement() == other.chain.placement()
    }
}

#[derive(PartialEq)]
pub struct PlacedMark {
    pub name: String,
    pub at: Pos,
    pub score: String,
    pub line: usize,
}

/// What an equation lost its song time to, for the refusal that needs one.
#[derive(Clone)]
pub(crate) struct Lost {
    pub path: String,
    pub span: ByteSpan,
}

pub(crate) struct Placing<'g> {
    pub graph: &'g Graph,
    pub curve: &'g Curve,
    containers: BTreeMap<String, bool>,
    stack: Vec<String>,
    pub instances: Vec<Instance>,
    pub moves: Vec<Reached>,
    pub marks: Vec<PlacedMark>,
    pub refusals: Vec<Refusal>,
    pub bars_lost: Vec<(String, usize, usize, Lost)>,
    pub instances_lost: Vec<(usize, Option<(String, ByteSpan)>)>,
    /// Each container placed, as a lane on its own parameter reaches it.
    pub placed: BTreeSet<String>,
    lost: Option<Lost>,
}

/// A refusal at a byte of score `path`'s file, led by its line and column.
pub(crate) fn at_line(
    graph: &Graph,
    path: &str,
    at: usize,
    code: DiagCode,
    message: &str,
) -> Refusal {
    let text = graph.text(path).unwrap_or("");
    crate::score::refused(format!("{path}{EXTENSION}"), text, at, code, message)
}

/// Each parameter `path` reads: its default lines, then what its body reads free.
pub fn parameters(graph: &Graph, path: &str) -> Vec<String> {
    let Some(source) = graph.source(path) else {
        return Vec::new();
    };
    let is_node = |name: &str| resolve_ref_path(path, name).is_some_and(|n| graph.defines(&n));
    let mut out: Vec<String> = source.defaults.iter().map(|(k, _)| k.clone()).collect();
    out.extend(crate::parameters::free_parameters(
        &source.body,
        &source.defaults,
        &[],
        is_node,
    ));
    out
}

/// The constant shift `t - d` reads at, in seconds; `None` for any other time.
pub(crate) fn constant_shift(arg: &Expr) -> Option<f64> {
    let t = |e: &Expr| matches!(e, Expr::Var(n) if n == "t");
    match arg {
        e if t(e) => Some(0.0),
        Expr::Bin(BinOp::Sub, l, r) if t(l) => folded(r, &|_| None),
        Expr::Bin(BinOp::Add, l, r) if t(l) => folded(r, &|_| None).map(|d| -d),
        _ => None,
    }
}

/// Every invocation of a node in `e`, in written order: a read, a bareword call to a node,
/// and a `concat` or `repeat` as one site whose arguments are its parts.
pub(crate) fn sites<'e>(graph: &Graph, at: &str, e: &'e Expr, out: &mut Vec<&'e Expr>) {
    match e {
        Expr::Call { name, .. } if name == "concat" || name == "repeat" => {
            out.push(e);
            return;
        }
        Expr::Ref { .. } => out.push(e),
        Expr::Call { name, .. } if is_node_call(graph, at, name) => out.push(e),
        _ => {}
    }
    for c in crate::expr::children(e, crate::expr::Binds::Substitute) {
        sites(graph, at, c, out);
    }
}

pub(crate) fn is_node_call(graph: &Graph, at: &str, name: &str) -> bool {
    !is_builtin(name) && resolve_ref_path(at, name).is_some_and(|p| graph.defines(&p))
}

impl<'g> Placing<'g> {
    pub fn new(graph: &'g Graph, curve: &'g Curve) -> Placing<'g> {
        Placing {
            graph,
            curve,
            containers: BTreeMap::new(),
            stack: Vec::new(),
            instances: Vec::new(),
            moves: Vec::new(),
            marks: Vec::new(),
            refusals: Vec::new(),
            bars_lost: Vec::new(),
            instances_lost: Vec::new(),
            placed: BTreeSet::new(),
            lost: None,
        }
    }

    /// A score, or an equation that reaches one or arranges with `concat` or `repeat`: what is
    /// placed line by line, its lowering depending on where.
    pub fn is_container(&mut self, path: &str) -> bool {
        if let Some(known) = self.containers.get(path) {
            return *known;
        }
        self.containers.insert(path.to_string(), false);
        let found = match (
            self.graph.score(path),
            self.graph.source(path).map(|d| &d.body),
        ) {
            (Some(_), _) => true,
            (None, Some(body)) => {
                let mut held = Vec::new();
                sites(self.graph, path, body, &mut held);
                held.iter().any(|site| match site {
                    Expr::Call { name, .. } if name == "concat" || name == "repeat" => true,
                    Expr::Ref { path: to, .. } | Expr::Call { name: to, .. } => {
                        resolve_ref_path(path, to).is_some_and(|to| self.is_container(&to))
                    }
                    _ => false,
                })
            }
            (None, None) => false,
        };
        self.containers.insert(path.to_string(), found);
        found
    }

    fn refuse(&mut self, refusal: Refusal) {
        self.refusals.push(refusal);
    }

    /// `at` moved by the time written at byte `byte` of score `path`, refused there where
    /// the exact bar it lands on holds more than a fraction of 62 bits can.
    fn exact(&mut self, path: &str, byte: usize, at: Option<Pos>, by: Offset) -> Option<Pos> {
        let moved = shifted(at, by, self.curve);
        if let (Some(from), None) = (at, moved) {
            let message = format!(
                "this time, counted from {} where its score is placed, lands on a bar no \
                 fraction of 62-bit integers holds; write it with a smaller denominator",
                from.spelled(self.curve)
            );
            self.refuse(at_line(
                self.graph,
                path,
                byte,
                DiagCode::BadScore,
                &message,
            ));
        }
        moved
    }

    pub fn place(
        &mut self,
        path: &str,
        at: Option<Pos>,
        crop: Option<Pos>,
        chain: Chain,
    ) -> Container {
        self.placed.insert(path.to_string());
        let body = match self.graph.score(path) {
            Some(score) => Body::Score(self.score(path, score, at, crop, &chain)),
            None => Body::Equation(self.equation(path, at, &chain)),
        };
        Container {
            path: path.to_string(),
            at,
            body,
        }
    }

    fn cycle(&mut self, path: &str, refusal: impl FnOnce(String) -> Refusal) -> bool {
        if !self.stack.iter().any(|held| held == path) {
            return false;
        }
        let mut cycle: Vec<String> = self.stack.clone();
        cycle.push(path.to_string());
        let at = cycle
            .iter()
            .position(|held| held == path)
            .expect("on the stack");
        self.refuse(refusal(cycle[at..].join(" places ")));
        true
    }

    fn score(
        &mut self,
        path: &str,
        score: &'g Score,
        at: Option<Pos>,
        crop: Option<Pos>,
        chain: &Chain,
    ) -> Vec<Term> {
        self.stack.push(path.to_string());
        let mut terms = Vec::new();
        let curve = self.curve;
        if at.is_none()
            && score.unit == SpanUnit::Bars
            && curve.moves()
            && let Some(lost) = self.lost.clone()
            && !score.lines.is_empty()
        {
            self.bars_lost.push((path.to_string(), 0, 0, lost));
        }
        for (li, line) in score.lines.iter().enumerate() {
            for (ii, item) in line.items.iter().enumerate() {
                let pos = match score.unit {
                    SpanUnit::Bars => {
                        self.exact(path, item.at().at, at, Offset::Bars(item.at().amount))
                    }
                    SpanUnit::Seconds => {
                        at.map(|p| Pos::Seconds(p.seconds(curve) + item.at().amount.to_f64()))
                    }
                };
                if let (Some(pos), Some(crop)) = (pos, crop)
                    && pos.cmp(crop, curve) != Ordering::Less
                {
                    continue;
                }
                match item {
                    Item::Comment(_) => {}
                    Item::Mark(mark) => {
                        if let Some(pos) = pos {
                            let placed = PlacedMark {
                                name: mark.name.clone(),
                                at: pos,
                                score: path.to_string(),
                                line: line.number,
                            };
                            if !self.marks.contains(&placed) {
                                self.marks.push(placed);
                            }
                        }
                    }
                    Item::Move(m) => self.moved(path, m, (li, ii), pos, crop, chain),
                    Item::Event(event) => {
                        let placed = (path, score, (li, ii), pos);
                        if let Some(target) = self.event(placed, event, crop, chain) {
                            terms.push(Term {
                                line: li,
                                item: ii,
                                target,
                            });
                        }
                    }
                }
            }
        }
        self.stack.pop();
        terms
    }

    /// What one event plays: a score or container placed under any `len` crop, or an instance.
    fn event(
        &mut self,
        (path, score, (li, ii), pos): (&str, &'g Score, (usize, usize), Option<Pos>),
        event: &crate::score::Event,
        crop: Option<Pos>,
        chain: &Chain,
    ) -> Option<Target> {
        let curve = self.curve;
        let line = &score.lines[li];
        let target = resolve_ref_path(path, &event.node.name)?;
        if !self.graph.defines(&target) {
            return None;
        }
        let container = self.is_container(&target);
        if let Some(len) = event.len
            && ii == 0
            && let Some(other) = line.items.iter().find_map(|other| match other {
                Item::Event(o) => resolve_ref_path(path, &o.node.name)
                    .filter(|p| self.graph.defines(p) && self.is_container(p) != container),
                _ => None,
            })
        {
            self.refuse(at_line(
                self.graph,
                path,
                len.at,
                DiagCode::BadScore,
                &format!(
                    "`len` crops a placed score and binds a sound's `len`, and this line's \
                     `node` list holds both (`@{target}` and `@{other}`); split the line, one \
                     for the scores, one for the sounds"
                ),
            ));
            return None;
        }
        let here = chain.by(format!("{path}{EXTENSION}"), (line.number, ii), pos);
        if !container {
            let mut bound: BTreeSet<String> = event.binds.iter().map(|b| b.name.clone()).collect();
            if event.deg.is_some() {
                bound.insert("f0".to_string());
            }
            if event.len.is_some() {
                bound.insert("len".to_string());
            }
            return Some(Target::Instance(self.instance(Instance {
                node: target,
                at: pos,
                bound,
                chain: here,
                line: Some((path.to_string(), li, ii)),
                lanes: Vec::new(),
            })));
        }
        let bad = event
            .deg
            .map(|(_, at)| ("deg", at))
            .or_else(|| event.binds.first().map(|b| (b.name.as_str(), b.key)));
        if let Some((key, byte)) = bad {
            self.refuse(at_line(
                self.graph,
                path,
                byte,
                DiagCode::BadScore,
                &format!(
                    "`{key}` binds a parameter, and `@{target}` places a score, which reads none; \
                     a line placing a score takes `len` alone, which crops it"
                ),
            ));
            return None;
        }
        let graph = self.graph;
        let refused = |cycle: String| {
            let message = format!("a score places itself: {cycle}");
            at_line(graph, path, event.node.at, DiagCode::BadScore, &message)
        };
        if self.cycle(&target, refused) {
            return None;
        }
        let inner = event.len.and_then(|len| {
            let by = match score.unit {
                SpanUnit::Bars => Offset::Bars(len.amount),
                SpanUnit::Seconds => Offset::Seconds(len.amount.to_f64()),
            };
            self.exact(path, len.at, pos, by)
        });
        let cut = match (crop, inner) {
            (Some(a), Some(b)) => Some(match a.cmp(b, curve) {
                Ordering::Less => a,
                _ => b,
            }),
            (a, b) => a.or(b),
        };
        Some(Target::Container(Box::new(
            self.place(&target, pos, cut, here),
        )))
    }

    fn instance(&mut self, instance: Instance) -> usize {
        if instance.at.is_none() {
            let lost = self.lost.clone().map(|l| (l.path, l.span));
            self.instances_lost.push((self.instances.len(), lost));
        }
        self.instances.push(instance);
        self.instances.len() - 1
    }

    fn moved(
        &mut self,
        path: &str,
        m: &crate::score::Move,
        (li, ii): (usize, usize),
        pos: Option<Pos>,
        crop: Option<Pos>,
        chain: &Chain,
    ) {
        let graph = self.graph;
        let refuse =
            |at: usize, code: DiagCode, message: String| at_line(graph, path, at, code, &message);
        let lane = match &m.node {
            Some(node) => {
                let Some(target) = resolve_ref_path(path, &node.name) else {
                    return;
                };
                if graph.score(&target).is_some() {
                    self.refuse(refuse(
                        node.at,
                        DiagCode::BadScore,
                        format!(
                            "a move's node is a sound or an equation, and `@{}` is a score, \
                             which reads no parameter; move a lane of a node it plays",
                            node.name
                        ),
                    ));
                    return;
                }
                let params = parameters(graph, &target);
                if !params.contains(&m.lane.name) {
                    self.refuse(refuse(
                        m.lane.at,
                        DiagCode::BadScore,
                        format!(
                            "`@{}` reads no `{}`; its parameters are `{}`",
                            node.name,
                            m.lane.name,
                            params.join("`, `")
                        ),
                    ));
                    return;
                }
                LaneId::Sound(target, m.lane.name.clone())
            }
            None => {
                let variable = format!("{VARIABLES}/{}", m.lane.name);
                if !graph.defines(&variable) {
                    let mut held: Vec<&str> = graph
                        .paths()
                        .filter_map(|p| p.strip_prefix(&format!("{VARIABLES}/")))
                        .collect();
                    held.sort_unstable();
                    self.refuse(refuse(
                        m.lane.at,
                        DiagCode::BadScore,
                        format!(
                            "a move with no `node` moves a variable, and there is no \
                             `{variable}`; the variables there are {}",
                            match held.is_empty() {
                                true => "none".to_string(),
                                false => format!("`{}`", held.join("`, `")),
                            }
                        ),
                    ));
                    return;
                }
                LaneId::Variable(m.lane.name.clone())
            }
        };
        let Some(start) = pos else {
            let lost = self.lost.clone();
            let why = match lost {
                Some(l) => format!(
                    "`{}` reads what places it at a time that is no constant shift of `t`",
                    l.path
                ),
                None => "nothing places it in song time".to_string(),
            };
            self.refuse(refuse(
                m.at.at,
                DiagCode::BadLane,
                format!(
                    "this move on `{}` has no song time: {why}; a lane moves in song time, so \
                     place this score at a constant shift, as `@x(t - 8b)`, or from a score",
                    lane.name()
                ),
            ));
            return;
        };
        let end = m.ramp.and_then(|ramp| {
            let by = match graph.score(path).expect("a score").unit {
                SpanUnit::Bars => Offset::Bars(ramp.len.amount),
                SpanUnit::Seconds => Offset::Seconds(ramp.len.amount.to_f64()),
            };
            self.exact(path, ramp.len.at, pos, by)
        });
        if let (Some(end), Some(crop)) = (end, crop)
            && end.cmp(crop, self.curve) == Ordering::Greater
        {
            self.refuse(refuse(
                m.at.at,
                DiagCode::BadLane,
                format!(
                    "a `len` crop at {} cuts this ramp on `{}`, from {} to {}, leaving its lane \
                     mid-way; end the ramp by the crop or lengthen the crop ({})",
                    crop.spelled(self.curve),
                    lane.name(),
                    start.spelled(self.curve),
                    end.spelled(self.curve),
                    chain.text(self.curve)
                ),
            ));
            return;
        }
        let file = format!("{path}{EXTENSION}");
        let line_number = graph.score(path).expect("a score").lines[li].number;
        let reached = Reached {
            lane,
            score: path.to_string(),
            line: li,
            item: ii,
            start,
            end,
            chain: chain.by(file, (line_number, ii), Some(start)),
        };
        // Lanes are the song's: one placement read by several paths is one move.
        if !self.moves.iter().any(|m| m.same(&reached)) {
            self.moves.push(reached);
        }
    }

    fn equation(&mut self, path: &str, at: Option<Pos>, chain: &Chain) -> Vec<Site> {
        let Some(body) = self.graph.source(path).map(|d| &d.body) else {
            return Vec::new();
        };
        self.stack.push(path.to_string());
        let mut out = Vec::new();
        self.sites_of(path, body, at, chain, &mut out);
        self.stack.pop();
        out
    }

    /// Each site under `e`, read at `at`.
    fn sites_of(
        &mut self,
        path: &str,
        e: &'g Expr,
        at: Option<Pos>,
        chain: &Chain,
        out: &mut Vec<Site>,
    ) {
        match e {
            Expr::Call { name, args, span } if name == "concat" || name == "repeat" => {
                out.push(self.concat(path, (name, args, *span), at, chain));
            }
            Expr::Ref {
                path: to,
                arg,
                binds,
                span,
                address,
            } => {
                let shift = (*address == Address::Time)
                    .then(|| constant_shift(arg))
                    .flatten();
                // A score read at `t - d` is a view: its lines keep song time and sound d later.
                let view = resolve_ref_path(path, to).is_some_and(|n| self.is_container(&n));
                let (shift, delay) = match (shift, view) {
                    (Some(d), true) => (Some(0.0), d),
                    (shift, _) => (shift, 0.0),
                };
                let placed = match shift {
                    Some(d) => shifted(at, Offset::Seconds(d), self.curve),
                    None => None,
                };
                let lost = self.lost.clone();
                if shift.is_none() && at.is_some() {
                    self.lost = Some(Lost {
                        path: path.to_string(),
                        span: *span,
                    });
                }
                let names: BTreeSet<String> = binds.iter().map(|(k, _)| k.clone()).collect();
                let target = self.target(path, to, placed, names, chain, delay);
                out.push(Site::Read(shift, target));
                self.sites_of(path, arg, None, chain, out);
                for (_, value) in binds {
                    self.sites_of(path, value, placed, chain, out);
                }
                self.lost = lost;
            }
            Expr::Call { name, span, args } if is_node_call(self.graph, path, name) => {
                let target = self.target(path, name, at, BTreeSet::new(), chain, 0.0);
                out.push(Site::Read(Some(0.0), target));
                let _ = span;
                for a in args {
                    let (Arg::Pos(x) | Arg::Named(_, x)) = a;
                    self.sites_of(path, x, at, chain, out);
                }
            }
            _ => {
                for c in crate::expr::children(e, crate::expr::Binds::Substitute) {
                    self.sites_of(path, c, at, chain, out);
                }
            }
        }
    }

    fn target(
        &mut self,
        from: &str,
        to: &str,
        at: Option<Pos>,
        bound: BTreeSet<String>,
        chain: &Chain,
        delay: f64,
    ) -> Target {
        let Some(node) = resolve_ref_path(from, to) else {
            return Target::Instance(self.instance(Instance {
                node: to.to_string(),
                at,
                bound,
                chain: chain.clone(),
                line: None,
                lanes: Vec::new(),
            }));
        };
        let here = chain.read(from.to_string(), at, delay);
        if self.is_container(&node) {
            let graph = self.graph;
            if !self.cycle(&node, |cycle| {
                Refusal::new(
                    from,
                    ByteSpan::at(0),
                    DiagCode::BadScore,
                    format!("a score places itself: {cycle}"),
                )
            }) {
                let _ = graph;
                return Target::Container(Box::new(self.place(&node, at, None, here)));
            }
        }
        Target::Instance(self.instance(Instance {
            node,
            at,
            bound,
            chain: here,
            line: None,
            lanes: Vec::new(),
        }))
    }

    /// `concat`'s parts, each placed after the spans before it on the curve:
    /// it: part k is `crop(@x(t - dₖ), dₖ, dₖ + Lₖ)`; `repeat(@x, n)` is n parts of `@x`.
    fn concat(
        &mut self,
        path: &str,
        (name, args, span): (&str, &'g [Arg], ByteSpan),
        at: Option<Pos>,
        chain: &Chain,
    ) -> Site {
        let refuse = |why: String| Refusal::new(path, span, DiagCode::BadArrangementArg, why);
        let arg = |k: usize| args.get(k).map(|(Arg::Pos(x) | Arg::Named(_, x))| x);
        let parts: Vec<&Expr> = match name {
            "repeat" => {
                let (Some(x), Some(n), 2) = (arg(0), arg(1), args.len()) else {
                    self.refuse(refuse(
                        "`repeat` takes exactly two arguments: a ref and a whole copy count"
                            .to_string(),
                    ));
                    return Site::Concat(Vec::new());
                };
                match n {
                    Expr::Lit(Literal::Num(n)) if *n >= 0.0 && n.fract() == 0.0 => {
                        std::iter::repeat_n(x, *n as usize).collect()
                    }
                    _ => {
                        self.refuse(refuse(
                            "`repeat`'s second argument is a whole-number literal, 0 or more"
                                .to_string(),
                        ));
                        return Site::Concat(Vec::new());
                    }
                }
            }
            _ => (0..args.len()).filter_map(arg).collect(),
        };
        let mut out = Vec::with_capacity(parts.len());
        let (mut offset, mut bars) = (0.0f64, Some(Ratio::ZERO));
        for part in parts {
            let Expr::Ref {
                path: to,
                arg,
                binds,
                address,
                span: at_ref,
            } = part
            else {
                self.refuse(refuse(
                    "`concat`'s arguments are each a plain `@path` ref to a file with a declared \
                     span"
                        .to_string(),
                ));
                return Site::Concat(Vec::new());
            };
            if **arg != Expr::Var("t".to_string()) || *address != Address::Time {
                self.refuse(Refusal::new(
                    path,
                    *at_ref,
                    DiagCode::BadArrangementArg,
                    format!(
                        "`@{to}` inside `concat`/`repeat` must be a bare ref (no custom time \
                         argument); a shift is added automatically"
                    ),
                ));
                return Site::Concat(Vec::new());
            }
            let Some(node) = resolve_ref_path(path, to) else {
                self.refuse(Refusal::new(
                    path,
                    *at_ref,
                    DiagCode::RefAboveRoot,
                    format!("`@{to}` in `{path}` walks above the composition root"),
                ));
                return Site::Concat(Vec::new());
            };
            let Some(span) = self.graph.span(&node) else {
                self.refuse(Refusal::new(
                    path,
                    *at_ref,
                    DiagCode::BadArrangementArg,
                    format!(
                        "`@{to}` (resolved to `{node}`) has no declared span; `concat` needs one \
                         to know its duration"
                    ),
                ));
                return Site::Concat(Vec::new());
            };
            let start_at = match bars {
                Some(b) => shifted(at, Offset::Bars(b), self.curve),
                None => shifted(at, Offset::Seconds(offset), self.curve),
            };
            let length = match span.unit {
                SpanUnit::Seconds => span.amount,
                SpanUnit::Bars => {
                    let len = Ratio::of_f64(span.amount).unwrap_or(Ratio::ZERO);
                    let home = len.to_f64() * self.curve.seconds_per_bar();
                    match start_at {
                        Some(Pos::Bars(from)) => self.curve.duration(from, len),
                        Some(Pos::Seconds(s)) => self.curve.after(s, len).unwrap_or(home),
                        None => home,
                    }
                }
            };
            bars = match (bars, span.unit) {
                (Some(b), SpanUnit::Bars) => Ratio::of_f64(span.amount).and_then(|l| b.plus(l)),
                _ => None,
            };
            let names: BTreeSet<String> = binds.iter().map(|(k, _)| k.clone()).collect();
            let target = self.target(path, to, start_at, names, chain, 0.0);
            out.push((offset, offset + length, target));
            offset += length;
        }
        Site::Concat(out)
    }
}
