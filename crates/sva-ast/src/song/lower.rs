// Concern: lowers a placed song to equations: each container per placement, lanes folded or passed, each lane's closed form | Non-concern: placing (place.rs) | IO: (&Song, spans) -> nodes or Refusals

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::{Address, Arg, BinOp, Binds, DEGREE, Expr, Literal, map_children};
use crate::filename::SpanUnit;
use crate::graph::{Defined, Graph};
use crate::refusal::Refusal;
use crate::reserved::{Values, key_ref, tonic_times};
use crate::score::{Event, Item, Ratio, Shape};
use crate::series::folded;
use crate::song::lanes::{Held, Hold, Home, Lane};
use crate::song::place::{Body, Container, Pos, Site, Target, at_line, sites};
use crate::song::reach::{retimed, with_binds};
use crate::song::{LaneId, Lowered, Song, Span};

/// Each instance and lane a lowering passed on as a moving value.
pub(crate) type Passed = Vec<(usize, LaneId)>;

#[derive(Clone, Debug)]
enum Fold {
    Home,
    Value(f64),
    Pass,
}

pub(crate) struct Lowering<'a> {
    graph: &'a Graph,
    song: &'a Song,
    spans: BTreeMap<usize, Span>,
    folds: Vec<Vec<(LaneId, Fold)>>,
    variants: BTreeMap<String, Vec<(Defined, String)>>,
    /// Each placement's lowered node and lane parameters, by the placement every read of it
    /// shares.
    lowered: HashMap<*const Container, (String, BTreeSet<LaneId>)>,
    out: Vec<(String, Defined)>,
    values: Option<Values>,
}

fn num(x: f64) -> Expr {
    Expr::Lit(Literal::Num(x))
}

fn t() -> Expr {
    Expr::Var("t".to_string())
}

/// `t - d`: `t` at no shift, `t + d` before zero.
fn shift(d: Expr, negative: bool, zero: bool) -> Expr {
    match (zero, negative) {
        (true, _) => t(),
        (false, true) => Expr::Bin(BinOp::Add, Box::new(t()), Box::new(d)),
        (false, false) => Expr::Bin(BinOp::Sub, Box::new(t()), Box::new(d)),
    }
}

fn balanced(mut terms: Vec<Expr>) -> Expr {
    match terms.len() {
        0 => num(0.0),
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

/// A child node's ref as its parent writes it: its variant's suffix after the written path.
fn written(path: &str, source: &str, variant: &str) -> String {
    format!("{path}{}", &variant[source.len()..])
}

impl Song {
    /// `@node(t, ...)` for instance `k`, every lane at its value at second `at`.
    pub(crate) fn probe(&self, graph: &Graph, k: usize, at: f64) -> Result<Expr, Vec<Refusal>> {
        let instance = &self.instances[k];
        let mut binds = self.line_binds(graph, k)?;
        for lane in &instance.lanes {
            let l = self.lane(lane).expect("a moved lane");
            let own = matches!(lane, LaneId::Sound(node, _) if *node == instance.node);
            match (l.at_home(at), own, l.at(at).and_then(|h| h.number())) {
                (true, true, _) => {}
                (_, _, Some(v)) => binds.push((bound_name(lane, own), num(v))),
                (_, _, None) => {}
            }
        }
        Ok(Expr::Ref {
            path: instance.node.clone(),
            arg: Box::new(t()),
            binds,
            address: Address::Time,
            span: ByteSpan::at(0),
        })
    }

    /// The binds a score line writes for instance `k`: its values in written order, `len` on
    /// the curve, and `deg` as `f0`, read at the line's position.
    fn line_binds(&self, graph: &Graph, k: usize) -> Result<Vec<(String, Expr)>, Vec<Refusal>> {
        let instance = &self.instances[k];
        let Some((score, li, ii)) = &instance.line else {
            return Ok(Vec::new());
        };
        let file = graph.score(score).expect("a score");
        let Item::Event(event) = &file.lines[*li].items[*ii] else {
            unreachable!("an instance's line is an event")
        };
        let at = instance.at;
        let mut out: Vec<(usize, String, Expr)> = Vec::new();
        for bind in &event.binds {
            out.push((
                bind.at,
                bind.name.clone(),
                self.degrees(graph, score, &bind.value, at)?,
            ));
        }
        if let Some(len) = event.len {
            let value = match file.unit {
                SpanUnit::Seconds => num(len.amount.to_f64()),
                SpanUnit::Bars => self.bars(at, len.amount).0,
            };
            out.push((len.at, "len".to_string(), value));
        }
        out.sort_by_key(|(at, ..)| *at);
        let mut binds: Vec<(String, Expr)> = out.into_iter().map(|(_, k, v)| (k, v)).collect();
        if let Some((n, byte)) = event.deg {
            let call = Expr::Call {
                name: DEGREE.to_string(),
                args: vec![Arg::Pos(num(n as f64))],
                span: ByteSpan::new(byte, byte + 1),
            };
            binds.push(("f0".to_string(), self.degrees(graph, score, &call, at)?));
        }
        Ok(binds)
    }

    /// `len` bars from `at` in seconds on the curve, as a bar literal while the tempo holds its
    /// home, which the reserved pass writes as the home product.
    fn bars(&self, at: Option<Pos>, len: Ratio) -> (Expr, f64) {
        let curve = &self.curve;
        let moving = match at {
            Some(Pos::Bars(b)) => {
                let to = b.plus(len).unwrap_or(b);
                (!curve.steady_until(b.max(to))).then(|| curve.duration(b, len))
            }
            Some(Pos::Seconds(s)) => curve.after(s, len),
            None => None,
        };
        match moving {
            Some(d) => (num(d), d),
            None => (
                Expr::Lit(Literal::Bars(len.to_f64())),
                len.to_f64() * curve.seconds_per_bar(),
            ),
        }
    }

    /// `e` with each `deg(n)` read at the line's position where the key or scale moves: a
    /// number, or the tonic's read times the scale's ratio there.
    fn degrees(
        &self,
        graph: &Graph,
        score: &str,
        e: &Expr,
        at: Option<Pos>,
    ) -> Result<Expr, Vec<Refusal>> {
        let key = LaneId::Variable("key".to_string());
        let scale = LaneId::Variable("scale".to_string());
        let (key_moves, scale_moves) = (self.lane(&key).is_some(), self.lane(&scale).is_some());
        if !key_moves && !scale_moves {
            return Ok(e.clone());
        }
        let seconds = at.map(|p| p.seconds(&self.curve));
        let refuse = |span: ByteSpan, why: String| {
            vec![at_line(graph, score, span.start, DiagCode::BadScale, &why)]
        };
        let lowered = self.substitute(e, &mut |name_span, n| {
            let Some(s) = seconds else {
                return Err(refuse(
                    name_span,
                    "this `deg` reads the key where it sits, and the key moves, so it needs a \
                     song time; place its score at a constant shift"
                        .to_string(),
                ));
            };
            let ratio = self.scale_at(s).and_then(|sc| sc.ratio(n)).ok_or_else(|| {
                refuse(
                    name_span,
                    "`deg` counts in `variables/scale`, and none holds here".to_string(),
                )
            })?;
            match key_moves {
                true => {
                    let k = self.key_at(s).ok_or_else(|| {
                        refuse(
                            name_span,
                            "`deg` counts from the key, and the key holds no note here".to_string(),
                        )
                    })?;
                    Ok(num(k * ratio))
                }
                false => Ok(tonic_times(&key_ref(score), ratio, name_span)),
            }
        })?;
        Ok(match folded(&lowered, &|_| None) {
            Some(n) => num(n),
            None => lowered,
        })
    }

    fn substitute(
        &self,
        e: &Expr,
        f: &mut dyn FnMut(ByteSpan, i64) -> Result<Expr, Vec<Refusal>>,
    ) -> Result<Expr, Vec<Refusal>> {
        if let Expr::Call { name, args, span } = e
            && name == DEGREE
            && let Ok(n) = crate::reserved::degree(args)
        {
            return f(*span, n);
        }
        map_children(e, Binds::Substitute, |c| self.substitute(c, f))
    }
}

/// The parameter a lane binds on an instance: its own name for the node's own parameter, the
/// lane's for a lane parameter.
fn bound_name(lane: &LaneId, own: bool) -> String {
    match (lane, own) {
        (LaneId::Sound(_, param), true) => param.clone(),
        _ => lane.name(),
    }
}

impl<'a> Lowering<'a> {
    pub fn new(graph: &'a Graph, song: &'a Song, spans: &[Span]) -> Lowering<'a> {
        let spans = song
            .asked
            .iter()
            .zip(spans)
            .map(|((k, _), span)| (*k, *span))
            .collect();
        Lowering {
            graph,
            song,
            spans,
            folds: Vec::new(),
            variants: BTreeMap::new(),
            lowered: HashMap::new(),
            out: Vec::new(),
            values: None,
        }
    }

    pub fn lowered(mut self) -> Result<(Lowered, Passed), Vec<Refusal>> {
        self.values = Some(self.graph.values(None)?);
        self.folds = self.folds()?;
        self.out.extend(self.song.instruments.iter().cloned());
        let root = &self.song.tree;
        let (body, _) = self.equation(root, true)?;
        self.out.push((self.song.root.clone(), body));
        for lane in &self.song.lanes {
            if let Some(defined) = closed_form(lane) {
                self.out.push((lane.id.name(), defined));
            }
        }
        let passed = self
            .folds
            .iter()
            .enumerate()
            .flat_map(|(k, folds)| {
                folds
                    .iter()
                    .filter(|(_, fold)| matches!(fold, Fold::Pass))
                    .map(move |(lane, _)| (k, lane.clone()))
            })
            .collect();
        Ok((Lowered { nodes: self.out }, passed))
    }

    fn folds(&self) -> Result<Vec<Vec<(LaneId, Fold)>>, Vec<Refusal>> {
        let song = self.song;
        let mut out = Vec::with_capacity(song.instances.len());
        let mut refusals = Vec::new();
        for (k, instance) in song.instances.iter().enumerate() {
            let mut held = Vec::with_capacity(instance.lanes.len());
            let Some(at) = instance.at.map(|p| p.seconds(&song.curve)) else {
                out.push(held);
                continue;
            };
            let (a, b) = match self.spans.get(&k) {
                Some(span) => (
                    span.start.map_or(f64::NEG_INFINITY, |s| at + s),
                    span.end.map(|e| at + e),
                ),
                None => (at, None),
            };
            for lane in &instance.lanes {
                let l = song.lane(lane).expect("a moved lane");
                let own = matches!(lane, LaneId::Sound(node, _) if *node == instance.node);
                let fold = match l.constant_over(a, b) {
                    Some(Hold::Home) if own => Fold::Home,
                    Some(Hold::Home) => match &l.home {
                        Home::Number(v, _) => Fold::Value(*v),
                        _ => {
                            refusals.push(self.home_refusal(l, instance));
                            Fold::Pass
                        }
                    },
                    Some(Hold::Value(Held::Number(v))) => Fold::Value(v),
                    Some(Hold::Value(Held::Scale(_))) => Fold::Pass,
                    None => {
                        if !matches!(l.home, Home::Number(..)) && l.at_home(at) {
                            refusals.push(self.home_refusal(l, instance));
                        }
                        Fold::Pass
                    }
                };
                held.push((lane.clone(), fold));
            }
            out.push(held);
        }
        refusals.dedup_by(|a, b| a.reason == b.reason);
        match refusals.is_empty() {
            true => Ok(out),
            false => Err(refusals),
        }
    }

    fn home_refusal(&self, lane: &Lane, instance: &crate::song::Instance) -> Refusal {
        let first = &lane.moves[0].reached;
        let at = match &self.graph.score(&first.score).expect("a score").lines[first.line].items
            [first.item]
        {
            Item::Move(m) => m.at.at,
            _ => 0,
        };
        at_line(
            self.graph,
            &first.score,
            at,
            DiagCode::BadLane,
            &format!(
                "this set on `{}` lands inside the span of `@{}` ({}), which reads the lane's \
                 home, `{}`, no constant: the note would need its own default before the set and \
                 the set's value after it; set the lane before the note",
                lane.id.name(),
                instance.node,
                instance.chain.text(&self.song.curve),
                match &lane.home {
                    Home::Default(text) => text.clone(),
                    _ => "unset".to_string(),
                }
            ),
        )
    }

    /// The lane read an invoker passes at shift `d`: the root reads the lane itself, any other
    /// container its own lane parameter.
    fn read(&self, root: bool, lane: &LaneId, d: f64) -> Expr {
        match root {
            true => Expr::Ref {
                path: format!(
                    "{}{}",
                    "../".repeat(self.song.root.matches('/').count()),
                    lane.name()
                ),
                arg: Box::new(match d == 0.0 {
                    true => t(),
                    false => Expr::Bin(BinOp::Add, Box::new(t()), Box::new(num(d))),
                }),
                binds: Vec::new(),
                address: Address::Time,
                span: ByteSpan::at(0),
            },
            false => retimed(&lane.name(), d),
        }
    }

    /// Instance `k`'s lane binds at shift `d`, each lane it passes on added to `params`.
    fn lane_binds(
        &self,
        k: usize,
        root: bool,
        d: f64,
        params: &mut BTreeSet<LaneId>,
    ) -> Vec<(String, Expr)> {
        let instance = &self.song.instances[k];
        let mut out = Vec::new();
        for (lane, fold) in &self.folds[k] {
            let own = matches!(lane, LaneId::Sound(node, _) if *node == instance.node);
            match fold {
                Fold::Home => {}
                Fold::Value(v) => out.push((bound_name(lane, own), num(*v))),
                Fold::Pass => {
                    if !root {
                        params.insert(lane.clone());
                    }
                    out.push((bound_name(lane, own), self.read(root, lane, d)));
                }
            }
        }
        out
    }

    /// A container's lowered node, and the lanes its body reads as parameters: lowered once
    /// per placement, however many reads share it.
    fn container(
        &mut self,
        c: &Arc<Container>,
    ) -> Result<(String, BTreeSet<LaneId>), Vec<Refusal>> {
        if let Some(held) = self.lowered.get(&Arc::as_ptr(c)) {
            return Ok(held.clone());
        }
        let (defined, params) = match &c.body {
            Body::Score(_) => self.score(c)?,
            Body::Equation(_) => self.equation(c, false)?,
        };
        let values = self.values.as_ref().expect("read before lowering");
        let defined = self.graph.lowered(&c.path, &defined, values)?;
        let held = (self.variant(&c.path, defined), params);
        self.lowered.insert(Arc::as_ptr(c), held.clone());
        Ok(held)
    }

    fn variant(&mut self, path: &str, defined: Defined) -> String {
        let held = self.variants.entry(path.to_string()).or_default();
        if let Some((_, name)) = held.iter().find(|(d, _)| *d == defined) {
            return name.clone();
        }
        let name = match held.len() {
            0 => path.to_string(),
            n => format!("{path}#{}", n + 1),
        };
        held.push((defined.clone(), name.clone()));
        self.out.push((name.clone(), defined));
        name
    }

    fn score(&mut self, c: &Container) -> Result<(Defined, BTreeSet<LaneId>), Vec<Refusal>> {
        let Body::Score(terms) = &c.body else {
            unreachable!("a score's body")
        };
        let graph = self.graph;
        let score = graph.score(&c.path).expect("a score");
        let mut params = BTreeSet::new();
        let mut out = Vec::with_capacity(terms.len());
        for term in terms {
            let Item::Event(event) = &score.lines[term.line].items[term.item] else {
                unreachable!("a term is an event")
            };
            let (d_expr, d) = match score.unit {
                SpanUnit::Seconds => {
                    let d = event.at.amount.to_f64();
                    (num(d.abs()), d)
                }
                SpanUnit::Bars => {
                    let (e, d) = self.song.bars(c.at, event.at.amount);
                    let e = match e {
                        Expr::Lit(Literal::Bars(b)) => Expr::Lit(Literal::Bars(b.abs())),
                        Expr::Lit(Literal::Num(n)) => num(n.abs()),
                        other => other,
                    };
                    (e, d)
                }
            };
            let arg = shift(
                d_expr,
                event.at.amount.is_negative(),
                event.at.amount == Ratio::ZERO,
            );
            let span = score.lines[term.line].span;
            let expr = match &term.target {
                Target::Instance(k) => {
                    let mut binds = self.song.line_binds(graph, *k)?;
                    binds.extend(self.lane_binds(*k, false, d, &mut params));
                    Expr::Ref {
                        path: event.node.name.clone(),
                        arg: Box::new(arg),
                        binds,
                        address: Address::Time,
                        span,
                    }
                }
                Target::Container(child) => {
                    let (name, child_params) = self.container(child)?;
                    let mut binds = Vec::new();
                    for lane in &child_params {
                        params.insert(lane.clone());
                        binds.push((lane.name(), self.read(false, lane, d)));
                    }
                    let placed = Expr::Ref {
                        path: written(&event.node.name, &child.path, &name),
                        arg: Box::new(arg),
                        binds,
                        address: Address::Time,
                        span,
                    };
                    match event.len {
                        None => placed,
                        Some(len) => {
                            self.cropped(placed, c.at, event, len.amount, score.unit, span)
                        }
                    }
                }
            };
            out.push(expr);
        }
        Ok((
            Defined {
                body: balanced(out),
                defaults: Vec::new(),
            },
            params,
        ))
    }

    /// `crop(@score(t - d), -inf, d + L)`.
    fn cropped(
        &self,
        placed: Expr,
        at: Option<Pos>,
        event: &Event,
        len: Ratio,
        unit: SpanUnit,
        span: ByteSpan,
    ) -> Expr {
        let end = match unit {
            SpanUnit::Seconds => num(event.at.amount.to_f64() + len.to_f64()),
            SpanUnit::Bars => {
                let to = event.at.amount.plus(len).expect("a bar span");
                let (e, d) = self.song.bars(at, to);
                match e {
                    Expr::Lit(Literal::Bars(_)) => e,
                    _ => num(d),
                }
            }
        };
        Expr::Call {
            name: "crop".to_string(),
            args: vec![
                Arg::Pos(placed),
                Arg::Pos(crate::parser::parse("-inf").expect("a literal")),
                Arg::Pos(end),
            ],
            span,
        }
    }

    /// An equation container's body with every site lowered; the root reads lanes itself.
    fn equation(
        &mut self,
        c: &Container,
        root: bool,
    ) -> Result<(Defined, BTreeSet<LaneId>), Vec<Refusal>> {
        let Body::Equation(placed) = &c.body else {
            unreachable!("an equation's body")
        };
        let defined = self.graph.source(&c.path).expect("a container").clone();
        let mut order = Vec::new();
        sites(self.graph, &c.path, &defined.body, &mut order);
        let mut params = BTreeSet::new();
        let mut next = 0usize;
        let body = self.rebuild(
            (&c.path, &defined.body),
            placed,
            &mut next,
            root,
            &mut params,
        )?;
        Ok((
            Defined {
                body,
                defaults: defined.defaults.clone(),
            },
            params,
        ))
    }

    fn rebuild(
        &mut self,
        (at, e): (&str, &Expr),
        placed: &[Site],
        next: &mut usize,
        root: bool,
        params: &mut BTreeSet<LaneId>,
    ) -> Result<Expr, Vec<Refusal>> {
        let is_site = match e {
            Expr::Call { name, .. } if name == "concat" || name == "repeat" => true,
            Expr::Ref { .. } => true,
            Expr::Call { name, .. } => crate::song::place::is_node_call(self.graph, at, name),
            _ => false,
        };
        if !is_site {
            return map_children(e, Binds::Substitute, |child| {
                self.rebuild((at, child), placed, next, root, params)
            });
        }
        let site = &placed[*next];
        *next += 1;
        match (site, e) {
            (Site::Concat(parts), Expr::Call { args, span, .. }) => {
                let arg = |k: usize| args.get(k).map(|(Arg::Pos(x) | Arg::Named(_, x))| x);
                let mut terms = Vec::with_capacity(parts.len());
                for (k, (start, end, target)) in parts.iter().enumerate() {
                    let part = match e {
                        Expr::Call { name, .. } if name == "repeat" => arg(0),
                        _ => arg(k),
                    };
                    let Some(Expr::Ref {
                        path,
                        binds,
                        span: at,
                        ..
                    }) = part
                    else {
                        unreachable!("a part is a ref")
                    };
                    let shifted = Expr::Bin(BinOp::Sub, Box::new(t()), Box::new(num(*start)));
                    let read = Expr::Ref {
                        path: path.clone(),
                        arg: Box::new(shifted),
                        binds: binds.clone(),
                        address: Address::Time,
                        span: *at,
                    };
                    let read = self.placed(read, target, root, *start, params)?;
                    terms.push(Expr::Call {
                        name: "crop".to_string(),
                        args: vec![Arg::Pos(read), Arg::Pos(num(*start)), Arg::Pos(num(*end))],
                        span: *span,
                    });
                }
                Ok(terms
                    .into_iter()
                    .reduce(|acc, t| Expr::Bin(BinOp::Add, Box::new(acc), Box::new(t)))
                    .unwrap_or(num(0.0)))
            }
            (Site::Read(d, target), e) => {
                let inner = map_children(e, Binds::Substitute, |child| {
                    self.rebuild((at, child), placed, next, root, params)
                })?;
                self.placed(inner, target, root, d.unwrap_or(0.0), params)
            }
            _ => unreachable!("sites are visited in the order they were placed"),
        }
    }

    /// A read of `target` at shift `d`, its lanes bound or passed on.
    fn placed(
        &mut self,
        read: Expr,
        target: &Target,
        root: bool,
        d: f64,
        params: &mut BTreeSet<LaneId>,
    ) -> Result<Expr, Vec<Refusal>> {
        match target {
            Target::Instance(k) => {
                let instance = &self.song.instances[*k];
                if let Some(name) = instance.node.strip_prefix("variables/")
                    && self
                        .song
                        .lane(&LaneId::Variable(name.to_string()))
                        .is_some()
                {
                    if root {
                        return Ok(read);
                    }
                    let lane = LaneId::Variable(name.to_string());
                    params.insert(lane.clone());
                    let arg = match &read {
                        Expr::Ref { arg, .. } => (**arg).clone(),
                        _ => t(),
                    };
                    return Ok(match arg == t() {
                        true => Expr::Var(lane.name()),
                        false => Expr::Call {
                            name: lane.name(),
                            args: vec![Arg::Pos(arg)],
                            span: ByteSpan::at(0),
                        },
                    });
                }
                let binds = self.lane_binds(*k, root, d, params);
                Ok(match binds.is_empty() {
                    true => read,
                    false => with_binds(read, binds),
                })
            }
            Target::Container(child) => {
                let (name, child_params) = self.container(child)?;
                let mut binds = Vec::new();
                for lane in &child_params {
                    if !root {
                        params.insert(lane.clone());
                    }
                    binds.push((lane.name(), self.read(root, lane, d)));
                }
                let written_binds: Vec<&str> = match &read {
                    Expr::Ref { binds, .. } => binds.iter().map(|(k, _)| k.as_str()).collect(),
                    _ => Vec::new(),
                };
                for lane in self.song.own_lanes(self.graph, &child.path) {
                    let LaneId::Sound(_, param) = &lane else {
                        unreachable!("a node's own lane")
                    };
                    if written_binds.contains(&param.as_str()) {
                        continue;
                    }
                    if !root {
                        params.insert(lane.clone());
                    }
                    binds.push((param.clone(), self.read(root, &lane, d)));
                }
                let read = match read {
                    Expr::Ref {
                        path,
                        arg,
                        binds: written_binds,
                        address,
                        span,
                    } => Expr::Ref {
                        path: written(&path, &child.path, &name),
                        arg,
                        binds: written_binds,
                        address,
                        span,
                    },
                    Expr::Call {
                        name: called, span, ..
                    } => Expr::Ref {
                        path: written(&called, &child.path, &name),
                        arg: Box::new(t()),
                        binds: Vec::new(),
                        address: Address::Time,
                        span,
                    },
                    other => other,
                };
                Ok(match binds.is_empty() {
                    true => read,
                    false => with_binds(read, binds),
                })
            }
        }
    }
}

/// A moved lane as one closed form in song time: its home until its first move, each ramp
/// over its span, each value held until the next move. A scale has no number, so no form.
fn closed_form(lane: &Lane) -> Option<Defined> {
    if matches!(lane.home, Home::Scale(_)) {
        return None;
    }
    let n = |x: f64| match x < 0.0 {
        true => format!("({x})"),
        false => format!("{x}"),
    };
    let s = |x: f64| match x < 0.0 {
        true => format!("(-{}s)", -x),
        false => format!("{x}s"),
    };
    let mut terms: Vec<String> = Vec::new();
    let first = s(lane.moves.first()?.start);
    if let Home::Number(home, _) = lane.home {
        terms.push(format!("crop({}, -inf, {first})", n(home)));
    }
    for (k, m) in lane.moves.iter().enumerate() {
        let next = lane.moves.get(k + 1).map(|n| n.start);
        let to = m.to.number()?;
        if m.end > m.start {
            let from = m.from.as_ref().and_then(Held::number)?;
            let u = format!("(t - {})/{}", s(m.start), s(m.end - m.start));
            let value = match (m.shape, from == to) {
                (_, true) => n(from),
                (Some(Shape::Exp), false) => format!("{}*pow({}, {u})", n(from), n(to / from)),
                _ => format!("{} + {}*{u}", n(from), n(to - from)),
            };
            terms.push(format!("crop({value}, {}, {})", s(m.start), s(m.end)));
        }
        let end = next.map_or("inf".to_string(), s);
        if next.is_none_or(|next| next > m.end) {
            terms.push(format!("crop({}, {}, {end})", n(to), s(m.end)));
        }
    }
    let text = terms.join(" + ");
    let body = crate::parser::parse(&text).expect("a lane's closed form parses");
    Some(Defined {
        body,
        defaults: Vec::new(),
    })
}
