// Concern: the moved lanes reaching each node no score sits under, and its body passing each on as a parameter | Non-concern: folding a lane (lower.rs) | IO: (&Graph, node) -> lanes, Defined

use std::collections::{BTreeMap, BTreeSet};

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::{Arg, BinOp, Binds, Expr, Literal, map_children};
use std::sync::Arc;

use crate::graph::{Defined, Graph, VARIABLES, resolve_ref_path};
use crate::refusal::Refusal;
use crate::reserved::Values;
use crate::song::LaneId;
use crate::song::place::{constant_shift, is_node_call, parameters};

pub(crate) struct Reach<'g> {
    graph: &'g Graph,
    moved: &'g BTreeSet<LaneId>,
    held: BTreeMap<(String, Vec<String>), Vec<LaneId>>,
    values: &'g Values,
}

/// What `l` passes on read `d` seconds later: `l` itself at no shift.
pub(crate) fn retimed(l: &str, d: f64) -> Expr {
    let name = l.to_string();
    if d == 0.0 {
        return Expr::Var(name);
    }
    let t = Box::new(Expr::Var("t".to_string()));
    let arg = match d > 0.0 {
        true => Expr::Bin(BinOp::Add, t, Box::new(Expr::Lit(Literal::Num(d)))),
        false => Expr::Bin(BinOp::Sub, t, Box::new(Expr::Lit(Literal::Num(-d)))),
    };
    Expr::Call {
        name,
        args: vec![Arg::Pos(arg)],
        span: ByteSpan::at(0),
    }
}

/// The node a read or a bareword call names, its time argument and its binds' names.
pub(crate) fn invoked<'e>(
    graph: &Graph,
    at: &str,
    e: &'e Expr,
) -> Option<(String, &'e Expr, BTreeSet<&'e str>)> {
    match e {
        Expr::Ref {
            path, arg, binds, ..
        } => Some((
            resolve_ref_path(at, path)?,
            arg,
            binds.iter().map(|(k, _)| k.as_str()).collect(),
        )),
        Expr::Call { name, .. } if is_node_call(graph, at, name) => {
            Some((resolve_ref_path(at, name)?, &T, BTreeSet::new()))
        }
        _ => None,
    }
}

static T: Expr = Expr::Lit(Literal::Num(0.0));

fn is_t(e: &Expr) -> bool {
    std::ptr::eq(e, &T) || matches!(e, Expr::Var(n) if n == "t")
}

fn moved_variable(path: &str) -> Option<&str> {
    path.strip_prefix(&format!("{VARIABLES}/"))
}

impl<'g> Reach<'g> {
    pub fn new(graph: &'g Graph, moved: &'g BTreeSet<LaneId>, values: &'g Values) -> Reach<'g> {
        Reach {
            graph,
            moved,
            held: BTreeMap::new(),
            values,
        }
    }

    /// `path` as its file writes it, where it counts degrees in a scale that moves: each
    /// `deg(n)` reading degree n's ratio lane. `None` where no degree of it follows the scale.
    fn degrees(&self, path: &str) -> Option<Result<Defined, Refusal>> {
        let moves = self.moved.contains(&LaneId::Variable("scale".to_string()));
        if !moves || !self.graph.reserved_reads(path).contains(&"scale") {
            return None;
        }
        let text = self.graph.text(path)?;
        let parsed = match crate::ingest::parse_file(text) {
            Ok(parsed) => parsed,
            Err(d) => return Some(Err(Refusal::new(path, d.span, d.code, d.message))),
        };
        let written = Defined {
            body: parsed.expr,
            defaults: parsed.defaults,
        };
        let ratio = |n: i64| Expr::Var(LaneId::Degree(n).name());
        let lowered = self
            .graph
            .lowered_with(path, &written, self.values, Some(&ratio));
        Some(lowered.map_err(|mut refused| refused.remove(0)))
    }

    /// The sound lanes of `node`'s parameters that a site binding `bound` leaves to reach it.
    pub fn sound(&self, node: &str, bound: &BTreeSet<&str>) -> Vec<LaneId> {
        parameters(self.graph, node)
            .into_iter()
            .filter(|p| !bound.contains(p.as_str()))
            .map(|p| LaneId::Sound(node.to_string(), p))
            .filter(|l| self.moved.contains(l))
            .collect()
    }

    /// Every moved lane reaching what `path` computes where its invoker binds `bound`, sorted:
    /// a default line the invoker binds is never read, so what it reads reaches nothing.
    pub fn of(&mut self, path: &str, bound: &BTreeSet<&str>) -> Vec<LaneId> {
        let key = (
            path.to_string(),
            bound.iter().map(|b| b.to_string()).collect::<Vec<_>>(),
        );
        if let Some(held) = self.held.get(&key) {
            return held.clone();
        }
        self.held.insert(key.clone(), Vec::new());
        let mut out: BTreeSet<LaneId> = BTreeSet::new();
        let source = match self.degrees(path) {
            Some(Ok(defined)) => Some(Arc::new(defined)),
            _ => self.graph.source(path).cloned(),
        };
        if let Some(defined) = source {
            let read: Vec<&Expr> = std::iter::once(&defined.body)
                .chain(
                    defined
                        .defaults
                        .iter()
                        .filter(|(name, _)| !bound.contains(name.as_str()))
                        .map(|(_, v)| v),
                )
                .collect();
            let mut sites = Vec::new();
            for e in read {
                read_degrees(e, &mut out);
                crate::song::place::sites(self.graph, path, e, &mut sites);
            }
            for site in sites {
                let Some((node, _, bound)) = invoked(self.graph, path, site) else {
                    continue;
                };
                if let Some(name) = moved_variable(&node)
                    && self.moved.contains(&LaneId::Variable(name.to_string()))
                {
                    out.insert(LaneId::Variable(name.to_string()));
                    continue;
                }
                out.extend(self.sound(&node, &bound));
                out.extend(self.of(&node, &bound));
            }
        }
        let out: Vec<LaneId> = out.into_iter().collect();
        self.held.insert(key, out.clone());
        out
    }

    /// `path`'s body and defaults with each lane reaching it a parameter: a moved variable's
    /// read becomes the parameter read at its time, and each invocation passes on the lanes
    /// reaching its node, retimed by its shift.
    pub fn lowered(&mut self, path: &str) -> Result<Defined, Refusal> {
        let defined = match self.degrees(path) {
            Some(written) => Arc::new(written?),
            None => self.graph.source(path).expect("a node reached").clone(),
        };
        let body = self.rewrite(path, &defined.body)?;
        let defaults = defined
            .defaults
            .iter()
            .map(|(k, v)| Ok((k.clone(), self.rewrite(path, v)?)))
            .collect::<Result<_, Refusal>>()?;
        Ok(Defined { body, defaults })
    }

    fn rewrite(&mut self, path: &str, e: &Expr) -> Result<Expr, Refusal> {
        let rebuilt = map_children(e, Binds::Substitute, |c| self.rewrite(path, c))?;
        let Some((node, arg, bound)) = invoked(self.graph, path, e) else {
            return Ok(rebuilt);
        };
        if let Some(name) = moved_variable(&node)
            && self.moved.contains(&LaneId::Variable(name.to_string()))
        {
            let lane = LaneId::Variable(name.to_string()).name();
            return Ok(match is_t(arg) {
                true => Expr::Var(lane),
                false => Expr::Call {
                    name: lane,
                    args: vec![Arg::Pos(arg.clone())],
                    span: ByteSpan::at(0),
                },
            });
        }
        let bound: BTreeSet<String> = bound.into_iter().map(str::to_string).collect();
        let bound_ref: BTreeSet<&str> = bound.iter().map(String::as_str).collect();
        let sound = self.sound(&node, &bound_ref);
        let inner = self.of(&node, &bound_ref);
        if sound.is_empty() && inner.is_empty() {
            return Ok(rebuilt);
        }
        let d = match is_t(arg) {
            true => 0.0,
            false => constant_shift(arg).ok_or_else(|| {
                let lanes: Vec<String> = sound.iter().chain(&inner).map(LaneId::name).collect();
                Refusal::new(
                    path,
                    span_of(e),
                    DiagCode::BadLane,
                    format!(
                        "`{path}` reads `@{node}` at a time that is no constant shift of `t`, \
                         so it has no song time, and the lanes `{}` move it; read it at \
                         `t - d`, or pass each lane's parameter on",
                        lanes.join("`, `")
                    ),
                )
            })?,
        };
        let mut binds: Vec<(String, Expr)> = Vec::new();
        for lane in &sound {
            let LaneId::Sound(_, param) = lane else {
                unreachable!("a sound lane")
            };
            binds.push((param.clone(), retimed(&lane.name(), d)));
        }
        for lane in &inner {
            binds.push((lane.name(), retimed(&lane.name(), d)));
        }
        Ok(with_binds(rebuilt, binds))
    }
}

/// Each degree ratio lane `e` reads.
fn read_degrees(e: &Expr, out: &mut BTreeSet<LaneId>) {
    if let Expr::Var(name) = e
        && let Some(n) = name.strip_prefix(&format!("{VARIABLES}/scale:"))
        && let Ok(n) = n.parse()
    {
        out.insert(LaneId::Degree(n));
    }
    for c in crate::expr::children(e, Binds::Substitute) {
        read_degrees(c, out);
    }
}

pub(crate) fn span_of(e: &Expr) -> ByteSpan {
    match e {
        Expr::Ref { span, .. } | Expr::Call { span, .. } => *span,
        _ => ByteSpan::at(0),
    }
}

/// `e`, a read or a bareword call, with `more` bound after what it binds.
pub(crate) fn with_binds(e: Expr, more: Vec<(String, Expr)>) -> Expr {
    match e {
        Expr::Ref {
            path,
            arg,
            mut binds,
            address,
            span,
        } => {
            binds.extend(more);
            Expr::Ref {
                path,
                arg,
                binds,
                address,
                span,
            }
        }
        Expr::Call { name, span, .. } => Expr::Ref {
            path: name,
            arg: Box::new(Expr::Var("t".to_string())),
            binds: more,
            address: crate::expr::Address::Time,
            span,
        },
        other => other,
    }
}
