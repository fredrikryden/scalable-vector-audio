// Concern: what each value a node's renderer reads contributed to it | Non-concern: building the renderer, sharing a target's energy out (sva-samples) | IO: (NodeId) -> a ref and its addend

use sva_formula::NodeId;
use sva_samples::{BufId, NodeRenderer, Slot};

use crate::error::EngineError;
use crate::render::Render;
use crate::render::value_graph::{Kind, ValueGraph};

/// Every ref a node's renderer reads, one row per name however many slots carry it.
pub(super) fn refs_read(
    render: &Render,
    node: NodeId,
    holds: &dyn Fn(NodeId) -> bool,
) -> Result<Vec<NodeId>, EngineError> {
    if let Some(only) = crate::refs::passes(&render.tys, node) {
        return Ok(holds(only).then_some(only).into_iter().collect());
    }
    let Some((value_graph, at)) = machine_run_of(render, node) else {
        return Ok(Vec::new());
    };
    let here = render.tys.name(node);
    let mut out: Vec<NodeId> = Vec::new();
    for source in sources(&value_graph.values[at]) {
        let Some(source) = source else {
            continue;
        };
        let Some(ref_node) = behind(render, here, source, holds) else {
            continue;
        };
        let name = render.tys.name(ref_node);
        if !out.iter().any(|held| render.tys.name(*held) == name) {
            out.push(ref_node);
        }
    }
    Ok(out)
}

/// The node each slot of a machine run was built for.
fn sources(value: &crate::render::value_graph::Value) -> Vec<Option<NodeId>> {
    match &value.kind {
        Kind::MachineRun(machine_run) => machine_run.sources.to_vec(),
        _ => Vec::new(),
    }
}

/// The value graph's machine run for `node`, where it computes one.
fn machine_run_of(render: &Render, node: NodeId) -> Option<(&ValueGraph, usize)> {
    let value_graph = render.value_graph.as_ref()?;
    let at = value_graph.of(node)?;
    matches!(value_graph.values[at].kind, Kind::MachineRun(_)).then_some((value_graph, at))
}

pub(super) fn reads_held(render: &Render, node: NodeId) -> Result<bool, EngineError> {
    Ok(render
        .value_graph
        .as_ref()
        .is_some_and(|t| t.of(node).is_some()))
}

/// The ref one slot stands for: its source, or — where that source is a subterm written here,
/// and so carries this node's name — the one ref it reads. A subterm summing several isolates
/// none, and one this render never held has no row.
fn behind(
    render: &Render,
    here: &str,
    source: NodeId,
    holds: &dyn Fn(NodeId) -> bool,
) -> Option<NodeId> {
    let held = |id: NodeId| holds(id).then_some(id);
    if render.tys.name(source) != here {
        return held(source);
    }
    let read = crate::schedule::read_operands(&render.tys, source);
    let mut names: Vec<&str> = read.iter().map(|id| render.tys.name(*id)).collect();
    names.dedup();
    match (names.len(), read.first()) {
        (1, Some(only)) if render.tys.name(*only) != here => held(*only),
        _ => None,
    }
}

/// Where what one ref put into the node reading it comes from: that node's own samples where
/// it only passes the ref on, else its renderer run with every other slot silenced, so the
/// refs of a sum add back to it. `None` is an edge no slot isolates.
pub(super) enum Contribution {
    Passed,
    Silenced(Box<NodeRenderer>),
}

pub(super) fn contribution(
    render: &Render,
    parent: NodeId,
    child: NodeId,
    holds: &dyn Fn(NodeId) -> bool,
) -> Result<Option<Contribution>, EngineError> {
    if crate::refs::passes(&render.tys, parent) == Some(child) {
        return Ok(holds(parent).then_some(Contribution::Passed));
    }
    let Some((renderer, kept)) = isolated(render, parent, child, holds)? else {
        return Ok(None);
    };
    let (value_graph, at) =
        machine_run_of(render, parent).expect("an isolated edge is a machine run's");
    let Kind::MachineRun(machine_run) = &value_graph.values[at].kind else {
        unreachable!("machine_run_of answers a machine run");
    };
    let others = Silenced {
        kept: &|id: BufId| kept.contains(&id),
        widths: &machine_run.layout.read_widths,
    };
    Ok(Some(Contribution::Silenced(Box::new(others.of(&renderer)))))
}

pub(super) fn isolated(
    render: &Render,
    parent: NodeId,
    child: NodeId,
    holds: &dyn Fn(NodeId) -> bool,
) -> Result<Option<(NodeRenderer, Vec<BufId>)>, EngineError> {
    let Some((value_graph, at)) = machine_run_of(render, parent) else {
        return Ok(None);
    };
    let Kind::MachineRun(machine_run) = &value_graph.values[at].kind else {
        return Ok(None);
    };
    let (here, name) = (render.tys.name(parent), render.tys.name(child));
    let kept: Vec<BufId> = machine_run
        .sources
        .iter()
        .enumerate()
        .filter(|(_, source)| {
            source
                .and_then(|source| behind(render, here, source, holds))
                .is_some_and(|id| render.tys.name(id) == name)
        })
        .map(|(slot, _)| BufId(slot as u32))
        .collect();
    let held = |id: BufId| kept.contains(&id);
    match kept.is_empty() || !separable(&machine_run.renderer, &held) {
        true => Ok(None),
        false => Ok(Some(((*machine_run.renderer).clone(), kept))),
    }
}

/// Two moving operands under one product have no addend apiece; a solver moves as a slot
/// does. A filter is linear in what it filters, so only its slots count.
fn separable(r: &NodeRenderer, kept: &dyn Fn(BufId) -> bool) -> bool {
    let parts = operands(r);
    match r {
        NodeRenderer::Add(_) | NodeRenderer::Sub(..) => parts.iter().all(|p| separable(p, kept)),
        _ => {
            let counts = |p: &NodeRenderer| match r {
                NodeRenderer::Filter { .. } => reads(p, &|_| true),
                _ => moves(p),
            };
            let mut holding = parts.iter().filter(|p| counts(p));
            match (holding.next(), holding.next()) {
                (None, _) => true,
                (Some(only), None) => separable(only, kept),
                _ => !reads(r, kept),
            }
        }
    }
}

fn moves(r: &NodeRenderer) -> bool {
    match r {
        NodeRenderer::Const(_) => false,
        NodeRenderer::Time
        | NodeRenderer::Read { .. }
        | NodeRenderer::Indexed { .. }
        | NodeRenderer::Noise(_)
        | NodeRenderer::Physics { .. } => true,
        other => operands(other).iter().any(|p| moves(p)),
    }
}

fn reads(r: &NodeRenderer, kept: &dyn Fn(BufId) -> bool) -> bool {
    match r {
        NodeRenderer::Read {
            slot: Slot::Read(id),
            ..
        }
        | NodeRenderer::Indexed {
            slot: Slot::Read(id),
            ..
        } if kept(*id) => true,
        NodeRenderer::Read { .. } => false,
        other => operands(other).iter().any(|p| reads(p, kept)),
    }
}

/// A renderer with every read but the kept ones at exact zero, each zero as wide as the
/// slot it stands for, so the renderer writes the width its value holds.
struct Silenced<'a> {
    kept: &'a dyn Fn(BufId) -> bool,
    widths: &'a [usize],
}

impl Silenced<'_> {
    fn of(&self, r: &NodeRenderer) -> NodeRenderer {
        match r {
            NodeRenderer::Read {
                slot: Slot::Read(id),
                ..
            }
            | NodeRenderer::Indexed {
                slot: Slot::Read(id),
                ..
            } if !(self.kept)(*id) => NodeRenderer::zeros(self.widths[id.0 as usize]),
            NodeRenderer::Add(set) => NodeRenderer::Add(self.each(set)),
            NodeRenderer::Mul(set) => NodeRenderer::Mul(self.each(set)),
            NodeRenderer::Join(set) => NodeRenderer::Join(self.each(set)),
            NodeRenderer::Sub(l, r) => NodeRenderer::Sub(self.one(l), self.one(r)),
            NodeRenderer::Div(l, r) => NodeRenderer::Div(self.one(l), self.one(r)),
            NodeRenderer::Pow(l, r) => NodeRenderer::Pow(self.one(l), self.one(r)),
            NodeRenderer::Zip(op, l, r) => NodeRenderer::Zip(*op, self.one(l), self.one(r)),
            NodeRenderer::Map(op, x) => NodeRenderer::Map(*op, self.one(x)),
            NodeRenderer::Crop {
                x,
                window,
                a,
                b,
                rise,
                fall,
            } => NodeRenderer::Crop {
                x: self.one(x),
                window: *window,
                a: *a,
                b: *b,
                rise: *rise,
                fall: *fall,
            },
            NodeRenderer::Channel { x, k } => NodeRenderer::Channel {
                x: self.one(x),
                k: *k,
            },
            NodeRenderer::Filter {
                site,
                from,
                x,
                cutoff,
                q,
                gain,
            } => NodeRenderer::Filter {
                site: *site,
                from: *from,
                x: self.one(x),
                cutoff: self.one(cutoff),
                q: self.one(q),
                gain: self.one(gain),
            },
            NodeRenderer::Physics { site, from, args } => NodeRenderer::Physics {
                site: *site,
                from: *from,
                args: self.each(args),
            },
            leaf => leaf.clone(),
        }
    }

    fn each(&self, set: &[NodeRenderer]) -> Vec<NodeRenderer> {
        set.iter().map(|p| self.of(p)).collect()
    }

    fn one(&self, r: &NodeRenderer) -> Box<NodeRenderer> {
        Box::new(self.of(r))
    }
}

fn operands(r: &NodeRenderer) -> Vec<&NodeRenderer> {
    match r {
        NodeRenderer::Add(set) | NodeRenderer::Mul(set) | NodeRenderer::Join(set) => {
            set.iter().collect()
        }
        NodeRenderer::Sub(l, r)
        | NodeRenderer::Div(l, r)
        | NodeRenderer::Pow(l, r)
        | NodeRenderer::Zip(_, l, r) => vec![l, r],
        NodeRenderer::Map(_, x)
        | NodeRenderer::Crop { x, .. }
        | NodeRenderer::Channel { x, .. } => {
            vec![x]
        }
        NodeRenderer::Filter {
            x, cutoff, q, gain, ..
        } => vec![x, cutoff, q, gain],
        NodeRenderer::Physics { args, .. } => args.iter().collect(),
        _ => Vec::new(),
    }
}
