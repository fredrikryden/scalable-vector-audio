// Concern: lays a ledger's contributions out as values the pull computes, tallying each block | Non-concern: what one entry says (sva-samples) | IO: (&mut Render) -> Ledgers; (block) -> sums

use std::collections::{BTreeMap, BTreeSet};

use sva_formula::NodeId;
use sva_samples::measure::ledger::{Sources, Tally};
use sva_samples::{Buffer, Extent, LedgerEntry, SignalKind};

use super::Render;
use super::slots::{self, Contribution};
use super::value_graph::{Kind, ValueGraph};
use crate::error::EngineError;
use crate::query::Representation;

/// Tallied a block at a time as the pull goes: no row's samples outlive their block.
pub(super) struct Ledgers(Vec<Ledger>);

struct Ledger {
    node: NodeId,
    depth: usize,
    tally: Result<Tallying, EngineError>,
}

struct Tallying {
    tally: Tally,
    values: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct Tallied {
    node: NodeId,
    depth: usize,
    entries: Result<Vec<LedgerEntry>, EngineError>,
}

impl Ledgers {
    /// Each contribution a value of its own beside the machine run it isolates a slot of.
    pub(super) fn laid(render: &mut Render) -> Option<Ledgers> {
        let mut asked: Vec<(NodeId, usize)> = Vec::new();
        for ask in &render.config.asks {
            let Representation::Ledger { depth } = ask.representation else {
                continue;
            };
            let Some(node) = render.tys.id(&ask.node) else {
                continue;
            };
            if !asked.contains(&(node, depth)) {
                asked.push((node, depth));
            }
        }
        if asked.is_empty() || render.value_graph.is_none() {
            return None;
        }
        let walked: Vec<_> = asked
            .iter()
            .map(|(node, depth)| walked(render, *node, *depth))
            .collect();
        let value_graph = render.value_graph.as_mut().expect("a ledger's value graph");
        let each = asked
            .into_iter()
            .zip(walked)
            .map(|((node, depth), walked)| Ledger {
                node,
                depth,
                tally: walked.and_then(|walked| walked.tallying(value_graph)),
            })
            .collect();
        Some(Ledgers(each))
    }

    pub(super) fn add(&mut self, value_graph: &ValueGraph, block: Extent) {
        for ledger in &mut self.0 {
            let Ok(tallying) = &mut ledger.tally else {
                continue;
            };
            let held: Vec<Option<Buffer>> = (0..tallying.values.len())
                .map(|source| Some(value_graph.samples(tallying.values[source], block)))
                .collect();
            let held: Vec<Option<&Buffer>> = held.iter().map(Option::as_ref).collect();
            tallying.tally.add(&held, 0..block.len());
        }
    }

    pub(super) fn tallied(self) -> Vec<Tallied> {
        self.0
            .into_iter()
            .map(|ledger| Tallied {
                node: ledger.node,
                depth: ledger.depth,
                entries: ledger.tally.map(|tallying| tallying.tally.entries()),
            })
            .collect()
    }
}

pub(super) fn entries(
    render: &Render,
    node: NodeId,
    depth: usize,
) -> Option<Result<Vec<LedgerEntry>, EngineError>> {
    let found = render.ledgers.iter();
    let mut found = found.filter(|held| held.node == node && held.depth == depth);
    found.next().map(|held| held.entries.clone())
}

struct Walked {
    target: String,
    depth: usize,
    deps: BTreeMap<String, Vec<String>>,
    kinds: BTreeMap<String, SignalKind>,
    own: BTreeMap<String, usize>,
    contributed: BTreeMap<String, (usize, Contribution)>,
}

/// A ref the closed form adds stands as what it contributed to its reader; one no addend
/// isolates is left unattributed.
fn walked(render: &Render, node: NodeId, depth: usize) -> Result<Walked, EngineError> {
    let value_graph = render.value_graph.as_ref().expect("a ledger's value graph");
    let schedule = &render.schedule;
    let rows: BTreeSet<NodeId> = (schedule.wanted.iter().chain(&schedule.tallied))
        .copied()
        .filter(|id| {
            value_graph
                .of(*id)
                .is_some_and(|at| !matches!(value_graph.values[at].kind, Kind::Frames { .. }))
        })
        .collect();
    let holds = |id: NodeId| rows.contains(&id);
    let edges = edges_under(render, node, depth, &holds)?;
    let under: BTreeSet<NodeId> = std::iter::once(node)
        .chain(edges.iter().map(|(_, child)| *child))
        .collect();
    let mut deps = BTreeMap::new();
    let mut kinds = BTreeMap::new();
    for id in rows.iter().filter(|id| under.contains(id)) {
        let name = render.tys.name(*id).to_string();
        let read = slots::refs_read(render, *id, &holds)?
            .into_iter()
            .map(|op| render.tys.name(op).to_string())
            .collect();
        deps.insert(name.clone(), read);
        kinds.insert(name, SignalKind::Audio);
    }
    let mut contributed = BTreeMap::new();
    for (parent, child) in &edges {
        if !slots::reads_held(render, *parent)? {
            return Err(super::answer::unmaterialized(
                render,
                node,
                Representation::Ledger { depth },
            ));
        }
        if let Some(held) = slots::contribution(render, *parent, *child, &holds)? {
            let at = value_graph.of(*parent).expect("a reader's value");
            contributed.insert(render.tys.name(*child).to_string(), (at, held));
        }
    }
    let readers: BTreeSet<NodeId> = edges.iter().map(|(parent, _)| *parent).collect();
    let mut own = BTreeMap::new();
    for id in rows.iter().filter(|id| under.contains(id)) {
        let name = render.tys.name(*id);
        if *id == node || readers.contains(id) || !contributed.contains_key(name) {
            own.insert(
                name.to_string(),
                value_graph.of(*id).expect("a row's value"),
            );
        }
    }
    Ok(Walked {
        target: render.tys.name(node).to_string(),
        depth,
        deps,
        kinds,
        own,
        contributed,
    })
}

impl Walked {
    fn tallying(self, value_graph: &mut ValueGraph) -> Result<Tallying, EngineError> {
        let mut values: Vec<usize> = Vec::new();
        let mut source = |at: usize| match values.iter().position(|held| *held == at) {
            Some(found) => found,
            None => {
                values.push(at);
                values.len() - 1
            }
        };
        let own: BTreeMap<String, usize> = (self.own.iter())
            .map(|(name, at)| (name.clone(), source(*at)))
            .collect();
        let mut read = own.clone();
        let mut attributed = BTreeSet::new();
        for (name, (at, held)) in self.contributed {
            let at = match held {
                Contribution::Passed => at,
                Contribution::Silenced(renderer) => value_graph.beside(at, *renderer)?,
            };
            read.insert(name.clone(), source(at));
            attributed.insert(name);
        }
        let widths: Vec<usize> = values.iter().map(|at| width(value_graph, *at)).collect();
        let sources = Sources {
            read: &read,
            own: &own,
            attributed: &attributed,
            widths: &widths,
        };
        let tally = Tally::new(&sources, &self.deps, &self.kinds, &self.target, self.depth);
        Ok(Tallying { tally, values })
    }
}

fn width(value_graph: &ValueGraph, mut at: usize) -> usize {
    while let Some((read, _)) = value_graph.values[at].alias() {
        at = read;
    }
    value_graph.values[at].width
}

/// The tree a ledger walks; a node several read is attributed to the first to reach it.
fn edges_under(
    render: &Render,
    node: NodeId,
    depth: usize,
    holds: &dyn Fn(NodeId) -> bool,
) -> Result<Vec<(NodeId, NodeId)>, EngineError> {
    let mut seen = BTreeSet::from([node]);
    let (mut level, mut out) = (vec![node], Vec::new());
    for _ in 0..depth {
        let mut next = Vec::new();
        for parent in level {
            for child in slots::refs_read(render, parent, holds)? {
                if seen.insert(child) {
                    out.push((parent, child));
                    next.push(child);
                }
            }
        }
        level = next;
    }
    Ok(out)
}
