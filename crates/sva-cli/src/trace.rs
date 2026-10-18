// Concern: seeds a trace from every entry point and names what the caller asked for | Non-concern: the traversal itself (sva-engine), JSON shape (output.rs) | IO: (dir, target) -> Traced or CliError

use std::path::Path;

use std::collections::BTreeSet;

use sva_ast::{Binds, Expr, Graph, children, resolve_ref_path};
use sva_core::{CliError, PROBE, define_probe_for, prepared};
use sva_engine::{EngineError, Traced};

pub struct Traceable {
    pub bpm: Option<f64>,
    pub meter: Option<String>,
    pub traced: Traced,
}

/// Every entry point is a root, and an uninstantiated bench refuses. A structure has no
/// time, so an interval the target reads is dropped.
pub fn trace(dir: &Path, target: &str) -> Result<Traceable, CliError> {
    let target = match sva_core::target(target)? {
        sva_core::Target {
            expr,
            interval: Some(_),
        } => expr,
        _ => target.to_string(),
    };
    let target = target.as_str();
    let mut graph = prepared(&sva_ast::Dir::at(dir))?;

    let mut roots = entry_points(&graph);
    if graph.defines(target)
        && !roots.iter().any(|r| r == target)
        && !referenced(&graph).contains(target)
    {
        roots.push(target.to_string());
    }
    let mut traced = match sva_engine::trace(&graph, &roots, target) {
        Err(EngineError::UnknownNode(ref name)) if name == target && !graph.defines(target) => {
            probe(&mut graph, target)?
        }
        other => other.map_err(CliError::Engine)?,
    };
    let readers = traced.up.iter_mut().map(|up| (&up.node, &mut up.expr));
    for (node, expr) in std::iter::once((&traced.node, &mut traced.expr)).chain(readers) {
        if let Some(score) = graph.score(node) {
            *expr = summary(score);
        }
    }
    let tempo = graph.tempo();
    Ok(Traceable {
        bpm: tempo.as_ref().map(|tempo| tempo.bpm),
        meter: tempo.map(|tempo| tempo.meter),
        traced,
    })
}

/// A score is its lines, not the sum they lower to: how many, when, and what they play.
fn summary(score: &sva_ast::Score) -> String {
    let mut nodes: Vec<&str> = Vec::new();
    for event in &score.events {
        if !nodes.contains(&event.node.as_str()) {
            nodes.push(&event.node);
        }
    }
    let (first, last) = (&score.events[0], &score.events[score.events.len() - 1]);
    format!(
        "a score of {} lines, `at` {} to {}, each playing @{} shifted to its `at`",
        score.events.len(),
        first.at.spelled(),
        last.at.spelled(),
        nodes.join(", @")
    )
}

/// `text` is neither node nor instance: argv math, an entry point of its own.
fn probe(graph: &mut Graph, text: &str) -> Result<Traced, CliError> {
    define_probe_for(graph, text)?;
    let roots = entry_points(graph);
    sva_engine::trace(graph, &roots, PROBE).map_err(CliError::Engine)
}

fn referenced(graph: &Graph) -> BTreeSet<String> {
    let mut reached: BTreeSet<String> = BTreeSet::new();
    for path in graph.paths() {
        if let Some(expr) = graph.expr(path) {
            collect_refs(path, expr, &mut reached);
        }
    }
    reached
}

/// Every node nothing references, bar a library node (one with a free parameter) and a scale,
/// which is data a score's degrees read and no signal.
pub(crate) fn entry_points(graph: &Graph) -> Vec<String> {
    let reached = referenced(graph);
    graph
        .paths()
        .filter(|p| !reached.contains(*p))
        .filter(|p| !reserved_variable(p) && graph.scale(p).is_none())
        .filter(|p| graph.free_parameters(p).is_empty())
        .map(str::to_string)
        .collect()
}

pub(crate) fn reserved_variable(path: &str) -> bool {
    sva_core::RESERVED_VARIABLES
        .iter()
        .any(|name| sva_ast::variable(name) == path)
}

fn collect_refs(from: &str, expr: &Expr, out: &mut BTreeSet<String>) {
    if let Expr::Ref { path, .. } = expr
        && let Some(resolved) = resolve_ref_path(from, path)
    {
        out.insert(resolved);
    }
    for child in children(expr, Binds::Substitute) {
        collect_refs(from, child, out);
    }
}
