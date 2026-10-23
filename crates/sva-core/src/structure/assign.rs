// Concern: k-medians over one kind's nodes: each line to its nearest centre in JNDs, each default to its lines' median | Non-concern: which kind | IO: (Model, varies, kind) -> edits

use super::{Edit, Proposal, alive, lines_of, played, push, varied};
use crate::model::Model;
use crate::solve::Vary;

const REASSIGN_ROUNDS: usize = 6;

/// Every kind's lines reassigned among its nodes, from the grouping the composition gives.
pub fn cluster(model: &Model, varies: &[Vary]) -> Proposal {
    let mut kinds: Vec<Vec<String>> = Vec::new();
    for n in model.nodes.keys().filter(|n| **n != model.root) {
        match kinds.iter_mut().find(|k| model.kin(&k[0], n)) {
            Some(k) => k.push(n.clone()),
            None => kinds.push(vec![n.clone()]),
        }
    }
    let (mut m, mut edits) = (model.clone(), Vec::new());
    for kind in kinds.iter().filter(|k| k.len() > 1) {
        reassign(&mut m, varies, kind, &mut edits);
    }
    Proposal {
        label: "cluster".to_string(),
        edits,
    }
}

/// How far line `line` plays from `node`'s defaults, in JNDs summed over each varied default.
fn distance(m: &Model, varies: &[Vary], line: usize, node: &str) -> f64 {
    let at = &m.lines[line].node;
    let to = &m.nodes[node].defaults;
    m.nodes[at]
        .defaults
        .iter()
        .filter_map(|(name, _)| {
            let v = varied(varies, at, name)?;
            let centre = varied(varies, node, name)?;
            let (_, c) = to.iter().find(|(d, _)| d == name)?;
            let res = v.resolution().expect("a priced fit's `--vary` has a step");
            let x = played(m, line, name).expect("a default");
            Some((v.coord(x) - centre.coord(*c)).abs() / res)
        })
        .sum()
}

/// `kind`'s lines each moved to its nearest node, then each node recentred, for every round.
pub(super) fn reassign(m: &mut Model, varies: &[Vary], kind: &[String], edits: &mut Vec<Edit>) {
    let lines: Vec<usize> = (0..m.lines.len())
        .filter(|l| kind.contains(&m.lines[*l].node) && alive(m, varies, *l))
        .collect();
    for _ in 0..REASSIGN_ROUNDS {
        let nearest: Vec<&String> = lines
            .iter()
            .map(|l| {
                let d = |n: &&String| distance(m, varies, *l, n);
                kind.iter()
                    .min_by(|a, b| d(a).total_cmp(&d(b)))
                    .expect("a kind holds nodes")
            })
            .collect();
        for to in kind {
            let moving: Vec<usize> = lines
                .iter()
                .zip(&nearest)
                .filter(|(l, n)| **n == to && m.lines[**l].node != *to)
                .map(|(l, _)| *l)
                .collect();
            if !moving.is_empty() {
                let to = to.clone();
                push(m, edits, Edit::Move { lines: moving, to });
            }
        }
        for n in kind {
            if let Some(e) = recentre(m, varies, n) {
                push(m, edits, e);
            }
        }
    }
}

/// `node`'s varied defaults each at the median of its sounding lines' values, where one moves.
pub(super) fn recentre(m: &Model, varies: &[Vary], node: &str) -> Option<Edit> {
    let lines: Vec<usize> = lines_of(m, node).filter(|l| alive(m, varies, *l)).collect();
    if lines.is_empty() {
        return None;
    }
    let defaults: Vec<(String, f64)> = m.nodes[node]
        .defaults
        .iter()
        .filter_map(|(name, now)| {
            let v = varied(varies, node, name)?;
            let mut th: Vec<(f64, f64)> = lines
                .iter()
                .map(|l| {
                    let x = played(m, *l, name).expect("a default");
                    (v.coord(x), x)
                })
                .collect();
            th.sort_by(|a, b| a.0.total_cmp(&b.0));
            let k = th.len();
            let median = match k % 2 {
                1 => th[k / 2].1,
                _ => v.value(0.5 * (th[k / 2 - 1].0 + th[k / 2].0)),
            };
            (median != *now).then(|| (name.clone(), median))
        })
        .collect();
    (!defaults.is_empty()).then(|| Edit::Recentre {
        node: node.to_string(),
        defaults,
    })
}
