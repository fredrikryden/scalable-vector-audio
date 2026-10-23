// Concern: the splits a composition offers: a node's lines halved by pitch with a new node of its kind | Non-concern: scoring them | IO: (Model, varies) -> Vec<Proposal>

use super::assign::{reassign, recentre};
use super::{Edit, Proposal, alive, lines_of, played, push};
use crate::model::Model;
use crate::solve::Vary;

/// One split per node but the root with two sounding lines or more.
pub fn proposals(model: &Model, varies: &[Vary]) -> Vec<Proposal> {
    let sounding = |m: &Model, n: &str| -> Vec<usize> {
        lines_of(m, n).filter(|l| alive(m, varies, *l)).collect()
    };
    model
        .nodes
        .keys()
        .filter(|n| **n != model.root && sounding(model, n).len() > 1)
        .map(|a| {
            let (mut m, mut edits) = (model.clone(), Vec::new());
            let b = fresh(model, a);
            let copy = Edit::Copy {
                from: a.clone(),
                path: b.clone(),
            };
            push(&mut m, &mut edits, copy);
            let mut lines = sounding(&m, a);
            lines.sort_by(|x, y| pitch(&m, *x).total_cmp(&pitch(&m, *y)));
            let upper = lines[lines.len() / 2..].to_vec();
            let to = b.clone();
            push(&mut m, &mut edits, Edit::Move { lines: upper, to });
            for n in [a, &b] {
                if let Some(e) = recentre(&m, varies, n) {
                    push(&mut m, &mut edits, e);
                }
            }
            reassign(&mut m, varies, &[a.clone(), b], &mut edits);
            Proposal {
                label: format!("split {a}"),
                edits,
            }
        })
        .collect()
}

fn pitch(m: &Model, line: usize) -> f64 {
    played(m, line, "f0").unwrap_or(m.lines[line].at)
}

/// `<source>-<k>` for the least free `k` from 2, `source` the kin node `node` was copied from.
fn fresh(model: &Model, node: &str) -> String {
    let source = node
        .rsplit_once('-')
        .filter(|(s, k)| {
            !k.is_empty()
                && k.chars().all(|c| c.is_ascii_digit())
                && model.nodes.contains_key(*s)
                && model.kin(s, node)
        })
        .map_or(node, |(s, _)| s);
    (2..)
        .map(|k| format!("{source}-{k}"))
        .find(|p| !model.nodes.contains_key(p))
        .expect("a free path")
}
