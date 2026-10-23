// Concern: the merges a composition offers: one node's lines folded into another of its kind | Non-concern: scoring them | IO: (Model, varies) -> Vec<Proposal>

use super::{Edit, Proposal, alive, lines_of};
use crate::model::Model;
use crate::solve::Vary;

/// One merge per ordered pair of kin nodes but the root that each have a sounding line: every
/// line of the first moved to the second.
pub fn proposals(model: &Model, varies: &[Vary]) -> Vec<Proposal> {
    let used: Vec<&String> = model
        .nodes
        .keys()
        .filter(|n| **n != model.root && lines_of(model, n).any(|l| alive(model, varies, l)))
        .collect();
    let mut out = Vec::new();
    for a in &used {
        for b in &used {
            if a != b && model.kin(a, b) {
                out.push(Proposal {
                    label: format!("merge {a} into {b}"),
                    edits: vec![Edit::Move {
                        lines: lines_of(model, a).collect(),
                        to: b.to_string(),
                    }],
                });
            }
        }
    }
    out
}
