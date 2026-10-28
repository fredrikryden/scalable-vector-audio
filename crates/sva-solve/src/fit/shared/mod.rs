// Concern: the trait each kind of shared variable meets, a link applied to a render, a variable's price, the active ones | Non-concern: one kind's math (periodic.rs) | IO: (links, render) -> render

mod periodic;

use super::problem::{Problem, Reading, State};
use super::render::{Held, Proposal};
use super::written::Written;
use crate::optimize::code::Price;
use crate::optimize::{self, Ledger, Scored};
use crate::sources::{self, Link, SharedKind};

trait Variable: Sync {
    /// A link of `own` numbers to a variable at `numbers` transforms a source's `render`.
    fn transform(&self, p: &Problem, numbers: &[f64], own: &[f64], render: &mut [Vec<f64>]);

    /// The price of a variable at `numbers` its `uses` reference.
    fn price(&self, p: &Problem, numbers: &[f64], uses: &[(&Written, &Link)]) -> Price;

    /// `None` where the clip holds no such variable.
    fn attach(
        &self,
        at: (&Problem, &[Held], &Ledger, &Reading),
        key: u64,
        proposal: (&Proposal, f64),
        s: usize,
    ) -> Option<Scored<Problem>>;
}

fn of(kind: SharedKind) -> &'static dyn Variable {
    match kind {
        SharedKind::Periodic => &periodic::Periodic,
    }
}

pub(super) fn linked(
    p: &Problem,
    links: &[Link],
    shared: &[optimize::Shared],
    render: &mut [Vec<f64>],
) {
    for link in links {
        let to = &shared[link.to];
        assert_eq!(link.kind as usize, to.kind, "a link's kind");
        of(link.kind).transform(p, &to.numbers, &link.own, render);
    }
}

pub(super) fn price<'a>(
    p: &Problem,
    written: impl Iterator<Item = &'a Written>,
    variable: &optimize::Shared,
    s: usize,
) -> Price {
    let uses: Vec<(&Written, &Link)> = written
        .flat_map(|w| {
            w.source
                .links
                .iter()
                .filter(|l| l.to == s)
                .map(move |l| (w, l))
        })
        .collect();
    of(SharedKind::ALL[variable.kind]).price(p, &variable.numbers, &uses)
}

pub(super) fn attach(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    key: u64,
    proposal: (&Proposal, f64),
    s: usize,
) -> Option<Scored<Problem>> {
    let kind = SharedKind::ALL[reading.shared[s].kind];
    of(kind).attach((p, kept, ledger, reading), key, proposal, s)
}

fn members(kept: &[Held], s: usize) -> Vec<usize> {
    (0..kept.len())
        .filter(|a| kept[*a].links().iter().any(|l| l.to == s))
        .collect()
}

pub(super) fn active(state: &State) -> Vec<(usize, sources::Shared)> {
    let shared = state.shared();
    (0..shared.len())
        .filter(|s| !members(state.kept(), *s).is_empty())
        .map(|s| {
            let node = sources::Shared {
                kind: SharedKind::ALL[shared[s].kind],
                numbers: shared[s].numbers.clone(),
            };
            (s, node)
        })
        .collect()
}
