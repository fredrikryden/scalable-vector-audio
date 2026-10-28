// Concern: the blind optimizer: passes of the best DL drop, the stop rule, kept things, shared variables, the ledger pricing moves | Non-concern: sound, threads | IO: (Problem, State) -> State, End

pub(crate) mod code;

use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;

use code::{Price, count};

/// How many candidates one turn fits, how a turn runs them, and whether the run is stopped.
pub(crate) trait Pace {
    fn width(&self) -> NonZeroUsize;
    fn each<T: Send, R: Send>(&self, tasks: Vec<T>, run: impl Fn(T) -> R + Sync) -> Vec<R>;
    fn stopped(&self) -> Pin<Box<dyn Future<Output = bool> + '_>>;
}

pub(crate) trait Problem: Sync + Sized {
    type Thing: Clone + Send + Sync;
    type Draft: Send;
    type Label: Copy + Send;
    type Candidate: Send;
    type Reading: Sync;

    fn missed(&self, kept: &[Self::Thing], shared: &[Shared]) -> f64;
    fn price(&self, thing: &Self::Thing) -> Price;
    fn attached(&self, thing: &Self::Thing) -> Vec<usize>;
    /// One unused spare of each kind of shared variable, the kind its index.
    fn spares(&self) -> Vec<Shared>;
    /// Shared variable `s`'s price, `kept` the things.
    fn price_shared(&self, s: usize, kept: &[Self::Thing]) -> Price;
    fn read(&self, kept: &[Self::Thing], shared: &[Shared]) -> Self::Reading;
    fn propose(&self, kept: &[Self::Thing], reading: &Self::Reading) -> Vec<Self::Candidate>;
    /// `candidate`'s moves, each priced by `ledger`.
    fn fit(
        &self,
        kept: &[Self::Thing],
        ledger: &Ledger,
        reading: &Self::Reading,
        key: u64,
        candidate: Self::Candidate,
    ) -> Vec<Scored<Self>>;
    fn make(&self, reading: &Self::Reading, key: u64, draft: Self::Draft) -> Self::Thing;
    /// A move whose thing is not admitted is skipped for the next best.
    fn admits(&self, thing: &Self::Thing) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Edit<A, T, L> {
    Add(A),
    Replace(usize, T, L),
    Remove(usize, L),
}

/// The edit a pass made, its things left out.
pub(crate) type Moved<L> = Edit<(), (), L>;

/// A list of numbers several things reference, each thing its own attachment numbers in its
/// own price; its kind the index of the spare `Problem::spares` gives.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Shared {
    pub(crate) kind: usize,
    pub(crate) numbers: Vec<f64>,
}

/// An add beside what its fit also sets: shared variables' numbers, and kept things other than
/// it in their attachment numbers.
pub(crate) struct Added<P: Problem> {
    pub(crate) draft: P::Draft,
    pub(crate) shared: Vec<(usize, Shared)>,
    pub(crate) others: Vec<(usize, P::Thing)>,
}

impl<P: Problem> Added<P> {
    pub(crate) fn alone(draft: P::Draft) -> Added<P> {
        Added {
            draft,
            shared: Vec::new(),
            others: Vec::new(),
        }
    }
}

pub(crate) struct Scored<P: Problem> {
    pub(crate) change: f64,
    pub(crate) edit: Edit<Added<P>, P::Thing, P::Label>,
}

/// An edit: each thing beside the shared variables it references, and shared variables'
/// prices after it.
#[derive(Default)]
pub(crate) struct Change<'a> {
    pub(crate) added: Option<(&'a Price, &'a [usize])>,
    pub(crate) removed: Option<usize>,
    pub(crate) replaced: Vec<(usize, &'a Price, &'a [usize])>,
    pub(crate) shared: Vec<(usize, &'a Price)>,
}

/// The one place a move's DL change is priced: the things' count and prices; per kind the
/// count of active shared variables, each thing's choice of none or one, each active one's
/// price once, a spare 0 bits. `missed` is the problem's change in the missed bits.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Ledger {
    bits: Vec<f64>,
    uses: Vec<Vec<usize>>,
    shared: Vec<(usize, f64)>,
    kinds: usize,
}

impl Ledger {
    fn of(
        prices: &[Price],
        uses: Vec<Vec<usize>>,
        shared: Vec<(usize, f64)>,
        kinds: usize,
    ) -> Ledger {
        Ledger {
            bits: prices.iter().map(Price::bits).collect(),
            uses,
            shared,
            kinds,
        }
    }

    fn total(&self, things: usize, priced: f64, refs: &[usize], shared: &[(usize, f64)]) -> f64 {
        let mut active = vec![0; self.kinds];
        let mut held = 0.0;
        for ((kind, bits), n) in shared.iter().zip(refs) {
            if *n > 0 {
                active[*kind] += 1;
                held += bits;
            }
        }
        let chosen: f64 = active
            .iter()
            .map(|n| count(*n) + things as f64 * libm::log2(*n as f64 + 1.0))
            .sum();
        count(things) + priced + chosen + held
    }

    fn refs(&self) -> Vec<usize> {
        let mut refs = vec![0; self.shared.len()];
        for s in self.uses.iter().flatten() {
            refs[*s] += 1;
        }
        refs
    }

    fn bits(&self) -> f64 {
        let priced = self.bits.iter().sum();
        self.total(self.bits.len(), priced, &self.refs(), &self.shared)
    }

    pub(crate) fn changed(&self, change: &Change, missed: f64) -> f64 {
        let (mut things, mut priced) = (self.bits.len(), 0.0);
        let mut refs = self.refs();
        let mut shared = self.shared.clone();
        let mut drop = |a: usize, refs: &mut Vec<usize>| {
            priced -= self.bits[a];
            for s in &self.uses[a] {
                refs[*s] -= 1;
            }
        };
        if let Some(a) = change.removed {
            drop(a, &mut refs);
            things -= 1;
        }
        for (a, _, _) in &change.replaced {
            drop(*a, &mut refs);
        }
        let mut take = |price: &Price, uses: &[usize], refs: &mut Vec<usize>| {
            priced += price.bits();
            for s in uses {
                refs[*s] += 1;
            }
        };
        for (_, price, uses) in &change.replaced {
            take(price, uses, &mut refs);
        }
        if let Some((price, uses)) = change.added {
            take(price, uses, &mut refs);
            things += 1;
        }
        for (s, price) in &change.shared {
            shared[*s].1 = price.bits();
        }
        let before = self.bits();
        let after = self.bits.iter().sum::<f64>() + priced;
        self.total(things, after, &refs, &shared) - before + missed
    }

    pub(crate) fn added(&self, price: &Price, missed: f64) -> f64 {
        let change = Change {
            added: Some((price, &[])),
            ..Change::default()
        };
        self.changed(&change, missed)
    }

    pub(crate) fn without(&self, a: usize) -> Ledger {
        let mut out = self.clone();
        out.bits.remove(a);
        out.uses.remove(a);
        out
    }
}

pub(crate) struct State<P: Problem> {
    kept: Vec<P::Thing>,
    shared: Vec<Shared>,
    born: u64,
    passes: usize,
    ledger: Ledger,
    missed_bits: f64,
    moved: Option<Moved<P::Label>>,
    skipped: Vec<P::Thing>,
}

impl<P: Problem> Clone for State<P> {
    fn clone(&self) -> State<P> {
        State {
            kept: self.kept.clone(),
            shared: self.shared.clone(),
            born: self.born,
            passes: self.passes,
            ledger: self.ledger.clone(),
            missed_bits: self.missed_bits,
            moved: self.moved,
            skipped: self.skipped.clone(),
        }
    }
}

impl<P: Problem> State<P> {
    pub(crate) fn of(
        problem: &P,
        kept: Vec<P::Thing>,
        shared: Vec<Shared>,
        born: u64,
        passes: usize,
    ) -> State<P> {
        let mut state = State {
            kept,
            shared,
            born,
            passes,
            ledger: Ledger::of(&[], Vec::new(), Vec::new(), 0),
            missed_bits: 0.0,
            moved: None,
            skipped: Vec::new(),
        };
        state.price(problem);
        state
    }

    pub(crate) fn kept(&self) -> &[P::Thing] {
        &self.kept
    }

    pub(crate) fn shared(&self) -> &[Shared] {
        &self.shared
    }

    pub(crate) fn passes(&self) -> usize {
        self.passes
    }

    pub(crate) fn thing_bits(&self) -> f64 {
        self.ledger.bits()
    }

    pub(crate) fn missed_bits(&self) -> f64 {
        self.missed_bits
    }

    pub(crate) fn moved(&self) -> Option<Moved<P::Label>> {
        self.moved
    }

    pub(crate) fn skipped(&self) -> &[P::Thing] {
        &self.skipped
    }

    pub(crate) fn dl(&self) -> f64 {
        self.thing_bits() + self.missed_bits
    }

    /// The ledger afresh, a spare added of each kind none is free of.
    fn price(&mut self, problem: &P) {
        let uses: Vec<Vec<usize>> = self.kept.iter().map(|t| problem.attached(t)).collect();
        let spares = problem.spares();
        for (kind, spare) in spares.iter().enumerate() {
            let free = (0..self.shared.len())
                .any(|s| self.shared[s].kind == kind && !uses.iter().flatten().any(|u| *u == s));
            if !free {
                self.shared.push(spare.clone());
            }
        }
        let shared = (0..self.shared.len())
            .map(|s| {
                let held = uses.iter().flatten().any(|u| *u == s);
                let bits = match held {
                    true => problem.price_shared(s, &self.kept).bits(),
                    false => 0.0,
                };
                (self.shared[s].kind, bits)
            })
            .collect();
        let prices: Vec<Price> = self.kept.iter().map(|t| problem.price(t)).collect();
        self.ledger = Ledger::of(&prices, uses, shared, spares.len());
        self.missed_bits = problem.missed(&self.kept, &self.shared);
    }
}

pub const TOLERANCE: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Converged,
    Tolerance,
    Stopped,
}

impl End {
    pub fn name(self) -> &'static str {
        match self {
            End::Converged => "converged",
            End::Tolerance => "tolerance",
            End::Stopped => "stopped",
        }
    }
}

/// `tasks` run `pace.width()` at a time; `None` if `pace` stops before a turn.
async fn paced<T: Send, R: Send>(
    pace: &impl Pace,
    tasks: Vec<T>,
    run: impl Fn(T) -> R + Sync,
) -> Option<Vec<R>> {
    let mut tasks = tasks.into_iter().peekable();
    let mut out = Vec::new();
    while tasks.peek().is_some() {
        if pace.stopped().await {
            return None;
        }
        let turn: Vec<T> = tasks.by_ref().take(pace.width().get()).collect();
        out.extend(pace.each(turn, &run));
    }
    Some(out)
}

/// One pass: the move lowering the DL most, ties in the moves' order; `None` if `pace` stops it.
async fn pass<P: Problem>(problem: &P, state: &State<P>, pace: &impl Pace) -> Option<State<P>> {
    let (kept, key, ledger) = (&state.kept[..], state.born, &state.ledger);
    let reading = problem.read(kept, &state.shared);
    let candidates = problem.propose(kept, &reading);
    let fitted = paced(pace, candidates, |candidate| {
        problem.fit(kept, ledger, &reading, key, candidate)
    })
    .await?;
    let mut ranked: Vec<Scored<P>> = fitted
        .into_iter()
        .flatten()
        .filter(|s| s.change < 0.0)
        .collect();
    // Stable, so ties keep the moves' order.
    ranked.sort_by(|x, y| x.change.total_cmp(&y.change));
    let mut next = state.clone();
    next.moved = None;
    next.skipped = Vec::new();
    let mut best = None;
    for Scored { edit, .. } in ranked {
        let made = match edit {
            Edit::Add(added) => Edit::Add((
                problem.make(&reading, key, added.draft),
                added.shared,
                added.others,
            )),
            Edit::Replace(a, thing, label) => Edit::Replace(a, thing, label),
            Edit::Remove(a, label) => Edit::Remove(a, label),
        };
        match made {
            Edit::Add((thing, ..)) | Edit::Replace(_, thing, _) if !problem.admits(&thing) => {
                next.skipped.push(thing);
            }
            Edit::Add((_, _, others)) if others.iter().any(|(_, t)| !problem.admits(t)) => {
                next.skipped.extend(others.into_iter().map(|(_, t)| t));
            }
            made => {
                best = Some(made);
                break;
            }
        }
    }
    let Some(made) = best else {
        return Some(next);
    };
    let moved = match made {
        Edit::Add((thing, shared, others)) => {
            for (a, other) in others {
                next.kept[a] = other;
            }
            for (s, to) in shared {
                next.shared[s] = to;
            }
            next.kept.push(thing);
            next.born += 1;
            Edit::Add(())
        }
        Edit::Replace(a, thing, label) => {
            next.kept[a] = thing;
            Edit::Replace(a, (), label)
        }
        Edit::Remove(a, label) => {
            next.kept.remove(a);
            Edit::Remove(a, label)
        }
    };
    next.price(problem);
    // A drop under the rounding of the repricing is no drop.
    if next.dl() >= state.dl() {
        let skipped = std::mem::take(&mut next.skipped);
        next = state.clone();
        next.moved = None;
        next.skipped = skipped;
        return Some(next);
    }
    next.moved = Some(moved);
    next.passes += 1;
    Some(next)
}

pub(crate) type Heard<'a, P> = dyn FnMut(&State<P>, &State<P>, f64) + 'a;

/// Passes till one makes no move, drops the DL by no more than `tolerance` times the larger of
/// the DL before and after it and 1, or `pace` stops it.
pub(crate) async fn run<P: Problem>(
    problem: &P,
    mut state: State<P>,
    tolerance: f64,
    pace: &impl Pace,
    clock: &dyn Fn() -> f64,
    heard: &mut Heard<'_, P>,
) -> (State<P>, End) {
    let ended = loop {
        if pace.stopped().await {
            break End::Stopped;
        }
        let began = clock();
        let Some(next) = pass(problem, &state, pace).await else {
            break End::Stopped;
        };
        if next.moved.is_none() {
            break End::Converged;
        }
        let (before, after) = (state.dl(), next.dl());
        heard(&state, &next, began);
        state = next;
        // Optimisers' `ftol` (scipy.optimize's L-BFGS-B).
        if before - after <= tolerance * before.abs().max(after.abs()).max(1.0) {
            break End::Tolerance;
        }
    };
    (state, ended)
}
