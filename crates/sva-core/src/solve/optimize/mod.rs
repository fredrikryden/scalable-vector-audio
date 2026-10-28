// Concern: the blind optimizer: passes of the best DL drop, the stop rule, kept things, shared variables, the ledger pricing moves, a resume | Non-concern: sound | IO: (Problem, State) -> State, End

pub mod code;

use std::num::NonZeroUsize;

use sva_engine::{Abandon, each};

use code::{Price, count};

/// What the optimizer reaches a problem through.
pub trait Problem: Sync + Sized {
    type Thing: Clone + Send + Sync;
    type Draft: Send;
    type Label: Copy + Send;
    type Candidate: Send;
    type Reading: Sync;

    fn missed(&self, kept: &[Self::Thing]) -> f64;
    fn price(&self, thing: &Self::Thing) -> Price;
    /// The shared variables `thing` references.
    fn attached(&self, thing: &Self::Thing) -> Vec<usize>;
    /// One unused spare of each kind of shared variable, the kind its index.
    fn spares(&self) -> Vec<Shared>;
    /// Shared variable `s`'s price, `kept` the things.
    fn price_shared(&self, s: usize, kept: &[Self::Thing]) -> Price;
    fn read(&self, kept: &[Self::Thing], shared: &[Shared]) -> Self::Reading;
    fn propose(&self, kept: &[Self::Thing], reading: &Self::Reading) -> Vec<Self::Candidate>;
    /// `candidate`'s moves, each priced by `ledger`, or a candidate fitted next round in its place.
    fn fit(
        &self,
        kept: &[Self::Thing],
        ledger: &Ledger,
        reading: &Self::Reading,
        key: u64,
        candidate: Self::Candidate,
    ) -> Fitting<Self>;
    fn make(&self, reading: &Self::Reading, key: u64, draft: Self::Draft) -> Self::Thing;
    /// A move whose thing is not admitted is skipped for the next best.
    fn admits(&self, thing: &Self::Thing) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edit<A, T, L> {
    Add(A),
    Replace(usize, T, L),
    Remove(usize, L),
}

/// The edit a pass made, its things left out.
pub type Moved<L> = Edit<(), (), L>;

/// A list of numbers several things reference, each thing its own attachment numbers in its
/// own price; its kind the index of the spare `Problem::spares` gives.
#[derive(Clone, Debug, PartialEq)]
pub struct Shared {
    pub kind: usize,
    pub numbers: Vec<f64>,
}

/// An add beside what its fit also sets: shared variables' numbers, and kept things other than
/// it in their attachment numbers, every thing attached to a variable set among them.
pub struct Added<P: Problem> {
    pub draft: P::Draft,
    pub shared: Vec<(usize, Shared)>,
    pub others: Vec<(usize, P::Thing)>,
}

impl<P: Problem> Added<P> {
    pub fn alone(draft: P::Draft) -> Added<P> {
        Added {
            draft,
            shared: Vec::new(),
            others: Vec::new(),
        }
    }
}

pub struct Scored<P: Problem> {
    pub change: f64,
    pub edit: Edit<Added<P>, P::Thing, P::Label>,
}

pub enum Fitting<P: Problem> {
    Moves(Vec<Scored<P>>),
    Again(P::Candidate),
}

/// An edit: each thing beside the shared variables it references, and shared variables'
/// prices after it.
#[derive(Default)]
pub struct Change<'a> {
    pub added: Option<(&'a Price, &'a [usize])>,
    pub removed: Option<usize>,
    pub replaced: Vec<(usize, &'a Price, &'a [usize])>,
    pub shared: Vec<(usize, &'a Price)>,
}

/// The one place a move's DL change is priced: the things' count and prices; per kind the
/// count of active shared variables, each thing's choice of none or one, each active one's
/// price once, a spare 0 bits. `missed` is the problem's change in the missed bits.
#[derive(Clone, Debug, PartialEq)]
pub struct Ledger {
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

    pub fn bits(&self) -> f64 {
        let priced = self.bits.iter().sum();
        self.total(self.bits.len(), priced, &self.refs(), &self.shared)
    }

    pub fn changed(&self, change: &Change, missed: f64) -> f64 {
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

    pub fn added(&self, price: &Price, missed: f64) -> f64 {
        let change = Change {
            added: Some((price, &[])),
            ..Change::default()
        };
        self.changed(&change, missed)
    }

    pub fn removed(&self, a: usize, missed: f64) -> f64 {
        let change = Change {
            removed: Some(a),
            ..Change::default()
        };
        self.changed(&change, missed)
    }

    pub fn replaced(&self, a: usize, after: &Price, missed: f64) -> f64 {
        let change = Change {
            replaced: vec![(a, after, &self.uses[a])],
            ..Change::default()
        };
        self.changed(&change, missed)
    }

    /// The ledger with thing `a` gone.
    pub fn without(&self, a: usize) -> Ledger {
        let mut out = self.clone();
        out.bits.remove(a);
        out.uses.remove(a);
        out
    }
}

pub struct State<P: Problem> {
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
    pub fn of(
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

    pub fn kept(&self) -> &[P::Thing] {
        &self.kept
    }

    pub fn shared(&self) -> &[Shared] {
        &self.shared
    }

    pub fn passes(&self) -> usize {
        self.passes
    }

    pub fn thing_bits(&self) -> f64 {
        self.ledger.bits()
    }

    pub fn missed_bits(&self) -> f64 {
        self.missed_bits
    }

    pub fn moved(&self) -> Option<Moved<P::Label>> {
        self.moved
    }

    pub fn skipped(&self) -> &[P::Thing] {
        &self.skipped
    }

    pub fn dl(&self) -> f64 {
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
        self.missed_bits = problem.missed(&self.kept);
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

async fn paced<T: Send, R: Send>(
    threads: NonZeroUsize,
    stop: &dyn Abandon,
    tasks: Vec<T>,
    run: impl Fn(T) -> R + Sync,
) -> Option<Vec<R>> {
    let mut tasks = tasks.into_iter().peekable();
    let mut out = Vec::new();
    while tasks.peek().is_some() {
        if stop.abandoned().await {
            return None;
        }
        let turn: Vec<T> = tasks.by_ref().take(threads.get()).collect();
        out.extend(each(threads, turn, &run));
    }
    Some(out)
}

/// One pass: the move lowering the DL most, ties in the moves' order; `None` if `stop` ends it.
async fn pass<P: Problem>(
    problem: &P,
    state: &State<P>,
    threads: NonZeroUsize,
    stop: &dyn Abandon,
) -> Option<State<P>> {
    let (kept, key, ledger) = (&state.kept[..], state.born, &state.ledger);
    let reading = problem.read(kept, &state.shared);
    let mut round: Vec<(usize, P::Candidate)> = problem
        .propose(kept, &reading)
        .into_iter()
        .enumerate()
        .collect();
    let mut fitted: Vec<Vec<Scored<P>>> = round.iter().map(|_| Vec::new()).collect();
    while !round.is_empty() {
        let fits = paced(threads, stop, round, |(i, candidate)| {
            (i, problem.fit(kept, ledger, &reading, key, candidate))
        })
        .await?;
        round = Vec::new();
        for (i, fitting) in fits {
            match fitting {
                Fitting::Moves(moves) => fitted[i] = moves,
                Fitting::Again(candidate) => round.push((i, candidate)),
            }
        }
    }
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
            let set: Vec<usize> = shared.iter().map(|(s, _)| *s).collect();
            assert!(
                state.kept.iter().enumerate().all(|(a, t)| {
                    !problem.attached(t).iter().any(|s| set.contains(s))
                        || others.iter().any(|(o, _)| *o == a)
                }),
                "every thing attached to a variable set is set beside it"
            );
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

pub type Heard<'a, P> = dyn FnMut(&State<P>, &State<P>, f64) + 'a;

/// Passes till one makes no move, drops the DL by no more than `tolerance` times the larger of
/// the DL before and after it and 1, or `stop` ends it.
pub async fn run<P: Problem>(
    problem: &P,
    mut state: State<P>,
    tolerance: f64,
    threads: NonZeroUsize,
    stop: &dyn Abandon,
    clock: &dyn Fn() -> f64,
    heard: &mut Heard<'_, P>,
) -> (State<P>, End) {
    let ended = loop {
        if stop.abandoned().await {
            break End::Stopped;
        }
        let began = clock();
        let Some(next) = pass(problem, &state, threads, stop).await else {
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
