// Concern: the blind optimizer: one move per pass, the best DL drop winning, the stop rule, the kept things and their bits, a resume | Non-concern: sound (fit.rs) | IO: (Problem, State) -> State, End

pub mod code;

use std::num::NonZeroUsize;

use sva_engine::{Abandon, each};

use code::Price;

/// What the optimizer reaches a problem through.
pub trait Problem: Sync + Sized {
    type Thing: Clone + Send + Sync;
    type Draft: Send;
    type Label: Copy + Send;
    type Candidate: Send;
    type Reading: Sync;

    fn missed(&self, kept: &[Self::Thing]) -> f64;
    fn price(&self, thing: &Self::Thing) -> Price;
    fn read(&self, kept: &[Self::Thing]) -> Self::Reading;
    fn propose(&self, kept: &[Self::Thing], reading: &Self::Reading) -> Vec<Self::Candidate>;
    /// `candidate`'s moves, or a candidate fitted next round in its place.
    fn fit(
        &self,
        kept: &[Self::Thing],
        reading: &Self::Reading,
        key: u64,
        candidate: Self::Candidate,
    ) -> Fitting<Self>;
    fn make(
        &self,
        kept: &[Self::Thing],
        reading: &Self::Reading,
        key: u64,
        draft: Self::Draft,
    ) -> Self::Thing;
    /// A move whose thing is not admitted is skipped for the next best.
    fn admits(&self, thing: &Self::Thing) -> bool;
}

pub enum Edit<A, T, L> {
    Add(A),
    Replace(usize, T, L),
    Remove(usize, L),
}

pub struct Scored<P: Problem> {
    pub change: f64,
    pub edit: Edit<P::Draft, P::Thing, P::Label>,
}

pub enum Fitting<P: Problem> {
    Moves(Vec<Scored<P>>),
    Again(P::Candidate),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Moved<L> {
    Add,
    Replace(usize, L),
    Remove(usize, L),
}

pub struct State<P: Problem> {
    pub kept: Vec<P::Thing>,
    pub born: u64,
    pub passes: usize,
    pub thing_bits: f64,
    pub missed_bits: f64,
    pub moved: Option<Moved<P::Label>>,
    pub skipped: Vec<P::Thing>,
}

impl<P: Problem> Clone for State<P> {
    fn clone(&self) -> State<P> {
        State {
            kept: self.kept.clone(),
            born: self.born,
            passes: self.passes,
            thing_bits: self.thing_bits,
            missed_bits: self.missed_bits,
            moved: self.moved,
            skipped: self.skipped.clone(),
        }
    }
}

impl<P: Problem> State<P> {
    pub fn of(problem: &P, kept: Vec<P::Thing>, born: u64, passes: usize) -> State<P> {
        let mut state = State {
            kept,
            born,
            passes,
            thing_bits: 0.0,
            missed_bits: 0.0,
            moved: None,
            skipped: Vec::new(),
        };
        state.price(problem);
        state
    }

    pub fn dl(&self) -> f64 {
        self.thing_bits + self.missed_bits
    }

    fn price(&mut self, problem: &P) {
        let prices: Vec<Price> = self.kept.iter().map(|t| problem.price(t)).collect();
        self.thing_bits = code::bits(&prices);
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
    let (kept, key) = (&state.kept[..], state.born);
    let reading = problem.read(kept);
    let mut round: Vec<(usize, P::Candidate)> = problem
        .propose(kept, &reading)
        .into_iter()
        .enumerate()
        .collect();
    let mut fitted: Vec<Vec<Scored<P>>> = round.iter().map(|_| Vec::new()).collect();
    while !round.is_empty() {
        let fits = paced(threads, stop, round, |(i, candidate)| {
            (i, problem.fit(kept, &reading, key, candidate))
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
            Edit::Add(draft) => Edit::Add(problem.make(kept, &reading, key, draft)),
            Edit::Replace(a, thing, label) => Edit::Replace(a, thing, label),
            Edit::Remove(a, label) => Edit::Remove(a, label),
        };
        match made {
            Edit::Add(thing) | Edit::Replace(_, thing, _) if !problem.admits(&thing) => {
                next.skipped.push(thing);
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
        Edit::Add(thing) => {
            next.kept.push(thing);
            next.born += 1;
            Moved::Add
        }
        Edit::Replace(a, thing, label) => {
            next.kept[a] = thing;
            Moved::Replace(a, label)
        }
        Edit::Remove(a, label) => {
            next.kept.remove(a);
            Moved::Remove(a, label)
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
