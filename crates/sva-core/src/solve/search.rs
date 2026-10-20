// Concern: a bounded compass search, asked for each point and told its cost, steps halving onto a grid | Non-concern: what a point costs (mod.rs) | IO: (axes, stops) -> points

use super::vary::{Precision, Vary};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stopped {
    Converged,
    Budget,
    Cancelled,
}

impl Stopped {
    pub fn name(self) -> &'static str {
        match self {
            Stopped::Converged => "converged",
            Stopped::Budget => "budget",
            Stopped::Cancelled => "cancelled",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stops {
    pub max_renders: usize,
    /// Each axis stops once its step is under this fraction of its first step.
    pub step_tol: f64,
    pub cost_tol: f64,
}

pub const DEFAULT_MAX_RENDERS: usize = 200;
pub const DEFAULT_STEP_TOL: f64 = 1.0 / 64.0;
pub const DEFAULT_COST_TOL: f64 = 0.0;
/// Without `/<step>`, the first step is this fraction of the bounds' span.
pub const FIRST_STEP_OF_SPAN: f64 = 0.25;

/// One variable's coordinate: itself, or its log on a log scale.
#[derive(Clone, Debug)]
struct Axis {
    lo: f64,
    hi: f64,
    log: bool,
    first: f64,
    /// Decimals the first step is written to, where values land on a decimal grid.
    decimals: Option<usize>,
}

impl Axis {
    fn of(vary: &Vary, start: f64) -> Axis {
        let (lo, hi) = match vary.log {
            true => (libm::log(vary.lo), libm::log(vary.hi)),
            false => (vary.lo, vary.hi),
        };
        let first = match (vary.step, vary.log) {
            (Some(ratio), true) => libm::log(ratio),
            (Some(step), false) => step,
            (None, _) => (hi - lo) * FIRST_STEP_OF_SPAN,
        };
        let decimals = match vary.log {
            false => Some(decimals(first).max(decimals(vary.lo)).max(decimals(start))),
            true => None,
        };
        Axis {
            lo,
            hi,
            log: vary.log,
            first,
            decimals,
        }
    }

    fn value(&self, u: f64) -> f64 {
        match self.log {
            true => libm::exp(u).clamp(libm::exp(self.lo), libm::exp(self.hi)),
            false => u,
        }
    }

    fn coordinate(&self, value: f64) -> f64 {
        match self.log {
            true => libm::log(value),
            false => value,
        }
    }

    /// On the grid `step` lays from the low bound, inside the bounds.
    fn snapped(&self, u: f64, step: f64) -> f64 {
        let k = ((u - self.lo) / step).round();
        (self.lo + k * step).clamp(self.lo, self.hi)
    }
}

fn decimals(v: f64) -> usize {
    let text = format!("{v}");
    text.split_once('.').map_or(0, |(_, frac)| frac.len())
}

pub struct Search {
    axes: Vec<Axis>,
    stops: Stops,
    x: Vec<f64>,
    best: Option<f64>,
    steps: Vec<f64>,
    halvings: u32,
    poll: Vec<Vec<f64>>,
    polled: Vec<(Vec<f64>, f64)>,
    seen: Vec<(Vec<f64>, f64)>,
    asked: Option<Vec<f64>>,
    renders: usize,
    stopped: Option<Stopped>,
}

pub enum Next {
    /// The values to render at, one per variable, in the order they were given.
    Render(Vec<f64>),
    Done(Stopped),
}

impl Search {
    pub fn new(vary: &[Vary], start: &[f64], stops: Stops) -> Search {
        let axes: Vec<Axis> = vary
            .iter()
            .zip(start)
            .map(|(v, s)| Axis::of(v, *s))
            .collect();
        let x = axes
            .iter()
            .zip(vary.iter().zip(start))
            .map(|(axis, (one, v))| {
                let u = axis.coordinate(*v).clamp(axis.lo, axis.hi);
                match (one.step, one.log) {
                    (Some(_), false) => axis.snapped(u, axis.first),
                    _ => u,
                }
            })
            .collect();
        let steps = axes.iter().map(|a| a.first).collect();
        Search {
            axes,
            stops,
            x,
            best: None,
            steps,
            halvings: 0,
            poll: Vec::new(),
            polled: Vec::new(),
            seen: Vec::new(),
            asked: None,
            renders: 0,
            stopped: None,
        }
    }

    pub fn values(&self, u: &[f64]) -> Vec<f64> {
        self.axes.iter().zip(u).map(|(a, u)| a.value(*u)).collect()
    }

    /// A linear grid's decimals, finer by one per halving; on a log scale, enough significant
    /// digits that a step still moves the value.
    pub fn digits(&self) -> Vec<Precision> {
        self.axes
            .iter()
            .zip(&self.steps)
            .map(|(a, step)| match a.decimals {
                Some(d) => Precision::Decimals(d + self.halvings as usize),
                None => {
                    Precision::Significant((2.0 - libm::log10(*step).floor()).max(3.0) as usize)
                }
            })
            .collect()
    }

    pub fn cancel(&mut self) {
        self.stopped = Some(Stopped::Cancelled);
    }

    pub fn tell(&mut self, cost: f64) {
        let point = self
            .asked
            .take()
            .expect("told the cost of a point it asked");
        self.renders += 1;
        self.seen.push((point.clone(), cost));
        match self.best {
            None => self.best = Some(cost),
            Some(_) => self.polled.push((point, cost)),
        }
    }

    pub fn next(&mut self) -> Next {
        loop {
            if let Some(stopped) = self.stopped {
                return Next::Done(stopped);
            }
            if self.best.is_none() {
                return self.render(self.x.clone());
            }
            if let Some(point) = self.poll.pop() {
                match self.seen.iter().find(|(p, _)| *p == point) {
                    Some((_, cost)) => self.polled.push((point, *cost)),
                    None => return self.render(point),
                }
                continue;
            }
            if !self.polled.is_empty() {
                self.moved();
                continue;
            }
            let best = self.best.expect("a first cost");
            let floors = self.axes.iter().map(|a| a.first * self.stops.step_tol);
            let fine = self.steps.iter().zip(floors).all(|(s, floor)| *s < floor);
            if best <= self.stops.cost_tol || fine {
                self.stopped = Some(Stopped::Converged);
                continue;
            }
            self.poll = self.candidates();
            if self.poll.is_empty() {
                self.halve();
            }
        }
    }

    fn render(&mut self, point: Vec<f64>) -> Next {
        if self.renders >= self.stops.max_renders {
            self.stopped = Some(Stopped::Budget);
            return Next::Done(Stopped::Budget);
        }
        self.asked = Some(point.clone());
        Next::Render(self.values(&point))
    }

    /// `+step` then `-step` on each axis in order, reversed so `pop` takes them in order.
    fn candidates(&self) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        for (i, axis) in self.axes.iter().enumerate() {
            for sign in [1.0, -1.0] {
                let mut y = self.x.clone();
                y[i] = axis.snapped(self.x[i] + sign * self.steps[i], self.steps[i]);
                if y[i] != self.x[i] && !out.contains(&y) {
                    out.push(y);
                }
            }
        }
        out.reverse();
        out
    }

    /// The best of a whole poll, ties to the earliest, moves the point; else every step halves.
    fn moved(&mut self) {
        let polled = std::mem::take(&mut self.polled);
        let best = self.best.expect("a first cost");
        let winner = polled
            .into_iter()
            .fold(None::<(Vec<f64>, f64)>, |held, (p, c)| match &held {
                Some((_, h)) if *h <= c => held,
                _ => Some((p, c)),
            });
        match winner {
            Some((point, cost)) if cost < best => {
                self.x = point;
                self.best = Some(cost);
            }
            _ => self.halve(),
        }
    }

    fn halve(&mut self) {
        self.steps.iter_mut().for_each(|s| *s /= 2.0);
        self.halvings += 1;
    }
}
