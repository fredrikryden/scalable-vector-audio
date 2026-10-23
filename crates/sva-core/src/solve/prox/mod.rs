// Concern: the descent at one price from a start, weights off it, a debias holding every zero, each ended once flat | Non-concern: one step, an evaluation, which price | IO: (Fit, State, price) -> State

pub mod draws;
pub mod eval;
pub mod lines;
pub mod measure;
pub mod played;
pub mod price;
pub mod run;
pub mod state;
pub mod step;
pub mod write;

pub use measure::measure;
pub use run::{Settings, Shape, Solution, checked, shape, solve};

use sva_engine::Abandon;

use crate::cli_error::CliError;
use eval::{Evaluated, Fit, evaluate, scales};
use state::State;
use step::{Adam, Priced, step, zeros};

pub const DEFAULT_STEPS: usize = 600;
/// A cost bound: an evaluation's work grows with its draws.
pub const DEFAULT_DRAWS: usize = 16;
/// The exchange rate of a reading's residual into bits: each element a cell coded as a
/// Gaussian of one JND's spread, `d^2/2` nats.
pub const BITS_PER_JND2: f64 = 0.5 * std::f64::consts::LOG2_E;
pub const DEBIAS_STEPS: f64 = 150.0 / 600.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Converged,
    Cap,
    Signal,
}

impl Ended {
    pub fn name(self) -> &'static str {
        match self {
            Ended::Converged => "converged",
            Ended::Cap => "cap",
            Ended::Signal => "signal",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Descended {
    pub phase: &'static str,
    pub steps: usize,
    pub cap: usize,
    pub draws: usize,
    pub ended: Ended,
}

pub struct Progress<'a> {
    pub phase: &'a str,
    pub lambda: f64,
    pub step: usize,
    pub draws: usize,
    pub residual_bits: f64,
    pub numbers: usize,
    pub bits: f64,
    pub ended: Option<Ended>,
}

/// Stopped falling: the least objective over the later half of the `k` steps run fell less
/// than its draws' standard error. Halves shorter than Adam's first-moment memory,
/// `1/(1 - B1)` steps, are its warm-up and never read.
fn flat(least: &[f64], error: f64) -> bool {
    let k = least.len() - 1;
    k >= 2 * step::memory() && least[k / 2] - least[k] < error
}

struct Descent<'a> {
    fit: &'a Fit,
    weights: Vec<f64>,
    scales: Vec<f64>,
    abandon: &'a dyn Abandon,
    report: &'a mut dyn FnMut(&Progress),
}

impl Descent<'_> {
    async fn descend(
        &mut self,
        s: &mut State,
        adam: &mut Adam,
        lambda: f64,
        cap: usize,
        held: Option<&[bool]>,
        phase: &'static str,
    ) -> Result<Descended, CliError> {
        let draws = self.fit.draws;
        let ran = |steps, ended| Descended {
            phase,
            steps,
            cap,
            draws,
            ended,
        };
        if cap == 0 {
            return Ok(ran(0, Ended::Cap));
        }
        let mut least: Vec<f64> = Vec::with_capacity(cap + 1);
        for k in 0..=cap {
            if self.abandon.abandoned().await {
                return Ok(ran(k, Ended::Signal));
            }
            let e = evaluate(self.fit, s, k < cap, self.fit.threads)?;
            let (numbers, bits) = price::numbers(s);
            let objective = e.residual.bits + lambda * bits;
            least.push(least.last().map_or(objective, |l| l.min(objective)));
            let ended = match k == cap {
                true => Some(Ended::Cap),
                false => flat(&least, e.residual.error).then_some(Ended::Converged),
            };
            if k % 50 == 0 || ended.is_some() {
                (self.report)(&Progress {
                    phase,
                    lambda,
                    step: k,
                    draws,
                    residual_bits: e.residual.bits,
                    numbers,
                    bits,
                    ended,
                });
            }
            if let Some(ended) = ended {
                return Ok(ran(k, ended));
            }
            let grad = e.grad.expect("a gradient was asked");
            let priced = Priced {
                lambda,
                weights: &self.weights,
                scales: &self.scales,
                held,
            };
            step(s, adam, &grad, &priced);
        }
        unreachable!("the step at the cap ends the descent")
    }
}

/// Each number's price weight and gradient scale, as a descent from `s` steps by.
fn primed(fit: &Fit, s: &State, lambda: f64) -> Result<(Vec<f64>, Vec<f64>), CliError> {
    let scales = scales(fit, s)?;
    let weights = match lambda > 0.0 {
        true => price::weights(s, &scales),
        false => vec![0.0; s.x.len()],
    };
    Ok((weights, scales))
}

/// One evaluation at `s`, its gradient asked, and the state a descent's first step reaches.
pub fn stepped(fit: &Fit, s: &State, lambda: f64) -> Result<(Evaluated, State), CliError> {
    let e = evaluate(fit, s, true, fit.threads)?;
    let (weights, scales) = primed(fit, s, lambda)?;
    let priced = Priced {
        lambda,
        weights: &weights,
        scales: &scales,
        held: None,
    };
    let mut next = s.clone();
    let grad = e.grad.as_deref().expect("a gradient was asked");
    step(&mut next, &mut Adam::of(s), grad, &priced);
    Ok((e, next))
}

pub async fn descended(
    fit: &Fit,
    s: &mut State,
    lambda: f64,
    steps: usize,
    abandon: &dyn Abandon,
    report: &mut dyn FnMut(&Progress),
) -> Result<Vec<Descended>, CliError> {
    let mut adam = Adam::of(s);
    let (weights, scales) = primed(fit, s, lambda)?;
    let mut run = Descent {
        fit,
        weights,
        scales,
        abandon,
        report,
    };
    let descent = run
        .descend(s, &mut adam, lambda, steps, None, "descend")
        .await?;
    let mut ran = vec![descent];
    if lambda > 0.0 && ran[0].ended != Ended::Signal {
        let held = zeros(s, &run.weights);
        let debias = match steps {
            0 => 0,
            _ => ((steps as f64 * DEBIAS_STEPS).round() as usize).max(1),
        };
        ran.push(
            run.descend(s, &mut adam, 0.0, debias, Some(&held), "debias")
                .await?,
        );
    }
    Ok(ran)
}
