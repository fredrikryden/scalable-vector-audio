// Concern: the descent at one price from a start, weights off it, a debias holding every zero, each ended once flat | Non-concern: one step, an evaluation, which price | IO: (Fit, State, price) -> State

pub mod draws;
pub mod eval;
pub mod heard;
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

use std::rc::Rc;

use sva_engine::Abandon;

use crate::cli_error::CliError;
use eval::{Evaluated, Fit, evaluate, scales, standard_error};
use heard::{Act, Action, Hearing};
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
    pub noise_limited: bool,
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

struct Least {
    objective: f64,
    drawn: Rc<[f64]>,
}

/// The later half's first and last least, once halves pass Adam's memory, `1/(1 - B1)` steps.
fn halves(least: &[Least]) -> Option<(&Least, &Least)> {
    let k = least.len() - 1;
    (k >= 2 * step::memory()).then(|| (&least[k / 2], &least[k]))
}

/// Stopped falling: the later half's fall within its standard error paired over the draws.
fn flat(least: &[Least]) -> bool {
    halves(least).is_some_and(|(then, now)| {
        let fell: Vec<f64> = then
            .drawn
            .iter()
            .zip(now.drawn.iter())
            .map(|(a, b)| a - b)
            .collect();
        then.objective - now.objective <= standard_error(&fell)
    })
}

/// Limited by the draws' noise, however it ended: the later half's fall within the standard
/// error of its last least's residual over the draws.
fn noise_limited(least: &[Least]) -> bool {
    halves(least)
        .is_some_and(|(then, now)| then.objective - now.objective <= standard_error(&now.drawn))
}

struct Best {
    state: State,
    residual_bits: f64,
    numbers: usize,
    bits: f64,
}

struct Descent<'a, 'h> {
    fit: &'a Fit,
    weights: Vec<f64>,
    scales: Vec<f64>,
    abandon: &'a dyn Abandon,
    hearing: &'a Hearing<'h>,
}

impl Descent<'_, '_> {
    /// Leaves `s` at its least evaluated objective, start included: Adam may climb.
    async fn descend(
        &mut self,
        s: &mut State,
        adam: &mut Adam,
        lambda: f64,
        cap: usize,
        held: Option<&[bool]>,
        phase: &'static str,
    ) -> Result<Descended, CliError> {
        let action = match held {
            None => Action::Descending,
            Some(_) => Action::Debiasing,
        };
        let hearing = self.hearing;
        let run = self.steps(s, adam, lambda, cap, held, phase);
        hearing.during(Act::new(action).of(cap), run).await
    }

    async fn steps(
        &mut self,
        s: &mut State,
        adam: &mut Adam,
        lambda: f64,
        cap: usize,
        held: Option<&[bool]>,
        phase: &'static str,
    ) -> Result<Descended, CliError> {
        let draws = self.fit.draws;
        let ran = |steps, ended, noise_limited| Descended {
            phase,
            steps,
            cap,
            draws,
            ended,
            noise_limited,
        };
        if cap == 0 {
            return Ok(ran(0, Ended::Cap, false));
        }
        let mut least: Vec<Least> = Vec::with_capacity(cap + 1);
        let mut best: Option<Best> = None;
        for k in 0..=cap {
            if self.abandon.abandoned().await {
                if let Some(b) = best {
                    *s = b.state;
                }
                return Ok(ran(k, Ended::Signal, false));
            }
            self.hearing.tick(k);
            let e = evaluate(self.fit, s, k < cap, self.fit.threads)?;
            let (numbers, bits) = price::numbers(s);
            let objective = e.residual.bits + lambda * bits;
            let next = match least.last() {
                Some(l) if l.objective <= objective => Least {
                    objective: l.objective,
                    drawn: l.drawn.clone(),
                },
                _ => {
                    best = Some(Best {
                        state: s.clone(),
                        residual_bits: e.residual.bits,
                        numbers,
                        bits,
                    });
                    Least {
                        objective,
                        drawn: e.drawn.as_slice().into(),
                    }
                }
            };
            least.push(next);
            let ended = match k == cap {
                true => Some(Ended::Cap),
                false => flat(&least).then_some(Ended::Converged),
            };
            if k % 50 == 0 || ended.is_some() {
                let (residual_bits, numbers, bits) = match (ended, &best) {
                    (Some(_), Some(b)) => (b.residual_bits, b.numbers, b.bits),
                    _ => (e.residual.bits, numbers, bits),
                };
                self.hearing.step(&Progress {
                    phase,
                    lambda,
                    step: k,
                    draws,
                    residual_bits,
                    numbers,
                    bits,
                    ended,
                });
            }
            if let Some(ended) = ended {
                *s = best.expect("the first step sets the least").state;
                return Ok(ran(k, ended, noise_limited(&least)));
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
    hearing: &Hearing<'_>,
) -> Result<Vec<Descended>, CliError> {
    let mut adam = Adam::of(s);
    let (weights, scales) = primed(fit, s, lambda)?;
    let mut run = Descent {
        fit,
        weights,
        scales,
        abandon,
        hearing,
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
