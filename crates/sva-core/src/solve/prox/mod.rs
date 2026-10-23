// Concern: the descent at one price from a given start, its weights off that start, a debias holding every zero | Non-concern: one step, one evaluation, which price | IO: (Fit, State, price) -> State

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
use eval::{Fit, evaluate, scales};
use state::State;
use step::{Adam, Priced, step, zeros};

pub const DEFAULT_STEPS: usize = 600;
/// The exchange rate of a reading's residual into bits: each element a cell coded as a
/// Gaussian of one JND's spread, `d^2/2` nats.
pub const BITS_PER_JND2: f64 = 0.5 * std::f64::consts::LOG2_E;
pub const DEBIAS_STEPS: f64 = 150.0 / 600.0;

pub struct Progress<'a> {
    pub phase: &'a str,
    pub lambda: f64,
    pub step: usize,
    pub residual_bits: f64,
    pub numbers: usize,
    pub bits: f64,
}

struct Descent<'a> {
    fit: &'a Fit,
    weights: Vec<f64>,
    scales: Vec<f64>,
    abandon: &'a dyn Abandon,
    report: &'a mut dyn FnMut(&Progress),
    cancelled: bool,
}

impl Descent<'_> {
    async fn descend(
        &mut self,
        s: &mut State,
        adam: &mut Adam,
        lambda: f64,
        count: usize,
        held: Option<&[bool]>,
        phase: &str,
    ) -> Result<(), CliError> {
        for k in 0..count {
            if self.cancelled || self.abandon.abandoned().await {
                self.cancelled = true;
                break;
            }
            let e = evaluate(self.fit, s, true, self.fit.threads)?;
            let grad = e.grad.expect("a gradient was asked");
            let priced = Priced {
                lambda,
                weights: &self.weights,
                scales: &self.scales,
                held,
            };
            step(s, adam, &grad, &priced);
            if k % 50 == 0 || k + 1 == count {
                let (numbers, bits) = price::numbers(s);
                (self.report)(&Progress {
                    phase,
                    lambda,
                    step: k,
                    residual_bits: e.residual.bits,
                    numbers,
                    bits,
                });
            }
        }
        Ok(())
    }
}

/// Whether a signal stopped it.
pub async fn descended(
    fit: &Fit,
    s: &mut State,
    lambda: f64,
    steps: usize,
    abandon: &dyn Abandon,
    report: &mut dyn FnMut(&Progress),
) -> Result<bool, CliError> {
    let mut adam = Adam::of(s);
    let scales = scales(fit, s)?;
    let weights = match lambda > 0.0 {
        true => price::weights(s, &scales),
        false => vec![0.0; s.x.len()],
    };
    let mut run = Descent {
        fit,
        weights,
        scales,
        abandon,
        report,
        cancelled: false,
    };
    run.descend(s, &mut adam, lambda, steps, None, "descend")
        .await?;
    if lambda > 0.0 {
        let held = zeros(s, &run.weights);
        let debias = match steps {
            0 => 0,
            _ => ((steps as f64 * DEBIAS_STEPS).round() as usize).max(1),
        };
        run.descend(s, &mut adam, 0.0, debias, Some(&held), "debias")
            .await?;
    }
    Ok(run.cancelled)
}
