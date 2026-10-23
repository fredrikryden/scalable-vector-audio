// Concern: the price homotopy: an unpriced start, a cluster, then one stage a price, each restructured by split and merge | Non-concern: one price's descent | IO: (prices) -> stages

use sva_engine::Backend;

use super::improve::improve;
use super::{Cx, Reached, Stage, Stages, merged, scaled};
use crate::cli_error::CliError;
use crate::solve::prox::DEBIAS_STEPS;
use crate::solve::with_price;
use crate::structure::{applied, assign, merge, split};

/// Prices in bits per bit: a homotopy from nearly unpriced to 1, where the objective is the
/// description length itself.
pub const DEFAULT_PRICES: [f64; 5] = [0.01, 0.03, 0.1, 0.3, 1.0];
pub(super) const START_STEPS: f64 = 500.0 / 600.0;
const SETTLE_STEPS: f64 = 200.0 / 600.0;

enum Next {
    Start,
    Settle,
    Price(usize),
    Restructure(usize),
}

pub(super) struct Path {
    prices: Vec<f64>,
    rounds: usize,
    steps: usize,
    next: Next,
}

impl Path {
    pub(super) fn new(prices: Vec<f64>, rounds: usize, steps: usize) -> Path {
        Path {
            prices,
            rounds,
            steps,
            next: Next::Start,
        }
    }
}

impl<B: Backend> Stages<B> for Path {
    async fn next(
        &mut self,
        last: Option<&Reached>,
        cx: &Cx<'_, B>,
    ) -> Result<Option<Stage>, CliError> {
        let unpriced = || with_price(&cx.problem.cost, 0.0);
        let files = last.map(Reached::files).unwrap_or_default();
        loop {
            match self.next {
                Next::Start => {
                    self.next = Next::Settle;
                    return Ok(Some(Stage {
                        label: "start".to_string(),
                        files,
                        cost: unpriced()?,
                        steps: scaled(self.steps, START_STEPS),
                        edits: Vec::new(),
                    }));
                }
                Next::Settle => {
                    self.next = Next::Price(0);
                    let model = cx.model(&files)?;
                    let cluster = assign::cluster(&model, &cx.problem.vary, self.rounds);
                    let edited = applied(&model, &cx.texts(&files), &cx.problem.vary, &cluster)?;
                    let edits = match cluster.edits.is_empty() {
                        true => Vec::new(),
                        false => vec![cluster.label],
                    };
                    return Ok(Some(Stage {
                        label: "settle".to_string(),
                        files: merged(&files, &edited),
                        cost: unpriced()?,
                        steps: scaled(self.steps, SETTLE_STEPS),
                        edits,
                    }));
                }
                Next::Price(k) if k == self.prices.len() => return Ok(None),
                Next::Price(k) => {
                    self.next = Next::Restructure(k);
                    let price = self.prices[k];
                    return Ok(Some(Stage {
                        label: format!("price {price}"),
                        files,
                        cost: with_price(&cx.problem.cost, price)?,
                        steps: self.steps,
                        edits: Vec::new(),
                    }));
                }
                Next::Restructure(k) => {
                    self.next = Next::Price(k + 1);
                    let price = self.prices[k];
                    let cost = with_price(&cx.problem.cost, price)?;
                    let (varies, rounds) = (&cx.problem.vary, self.rounds);
                    let split = improve(cx, files.clone(), &cost, &|m, _, _| {
                        Ok(split::proposals(m, varies, rounds))
                    })
                    .await?;
                    let merged = improve(cx, split.files, &cost, &|m, _, _| {
                        Ok(merge::proposals(m, varies))
                    })
                    .await?;
                    if split.kept.is_empty() && merged.kept.is_empty() {
                        continue;
                    }
                    return Ok(Some(Stage {
                        label: format!("price {price}, restructured"),
                        files: merged.files,
                        cost,
                        steps: scaled(self.steps, DEBIAS_STEPS),
                        edits: split.kept.into_iter().chain(merged.kept).collect(),
                    }));
                }
            }
        }
    }
}
