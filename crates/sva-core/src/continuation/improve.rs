// Concern: structure edits ranked by one evaluation each, the best solved, kept where it falls | Non-concern: proposing edits, the descent | IO: (Problem, edits, price) -> files, kept

//! A proposal is evaluated once and predicted `f(x) + g·(x⁺ - x) + λ·(b(x⁺) - b(x))`, `x⁺` the
//! solver's first proximal step; the current files alike. Those predicted below them are solved
//! best first, the first whose objective falls kept.

use sva_engine::Backend;

use super::{Cx, merged};
use crate::cli_error::CliError;
use crate::model::Model;
use crate::solve::Cost;
use crate::solve::prox::run::Prepared;
use crate::solve::prox::{self, Progress, price};
use crate::structure::{Proposal, applied};

pub struct Improved {
    pub files: Vec<(String, String)>,
    pub kept: Vec<String>,
}

/// A round's edits, given the model, its files' texts and the labels kept before.
pub type Propose<'a> =
    dyn Fn(&Model, &Texts<'_>, &[String]) -> Result<Vec<Proposal>, CliError> + 'a;
type Texts<'t> = dyn Fn(&str) -> Option<String> + 't;

struct Predicted {
    objective: f64,
    predicted: f64,
}

fn predicted(at: &Prepared) -> Result<Predicted, CliError> {
    let (lambda, s) = (at.shape.price, &at.state);
    let (e, next) = prox::stepped(&at.fit, s, lambda)?;
    let bits = |s| price::numbers(s).1;
    let linear: f64 = e
        .grad
        .expect("a gradient was asked")
        .iter()
        .zip(next.x.iter().zip(&s.x))
        .filter(|(g, _)| g.is_finite())
        .map(|(g, (to, from))| g * (to - from))
        .sum();
    let objective = e.residual.bits + lambda * bits(s);
    Ok(Predicted {
        objective,
        predicted: objective + linear + lambda * (bits(&next) - bits(s)),
    })
}

pub async fn improve<B: Backend>(
    cx: &Cx<'_, B>,
    files: Vec<(String, String)>,
    cost: &Cost,
    propose: &Propose<'_>,
) -> Result<Improved, CliError> {
    let mut quiet = |_: &Progress| {};
    let (mut files, mut kept) = (files, Vec::new());
    'round: loop {
        if cx.abandon.abandoned().await {
            break;
        }
        let model = cx.model(&files)?;
        let proposals = propose(&model, &cx.texts(&files), &kept)?;
        if proposals.is_empty() {
            break;
        }
        let current = predicted(&cx.prepared(&files, cost).await?)?;
        let mut ranked = Vec::new();
        for p in proposals {
            if cx.abandon.abandoned().await {
                break 'round;
            }
            let edited = applied(&model, &cx.texts(&files), &cx.problem.vary, &p)?;
            let start = merged(&files, &edited);
            let at = predicted(&cx.prepared(&start, cost).await?)?;
            if at.predicted < current.predicted {
                ranked.push((at.predicted, p.label, start));
            }
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, label, start) in ranked {
            let solved = cx
                .solved(&start, cost, cx.settings.steps, &mut quiet)
                .await?;
            if solved.cancelled() {
                break 'round;
            }
            if solved.objective() < current.objective {
                files = merged(&start, &solved.files);
                kept.push(label);
                continue 'round;
            }
        }
        break;
    }
    Ok(Improved { files, kept })
}
