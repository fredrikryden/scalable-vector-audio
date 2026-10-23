// Concern: structure edits ranked by one evaluation, solved best first, the first that falls kept | Non-concern: proposing edits, the descent | IO: (Problem, edits, price, draws, report) -> files, kept

//! Each proposal is predicted from one proximal step (`f + g·Δx + λ·Δb`), as are the current
//! files. The best-ranked is always solved, the rest only while predicted below the current
//! files; the first solved below them is kept.

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
    stage: &str,
    files: Vec<(String, String)>,
    cost: &Cost,
    draws: usize,
    propose: &Propose<'_>,
    report: &mut dyn FnMut(&str, &Progress),
) -> Result<Improved, CliError> {
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
        let current = predicted(&cx.prepared(&files, cost, draws).await?)?;
        let mut ranked = Vec::new();
        for p in proposals {
            if cx.abandon.abandoned().await {
                break 'round;
            }
            let edited = applied(&model, &cx.texts(&files), &cx.problem.vary, &p)?;
            let start = merged(&files, &edited);
            let at = predicted(&cx.prepared(&start, cost, draws).await?)?;
            ranked.push((at.predicted, p.label, start));
        }
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (rank, (predicted, label, start)) in ranked.into_iter().enumerate() {
            if rank > 0 && predicted >= current.predicted {
                break;
            }
            let trial = format!("{stage} trial {label}");
            let mut heard = |p: &Progress| report(&trial, p);
            let solved = cx
                .solved(&start, cost, cx.settings.steps, draws, &mut heard)
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
