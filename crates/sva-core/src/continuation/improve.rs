// Concern: structure edits ranked by one evaluation each, in parallel, solved best first, the first that falls kept | Non-concern: proposing edits, the descent | IO: (Problem, edits) -> files, kept

//! Each proposal is predicted from one proximal step (`f + g·Δx + λ·Δb`), as are the current
//! files, ties in proposal order; one whose files, raw or as `--write` writes them, repeat
//! earlier ones is skipped. The best-ranked is solved, the rest while predicted below the
//! current; the first solved below them is kept.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use sva_engine::Backend;

use super::{Cx, Reached, Stage, merged};
use crate::cli_error::CliError;
use crate::model::Model;
use crate::solve::Cost;
use crate::solve::prox::heard::{Act, Action};
use crate::solve::prox::played::Renders;
use crate::solve::prox::run::Prepared;
use crate::solve::prox::{self, price, write};
use crate::structure::{Proposal, applied};

pub struct Improved {
    pub files: Vec<(String, String)>,
    pub kept: Vec<String>,
    /// The trial a signal stopped, labelled as its progress lines are, its edits the kept then
    /// its own.
    pub stopped: Option<Reached>,
}

/// A round's edits, given the model, its files' texts and the labels kept before.
pub type Propose<'a> =
    dyn Fn(&Model, &Texts<'_>, &[String]) -> Result<Vec<Proposal>, CliError> + 'a;
type Texts<'t> = dyn Fn(&str) -> Option<String> + 't;

struct Predicted {
    objective: f64,
    predicted: f64,
}

/// `files` as `--write` writes them from `at`, prepared from them.
fn as_written<B: Backend>(
    cx: &Cx<'_, B>,
    files: &[(String, String)],
    at: &Prepared,
) -> Result<Vec<(String, String)>, CliError> {
    Ok(merged(files, &write::files(&at.state, &cx.texts(files))?))
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

type Ranked = Vec<(f64, String, Vec<(String, String)>)>;

/// The current files' prediction and each proposal's, none where a signal stopped the round;
/// a proposal's composition prepared, its noise rendered, then a thread's worth predicted at once.
async fn ranked<B: Backend>(
    cx: &Cx<'_, B>,
    files: &[(String, String)],
    model: &Model,
    proposals: Vec<Proposal>,
    cost: &Cost,
    draws: usize,
) -> Result<Option<(Predicted, Ranked)>, CliError> {
    let here = cx.prepared(files, cost, draws).await?;
    here.fit.noised(&here.state, cx.hearing)?;
    let threads = here.fit.threads;
    let current = predicted(&here)?;
    let mut edited = BTreeSet::from([merged(files, &[])]);
    let mut written = BTreeSet::from([as_written(cx, files, &here)?]);
    let mut ranked = Vec::new();
    let mut proposals = proposals.into_iter().enumerate().peekable();
    while proposals.peek().is_some() {
        let mut starts = Vec::new();
        let mut jobs = Vec::new();
        while jobs.len() < threads.get()
            && let Some((i, p)) = proposals.next()
        {
            cx.hearing.tick(i);
            if cx.abandon.abandoned().await {
                return Ok(None);
            }
            let start = merged(
                files,
                &applied(model, &cx.texts(files), &cx.problem.vary, &p)?,
            );
            if !edited.insert(start.clone()) {
                continue;
            }
            let mut job = cx.prepared(&start, cost, draws).await?;
            if !written.insert(as_written(cx, &start, &job)?) {
                continue;
            }
            assert_eq!(
                job.fit.grid(),
                here.fit.grid(),
                "a round reads one reference"
            );
            job.fit.renders = Arc::new(Renders::over(Arc::clone(&here.fit.renders)));
            job.fit.noised(&job.state, cx.hearing)?;
            jobs.push(job);
            starts.push((p.label, start));
        }
        if jobs.is_empty() {
            continue;
        }
        let per_job = NonZeroUsize::new(threads.get() / jobs.len()).expect("a worker per thread");
        let predictions = sva_engine::each(threads, jobs, |mut at| {
            at.fit.threads = per_job;
            predicted(&at).map(|p| p.predicted)
        });
        for (p, (label, start)) in predictions.into_iter().zip(starts) {
            ranked.push((p?, label, start));
        }
    }
    Ok(Some((current, ranked)))
}

pub async fn improve<B: Backend>(
    cx: &Cx<'_, B>,
    stage: &str,
    files: Vec<(String, String)>,
    cost: &Cost,
    draws: usize,
    propose: &Propose<'_>,
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
        let act = Act::new(Action::RankingProposals).of(proposals.len());
        let Some((current, mut ranked)) = cx
            .hearing
            .during(act, ranked(cx, &files, &model, proposals, cost, draws))
            .await?
        else {
            break 'round;
        };
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let tried = ranked.len();
        for (rank, (predicted, label, start)) in ranked.into_iter().enumerate() {
            if rank > 0 && predicted >= current.predicted {
                break;
            }
            let trial = format!("{stage} trial {label}");
            let act = Act::new(Action::TryingProposal)
                .of(tried)
                .from(rank)
                .edit(&label, trial.clone());
            let solving = cx.solved(&start, cost, cx.settings.steps, draws);
            let solved = cx.hearing.during(act, solving).await?;
            if solved.cancelled() {
                let stage = Stage {
                    label: trial,
                    files: start,
                    cost: cost.clone(),
                    steps: cx.settings.steps,
                    edits: [kept.as_slice(), &[label]].concat(),
                };
                let stopped = Some(Reached {
                    stage,
                    solution: solved,
                    at_cap: None,
                    measured: None,
                });
                return Ok(Improved {
                    files,
                    kept,
                    stopped,
                });
            }
            if solved.objective() < current.objective {
                files = merged(&start, &solved.files);
                kept.push(label);
                continue 'round;
            }
        }
        break;
    }
    Ok(Improved {
        files,
        kept,
        stopped: None,
    })
}
