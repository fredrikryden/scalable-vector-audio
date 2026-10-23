// Concern: structure edits kept one at a time while a short solve says the objective falls | Non-concern: proposing edits, the descent | IO: (Problem, edits, price) -> files, kept

use sva_engine::Backend;

use super::{Cx, merged, scaled};
use crate::cli_error::CliError;
use crate::model::Model;
use crate::solve::Cost;
use crate::structure::{Proposal, applied};

const TRIAL_STEPS: f64 = 80.0 / 600.0;

pub struct Improved {
    pub files: Vec<(String, String)>,
    pub kept: Vec<String>,
}

/// Each round keeps the proposal whose trial lowers the objective most, with its trial's numbers.
pub async fn improve<B: Backend>(
    cx: &Cx<'_, B>,
    files: Vec<(String, String)>,
    cost: &Cost,
    propose: &dyn Fn(&Model) -> Vec<Proposal>,
) -> Result<Improved, CliError> {
    let trial = scaled(cx.settings.steps, TRIAL_STEPS);
    let mut quiet = |_: &crate::solve::prox::Progress| {};
    let (mut files, mut kept) = (files, Vec::new());
    loop {
        if cx.abandon.abandoned().await {
            break;
        }
        let model = cx.model(&files)?;
        let proposals = propose(&model);
        if proposals.is_empty() {
            break;
        }
        let base = cx.solved(&files, cost, 0, &mut quiet).await?;
        if base.cancelled {
            break;
        }
        let base = base.objective();
        let mut best: Option<(f64, Improved)> = None;
        for p in &proposals {
            let edited = applied(&model, &cx.texts(&files), &cx.problem.vary, p)?;
            let start = merged(&files, &edited);
            let solved = cx.solved(&start, cost, trial, &mut quiet).await?;
            if solved.cancelled {
                return Ok(Improved { files, kept });
            }
            let o = solved.objective();
            if o < base && best.as_ref().is_none_or(|b| o < b.0) {
                let files = merged(&start, &solved.files);
                let kept = vec![p.label.clone()];
                best = Some((o, Improved { files, kept }));
            }
        }
        match best {
            Some((_, better)) => {
                files = better.files;
                kept.extend(better.kept);
            }
            None => break,
        }
    }
    Ok(Improved { files, kept })
}
