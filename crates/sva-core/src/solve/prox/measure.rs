// Concern: a reached composition rendered once per draw and read against the reference, the draws rendered in parallel | Non-concern: minimising | IO: (Problem, files, draws, Tier) -> Residual

use std::num::NonZeroUsize;

use sva_ast::Source;
use sva_engine::{Abandon, Backend, EngineError, Tier};

use super::eval::{Residual, residual, weights};
use super::run::{FIT_RATE, Prepared, prepared, read_off};
use super::write;
use crate::cli_error::CliError;
use crate::query::{asked, call};
use crate::solve::{Problem, recorded};
use crate::{Job, Settled};

/// `files` laid over the problem's composition, each draw's `ph` and `seed` written and rendered.
/// Draws render in parallel, each over its own memory tier; a stop is answered once all end.
pub async fn measure<B: Backend>(
    problem: &Problem<'_>,
    files: &[(String, String)],
    draws: usize,
    tier: &Tier<B>,
    abandon: &dyn Abandon,
) -> Result<Residual, CliError> {
    let base = recorded(&problem.cost, problem.source);
    let set = crate::overlaid(&base, &problem.sets)?;
    let source = sva_ast::Overlay::new(&set, files.iter().cloned().collect());
    let Prepared {
        shape,
        fit,
        state,
        frames,
    } = prepared(problem, &source, draws, tier, abandon).await?;
    let leaf = &problem.cost.leaves[shape.model_leaf];
    let samples = asked(&call("samples")?)?;
    let text = |p: &str| -> Option<String> { source.get(p).ok().flatten().map(|t| t.into_owned()) };
    let model = format!("@{}([0, {frames}sp])", shape.root);
    let workers = fit.threads.get().min(draws);
    let per_draw = NonZeroUsize::new(fit.threads.get() / workers).expect("a worker per thread");
    let settled = (0..draws)
        .map(|k| {
            let drawn = write::drawn(&state, &text, k)?;
            let over = sva_ast::Overlay::new(&source, drawn.into_iter().collect());
            Settled::of(&Job {
                rate: Some(FIT_RATE),
                threads: Some(per_draw),
                asked: std::slice::from_ref(&samples),
                ..Job::over(&over, &model)
            })
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    let read = sva_engine::each(fit.threads, settled, |draw| {
        let rendered = draw.execute(&Tier::default())?;
        Ok(read_off(leaf, &rendered, &samples)?.0.elements)
    });
    let read = read.into_iter().collect::<Result<Vec<_>, CliError>>()?;
    if abandon.abandoned().await {
        return Err(CliError::Engine(EngineError::Abandoned));
    }
    Ok(residual(&fit, &weights(&fit), &read))
}
