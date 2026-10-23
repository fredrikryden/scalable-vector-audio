// Concern: a reached composition rendered once per draw and read against the reference, as the cost's leaf reads a render | Non-concern: minimising | IO: (Problem, files, Tier) -> Residual

use sva_ast::Source;
use sva_engine::{Abandon, Backend, Tier};

use super::draws::DRAWS;
use super::eval::{Residual, residual, weights};
use super::run::{Prepared, Reading, prepared};
use super::write;
use crate::cli_error::CliError;
use crate::solve::{Problem, recorded};

/// `files` laid over the problem's composition, each draw's `ph` and `seed` written and rendered.
pub async fn measure<B: Backend>(
    problem: &Problem<'_>,
    files: &[(String, String)],
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
    } = prepared(problem, &source, tier, abandon).await?;
    let reading = Reading {
        problem,
        tier,
        abandon,
        leaf: &problem.cost.leaves[shape.model_leaf],
    };
    let text = |p: &str| -> Option<String> { source.get(p).ok().flatten().map(|t| t.into_owned()) };
    let model = format!("@{}([0, {frames}sp])", shape.root);
    let mut draws = Vec::with_capacity(DRAWS);
    for k in 0..DRAWS {
        let drawn = write::drawn(&state, &text, k)?;
        let over = sva_ast::Overlay::new(&source, drawn.into_iter().collect());
        let (rendered, _) = reading.read(&over, &model).await?;
        draws.push(rendered.elements);
    }
    Ok(residual(&fit, &weights(&fit), &draws))
}
