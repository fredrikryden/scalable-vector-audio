// Concern: coarse to fine: the composition solved, then its limits doubled a level at a time, growths kept where the objective falls | Non-concern: what a growth writes | IO: (clip, ceiling) -> stages

use sva_ast::{Arg, Expr, Literal};
use sva_engine::Backend;

use super::improve::improve;
use super::path::START_STEPS;
use super::{Cx, Reached, Stage, Stages, merged, scaled, started};
use crate::cli_error::CliError;
use crate::seed::{self, Limits, Readings};
use crate::solve::{recorded, with_price};
use crate::structure::{applied, assign, grow, merge, split};

enum Next {
    Start,
    Settle,
    /// A level after the first carries the length of the stage before it.
    Level(Option<f64>),
}

pub(super) struct Grow {
    readings: Readings,
    ceiling: Limits,
    at: Limits,
    rounds: usize,
    steps: usize,
    next: Next,
}

impl Grow {
    pub(super) fn new<B: Backend>(
        cx: &Cx<'_, B>,
        ceiling: Limits,
        rounds: usize,
    ) -> Result<Grow, CliError> {
        Ok(Grow {
            readings: readings(cx)?,
            ceiling,
            at: grow::written(&cx.model(&[])?),
            rounds,
            steps: cx.settings.steps,
            next: Next::Start,
        })
    }
}

/// The clip the cost's reference samples, read as a starter reads it.
fn readings<B: Backend>(cx: &Cx<'_, B>) -> Result<Readings, CliError> {
    let target = &cx.shape.reference;
    let refused = || {
        CliError::usage(
            "solve.grow_reference",
            format!(
                "`{target}` is no `sample(\"<clip>\")`; `grow` reads its growths off the \
                 recording the cost's reference samples"
            ),
        )
    };
    let source = recorded(&cx.problem.cost, cx.problem.source);
    let parsed = sva_ast::parse_expr(target).map_err(|_| refused())?;
    let resolved =
        sva_ast::resolve_recordings(&parsed, &source).map_err(|unheld| match unheld.why {
            None => CliError::NotFound(unheld.message()),
            Some(_) => CliError::usage(crate::RECORDING_UNREADABLE, unheld.message()),
        })?;
    match resolved {
        Expr::Call { name, args, .. } if name == "sample" => match args.as_slice() {
            [Arg::Pos(Expr::Lit(Literal::Recorded(r)))] => {
                Ok(seed::read(&r.planes, f64::from(r.rate)))
            }
            _ => Err(refused()),
        },
        _ => Err(refused()),
    }
}

/// Each limit doubled, lone partials from none to one, held at the ceiling's.
fn finer(at: Limits, ceiling: Limits) -> Limits {
    let up = |n: usize, top: usize| n.max((2 * n).max(1).min(top));
    Limits {
        harmonics: up(at.harmonics, ceiling.harmonics),
        lone: up(at.lone, ceiling.lone),
    }
}

impl<B: Backend> Stages<B> for Grow {
    async fn next(
        &mut self,
        last: Option<&Reached>,
        cx: &Cx<'_, B>,
    ) -> Result<Option<Stage>, CliError> {
        let files = last.map(Reached::files).unwrap_or_default();
        let cost = &cx.problem.cost;
        let varies = &cx.problem.vary;
        let draws = started(last);
        match self.next {
            Next::Start => {
                self.next = Next::Settle;
                Ok(Some(Stage {
                    label: "start".to_string(),
                    files,
                    cost: with_price(cost, 0.0)?,
                    steps: scaled(self.steps, START_STEPS),
                    edits: Vec::new(),
                }))
            }
            Next::Settle => {
                self.next = Next::Level(None);
                let model = cx.model(&files)?;
                let cluster = assign::cluster(&model, varies, self.rounds);
                let edited = applied(&model, &cx.texts(&files), varies, &cluster)?;
                let edits = match cluster.edits.is_empty() {
                    true => Vec::new(),
                    false => vec![cluster.label],
                };
                Ok(Some(Stage {
                    label: "settle".to_string(),
                    files: merged(&files, &edited),
                    cost: cost.clone(),
                    steps: self.steps,
                    edits,
                }))
            }
            Next::Level(before) => {
                let length = last.map(Reached::length);
                if let (Some(before), Some(length)) = (before, length)
                    && length >= before
                {
                    return Ok(None);
                }
                self.next = Next::Level(length);
                let (from, to) = (self.at, finer(self.at, self.ceiling));
                if to == from {
                    return Ok(None);
                }
                let (readings, rounds) = (&self.readings, self.rounds);
                let grown = improve(cx, files, cost, draws, &|m, texts, kept| {
                    let offered = grow::proposals(m, texts, varies, readings, from, to)?;
                    Ok(offered
                        .into_iter()
                        .filter(|p| !kept.contains(&p.label))
                        .collect())
                })
                .await?;
                if grown.kept.is_empty() {
                    return Ok(None);
                }
                let split = improve(cx, grown.files, cost, draws, &|m, _, _| {
                    Ok(split::proposals(m, varies, rounds))
                })
                .await?;
                let merged = improve(cx, split.files, cost, draws, &|m, _, _| {
                    Ok(merge::proposals(m, varies))
                })
                .await?;
                self.at = to;
                Ok(Some(Stage {
                    label: format!("grow h={} lone={}", to.harmonics, to.lone),
                    files: merged.files,
                    cost: cost.clone(),
                    steps: self.steps,
                    edits: [grown.kept, split.kept, merged.kept].concat(),
                }))
            }
        }
    }
}
