// Concern: coarse to fine: the composition solved, then its limits doubled a level at a time, growths kept where the objective falls | Non-concern: what a growth writes | IO: (clip, ceiling) -> stages

use sva_ast::{Arg, Expr, Literal};
use sva_engine::Backend;

use super::improve::improve;
use super::path::START_STEPS;
use super::{Built, Cx, Reached, Stage, Stages, merged, scaled, started};
use crate::cli_error::CliError;
use crate::seed::{self, Limits, Readings};
use crate::solve::prox::heard::{Act, Action};
use crate::solve::prox::run::FIT_RATE;
use crate::solve::{recorded, with_price};
use crate::structure::{applied, assign, grow, merge, split};

enum Next {
    Start,
    Settle,
    /// A level after the first carries the length, at the cap's draws, of the stage before it.
    Level(Option<f64>),
}

pub(super) struct Grow {
    readings: Readings,
    ceiling: Limits,
    at: Limits,
    /// The levels built, and the most from the composition's limits to the ceiling.
    levels: (usize, usize),
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
        let readings = cx
            .hearing
            .within(Act::new(Action::Seeding), || readings(cx))?;
        let at = grow::written(&cx.model(&[])?);
        let mut levels = 0;
        let mut to = at;
        while finer(to, ceiling) != to {
            (to, levels) = (finer(to, ceiling), levels + 1);
        }
        Ok(Grow {
            readings,
            ceiling,
            at,
            levels: (0, levels),
            rounds,
            steps: cx.settings.steps,
            next: Next::Start,
        })
    }
}

/// The clip the cost's reference samples up to its interval's end, read as a starter reads it.
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
    let expr = crate::target::target(target)?.expr;
    let parsed = sva_ast::parse_expr(&expr).map_err(|_| refused())?;
    let resolved =
        sva_ast::resolve_recordings(&parsed, &source).map_err(|unheld| match unheld.why {
            None => CliError::NotFound(unheld.message()),
            Some(_) => CliError::usage(crate::RECORDING_UNREADABLE, unheld.message()),
        })?;
    let r = match resolved {
        Expr::Call { name, args, .. } if name == "sample" => match args.as_slice() {
            [Arg::Pos(Expr::Lit(Literal::Recorded(r)))] => r.clone(),
            _ => return Err(refused()),
        },
        _ => return Err(refused()),
    };
    let rate = f64::from(r.rate);
    Ok(match cx.shape.span(cx.per_bar()?)? {
        (_, None) => seed::read(&r.planes, rate),
        (_, Some(at)) => {
            let end = (at as u64 * u64::from(r.rate)).div_ceil(u64::from(FIT_RATE)) as usize;
            let cut: Vec<Vec<f32>> = r
                .planes
                .iter()
                .map(|p| p[..end.min(p.len())].to_vec())
                .collect();
            seed::read(&cut, rate)
        }
    })
}

/// Each limit doubled, lone partials from none to one, held at the ceiling's.
fn finer(at: Limits, ceiling: Limits) -> Limits {
    let up = |n: usize, top: usize| n.max((2 * n).max(1).min(top));
    Limits {
        harmonics: up(at.harmonics, ceiling.harmonics),
        lone: up(at.lone, ceiling.lone),
    }
}

impl Grow {
    /// The most stages it builds: a level that does not shorten the length ends the run.
    fn parts(&self) -> usize {
        2 + self.levels.1
    }
}

impl<B: Backend> Stages<B> for Grow {
    async fn next(
        &mut self,
        last: Option<&Reached>,
        cx: &Cx<'_, B>,
    ) -> Result<Option<Built>, CliError> {
        let files = last.map(Reached::files).unwrap_or_default();
        let cost = &cx.problem.cost;
        let varies = &cx.problem.vary;
        let draws = started(last);
        match self.next {
            Next::Start => {
                self.next = Next::Settle;
                cx.hearing.staged("start", (1, self.parts()));
                Ok(Some(Built::Stage(Stage {
                    label: "start".to_string(),
                    files,
                    cost: with_price(cost, 0.0)?,
                    steps: scaled(self.steps, START_STEPS),
                    edits: Vec::new(),
                })))
            }
            Next::Settle => {
                self.next = Next::Level(None);
                cx.hearing.staged("settle", (2, self.parts()));
                let model = cx.model(&files)?;
                let cluster = assign::cluster(&model, varies, self.rounds);
                let edited = applied(&model, &cx.texts(&files), varies, &cluster)?;
                let edits = match cluster.edits.is_empty() {
                    true => Vec::new(),
                    false => vec![cluster.label],
                };
                Ok(Some(Built::Stage(Stage {
                    label: "settle".to_string(),
                    files: merged(&files, &edited),
                    cost: cost.clone(),
                    steps: self.steps,
                    edits,
                })))
            }
            Next::Level(before) => {
                let length = last.and_then(Reached::length);
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
                let label = format!("grow h={} lone={}", to.harmonics, to.lone);
                let part = (3 + self.levels.0, self.parts());
                cx.hearing.staged(&label, part);
                let grown = improve(cx, &label, files, cost, draws, &|m, texts, kept| {
                    let offered = grow::proposals(m, texts, varies, readings, from, to)?;
                    Ok(offered
                        .into_iter()
                        .filter(|p| !kept.contains(&p.label))
                        .collect())
                })
                .await?;
                if let Some(r) = grown.stopped {
                    return Ok(Some(Built::Stopped(r)));
                }
                if grown.kept.is_empty() {
                    return Ok(None);
                }
                let split = improve(cx, &label, grown.files, cost, draws, &|m, _, _| {
                    Ok(split::proposals(m, varies, rounds))
                })
                .await?;
                if let Some(r) = split.stopped {
                    return Ok(Some(Built::Stopped(r.after(&grown.kept))));
                }
                let merged = improve(cx, &label, split.files, cost, draws, &|m, _, _| {
                    Ok(merge::proposals(m, varies))
                })
                .await?;
                let kept = [grown.kept, split.kept].concat();
                if let Some(r) = merged.stopped {
                    return Ok(Some(Built::Stopped(r.after(&kept))));
                }
                self.at = to;
                self.levels.0 += 1;
                Ok(Some(Built::Stage(Stage {
                    label,
                    files: merged.files,
                    cost: cost.clone(),
                    steps: self.steps,
                    edits: [kept, merged.kept].concat(),
                })))
            }
        }
    }
}
