// Concern: runs a scheme: stages solved in turn, draws doubled while noise-limited, each predicted at the cap's draws, the least chosen | Non-concern: what a stage is | IO: (Problem, Scheme) -> Ran

//! A run's first stage is solved at [`FEWEST`] draws, a later one at the last's, doubled while
//! its descent is noise-limited, until `Settings::draws`, the cap. A prediction fitted to few
//! draws is optimistic, so each finished stage is evaluated once at the cap's, no descent, and
//! compared on that.

mod grow;
pub mod improve;
pub mod path;

use std::collections::BTreeMap;

use sva_ast::{Overlay, Source};
use sva_engine::{Abandon, Backend, Tier};

use crate::cli_error::CliError;
use crate::model::Model;
use crate::seed::{LIMITS, Limits};
use crate::solve::prox::eval::{Residual, evaluate};
use crate::solve::prox::heard::{Act, Action, Heard, Hearing};
use crate::solve::prox::run::Prepared;
use crate::solve::prox::{self, Settings, Shape, Solution};
use crate::solve::{Cost, Problem, recorded};

#[derive(Clone, Debug, PartialEq)]
pub enum Scheme {
    /// The price homotopy: `prices` in turn, the cost's own where `None`; `rounds` the k-medians
    /// rounds of each cluster and split.
    Path {
        prices: Option<Vec<f64>>,
        rounds: usize,
    },
    /// Coarse to fine: the composition's limits doubled a level at a time up to `ceiling`, at
    /// the cost's own price.
    Grow { ceiling: Limits, rounds: usize },
}

/// A scheme's arguments as a front end hands them over, each optional.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Asked {
    pub prices: Option<Vec<f64>>,
    pub harmonics: Option<usize>,
    pub lone: Option<usize>,
    pub rounds: Option<usize>,
}

const SCHEMES: &str = "the schemes are `path`, with prices above 0 or none for the cost's own: \
     `path:0.01,0.1` on the command line, `{ scheme: \"path\", prices: [0.01, 0.1] }` in wasm; \
     and `grow`, with its ceiling's harmonics above 0 and lone partials, or none for 48 and 8: \
     `grow:48,8`, `{ scheme: \"grow\", harmonics: 48, lone: 8 }`";

fn refused(what: String) -> CliError {
    CliError::usage("solve.bad_continuation", format!("{what}; {SCHEMES}"))
}

impl Scheme {
    pub fn of(name: &str, asked: Asked) -> Result<Scheme, CliError> {
        let Asked {
            prices,
            harmonics,
            lone,
            rounds,
        } = asked;
        let rounds = rounds.unwrap_or(crate::structure::assign::REASSIGN_ROUNDS);
        if rounds == 0 {
            return Err(refused(
                "`rounds` 0 clusters nothing; the k-medians rounds are 1 or more".to_string(),
            ));
        }
        match name {
            "path" => {
                if harmonics.is_some() || lone.is_some() {
                    return Err(refused("`path` takes prices, not limits".to_string()));
                }
                if let Some(p) = &prices
                    && (p.is_empty() || p.iter().any(|x| !x.is_finite() || *x <= 0.0))
                {
                    return Err(refused(format!("`path` takes prices above 0, not {p:?}")));
                }
                Ok(Scheme::Path { prices, rounds })
            }
            "grow" => {
                if prices.is_some() {
                    return Err(refused(
                        "`grow` runs at the cost's own price and takes no prices".to_string(),
                    ));
                }
                let ceiling = Limits {
                    harmonics: harmonics.unwrap_or(LIMITS.harmonics),
                    lone: lone.unwrap_or(LIMITS.lone),
                };
                if ceiling.harmonics == 0 {
                    return Err(refused(
                        "`grow`'s harmonics are 1 or more, a voice's first".to_string(),
                    ));
                }
                Ok(Scheme::Grow { ceiling, rounds })
            }
            _ => Err(refused(format!("`{name}` names no continuation"))),
        }
    }

    /// `path[:<p>,<p>...]` or `grow[:<harmonics>,<lone>]`.
    pub fn parse(text: &str, rounds: Option<usize>) -> Result<Scheme, CliError> {
        let (name, list) = match text.split_once(':') {
            Some((name, list)) => (name, Some(list)),
            None => (text, None),
        };
        let numbers = |what: &str| {
            list.map(|l| {
                l.split(',')
                    .map(|x| {
                        x.trim()
                            .parse::<f64>()
                            .map_err(|_| refused(format!("`{x}` in `{text}` is no {what}")))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()
        };
        let mut asked = Asked {
            rounds,
            ..Asked::default()
        };
        match name {
            "grow" => {
                if let Some(limits) = numbers("limit")? {
                    let whole = |x: f64| (x >= 0.0 && x.fract() == 0.0).then_some(x as usize);
                    let (Some(h), Some(l)) = (match limits.as_slice() {
                        &[h, l] => (whole(h), whole(l)),
                        _ => (None, None),
                    }) else {
                        return Err(refused(format!(
                            "`{text}` names no two whole limits, harmonics then lone partials"
                        )));
                    };
                    (asked.harmonics, asked.lone) = (Some(h), Some(l));
                }
            }
            _ => asked.prices = numbers("price")?,
        }
        Scheme::of(name, asked)
    }

    pub fn json(&self) -> String {
        use crate::json::num;
        match self {
            Scheme::Path { prices, rounds } => format!(
                "{{ \"scheme\": \"path\", \"prices\": {}, \"rounds\": {rounds} }}",
                prices.as_ref().map_or("null".to_string(), |p| format!(
                    "[{}]",
                    p.iter().map(|x| num(*x)).collect::<Vec<_>>().join(", ")
                ))
            ),
            Scheme::Grow { ceiling, rounds } => format!(
                "{{ \"scheme\": \"grow\", \"harmonics\": {}, \"lone\": {}, \"rounds\": {rounds} }}",
                ceiling.harmonics, ceiling.lone
            ),
        }
    }
}

pub struct Stage {
    pub label: String,
    /// The stage's composition: every file it changes over the problem's.
    pub files: Vec<(String, String)>,
    pub cost: Cost,
    pub steps: usize,
    pub edits: Vec<String>,
}

pub struct Reached {
    pub stage: Stage,
    pub solution: Solution,
    /// Predicted at the cap's draws, where the stage finished.
    pub at_cap: Option<Residual>,
    /// Off the stage's renders, the chosen stage's alone and only where the run was not stopped.
    pub measured: Option<Residual>,
}

impl Reached {
    /// The reached composition over the problem's: the stage's files, then the solution's.
    pub fn files(&self) -> Vec<(String, String)> {
        merged(&self.stage.files, &self.solution.files)
    }

    /// At the cap's draws, the predicted residual's bits plus the numbers'.
    pub fn length(&self) -> Option<f64> {
        self.at_cap.map(|r| r.bits + self.solution.bits)
    }

    /// The same at the draws the stage was solved at.
    pub fn solved_length(&self) -> f64 {
        self.solution.predicted.bits + self.solution.bits
    }

    fn after(mut self, kept: &[String]) -> Reached {
        self.stage.edits.splice(0..0, kept.iter().cloned());
        self
    }
}

/// A stage to solve, or a trial a signal stopped.
enum Built {
    Stage(Stage),
    Stopped(Reached),
}

/// The fewest draws with a standard error.
pub const FEWEST: usize = 2;

fn started(last: Option<&Reached>) -> usize {
    last.map_or(FEWEST, |r| r.solution.draws())
}

pub struct Ran {
    pub stages: Vec<Reached>,
    pub chosen: Option<usize>,
    pub cancelled: bool,
}

impl Ran {
    pub fn of(stages: Vec<Reached>, cancelled: bool) -> Ran {
        let mut least = None;
        let chosen = (0..stages.len())
            .filter(|&i| below(&stages[i], &mut least))
            .last();
        Ran {
            stages,
            chosen,
            cancelled,
        }
    }

    /// The chosen stage rendered once per draw up to `Settings::draws` and read, unless stopped.
    pub async fn measured<B: Backend>(mut self, cx: &Cx<'_, B>) -> Result<Ran, CliError> {
        if let (false, Some(c)) = (self.cancelled, self.chosen) {
            let r = &mut self.stages[c];
            r.measured = Some(cx.measured(&r.stage, &r.solution).await?);
        }
        Ok(self)
    }
}

/// `r` shorter than `least`, then the least; a run's last such stage is chosen.
pub(crate) fn below(r: &Reached, least: &mut Option<f64>) -> bool {
    let shorter = r.length().is_some_and(|l| least.is_none_or(|b| l < b));
    if shorter {
        *least = r.length();
    }
    shorter
}

/// `a` with `b` laid over it, `b` winning a path both write.
pub(crate) fn merged(a: &[(String, String)], b: &[(String, String)]) -> Vec<(String, String)> {
    let mut out: BTreeMap<String, String> = a.iter().cloned().collect();
    out.extend(b.iter().cloned());
    out.into_iter().collect()
}

pub struct Cx<'a, B: Backend> {
    pub problem: &'a Problem<'a>,
    pub shape: &'a Shape,
    pub settings: &'a Settings,
    pub tier: &'a Tier<B>,
    pub hearing: &'a Hearing<'a>,
    pub abandon: &'a dyn Abandon,
}

impl<B: Backend> Cx<'_, B> {
    async fn solved(
        &self,
        files: &[(String, String)],
        cost: &Cost,
        steps: usize,
        draws: usize,
    ) -> Result<Solution, CliError> {
        let over = Overlay::new(self.problem.source, files.iter().cloned().collect());
        let problem = self.over(&over, cost.clone());
        let settings = Settings {
            steps,
            draws,
            ..*self.settings
        };
        prox::solve(&problem, &settings, self.tier, self.hearing, self.abandon).await
    }

    pub(crate) async fn scheduled(
        &self,
        files: &[(String, String)],
        cost: &Cost,
        steps: usize,
        draws: usize,
    ) -> Result<Solution, CliError> {
        let cap = self.settings.draws;
        let mut solution = self.solved(files, cost, steps, draws).await?;
        let mut limited = solution.descents[0].noise_limited;
        while limited && !solution.cancelled() && solution.draws() < cap {
            let from = merged(files, &solution.files);
            let twice = (2 * solution.draws()).min(cap);
            let next = self.solved(&from, cost, steps, twice).await?;
            limited = next.descents[0].noise_limited;
            let descents = [solution.descents, next.descents].concat();
            solution = Solution {
                files: merged(&solution.files, &next.files),
                descents,
                ..next
            };
        }
        Ok(solution)
    }

    /// `files`' reference read and model's state, as a solve of them at `draws` starts.
    async fn prepared(
        &self,
        files: &[(String, String)],
        cost: &Cost,
        draws: usize,
    ) -> Result<Prepared, CliError> {
        let over = Overlay::new(self.problem.source, files.iter().cloned().collect());
        let problem = self.over(&over, cost.clone());
        let base = recorded(&problem.cost, problem.source);
        let source = crate::overlaid(&base, &problem.sets)?;
        prox::run::prepared(
            &problem,
            &source,
            draws,
            self.tier,
            self.hearing,
            self.abandon,
        )
        .await
    }

    /// `solution` predicted at the cap's draws, where it finished.
    pub(crate) async fn reached(
        &self,
        stage: Stage,
        solution: Solution,
    ) -> Result<Reached, CliError> {
        let at_cap = match solution.cancelled() || self.abandon.abandoned().await {
            true => None,
            false => {
                let files = merged(&stage.files, &solution.files);
                let evaluated = async {
                    let at = self
                        .prepared(&files, &stage.cost, self.settings.draws)
                        .await?;
                    at.fit.noised(&at.state, self.hearing)?;
                    Ok(evaluate(&at.fit, &at.state, false, at.fit.threads)?.residual)
                };
                let act = Act::new(Action::EvaluatingAtCap);
                Some(self.hearing.during(act, evaluated).await?)
            }
        };
        Ok(Reached {
            stage,
            solution,
            at_cap,
            measured: None,
        })
    }

    async fn measured(&self, stage: &Stage, solution: &Solution) -> Result<Residual, CliError> {
        let over = Overlay::new(self.problem.source, stage.files.iter().cloned().collect());
        let problem = self.over(&over, stage.cost.clone());
        let draws = self.settings.draws;
        let (tier, hearing) = (self.tier, self.hearing);
        prox::measure(
            &problem,
            &solution.files,
            draws,
            tier,
            hearing,
            self.abandon,
        )
        .await
    }

    fn over<'s>(&'s self, source: &'s dyn Source, cost: Cost) -> Problem<'s> {
        let p = self.problem;
        Problem {
            source,
            texts: p.texts,
            cost,
            vary: p.vary.clone(),
            sets: p.sets.clone(),
            rate: p.rate,
            threads: p.threads,
            stops: p.stops,
        }
    }

    fn model(&self, files: &[(String, String)]) -> Result<Model, CliError> {
        let over = Overlay::new(self.problem.source, files.iter().cloned().collect());
        let set = crate::overlaid(&over, &self.problem.sets)?;
        let root = self.shape.root.as_str();
        let graph = sva_ast::load_reaching(&set, &[root]).map_err(CliError::Refusals)?;
        Model::of(&graph, root)
    }

    /// The seconds a bar of the composition lasts, where it states a tempo.
    fn per_bar(&self) -> Result<Option<f64>, CliError> {
        let set = crate::overlaid(self.problem.source, &self.problem.sets)?;
        let root = self.shape.root.as_str();
        let graph = sva_ast::load_reaching(&set, &[root]).map_err(CliError::Refusals)?;
        Ok(graph.tempo().map(|t| t.seconds_per_bar()))
    }

    fn texts<'f>(&self, files: &'f [(String, String)]) -> impl Fn(&str) -> Option<String> + 'f
    where
        Self: 'f,
    {
        let source = self.problem.source;
        move |p: &str| match files.iter().find(|(f, _)| f == p) {
            Some((_, t)) => Some(t.clone()),
            None => source.get(p).ok().flatten().map(|t| t.into_owned()),
        }
    }
}

/// The stages of one scheme, each from what the last one reached, staged in `cx.hearing`.
trait Stages<B: Backend> {
    async fn next(
        &mut self,
        last: Option<&Reached>,
        cx: &Cx<'_, B>,
    ) -> Result<Option<Built>, CliError>;
}

fn scaled(steps: usize, fraction: f64) -> usize {
    ((steps as f64 * fraction).round() as usize).max(1)
}

impl Reached {
    /// The line that lists it.
    pub fn listed(&self, files: &[(String, String)], least: bool, hearing: &Hearing) {
        hearing.heard(&Heard::Listed {
            label: &self.stage.label,
            solution: &self.solution,
            at_cap: self.at_cap.as_ref(),
            length: self.length(),
            files,
            least,
        });
    }
}

/// `cx.hearing` hears each step, each activity and each stage as it is listed. The answer is
/// unmeasured: [`Ran::measured`].
pub async fn run<B: Backend>(scheme: &Scheme, cx: &Cx<'_, B>) -> Result<Ran, CliError> {
    match scheme {
        Scheme::Path { prices, rounds } => {
            let prices = prices.clone().unwrap_or_else(|| vec![cx.shape.price]);
            let mut stages = path::Path::new(prices, *rounds, cx.settings.steps);
            driven(&mut stages, cx).await
        }
        Scheme::Grow { ceiling, rounds } => {
            let mut stages = grow::Grow::new(cx, *ceiling, *rounds)?;
            driven(&mut stages, cx).await
        }
    }
}

async fn driven<B: Backend>(stages: &mut impl Stages<B>, cx: &Cx<'_, B>) -> Result<Ran, CliError> {
    let mut reached: Vec<Reached> = Vec::new();
    let mut least: Option<f64> = None;
    let mut listed = |r: &Reached| -> Result<(), CliError> {
        r.listed(&written(cx, r)?, below(r, &mut least), cx.hearing);
        Ok(())
    };
    let mut cancelled = false;
    loop {
        let Some(built) = stages.next(reached.last(), cx).await? else {
            break;
        };
        let stage = match built {
            Built::Stage(stage) => stage,
            Built::Stopped(r) => {
                listed(&r)?;
                reached.push(r);
                cancelled = true;
                break;
            }
        };
        if cx.abandon.abandoned().await {
            cancelled = true;
            break;
        }
        let draws = started(reached.last());
        let solution = cx
            .scheduled(&stage.files, &stage.cost, stage.steps, draws)
            .await?;
        let r = cx.reached(stage, solution).await?;
        cancelled = r.at_cap.is_none() || cx.abandon.abandoned().await;
        listed(&r)?;
        reached.push(r);
        if cancelled {
            break;
        }
    }
    let cancelled = cancelled || cx.abandon.abandoned().await;
    Ok(Ran::of(reached, cancelled))
}

/// The files the chosen composition changes over the problem's, a new node no line reads left out.
pub fn written<B: Backend>(
    cx: &Cx<'_, B>,
    reached: &Reached,
) -> Result<Vec<(String, String)>, CliError> {
    let files = reached.files();
    let model = cx.model(&files)?;
    let source = cx.problem.source;
    Ok(files
        .into_iter()
        .filter(|(path, text)| match source.get(path).ok().flatten() {
            Some(was) => *was != **text,
            None => path.ends_with(".jsonl") || model.nodes.contains_key(path),
        })
        .collect())
}
