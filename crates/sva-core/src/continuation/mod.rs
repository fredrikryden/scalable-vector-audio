// Concern: runs a scheme: each stage solved fresh and measured over the last one's files, stopped on a signal, the least length chosen | Non-concern: what any stage is | IO: (Problem, Scheme) -> Ran

mod grow;
pub mod improve;
pub mod path;

use std::collections::BTreeMap;

use sva_ast::{Overlay, Source};
use sva_engine::{Abandon, Backend, Tier};

use crate::cli_error::CliError;
use crate::model::Model;
use crate::seed::{LIMITS, Limits};
use crate::solve::prox::eval::Residual;
use crate::solve::prox::{self, Progress, Settings, Shape, Solution};
use crate::solve::{Cost, Problem};

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
    pub measured: Option<Residual>,
}

impl Reached {
    /// The reached composition over the problem's: the stage's files, then the solution's.
    pub fn files(&self) -> Vec<(String, String)> {
        merged(&self.stage.files, &self.solution.files)
    }

    pub fn residual_bits(&self) -> f64 {
        self.measured.unwrap_or(self.solution.predicted).bits
    }

    pub fn length(&self) -> f64 {
        self.residual_bits() + self.solution.bits
    }
}

pub struct Ran {
    pub stages: Vec<Reached>,
    pub chosen: Option<usize>,
    pub cancelled: bool,
}

impl Ran {
    /// A run stopped by a signal is chosen on its predictions alone.
    pub fn of(stages: Vec<Reached>, cancelled: bool) -> Ran {
        let length = |r: &Reached| match cancelled {
            true => r.solution.predicted.bits + r.solution.bits,
            false => r.length(),
        };
        let chosen =
            (0..stages.len()).min_by(|a, b| length(&stages[*a]).total_cmp(&length(&stages[*b])));
        Ran {
            stages,
            chosen,
            cancelled,
        }
    }
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
    pub abandon: &'a dyn Abandon,
}

impl<B: Backend> Cx<'_, B> {
    async fn solved(
        &self,
        files: &[(String, String)],
        cost: &Cost,
        steps: usize,
        report: &mut dyn FnMut(&Progress),
    ) -> Result<Solution, CliError> {
        let over = Overlay::new(self.problem.source, files.iter().cloned().collect());
        let problem = self.over(&over, cost.clone());
        let settings = Settings { steps };
        prox::solve(&problem, &settings, self.tier, report, self.abandon).await
    }

    async fn measured(&self, stage: &Stage, solution: &Solution) -> Result<Residual, CliError> {
        let over = Overlay::new(self.problem.source, stage.files.iter().cloned().collect());
        let problem = self.over(&over, stage.cost.clone());
        prox::measure(&problem, &solution.files, self.tier, self.abandon).await
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

/// The stages of one scheme, each from what the last one reached.
trait Stages<B: Backend> {
    async fn next(
        &mut self,
        last: Option<&Reached>,
        cx: &Cx<'_, B>,
    ) -> Result<Option<Stage>, CliError>;
}

fn scaled(steps: usize, fraction: f64) -> usize {
    ((steps as f64 * fraction).round() as usize).max(1)
}

/// `report` hears each step under its stage's label.
pub async fn run<B: Backend>(
    scheme: &Scheme,
    cx: &Cx<'_, B>,
    report: &mut dyn FnMut(&str, &Progress),
) -> Result<Ran, CliError> {
    match scheme {
        Scheme::Path { prices, rounds } => {
            let prices = prices.clone().unwrap_or_else(|| vec![cx.shape.price]);
            let mut stages = path::Path::new(prices, *rounds, cx.settings.steps);
            driven(&mut stages, cx, report).await
        }
        Scheme::Grow { ceiling, rounds } => {
            let mut stages = grow::Grow::new(cx, *ceiling, *rounds)?;
            driven(&mut stages, cx, report).await
        }
    }
}

async fn driven<B: Backend>(
    stages: &mut impl Stages<B>,
    cx: &Cx<'_, B>,
    report: &mut dyn FnMut(&str, &Progress),
) -> Result<Ran, CliError> {
    let mut reached: Vec<Reached> = Vec::new();
    let mut cancelled = false;
    while let Some(stage) = stages.next(reached.last(), cx).await? {
        if cx.abandon.abandoned().await {
            cancelled = true;
            break;
        }
        let label = stage.label.clone();
        let mut heard = |p: &Progress| report(&label, p);
        let solution = cx
            .solved(&stage.files, &stage.cost, stage.steps, &mut heard)
            .await?;
        cancelled = solution.cancelled() || cx.abandon.abandoned().await;
        let measured = match cancelled {
            true => None,
            false => Some(cx.measured(&stage, &solution).await?),
        };
        reached.push(Reached {
            stage,
            solution,
            measured,
        });
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
