// Concern: one priced solve: the cost read as a d' plus a price on bits, the reference read, the descent, the reached files | Non-concern: argv, stages | IO: (Problem, Settings, Tier) -> Solution

use sva_ast::Source;
use sva_engine::{Abandon, Backend, Output, Session, Tier};

use sva_samples::measure::internal::{self, Ear, Grid, Internal};

use super::eval::{Fit, Residual, evaluate};
use super::played::Renders;
use super::state::{State, Var, reaches};
use super::{DEFAULT_DRAWS, DEFAULT_STEPS, Descended, Ended, Progress, descended, price, write};
use crate::cli_error::CliError;
use crate::model::Model;
use crate::query::{asked, call};
use crate::solve::cost::{Cost, Node, Op};
use crate::solve::{Leaf, Problem, recorded};
use crate::target::Edge;
use crate::{Job, execute_over, overlaid};

/// `--method prox`'s own settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The priced descent's cap on its steps; the debias's is a quarter as many.
    pub steps: usize,
    /// The draws each loss averages, a prefix of each line's; a fit's cap on its solves' own.
    pub draws: usize,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            steps: DEFAULT_STEPS,
            draws: DEFAULT_DRAWS,
        }
    }
}

pub struct Solution {
    pub price: f64,
    /// Every file the reached composition changes over the problem's source.
    pub files: Vec<(String, String)>,
    pub numbers: usize,
    pub bits: f64,
    /// The residual the descent lowers: the reading of the render's own lowered lines.
    pub predicted: Residual,
    /// The reading's cells, weighted by the independent samples each carries.
    pub cells: f64,
    /// Each phase run: the descent, then the debias where priced and no signal ended it.
    pub descents: Vec<Descended>,
}

impl Solution {
    pub fn cancelled(&self) -> bool {
        self.descents.iter().any(|d| d.ended == Ended::Signal)
    }

    pub fn draws(&self) -> usize {
        self.descents
            .last()
            .expect("a solve runs its descent")
            .draws
    }

    pub fn objective(&self) -> f64 {
        self.predicted.bits + self.price * self.bits
    }
}

/// A priced cost's parts: its model's root, the price on its numbers, both leaves' targets.
pub struct Shape {
    pub root: String,
    pub price: f64,
    pub model: String,
    pub reference: String,
    /// The interval both readings are written over; `None` reads the reference whole.
    pub interval: Option<(Edge, Edge)>,
    pub(super) model_leaf: usize,
}

impl Shape {
    /// The reference as the fit renders it: from 0 to its interval's end, so the reading of
    /// each frame inside the interval holds what came before it.
    pub(crate) fn read(&self) -> Result<String, CliError> {
        crate::target::from_zero(&self.reference)
    }

    /// The interval's start and its end where closed, in samples at the rate the fit renders
    /// at, so an end in `sp` is that render's sample; [`shape`] refused a start before 0.
    pub(crate) fn span(&self, per_bar: Option<f64>) -> Result<(usize, Option<usize>), CliError> {
        let Some((start, end)) = self.interval else {
            return Ok((0, None));
        };
        let at = |e: Edge| e.sample(FIT_RATE, per_bar).map(|s| s.map(|s| s as usize));
        Ok((at(start)?.unwrap_or(0), at(end)?))
    }
}

const SHAPE: &str = "under `--method prox` a cost is `norm(repr.internal(@<model>) - \
     repr.internal(<reference>)) + <price>*repr.numbers(@<model>)`: the d' of the model from \
     the reference, plus a price in d' on each bit of the numbers the model writes; the d' may \
     read an interval, `@<model>([a, b])` and the reference over the same `[a, b]`, as \
     `sample(\"<clip>\")([a, b])`";

fn shaped(what: String) -> CliError {
    CliError::usage("solve.prox_cost", format!("{what}; {SHAPE}"))
}

/// `[a, b]` as a target writes it.
fn written((start, end): (Edge, Edge)) -> String {
    let edge = |e: Edge| match e {
        Edge::Secs(0.0) => "0".to_string(),
        Edge::Secs(s) => format!("{s}s"),
        Edge::Bars(b) => format!("{b}b"),
        Edge::Samples(n) => format!("{n}sp"),
        Edge::Inf => "inf".to_string(),
    };
    match end {
        Edge::Inf => format!("[{}, inf)", edge(start)),
        end => format!("[{}, {}]", edge(start), edge(end)),
    }
}

pub fn shape(cost: &Cost) -> Result<Shape, CliError> {
    let mut pair = None;
    let mut price = None;
    for term in &cost.terms {
        let node = match &term.node {
            Node::Reduce("norm", args) if args.len() == 1 => &args[0],
            other => other,
        };
        match node {
            Node::Bin(Op::Sub, a, b) => match (&**a, &**b) {
                (Node::Leaf(x), Node::Leaf(y))
                    if cost.leaves[*x].asked.name == "internal"
                        && matches!(term.node, Node::Reduce("norm", _)) =>
                {
                    pair = Some((*x, *y));
                }
                _ => {
                    return Err(shaped(format!(
                        "`{}` is no norm of two internal readings",
                        term.text
                    )));
                }
            },
            Node::Bin(Op::Mul, a, b) => match (&**a, &**b) {
                (Node::Num(n), Node::Leaf(x)) | (Node::Leaf(x), Node::Num(n))
                    if cost.leaves[*x].asked.name == "numbers" =>
                {
                    price = Some((cost.leaves[*x].target.clone(), n * term.sign));
                }
                _ => return Err(shaped(format!("`{}` is neither part", term.text))),
            },
            Node::Leaf(x) if cost.leaves[*x].asked.name == "numbers" => {
                price = Some((cost.leaves[*x].target.clone(), term.sign))
            }
            _ => return Err(shaped(format!("`{}` is neither part", term.text))),
        }
    }
    let (a, b) = pair.ok_or_else(|| shaped("the cost holds no internal d'".to_string()))?;
    let (sized, lambda) =
        price.ok_or_else(|| shaped("the cost prices no `repr.numbers`".to_string()))?;
    let root = sized
        .strip_prefix('@')
        .filter(|p| sva_ast::whole_ref_path(p))
        .ok_or_else(|| shaped(format!("`{sized}` is no bare `@<node>`")))?;
    let model = &cost.leaves[a].target;
    let reference = &cost.leaves[b].target;
    let read = crate::target::target(model)?;
    let tight = |t: &str| t.split_whitespace().collect::<String>();
    let bare = match read.interval {
        Some(_) => tight(&read.expr) == format!("{sized}(t)"),
        None => *model == sized,
    };
    if !bare {
        return Err(shaped(format!(
            "`repr.numbers({sized})` prices another target than the distance's model `{model}`; \
             the model is `{sized}`, or `{sized}([a, b])` over an interval"
        )));
    }
    let interval = read.interval;
    let theirs = crate::target::target(reference)?.interval;
    if theirs != interval {
        let fix = match interval {
            Some(i) => format!("read the reference over `{}` as well", written(i)),
            None => format!(
                "read the model over the reference's interval, `{sized}({})`",
                written(theirs.expect("the two differ"))
            ),
        };
        let shown = |i: Option<(Edge, Edge)>| i.map_or("its whole span".to_string(), written);
        return Err(shaped(format!(
            "the model `{model}` reads {} and the reference `{reference}` {}; {fix}",
            shown(interval),
            shown(theirs)
        )));
    }
    if let Some((start, _)) = interval {
        let negative = match start {
            Edge::Secs(x) | Edge::Bars(x) | Edge::Samples(x) => x < 0.0,
            Edge::Inf => false,
        };
        if negative {
            return Err(shaped(format!(
                "`{model}` starts its interval before 0, where a priced fit reads nothing; \
                 start it at 0 or later"
            )));
        }
    }
    Ok(Shape {
        root: root.to_string(),
        price: lambda,
        model: model.clone(),
        reference: reference.clone(),
        interval,
        model_leaf: a,
    })
}

/// Every check a priced fit needs before anything renders: the cost's shape, the rate, each
/// `--vary`'s step and what it reaches in the composition as given.
pub fn checked(problem: &Problem) -> Result<Shape, CliError> {
    let shape = shape(&problem.cost)?;
    if let Some(rate) = problem.rate.filter(|r| *r != FIT_RATE) {
        return Err(CliError::usage(
            "solve.prox_rate",
            format!(
                "`--rate {rate}`: a priced fit renders at {FIT_RATE} Hz; drop `--rate`, and a \
                 recording at another rate is resampled to it"
            ),
        ));
    }
    let vars = problem
        .vary
        .iter()
        .map(Var::of)
        .collect::<Result<Vec<_>, _>>()?;
    let base = recorded(&problem.cost, problem.source);
    let source = overlaid(&base, &problem.sets)?;
    let graph =
        sva_ast::load_reaching(&source, &[shape.root.as_str()]).map_err(CliError::Refusals)?;
    reaches(&Model::of(&graph, &shape.root)?, &vars)?;
    Ok(shape)
}

pub(super) struct Reading<'a, B: Backend> {
    pub problem: &'a Problem<'a>,
    pub tier: &'a Tier<B>,
    pub abandon: &'a dyn Abandon,
    /// The leaf whose reading every render here is taken as.
    pub leaf: &'a Leaf,
}

impl<B: Backend> Reading<'_, B> {
    /// `target` rendered over `source` and read as `internal`, and its channels averaged.
    pub async fn read(
        &self,
        source: &dyn Source,
        target: &str,
    ) -> Result<(Internal, Vec<f64>), CliError> {
        let samples = asked(&call("samples")?)?;
        let job = Job {
            rate: Some(FIT_RATE),
            threads: self.problem.threads,
            asked: std::slice::from_ref(&samples),
            abandon: self.abandon,
            ..Job::over(source, target)
        };
        let rendered = execute_over(job, self.tier, &mut Session::default()).await?;
        let Output::Samples(buffer) = rendered.answered(&samples)?.value else {
            unreachable!("`samples` answers samples");
        };
        let per_bar = rendered.graph.tempo().map(|t| t.seconds_per_bar());
        let representation = self
            .leaf
            .asked
            .representation(rendered.config.rate, per_bar)?;
        let profile = rendered.config.profile.name;
        let planes: Vec<&[f64]> = (0..buffer.width()).map(|c| buffer.plane(c)).collect();
        let mono = internal::mono(&planes);
        let answer =
            sva_engine::answer_buffer(&self.leaf.written, &buffer, representation, profile)?;
        let Output::Internal(read) = answer.value else {
            unreachable!("`internal` answers its reading");
        };
        Ok((*read, mono))
    }
}

/// The rate a priced fit reads its reference and renders its model at.
pub(crate) const FIT_RATE: u32 = 48_000;

/// What a solve and a measure both read before either renders the model.
pub(crate) struct Prepared {
    pub shape: Shape,
    pub fit: Fit,
    pub state: State,
    pub frames: usize,
}

/// The problem's cost read, its reference rendered and read, and its model's state over `source`.
pub(crate) async fn prepared<B: Backend>(
    problem: &Problem<'_>,
    source: &dyn Source,
    draws: usize,
    tier: &Tier<B>,
    abandon: &dyn Abandon,
) -> Result<Prepared, CliError> {
    let shape = shape(&problem.cost)?;
    let vars = problem
        .vary
        .iter()
        .map(Var::of)
        .collect::<Result<Vec<_>, _>>()?;
    let reading = Reading {
        problem,
        tier,
        abandon,
        leaf: &problem.cost.leaves[shape.model_leaf],
    };
    let (clip, mono) = reading.read(source, &shape.read()?).await?;
    let root = shape.root.as_str();
    let graph = sva_ast::load_reaching(source, &[root]).map_err(CliError::Refusals)?;
    let grid = Grid::of(clip.rate, mono.len());
    let per_bar = graph.tempo().map(|t| t.seconds_per_bar());
    let (start, _) = shape.span(per_bar)?;
    let from = start.div_ceil(grid.hop);
    if from >= grid.frames {
        return Err(shaped(format!(
            "`{}` starts its interval at sample {start}, past the reference's last frame at \
             sample {}; start it inside the reference",
            shape.model,
            (grid.frames - 1) * grid.hop
        )));
    }
    let model = Model::of(&graph, root)?;
    let state = State::of(&model, &vars)?;
    let threads = problem.threads.unwrap_or_else(sva_engine::default_threads);
    let mut texts = std::collections::BTreeMap::new();
    let paths = model
        .nodes
        .keys()
        .cloned()
        .chain(model.score.iter().map(|s| format!("{s}.jsonl")));
    for p in paths {
        if let Some(t) = source.get(&p).ok().flatten() {
            texts.insert(p, t.into_owned());
        }
    }
    let ear = Ear::of(grid, clip.spl_db);
    let fit = Fit {
        thirds: ear.thirds(&internal::bin_power(&mono, &grid)),
        reference: clip,
        from,
        ear: std::sync::Arc::new(ear),
        threads,
        draws,
        texts,
        renders: Renders::default(),
    };
    Ok(Prepared {
        shape,
        fit,
        state,
        frames: mono.len(),
    })
}

/// The problem's composition descended at its cost's price from the numbers it holds, each
/// phase until its objective stops falling or `settings.steps` caps it; zero steps answer the
/// start.
pub async fn solve<B: Backend>(
    problem: &Problem<'_>,
    settings: &Settings,
    tier: &Tier<B>,
    report: &mut dyn FnMut(&Progress),
    abandon: &dyn Abandon,
) -> Result<Solution, CliError> {
    let base = recorded(&problem.cost, problem.source);
    let source = overlaid(&base, &problem.sets)?;
    let Prepared {
        shape,
        fit,
        mut state,
        ..
    } = prepared(problem, &source, settings.draws, tier, abandon).await?;
    let lambda = shape.price;
    let descents = descended(&fit, &mut state, lambda, settings.steps, abandon, report).await?;
    let e = evaluate(&fit, &state, false, fit.threads)?;
    let (numbers, bits) = price::numbers(&state);
    let text = |p: &str| -> Option<String> { source.get(p).ok().flatten().map(|t| t.into_owned()) };
    Ok(Solution {
        price: lambda,
        files: write::files(&state, &text)?,
        numbers,
        bits,
        predicted: e.residual,
        cells: super::eval::weights(&fit).iter().sum(),
        descents,
    })
}
