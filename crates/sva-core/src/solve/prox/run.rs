// Concern: one priced solve: the cost read as a d' plus a price on bits, the reference read once, the path, its choice's files | Non-concern: argv, writing | IO: (Problem, Tier) -> Fitted

use sva_ast::Source;
use sva_engine::{Abandon, Backend, Output, Session, Tier};

use sva_samples::measure::internal::{self, Ear, Grid, Internal};

use super::draws::DRAWS;
use super::eval::{Fit, residual, weights};
use super::model::Model;
use super::played::Renders;
use super::state::{State, Var};
use super::{DEFAULT_STEPS, Point, Progress, chosen, path, write};
use crate::cli_error::CliError;
use crate::query::{asked, call};
use crate::solve::cost::{Node, Op};
use crate::solve::{Problem, recorded};
use crate::{Job, execute_over, overlaid};

/// `--method prox`'s own settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Priced {
    /// The prices per number the path visits; `None` takes the cost's own.
    pub path: Option<Vec<f64>>,
    pub steps: usize,
}

impl Default for Priced {
    fn default() -> Priced {
        Priced {
            path: None,
            steps: DEFAULT_STEPS,
        }
    }
}

pub struct Fitted {
    pub model: String,
    pub reference: String,
    pub points: Vec<Point>,
    pub chosen: Option<usize>,
    pub files: Vec<(String, String)>,
    pub cancelled: bool,
}

const SHAPE: &str = "under `--method prox` a cost is `norm(repr.internal(@<model>) - \
     repr.internal(<reference>)) + <price>*repr.numbers(@<model>)`: the d' of the model from \
     the reference, plus a price in d' on each bit of the numbers the model writes";

fn shaped(what: String) -> CliError {
    CliError::usage("solve.prox_cost", format!("{what}; {SHAPE}"))
}

/// The internal pair's leaves and the price on the model's written numbers, off the cost's terms.
fn read(problem: &Problem) -> Result<(usize, usize, String, f64), CliError> {
    let cost = &problem.cost;
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
    if cost.leaves[a].target != sized {
        return Err(shaped(format!(
            "`repr.numbers({sized})` prices another target than the distance's model `{}`",
            cost.leaves[a].target
        )));
    }
    let root = sized
        .strip_prefix('@')
        .filter(|p| sva_ast::whole_ref_path(p))
        .ok_or_else(|| shaped(format!("`{sized}` is no bare `@<node>`")))?;
    Ok((a, b, root.to_string(), lambda))
}

struct Reading<'a, B: Backend> {
    problem: &'a Problem<'a>,
    tier: &'a Tier<B>,
    abandon: &'a dyn Abandon,
    /// The leaf whose reading every render here is taken as.
    leaf: &'a crate::solve::Leaf,
}

impl<B: Backend> Reading<'_, B> {
    /// `target` rendered over `source` and read as `internal`, and its channels averaged.
    async fn read(
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
const FIT_RATE: u32 = 48_000;

pub async fn fitted<B: Backend>(
    problem: &Problem<'_>,
    priced: &Priced,
    tier: &Tier<B>,
    report: &mut dyn FnMut(&Progress),
    abandon: &dyn Abandon,
) -> Result<Fitted, CliError> {
    let (model_leaf, reference_leaf, root, lambda) = read(problem)?;
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
    let leaf = &problem.cost.leaves[reference_leaf];
    let reading = Reading {
        problem,
        tier,
        abandon,
        leaf: &problem.cost.leaves[model_leaf],
    };
    let (clip, mono) = reading.read(&source, &leaf.target).await?;
    let frames = mono.len();
    let graph = sva_ast::load_reaching(&source, &[root.as_str()]).map_err(CliError::Refusals)?;
    let model = Model::of(&graph, &root)?;
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
    let grid = Grid::of(clip.rate, mono.len());
    let ear = Ear::of(grid, clip.spl_db);
    let fit = Fit {
        thirds: ear.thirds(&internal::bin_power(&mono, &grid)),
        reference: clip,
        ear: std::sync::Arc::new(ear),
        threads,
        texts,
        renders: Renders::default(),
    };
    let lambdas = priced.path.clone().unwrap_or_else(|| vec![lambda]);
    let steps = priced.steps;
    let (mut points, cancelled) = path(&fit, state, &lambdas, steps, abandon, report).await?;
    let text = |p: &str| -> Option<String> { source.get(p).ok().flatten().map(|t| t.into_owned()) };
    let model = format!("@{root}([0, {frames}sp])");
    for point in &mut points {
        let mut draws = Vec::with_capacity(DRAWS);
        for k in 0..DRAWS {
            let files = write::drawn(&point.state, &text, k)?;
            let over = sva_ast::Overlay::new(&source, files.into_iter().collect());
            let (rendered, _) = reading.read(&over, &model).await?;
            draws.push(rendered.elements);
        }
        point.measured = Some(residual(&fit, &weights(&fit), &draws));
    }
    let chosen = chosen(&points);
    let files = match chosen {
        Some(c) => write::files(&points[c].state, &text)?,
        None => Vec::new(),
    };
    Ok(Fitted {
        model: problem.cost.leaves[model_leaf].target.clone(),
        reference: leaf.target.clone(),
        points,
        chosen,
        files,
        cancelled,
    })
}
