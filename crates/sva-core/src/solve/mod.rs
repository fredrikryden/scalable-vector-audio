// Concern: one solve, each point every cost target rendered through render's path and read | Non-concern: argv, signals, files (sva-cli) | IO: (Problem, Tier) -> Solved or CliError

mod cost;
mod residual;
mod search;
mod vary;

pub use cost::{Cost, Leaf, REDUCERS, SUBTRACTS, cost};
pub use residual::{FLOOR_DB, MISSED_NOTE_CENTS, MISSED_ONSET_SECS, RULES};
pub use search::{
    DEFAULT_COST_TOL, DEFAULT_MAX_RENDERS, DEFAULT_STEP_TOL, FIRST_STEP_OF_SPAN, Stopped, Stops,
};
pub use vary::{Precision, Vary, vary};

use std::num::NonZeroUsize;

use sva_ast::Source;
use sva_engine::{Abandon, Backend, EngineError, Output, Resampled, Session, Tier};

use crate::cli_error::CliError;
use crate::json::{escape, list, num};
use crate::output::{Diagnostic, Severity};
use crate::query::{Asked, asked, call};
use crate::{Job, execute_over, overlaid};
use cost::{Node, Op};
use search::{Next, Search};

pub struct Problem<'a> {
    pub source: &'a dyn Source,
    pub cost: Cost,
    pub vary: Vec<Vary>,
    /// Variables held at these texts throughout.
    pub sets: Vec<(String, String)>,
    pub rate: Option<u32>,
    pub threads: Option<NonZeroUsize>,
    pub stops: Stops,
}

/// One render of every target at one point, and what the cost made of it.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub render: usize,
    pub values: Vec<(String, f64, String)>,
    pub total: f64,
    /// The least cost of this render and every one before it.
    pub best: f64,
    pub terms: Vec<f64>,
    /// Each target's samples computed this render, beside the target: none where every value
    /// was read back from the store.
    pub computed: Vec<(String, u64)>,
}

pub struct Solved {
    pub cost: String,
    pub terms: Vec<String>,
    pub start: Vec<String>,
    pub best: Point,
    pub on_bound: Vec<bool>,
    pub renders: usize,
    pub stopped_by: Stopped,
    pub trace: Vec<Point>,
    pub resampled: Vec<Resampled>,
}

/// Every check the cost and the variables need, before anything renders.
pub fn checked(problem: &Problem) -> Result<Vec<f64>, CliError> {
    for one in &problem.vary {
        if let Some((name, _)) = problem.sets.iter().find(|(n, _)| *n == one.name) {
            return Err(CliError::usage(
                "solve.set_and_vary",
                format!(
                    "`{name}` is both held by `--set` and varied by `--vary`; a variable is one \
                     or the other"
                ),
            ));
        }
        if problem.vary.iter().filter(|v| v.name == one.name).count() > 1 {
            return Err(CliError::usage(
                "solve.vary_twice",
                format!("`{}` is varied twice; vary each variable once", one.name),
            ));
        }
    }
    let held = overlaid(problem.source, &problem.sets)?;
    let start = problem
        .vary
        .iter()
        .map(|one| one.start(&held))
        .collect::<Result<Vec<_>, _>>()?;
    for one in &problem.vary {
        reaches(problem, one)?;
    }
    Ok(start)
}

/// A variable no target's value or interval moves with refuses: no render could fit it.
fn reaches(problem: &Problem, one: &Vary) -> Result<(), CliError> {
    let planned = |value: f64| -> Result<Vec<String>, CliError> {
        let mut sets = problem.sets.clone();
        sets.push((one.name.clone(), one.text(value, Precision::Shortest)));
        let source = overlaid(problem.source, &sets)?;
        targets(&problem.cost)
            .iter()
            .map(|target| {
                let job = Job {
                    rate: problem.rate,
                    ..Job::over(&source, target)
                };
                let (render, _) = crate::plan(&job)?;
                let identity = sva_engine::identity(&render.tys, render.root)?;
                Ok(format!("{identity:?} {:?}", render.config.range))
            })
            .collect()
    };
    if planned(one.lo)? == planned(one.hi)? {
        return Err(CliError::usage(
            "solve.reaches_nothing",
            format!(
                "`{}` reaches no target of the cost: every target renders the same at `{}` as \
                 at `{}`; vary a variable a target reads, as `@variables/{}` in its node, or add \
                 a target that reads it",
                one.name,
                one.text(one.lo, Precision::Shortest),
                one.text(one.hi, Precision::Shortest),
                one.name
            ),
        ));
    }
    Ok(())
}

fn targets(cost: &Cost) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for leaf in &cost.leaves {
        if !out.contains(&leaf.target) {
            out.push(leaf.target.clone());
        }
    }
    out
}

/// `heard` hears each point as it is rendered; `abandon` stops the search between renders
/// and the render in flight.
pub async fn solve<B: Backend>(
    problem: &Problem<'_>,
    tier: &Tier<B>,
    heard: &mut dyn FnMut(&Point),
    abandon: &dyn Abandon,
) -> Result<Solved, CliError> {
    let start = checked(problem)?;
    let mut search = Search::new(&problem.vary, &start, problem.stops);
    let targets = targets(&problem.cost);
    let mut sessions: Vec<Session> = targets.iter().map(|_| Session::default()).collect();
    let samples = asked(&call("samples")?)?;
    let mut trace: Vec<Point> = Vec::new();
    let mut resampled: Vec<Resampled> = Vec::new();
    let stopped_by = loop {
        let values = match search.next() {
            Next::Done(stopped) => break stopped,
            Next::Render(values) => values,
        };
        if abandon.abandoned().await {
            search.cancel();
            continue;
        }
        let digits = search.digits();
        let written: Vec<(String, f64, String)> = problem
            .vary
            .iter()
            .zip(values.iter().zip(digits))
            .map(|(one, (v, d))| {
                let text = one.text(*v, d);
                (one.name.clone(), one.number(&text), text)
            })
            .collect();
        let point = Rendering {
            problem,
            targets: &targets,
            samples: &samples,
            abandon,
        };
        let evaluated = point
            .evaluated(&written, tier, &mut sessions, &mut resampled)
            .await;
        let (total, terms, computed) = match evaluated {
            Err(CliError::Engine(EngineError::Abandoned)) => {
                search.cancel();
                continue;
            }
            other => other?,
        };
        let point = Point {
            render: trace.len() + 1,
            values: written,
            total,
            best: trace
                .last()
                .map_or(total, |last: &Point| last.best.min(total)),
            terms,
            computed,
        };
        heard(&point);
        trace.push(point);
        search.tell(total);
    };
    let best = trace
        .iter()
        .reduce(|held, p| if p.total < held.total { p } else { held })
        .cloned()
        .ok_or_else(|| {
            CliError::usage(
                "solve.no_render",
                "the solve was stopped before its first render finished, so it holds no point",
            )
        })?;
    Ok(Solved {
        cost: problem.cost.text.clone(),
        terms: problem.cost.terms.iter().map(|t| t.text.clone()).collect(),
        start: problem
            .vary
            .iter()
            .zip(&start)
            .map(|(one, v)| one.text(*v, Precision::Shortest))
            .collect(),
        on_bound: best
            .values
            .iter()
            .zip(&problem.vary)
            .map(|((_, v, _), one)| *v <= one.lo || *v >= one.hi)
            .collect(),
        best,
        renders: trace.len(),
        stopped_by,
        trace,
        resampled,
    })
}

struct Rendering<'p, 'a> {
    problem: &'p Problem<'a>,
    targets: &'p [String],
    samples: &'p Asked,
    abandon: &'p dyn Abandon,
}

type Evaluated = (f64, Vec<f64>, Vec<(String, u64)>);

impl Rendering<'_, '_> {
    /// Every target rendered once at `written`, then every leaf read off its target's samples.
    async fn evaluated<B: Backend>(
        &self,
        written: &[(String, f64, String)],
        tier: &Tier<B>,
        sessions: &mut [Session],
        resampled: &mut Vec<Resampled>,
    ) -> Result<Evaluated, CliError> {
        let problem = self.problem;
        let mut sets = problem.sets.clone();
        sets.extend(written.iter().map(|(n, _, t)| (n.clone(), t.clone())));
        let source = overlaid(problem.source, &sets)?;
        let mut readings: Vec<Option<Output>> = vec![None; problem.cost.leaves.len()];
        let mut computed = Vec::new();
        for (target, session) in self.targets.iter().zip(sessions.iter_mut()) {
            let job = Job {
                rate: problem.rate,
                threads: problem.threads,
                asked: std::slice::from_ref(self.samples),
                abandon: self.abandon,
                ..Job::over(&source, target)
            };
            let rendered = execute_over(job, tier, session).await?;
            computed.push((target.clone(), rendered.render.work().computed_samples));
            if let Some(label) = rendered.label() {
                for one in &label.resampled {
                    if !resampled.contains(one) {
                        resampled.push(*one);
                    }
                }
            }
            let Output::Samples(buffer) = rendered.answered(self.samples)?.value else {
                unreachable!("`samples` answers samples");
            };
            let per_bar = rendered.graph.tempo().map(|tempo| tempo.seconds_per_bar());
            let profile = rendered.config.profile.name;
            for (at, leaf) in problem.cost.leaves.iter().enumerate() {
                if leaf.target != *target {
                    continue;
                }
                let representation = leaf.asked.representation(rendered.config.rate, per_bar)?;
                let answer =
                    sva_engine::answer_buffer(&leaf.written, &buffer, representation, profile)?;
                readings[at] = Some(answer.value);
            }
        }
        let readings: Vec<Output> = readings
            .into_iter()
            .map(|r| r.expect("every leaf's target rendered"))
            .collect();
        let mut terms = Vec::new();
        for term in &problem.cost.terms {
            terms.push(term.sign * scalar(&value(&term.node, &readings)?));
        }
        Ok((terms.iter().sum(), terms, computed))
    }
}

#[derive(Clone, Debug)]
enum Value {
    Scalar(f64),
    Vector(Vec<f64>),
}

fn scalar(v: &Value) -> f64 {
    match v {
        Value::Scalar(x) => *x,
        Value::Vector(_) => unreachable!("the cost typed every term to a number"),
    }
}

fn lengths(a: usize, b: usize, what: &str) -> CliError {
    CliError::usage(
        "solve.length_mismatch",
        format!(
            "{what} meets a vector of {a} values with one of {b}; give a weight vector one \
             weight per value of the residual it weighs"
        ),
    )
}

fn value(node: &Node, readings: &[Output]) -> Result<Value, CliError> {
    Ok(match node {
        Node::Num(n) => Value::Scalar(*n),
        Node::Vector(v) => Value::Vector(v.clone()),
        Node::Leaf(_) => unreachable!("the cost typed every leaf into a subtraction"),
        Node::Neg(x) => match value(x, readings)? {
            Value::Scalar(s) => Value::Scalar(-s),
            Value::Vector(v) => Value::Vector(v.iter().map(|x| -x).collect()),
        },
        Node::Bin(Op::Sub, a, b) if matches!((&**a, &**b), (Node::Leaf(_), Node::Leaf(_))) => {
            let (Node::Leaf(x), Node::Leaf(y)) = (&**a, &**b) else {
                unreachable!("matched above");
            };
            let vector = residual::residual(&readings[*x], &readings[*y])
                .expect("the cost typed both leaves to one reading");
            Value::Vector(vector)
        }
        Node::Bin(op, a, b) => {
            let (a, b) = (value(a, readings)?, value(b, readings)?);
            let apply = |x: f64, y: f64| match op {
                Op::Add => x + y,
                Op::Sub => x - y,
                Op::Mul => x * y,
                Op::Div => x / y,
            };
            match (a, b) {
                (Value::Scalar(x), Value::Scalar(y)) => Value::Scalar(apply(x, y)),
                (Value::Vector(v), Value::Scalar(y)) => {
                    Value::Vector(v.iter().map(|x| apply(*x, y)).collect())
                }
                (Value::Scalar(x), Value::Vector(v)) => {
                    Value::Vector(v.iter().map(|y| apply(x, *y)).collect())
                }
                (Value::Vector(v), Value::Vector(w)) if v.len() == w.len() => {
                    Value::Vector(v.iter().zip(&w).map(|(x, y)| apply(*x, *y)).collect())
                }
                (Value::Vector(v), Value::Vector(w)) => {
                    return Err(lengths(v.len(), w.len(), "a sum of two vectors"));
                }
            }
        }
        Node::Reduce(name, args) => {
            let args = args
                .iter()
                .map(|a| value(a, readings))
                .collect::<Result<Vec<_>, _>>()?;
            match (*name, args.as_slice()) {
                ("abs", [Value::Scalar(x)]) => Value::Scalar(x.abs()),
                ("abs", [Value::Vector(v)]) => Value::Vector(v.iter().map(|x| x.abs()).collect()),
                ("norm", [Value::Vector(v)]) => {
                    Value::Scalar(v.iter().map(|x| x * x).sum::<f64>().sqrt())
                }
                ("sum", [Value::Vector(v)]) => Value::Scalar(v.iter().sum()),
                ("max", [Value::Vector(v)]) => {
                    Value::Scalar(v.iter().copied().reduce(f64::max).unwrap_or(0.0))
                }
                ("dot", [Value::Vector(w), Value::Vector(v)]) if w.len() == v.len() => {
                    Value::Scalar(w.iter().zip(v).map(|(a, b)| a * b).sum())
                }
                ("dot", [Value::Vector(w), Value::Vector(v)]) => {
                    return Err(lengths(w.len(), v.len(), "`dot`"));
                }
                _ => unreachable!("the cost typed every reducer's arguments"),
            }
        }
    })
}

/// One render's progress line, as `sva-cli` writes it to stderr.
pub fn point_json(point: &Point) -> String {
    let values: Vec<String> = point
        .values
        .iter()
        .map(|(name, _, text)| format!("\"{}\": \"{}\"", escape(name), escape(text)))
        .collect();
    format!(
        "{{\"render\": {}, \"values\": {{{}}}, \"cost\": {}, \"best\": {}}}",
        point.render,
        values.join(", "),
        num(point.total),
        num(point.best)
    )
}

/// The answer's `data`, as both front ends print it.
pub fn solved_data(solved: &Solved, vary: &[Vary], extra: &str) -> String {
    let point_values = |p: &Point| {
        let fields: Vec<String> = p
            .values
            .iter()
            .map(|(n, v, t)| {
                format!(
                    "\"{}\": {{ \"value\": {}, \"text\": \"{}\" }}",
                    escape(n),
                    num(*v),
                    escape(t)
                )
            })
            .collect();
        format!("{{ {} }}", fields.join(", "))
    };
    let values: Vec<String> = solved
        .best
        .values
        .iter()
        .zip(vary)
        .zip(solved.start.iter().zip(&solved.on_bound))
        .map(|(((n, v, t), one), (start, bound))| {
            format!(
                "\"{}\": {{ \"value\": {}, \"text\": \"{}\", \"unit\": \"{}\", \"start\": \"{}\", \
                 \"on_bound\": {bound} }}",
                escape(n),
                num(*v),
                escape(t),
                one.unit,
                escape(start)
            )
        })
        .collect();
    let terms: Vec<(String, f64)> = solved
        .terms
        .iter()
        .cloned()
        .zip(solved.best.terms.iter().copied())
        .collect();
    let terms = list(&terms, |(text, v)| {
        format!(
            "{{ \"term\": \"{}\", \"value\": {} }}",
            escape(text),
            num(*v)
        )
    });
    let trace = list(&solved.trace, |p| {
        let computed: Vec<String> = p
            .computed
            .iter()
            .map(|(t, n)| format!("\"{}\": {n}", escape(t)))
            .collect();
        format!(
            "{{ \"render\": {}, \"values\": {}, \"cost\": {}, \"computed_samples\": {{ {} }} }}",
            p.render,
            point_values(p),
            num(p.total),
            computed.join(", ")
        )
    });
    let resampled = list(&solved.resampled, crate::answer::resampled_json);
    format!(
        "{{\"cost_expression\": \"{}\", \"values\": {{ {} }}, \"cost\": {{ \"total\": {}, \
         \"terms\": {terms} }}, \"renders\": {}, \"stopped_by\": \"{}\", \"resampled\": \
         {resampled}{extra}, \"trace\": {trace}}}",
        escape(&solved.cost),
        values.join(", "),
        num(solved.best.total),
        solved.renders,
        solved.stopped_by.name()
    )
}

/// What a fit says beside its values: a variable on a bound, a spent budget, a resampling.
pub fn findings(solved: &Solved, vary: &[Vary]) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for ((name, _, text), (bound, one)) in solved
        .best
        .values
        .iter()
        .zip(solved.on_bound.iter().zip(vary))
    {
        if *bound {
            out.push(
                Diagnostic::new(
                    "solve.on_bound",
                    format!(
                        "`{name}` ended on a bound of `{}..{}`, at `{text}`: the fit wants past it",
                        one.text(one.lo, Precision::Shortest),
                        one.text(one.hi, Precision::Shortest)
                    ),
                )
                .with_severity(Severity::Warning)
                .helped("widen the bound, or the model lacks what the reference holds"),
            );
        }
    }
    if solved.stopped_by == Stopped::Budget {
        out.push(
            Diagnostic::new(
                "solve.budget",
                format!(
                    "the solve spent `--max-renders` ({}) before its steps or cost converged",
                    solved.renders
                ),
            )
            .with_severity(Severity::Warning)
            .helped("raise `--max-renders`, narrow the bounds, or vary fewer variables at once"),
        );
    }
    for r in &solved.resampled {
        out.push(
            Diagnostic::new(
                "solve.resampled",
                format!(
                    "a recording at {} Hz was read at {} Hz through a windowed sinc: within \
                     {:.1e} dB up to {} Hz, at least {:.1} dB down from {} Hz",
                    r.from_hz, r.to_hz, r.ripple_db, r.passband_hz, r.stopband_db, r.stopband_hz
                ),
            )
            .with_severity(Severity::Warning)
            .helped("pass `--rate` at the recording's own rate to read its samples unchanged"),
        );
    }
    out
}
