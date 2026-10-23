// Concern: the priced path: an unpriced start, each price's proximal descent, the merges the cost keeps, a debias | Non-concern: one step, one evaluation | IO: (Fit, State, prices) -> Points

pub mod draws;
pub mod eval;
pub mod lines;
pub mod played;
pub mod price;
pub mod run;
pub mod state;
pub mod step;
pub mod write;

use std::num::NonZeroUsize;

use sva_engine::Abandon;

use crate::cli_error::CliError;
use eval::{Fit, Residual, evaluate, scales};
use state::{Slot, State};
use step::{Adam, Priced, step, zeros};

pub const DEFAULT_STEPS: usize = 600;
/// Prices in bits per bit: a homotopy from nearly unpriced to 1, where the objective is the
/// description length itself.
pub const DEFAULT_PATH: [f64; 5] = [0.01, 0.03, 0.1, 0.3, 1.0];
/// The exchange rate of a reading's residual into bits: each element a cell coded as a
/// Gaussian of one JND's spread, `d^2/2` nats.
pub const BITS_PER_JND2: f64 = 0.5 * std::f64::consts::LOG2_E;
const START_STEPS: f64 = 500.0 / 600.0;
const SETTLE_STEPS: f64 = 200.0 / 600.0;
const TRIAL_STEPS: f64 = 80.0 / 600.0;
const DEBIAS_STEPS: f64 = 150.0 / 600.0;
const REASSIGN_ROUNDS: usize = 6;

#[derive(Clone)]
pub struct Point {
    pub lambda: f64,
    pub numbers: usize,
    pub bits: f64,
    /// The residual the descent lowers: the reading of the render's own lowered lines.
    pub predicted: Residual,
    /// The residual read off its renders, as the cost's leaf reads a render.
    pub measured: Option<Residual>,
    /// The reading's cells, weighted by the independent samples each carries.
    pub cells: f64,
    pub state: State,
    pub steps: usize,
    /// The state a cancelled solve stopped at, inside `lambda`'s phases or before any.
    pub stopped: bool,
}

impl Point {
    pub fn objective(&self) -> f64 {
        self.predicted.bits + self.lambda * self.bits
    }

    /// The bits its reading's residual codes in, its renders' where it was rendered.
    pub fn residual_bits(&self) -> f64 {
        self.measured.unwrap_or(self.predicted).bits
    }

    /// Its description length: the residual's bits and the numbers' bits.
    pub fn length(&self) -> f64 {
        self.residual_bits() + self.bits
    }
}

/// The point the path keeps: its least description length.
pub fn chosen(points: &[Point]) -> Option<usize> {
    (0..points.len()).min_by(|a, b| points[*a].length().total_cmp(&points[*b].length()))
}

/// One line of progress: the phase, the lambda, the step and what the state costs there.
pub struct Progress<'a> {
    pub phase: &'a str,
    pub lambda: f64,
    pub step: usize,
    pub residual_bits: f64,
    pub numbers: usize,
    pub bits: f64,
}

struct Run<'a> {
    fit: &'a Fit,
    weights: Vec<f64>,
    scales: Vec<f64>,
    steps: usize,
    abandon: &'a dyn Abandon,
    report: &'a mut dyn FnMut(&Progress),
    cancelled: bool,
}

fn scaled(steps: usize, fraction: f64) -> usize {
    ((steps as f64 * fraction).round() as usize).max(1)
}

/// What a step reads beside the state: the fit, its threads and its prices.
#[derive(Clone, Copy)]
struct Stepping<'a> {
    fit: &'a Fit,
    threads: NonZeroUsize,
    weights: &'a [f64],
    scales: &'a [f64],
}

impl Stepping<'_> {
    fn one(
        &self,
        s: &mut State,
        adam: &mut Adam,
        lambda: f64,
        held: Option<&[bool]>,
    ) -> Result<f64, CliError> {
        let e = evaluate(self.fit, s, true, self.threads)?;
        let grad = e.grad.expect("a gradient was asked");
        let priced = Priced {
            lambda,
            weights: self.weights,
            scales: self.scales,
            held,
        };
        step(s, adam, &grad, &priced);
        Ok(e.residual.bits)
    }

    fn objective(&self, s: &State, lambda: f64) -> Result<Objective, CliError> {
        let e = evaluate(self.fit, s, false, self.threads)?;
        let (numbers, bits) = price::numbers(s);
        Ok(Objective {
            total: e.residual.bits + lambda * bits,
            residual: e.residual,
            numbers,
            bits,
        })
    }
}

/// A tried split: its cost, state, moments, and every coordinate's weight and scale.
type Split = (f64, State, Adam, Vec<f64>, Vec<f64>);

struct Objective {
    total: f64,
    residual: Residual,
    numbers: usize,
    bits: f64,
}

impl Run<'_> {
    fn point(&self, s: &State, lambda: f64, stopped: bool) -> Result<Point, CliError> {
        let o = self.stepping().objective(s, lambda)?;
        Ok(Point {
            lambda,
            numbers: o.numbers,
            bits: o.bits,
            predicted: o.residual,
            measured: None,
            cells: eval::weights(self.fit).iter().sum(),
            state: s.clone(),
            steps: self.steps,
            stopped,
        })
    }

    fn stepping(&self) -> Stepping<'_> {
        Stepping {
            fit: self.fit,
            threads: self.fit.threads,
            weights: &self.weights,
            scales: &self.scales,
        }
    }

    async fn descend(
        &mut self,
        s: &mut State,
        adam: &mut Adam,
        lambda: f64,
        count: usize,
        held: Option<&[bool]>,
        phase: &str,
    ) -> Result<(), CliError> {
        for k in 0..count {
            if self.cancelled || self.abandon.abandoned().await {
                self.cancelled = true;
                break;
            }
            let residual_bits = self.stepping().one(s, adam, lambda, held)?;
            if k % 50 == 0 || k + 1 == count {
                let (numbers, bits) = price::numbers(s);
                (self.report)(&Progress {
                    phase,
                    lambda,
                    step: k,
                    residual_bits,
                    numbers,
                    bits,
                });
            }
        }
        Ok(())
    }

    /// Kept only where the cost falls: a node's lines split with a new node of its kind,
    /// each split tried from the same state on its own thread.
    async fn split(&mut self, s: &mut State, adam: &mut Adam, lambda: f64) -> Result<(), CliError> {
        loop {
            if self.cancelled || self.abandon.abandoned().await {
                self.cancelled = true;
                return Ok(());
            }
            let base = self.stepping().objective(s, lambda)?.total;
            let trials: Vec<usize> = (0..s.nodes.len())
                .filter(|n| {
                    s.nodes[*n] != s.model.root
                        && (0..s.model.lines.len())
                            .filter(|l| s.alive(*l) && s.line_node(*l) == *n)
                            .nth(1)
                            .is_some()
                })
                .collect();
            if trials.is_empty() {
                return Ok(());
            }
            let count = scaled(self.steps, TRIAL_STEPS);
            let (from, from_adam, fit) = (&*s, &*adam, self.fit);
            let (weights, scales) = (&self.weights, &self.scales);
            let tried = sva_engine::each(self.fit.threads, trials, |a| {
                let (mut t, mut ta) = (from.clone(), from_adam.clone());
                let (b, pairs) = t.copy_node(a);
                ta.extend(&pairs);
                let (mut w, mut sc) = (weights.clone(), scales.clone());
                for (_, c) in &pairs {
                    w.push(w[*c]);
                    sc.push(sc[*c]);
                }
                halve(&mut t, a, b);
                let one = Stepping {
                    fit,
                    threads: NonZeroUsize::MIN,
                    weights: &w,
                    scales: &sc,
                };
                for _ in 0..count {
                    one.one(&mut t, &mut ta, lambda, None)?;
                }
                let o = one.objective(&t, lambda)?.total;
                Ok::<_, CliError>((o, t, ta, w, sc))
            });
            let mut best: Option<Split> = None;
            for trial in tried {
                let trial = trial?;
                if trial.0 < base && best.as_ref().is_none_or(|b| trial.0 < b.0) {
                    best = Some(trial);
                }
            }
            match best {
                Some((_, t, ta, w, sc)) => {
                    *s = t;
                    *adam = ta;
                    self.weights = w;
                    self.scales = sc;
                }
                None => return Ok(()),
            }
        }
    }

    /// Kept only where the cost falls: a node's lines folded into another of its kind, each
    /// fold tried from the same state on its own thread.
    async fn merge(&mut self, s: &mut State, adam: &mut Adam, lambda: f64) -> Result<(), CliError> {
        loop {
            if self.cancelled || self.abandon.abandoned().await {
                self.cancelled = true;
                return Ok(());
            }
            let base = self.stepping().objective(s, lambda)?.total;
            let used: Vec<usize> = (0..s.nodes.len())
                .filter(|n| s.nodes[*n] != s.model.root && price::used(s, *n))
                .collect();
            let mut trials: Vec<(usize, usize)> = Vec::new();
            for a in &used {
                for b in &used {
                    if a != b && s.model.kin(&s.nodes[*a], &s.nodes[*b]) {
                        trials.push((*a, *b));
                    }
                }
            }
            if trials.is_empty() {
                return Ok(());
            }
            let count = scaled(self.steps, TRIAL_STEPS);
            let one = Stepping {
                threads: NonZeroUsize::MIN,
                ..self.stepping()
            };
            let (from, from_adam) = (&*s, &*adam);
            let tried = sva_engine::each(self.fit.threads, trials, |(a, b)| {
                let (mut t, mut ta) = (from.clone(), from_adam.clone());
                for l in 0..t.model.lines.len() {
                    if t.line_node(l) == a && t.alive(l) {
                        t.move_line(l, b);
                    }
                }
                for _ in 0..count {
                    one.one(&mut t, &mut ta, lambda, None)?;
                }
                let o = one.objective(&t, lambda)?.total;
                Ok::<_, CliError>((o, t, ta))
            });
            let mut best: Option<(f64, State, Adam)> = None;
            for trial in tried {
                let (o, t, ta) = trial?;
                if o < base && best.as_ref().is_none_or(|b| o < b.0) {
                    best = Some((o, t, ta));
                }
            }
            match best {
                Some((_, t, ta)) => {
                    *s = t;
                    *adam = ta;
                }
                None => return Ok(()),
            }
        }
    }
}

fn fused_distance(s: &State, line: usize, node: usize) -> f64 {
    s.slots(line)
        .iter()
        .filter_map(|(name, slot)| match slot {
            Slot::Fused { centre, dev } => {
                let c = s.centre.get(&(node, name.clone()))?;
                let var = &s.vars[s.var_of[*centre]];
                Some(((s.x[*centre] + s.x[*dev]) - s.x[*c]).abs() / var.res)
            }
            _ => None,
        })
        .sum()
}

fn pitch(s: &State, l: usize) -> f64 {
    s.slots(l)
        .iter()
        .find(|(n, _)| n == "f0")
        .map_or(s.model.lines[l].at, |(_, slot)| s.value(*slot))
}

/// k-medians over one kind's nodes: each line to its nearest centre, each centre to its
/// lines' median.
fn reassign(s: &mut State, kind: &[usize]) {
    let lines: Vec<usize> = (0..s.model.lines.len())
        .filter(|l| kind.contains(&s.line_node(*l)) && s.alive(*l))
        .collect();
    for _ in 0..REASSIGN_ROUNDS {
        for l in &lines {
            let to = *kind
                .iter()
                .min_by(|a, b| fused_distance(s, *l, **a).total_cmp(&fused_distance(s, *l, **b)))
                .expect("a kind holds nodes");
            s.move_line(*l, to);
        }
        for n in kind {
            s.recentre(*n);
        }
    }
}

/// Every kind's lines reassigned among its nodes, from the grouping the readings gave.
fn cluster(s: &mut State) {
    let mut kinds: Vec<Vec<usize>> = Vec::new();
    for n in 0..s.nodes.len() {
        if s.nodes[n] == s.model.root {
            continue;
        }
        match kinds
            .iter_mut()
            .find(|k| s.model.kin(&s.nodes[k[0]], &s.nodes[n]))
        {
            Some(k) => k.push(n),
            None => kinds.push(vec![n]),
        }
    }
    for kind in kinds.into_iter().filter(|k| k.len() > 1) {
        reassign(s, &kind);
    }
}

/// `node`'s lines split with the new node `to`: the upper half by pitch moved, then k-medians.
fn halve(s: &mut State, node: usize, to: usize) {
    let mut lines: Vec<usize> = (0..s.model.lines.len())
        .filter(|l| s.line_node(*l) == node && s.alive(*l))
        .collect();
    lines.sort_by(|a, b| pitch(s, *a).total_cmp(&pitch(s, *b)));
    for l in &lines[lines.len() / 2..] {
        s.move_line(*l, to);
    }
    s.recentre(node);
    s.recentre(to);
    reassign(s, &[node, to]);
}

/// The path from the overparameterised start: one point per lambda, warm from the last.
pub async fn path(
    fit: &Fit,
    mut s: State,
    lambdas: &[f64],
    steps: usize,
    abandon: &dyn Abandon,
    report: &mut dyn FnMut(&Progress),
) -> Result<(Vec<Point>, bool), CliError> {
    let mut adam = Adam::of(&s);
    let gains = scales(fit, &s)?;
    let mut run = Run {
        fit,
        weights: vec![0.0; s.x.len()],
        scales: gains,
        steps,
        abandon,
        report,
        cancelled: false,
    };
    let start = scaled(steps, START_STEPS);
    run.descend(&mut s, &mut adam, 0.0, start, None, "start")
        .await?;
    cluster(&mut s);
    let settle = scaled(steps, SETTLE_STEPS);
    run.descend(&mut s, &mut adam, 0.0, settle, None, "start")
        .await?;
    run.weights = price::weights(&s, &run.scales);
    let mut points = Vec::new();
    let mut at = 0.0;
    for lambda in lambdas {
        if run.cancelled {
            break;
        }
        at = *lambda;
        run.descend(&mut s, &mut adam, *lambda, steps, None, "descend")
            .await?;
        run.split(&mut s, &mut adam, *lambda).await?;
        run.merge(&mut s, &mut adam, *lambda).await?;
        let held = zeros(&s, &run.weights);
        let debias = scaled(steps, DEBIAS_STEPS);
        run.descend(&mut s, &mut adam, 0.0, debias, Some(&held), "debias")
            .await?;
        if run.cancelled {
            break;
        }
        points.push(run.point(&s, *lambda, false)?);
    }
    if run.cancelled {
        points.push(run.point(&s, at, true)?);
    }
    Ok((points, run.cancelled))
}

fn residual_json(r: &Residual) -> String {
    use crate::json::num;
    format!(
        "{{ \"residual_bits\": {}, \"standard_error\": {}, \"d_prime\": {}, \"per_cell\": {} }}",
        num(r.bits),
        num(r.error),
        num(r.d_prime),
        num(r.per_cell)
    )
}

fn point_json(p: &Point) -> String {
    use crate::json::num;
    format!(
        "{{ \"price\": {}, \"numbers\": {}, \"bits\": {}, \"residual_bits\": {}, \"length_bits\": {}, \
         \"cells\": {}, \"objective\": {}, \"measured\": {}, \"predicted\": {}, \"error_bits\": {}, \
         \"stopped\": {} }}",
        num(p.lambda),
        p.numbers,
        num(p.bits),
        num(p.residual_bits()),
        num(p.length()),
        num(p.cells),
        num(p.objective()),
        p.measured
            .as_ref()
            .map_or("null".to_string(), residual_json),
        residual_json(&p.predicted),
        p.measured
            .map_or("null".to_string(), |m| num(m.bits - p.predicted.bits)),
        p.stopped,
    )
}

/// One progress line, as `sva-cli` writes it to stderr.
pub fn progress_json(p: &Progress) -> String {
    use crate::json::{escape, num};
    format!(
        "{{\"phase\": \"{}\", \"price\": {}, \"step\": {}, \"residual_bits\": {}, \"numbers\": {}, \"bits\": {}}}",
        escape(p.phase),
        num(p.lambda),
        p.step,
        num(p.residual_bits),
        p.numbers,
        num(p.bits)
    )
}

/// The answer's `data`, as both front ends print it.
pub fn fitted_data(f: &run::Fitted, priced: &run::Priced, cost: &str, extra: &str) -> String {
    use crate::json::{escape, list, num};
    let chosen = f
        .chosen
        .map_or("null".to_string(), |c| point_json(&f.points[c]));
    format!(
        "{{\"method\": \"prox\", \"cost_expression\": \"{}\", \"model\": \"{}\", \"reference\": \
         \"{}\", \"bits_per_jnd2\": {}, \"steps\": {}, \"stopped_by\": \"{}\", \"chosen\": {chosen}, \
         \"files\": {}{extra}, \"path\": {}}}",
        escape(cost),
        escape(&f.model),
        escape(&f.reference),
        num(BITS_PER_JND2),
        priced.steps,
        if f.cancelled {
            "cancelled"
        } else {
            "converged"
        },
        list(&f.files, |(p, t)| format!(
            "{{ \"path\": \"{}\", \"text\": \"{}\" }}",
            escape(p),
            escape(t)
        )),
        list(&f.points, point_json)
    )
}
