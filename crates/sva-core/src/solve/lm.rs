// Concern: SOLVE: projected Levenberg-Marquardt in precision units, its step halved till the cost falls | Non-concern: a number's meaning, schedules | IO: (Problem, start, precisions) -> Descent

use sva_engine::{Abandon, default_threads, each};

pub const MODEL: &str = "projected Levenberg-Marquardt with the Fisher information as its \
     curvature (Fisher scoring), in precision units y_i = x_i/delta_i, where D = I is Marquardt's \
     scaling: g the cost's gradient and S its Fisher matrix in y, the numbers whose S_ii is 0 \
     (seeded nowhere) left where they are; the Levenberg path p(mu) = -(S + mu I)^-1 g for mu >= \
     c, c = P eps max S_ii (P the numbers, eps = 2^-52; the numerical rank's cut, Golub and Van \
     Loan, Matrix Computations, 4th ed., sec. 5.4.1), doubled until S + c I factors, read through \
     S = Q T Q^T, T tridiagonal by Householder reflections once per step (Golub and Van Loan, \
     sec. 8.3.1), so p(mu) = Q z, z = -(T + mu I)^-1 Q^T g by the Cholesky factor of T + mu I and \
     |p| = |z|, a pivot not positive refusing that mu; mu the least double with |p| <= r, by \
     Newton's iteration on 1/|p| - 1/r from c (More and Sorensen 1983, SIAM J. Sci. Stat. Comput. \
     4(3) 553-572, doi 10.1137/0904038), which rises to the root, then bisection on the doubles' \
     order between the last iterate short of r and the first past it; each point projected onto \
     the feasible set nearest in y (the problem's own exact projection; Kanzow, Yamashita and \
     Fukushima 2004, J. Comput. Appl. Math. 172(2) 375-397, doi 10.1016/j.cam.2004.02.013); the \
     Gauss-Newton point r = |p(c)| first, then r halved until the true cost falls, the first that \
     falls taken (a search choice: plain backtracking, halving as bisection halves its bracket); \
     an atom whose last taken step moved none of its numbers by 1 precision unit (the least move \
     that counts) and none of whose neighbours (atoms sharing a frame) moved one is frozen: \
     seeded nowhere, still in the cost, revived when a neighbour moves one; it ends when a taken \
     step moves no number by 1 precision unit, or when none falls by the time the trial point is \
     the state itself, every number unmoved in its double, at a cap on steps only if the caller \
     sets one, or on a signal, and answers its least state and which atoms it left frozen";

/// The cost's gradient and Fisher matrix at a state, in written coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub gradient: Vec<f64>,
    pub fisher: Vec<Vec<f64>>,
}

/// A cost over numbers in groups (atoms); it knows no schedule.
pub trait Problem {
    fn cost(&self, x: &[f64]) -> f64;
    /// No number `frozen` marks seeded.
    fn linear(&self, x: &[f64], frozen: &[bool]) -> Linear;
    /// `None` where nothing is feasible.
    fn project(&self, x: &[f64], weights: &[f64]) -> Option<Vec<f64>>;
    fn groups(&self, x: &[f64]) -> Groups;
    /// The Fisher diagonal `A_ii` in written coordinates, read whole: what `sqrt(12/A_ii)`
    /// precisions read (bits.rs).
    fn curvatures(&self, x: &[f64]) -> Vec<f64>;
}

/// Each group's count of numbers and the groups sharing a frame with it.
#[derive(Clone, Debug, PartialEq)]
pub struct Groups {
    pub sizes: Vec<usize>,
    pub neighbours: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Converged,
    Cap,
    Signal,
}

impl Ended {
    pub fn name(self) -> &'static str {
        match self {
            Ended::Converged => "converged",
            Ended::Cap => "cap",
            Ended::Signal => "signal",
        }
    }
}

/// One step: the cost it ends at, the radius taken (none where nothing was), the largest move
/// in precision units, and the costs evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub cost: f64,
    pub radius: Option<f64>,
    pub moved: f64,
    pub evaluations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Descent {
    pub x: Vec<f64>,
    pub cost: f64,
    pub start_cost: f64,
    pub ended: Ended,
    pub steps: Vec<Step>,
    pub frozen: Vec<bool>,
}

/// A cap on steps where the caller sets one, and whether a signal has come, asked before each
/// step, where a caller that answers later lets its page have its turn.
pub struct Limits<'a> {
    pub steps: Option<usize>,
    pub stopped: &'a dyn Abandon,
}

/// `start` projected, then steps until the stop rule, a cap or a signal; `None` where nothing
/// is feasible. `deltas` are the numbers' precisions, each finite and positive.
pub async fn descend(
    problem: &dyn Problem,
    start: &[f64],
    deltas: &[f64],
    limits: &Limits<'_>,
) -> Option<Descent> {
    assert_eq!(start.len(), deltas.len(), "one precision per number");
    assert!(
        deltas.iter().all(|d| d.is_finite() && *d > 0.0),
        "every precision finite and positive"
    );
    let weights: Vec<f64> = deltas.iter().map(|d| 1.0 / (d * d)).collect();
    let mut x = problem.project(start, &weights)?;
    let mut cost = problem.cost(&x);
    let start_cost = cost;
    let mut steps = Vec::new();
    let mut frozen = vec![false; problem.groups(&x).sizes.len()];
    let ended = loop {
        if limits.steps.is_some_and(|cap| steps.len() >= cap) {
            break Ended::Cap;
        }
        if limits.stopped.abandoned().await {
            break Ended::Signal;
        }
        let still = numbered(&problem.groups(&x).sizes, &frozen);
        let path = Path::of(&problem.linear(&x, &still), deltas);
        let mut evaluations = 0;
        let mut radius = path.longest();
        let mut at = path.start;
        let taken = loop {
            if radius.is_nan() || radius <= 0.0 {
                break None;
            }
            evaluations += 1;
            let (mu, step) = path.at(radius, at);
            at = mu;
            let moved: Vec<f64> = x
                .iter()
                .zip(step)
                .zip(deltas)
                .map(|((x, p), d)| x + d * p)
                .collect();
            let held = problem
                .project(&moved, &weights)
                .expect("a feasible set holds a point");
            let moves: Vec<f64> = held
                .iter()
                .zip(&x)
                .zip(deltas)
                .map(|((a, b), d)| (a - b).abs() / d)
                .collect();
            let c = problem.cost(&held);
            if c < cost {
                break Some((c, radius, held, moves));
            }
            if held == x {
                break None;
            }
            radius /= 2.0;
        };
        match taken {
            None => {
                steps.push(Step {
                    cost,
                    radius: None,
                    moved: 0.0,
                    evaluations,
                });
                frozen.fill(true);
                break Ended::Converged;
            }
            Some((c, radius, held, moves)) => {
                let moved = moves.iter().fold(0.0, |m: f64, v| m.max(*v));
                steps.push(Step {
                    cost: c,
                    radius: Some(radius),
                    moved,
                    evaluations,
                });
                x = held;
                cost = c;
                let groups = problem.groups(&x);
                let mut at = 0;
                let counted: Vec<bool> = groups
                    .sizes
                    .iter()
                    .map(|n| {
                        let any = moves[at..at + n].iter().any(|m| *m >= 1.0);
                        at += n;
                        any
                    })
                    .collect();
                frozen = (0..counted.len())
                    .map(|g| !counted[g] && !groups.neighbours[g].iter().any(|h| counted[*h]))
                    .collect();
                if moved < 1.0 {
                    break Ended::Converged;
                }
            }
        }
    };
    Some(Descent {
        x,
        cost,
        start_cost,
        ended,
        steps,
        frozen,
    })
}

/// `a . b` in four interleaved partial sums, so it vectorises; only its rounding is the
/// order's.
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let (whole_a, whole_b) = (a.chunks_exact(4), b.chunks_exact(4));
    let tail: f64 = whole_a
        .remainder()
        .iter()
        .zip(whole_b.remainder())
        .map(|(x, y)| x * y)
        .sum();
    let mut lanes = [0.0; 4];
    for (x, y) in whole_a.zip(whole_b) {
        for k in 0..4 {
            lanes[k] += x[k] * y[k];
        }
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + tail
}

/// Each number marked as its group is.
fn numbered(sizes: &[usize], marks: &[bool]) -> Vec<bool> {
    sizes
        .iter()
        .zip(marks)
        .flat_map(|(n, m)| std::iter::repeat_n(*m, *n))
        .collect()
}

/// The Levenberg path in precision units: the kept numbers' Fisher matrix in precision units as
/// `Q T Q^T`, `T` tridiagonal (`diagonal`, `off`) and `Q` the Householder reflectors stored by
/// column in `reflectors`, beside the pull `Q^T g`.
struct Path {
    n: usize,
    kept: Vec<usize>,
    diagonal: Vec<f64>,
    off: Vec<f64>,
    reflectors: Vec<Vec<f64>>,
    pulls: Vec<f64>,
    start: f64,
    longest: f64,
}

/// A point in `T`'s basis, its length, and `|L^-1 z|^2`.
type Point = (Vec<f64>, f64, f64);

/// `x - v (v . x) 2/|v|^2` over the tail of `x` the reflector `v` spans.
fn reflect(v: &[f64], x: &mut [f64]) {
    let norm = dot(v, v);
    if norm == 0.0 {
        return;
    }
    let from = x.len() - v.len();
    let scale = 2.0 * dot(v, &x[from..]) / norm;
    for (x, v) in x[from..].iter_mut().zip(v) {
        *x -= scale * v;
    }
}

impl Path {
    fn of(linear: &Linear, deltas: &[f64]) -> Path {
        let n = deltas.len();
        let kept: Vec<usize> = (0..n).filter(|i| linear.fisher[*i][*i] > 0.0).collect();
        let mut a: Vec<Vec<f64>> = kept
            .iter()
            .map(|i| {
                kept.iter()
                    .map(|j| deltas[*i] * linear.fisher[*i][*j] * deltas[*j])
                    .collect()
            })
            .collect();
        let top = (0..kept.len()).fold(0.0, |m: f64, i| m.max(a[i][i]));
        let mut pulls: Vec<f64> = kept
            .iter()
            .map(|i| linear.gradient[*i] * deltas[*i])
            .collect();
        let m = kept.len();
        let mut reflectors = Vec::with_capacity(m.saturating_sub(2));
        let mut off = vec![0.0; m.saturating_sub(1)];
        let threads = default_threads();
        for k in 0..m.saturating_sub(1) {
            let mut v: Vec<f64> = (k + 1..m).map(|i| a[i][k]).collect();
            let norm = dot(&v, &v).sqrt();
            let alpha = match v[0] > 0.0 {
                true => -norm,
                false => norm,
            };
            off[k] = alpha;
            v[0] -= alpha;
            let vv = dot(&v, &v);
            if vv > 0.0 {
                let beta = 2.0 / vv;
                let size = (m - k - 1).div_ceil(threads.get());
                let p: Vec<f64> = each(threads, a[k + 1..].chunks(size).collect(), |rows| {
                    rows.iter()
                        .map(|row| beta * dot(&row[k + 1..], &v))
                        .collect::<Vec<f64>>()
                })
                .concat();
                let half = beta * dot(&p, &v) / 2.0;
                let w: Vec<f64> = p.iter().zip(&v).map(|(p, v)| p - half * v).collect();
                let tasks: Vec<(usize, &mut [Vec<f64>])> = a[k + 1..]
                    .chunks_mut(size)
                    .enumerate()
                    .map(|(c, rows)| (c * size, rows))
                    .collect();
                each(threads, tasks, |(at, rows)| {
                    for (r, row) in rows.iter_mut().enumerate() {
                        let (vi, wi) = (v[at + r], w[at + r]);
                        for ((x, wc), vc) in row[k + 1..].iter_mut().zip(&w).zip(&v) {
                            *x -= vi * wc + wi * vc;
                        }
                    }
                });
            } else {
                off[k] = v[0] + alpha;
                v.iter_mut().for_each(|x| *x = 0.0);
            }
            reflect(&v, &mut pulls);
            reflectors.push(v);
        }
        let diagonal = (0..m).map(|i| a[i][i]).collect();
        let mut path = Path {
            n,
            kept,
            diagonal,
            off,
            reflectors,
            pulls,
            start: 0.0,
            longest: 0.0,
        };
        if path.kept.is_empty() {
            return path;
        }
        let mut c = path.kept.len() as f64 * f64::EPSILON * top;
        loop {
            if let Some((_, length, _)) = path.solve(c) {
                (path.start, path.longest) = (c, length);
                return path;
            }
            c *= 2.0;
        }
    }

    fn longest(&self) -> f64 {
        self.longest
    }

    /// `p(mu)` in `T`'s basis as `MODEL` states, `None` where `T + mu I` does not factor.
    fn solve(&self, mu: f64) -> Option<Point> {
        let m = self.kept.len();
        let (mut diagonal, mut below) = (vec![0.0; m], vec![0.0; m]);
        for i in 0..m {
            if i > 0 {
                below[i] = self.off[i - 1] / diagonal[i - 1];
            }
            let pivot = self.diagonal[i] + mu - below[i] * below[i];
            if pivot.is_nan() || pivot <= 0.0 {
                return None;
            }
            diagonal[i] = pivot.sqrt();
        }
        let forward = |b: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0; m];
            for i in 0..m {
                let carried = if i > 0 { below[i] * y[i - 1] } else { 0.0 };
                y[i] = (b[i] - carried) / diagonal[i];
            }
            y
        };
        let minus: Vec<f64> = self.pulls.iter().map(|g| -g).collect();
        let mut z = forward(&minus);
        for i in (0..m).rev() {
            if i + 1 < m {
                z[i] -= below[i + 1] * z[i + 1];
            }
            z[i] /= diagonal[i];
        }
        let length = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        let q = forward(&z).iter().map(|v| v * v).sum();
        Some((z, length, q))
    }

    /// The length at `mu`, infinite where it does not factor.
    fn length(&self, mu: f64) -> f64 {
        self.solve(mu).map_or(f64::INFINITY, |(_, l, _)| l)
    }

    /// The least `mu >= from` whose point is at most `radius` long, and the point.
    fn at(&self, radius: f64, from: f64) -> (f64, Vec<f64>) {
        let mut out = vec![0.0; self.n];
        if self.kept.is_empty() {
            return (from, out);
        }
        let mut lo = from.max(self.start);
        let mu = match self.length(lo) <= radius {
            true => lo,
            false => {
                let mut hi = None;
                while let Some((_, length, q)) = self.solve(lo) {
                    let next = lo + length * length / q * (length - radius) / radius;
                    if next.is_nan() || next <= lo || !next.is_finite() {
                        break;
                    }
                    match self.length(next) <= radius {
                        true => {
                            hi = Some(next);
                            break;
                        }
                        false => lo = next,
                    }
                }
                let mut hi = match hi {
                    Some(hi) => hi,
                    None => {
                        let mut step = 1u64;
                        loop {
                            let up = f64::from_bits(lo.to_bits().saturating_add(step));
                            if self.length(up) <= radius || up == f64::MAX {
                                break up;
                            }
                            lo = up;
                            step = step.saturating_mul(2);
                        }
                    }
                };
                let (mut lo_bits, mut hi_bits) = (lo.to_bits(), hi.to_bits());
                while hi_bits - lo_bits > 1 {
                    let mid = lo_bits + (hi_bits - lo_bits) / 2;
                    match self.length(f64::from_bits(mid)) <= radius {
                        true => hi_bits = mid,
                        false => lo_bits = mid,
                    }
                }
                hi = f64::from_bits(hi_bits);
                hi
            }
        };
        let (mut p, _, _) = self
            .solve(mu)
            .expect("a damping past one that factors factors");
        for v in self.reflectors.iter().rev() {
            reflect(v, &mut p);
        }
        for (i, v) in self.kept.iter().zip(p) {
            out[*i] = v;
        }
        (mu, out)
    }
}
