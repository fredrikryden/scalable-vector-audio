// Concern: SOLVE: projected Levenberg-Marquardt in precision units, its step halved till the cost falls | Non-concern: a number's meaning, schedules | IO: (Problem, start, precisions) -> Descent

use sva_engine::Abandon;

pub const MODEL: &str = "projected Levenberg-Marquardt with the Fisher information as its \
     curvature (Fisher scoring), in precision units y_i = x_i/delta_i, where D = I is \
     Marquardt's scaling: g the cost's gradient and S its Fisher matrix in y, the numbers whose \
     S_ii is 0 (seeded nowhere) left where they are, the rest ordered as the problem gives \
     them (atoms by the last frame each reaches, so every atom's block row meets only atoms \
     sharing a frame with it, an interval graph's perfect elimination order: no fill); the \
     Levenberg path p(mu) = -(S + mu I)^-1 g for mu >= c, c = P eps max S_ii (P the numbers, \
     eps = 2^-52; the numerical rank's cut, Golub and Van Loan, Matrix Computations, 4th ed., \
     sec. 5.4.1), doubled until S + c I factors, by the Cholesky factor within each row's \
     envelope (George and Liu 1981, Computer Solution of Large Sparse Positive Definite \
     Systems, ch. 4), a pivot not positive refusing that mu; mu the least double with |p| <= \
     r, by Newton's iteration on 1/|p| - 1/r from c (More and Sorensen 1983, SIAM J. Sci. Stat. \
     Comput. 4(3) 553-572, doi 10.1137/0904038), which rises to the root, then bisection on \
     the doubles' order between the last iterate short of r and the first past it; each point \
     projected onto the feasible set nearest in y (the problem's own exact projection; Kanzow, \
     Yamashita and Fukushima 2004, J. Comput. Appl. Math. 172(2) 375-397, doi \
     10.1016/j.cam.2004.02.013); the Gauss-Newton point r = |p(c)| first, then r halved until \
     the true cost falls, the first that falls taken (a search choice: plain backtracking, \
     halving as bisection halves its bracket); an atom whose last taken step moved none of its \
     numbers by 1 precision unit (the least move that counts) and none of whose neighbours \
     (atoms sharing a frame) moved one is frozen: seeded nowhere, still in the cost, revived \
     when a neighbour moves one; it ends when a taken step moves no number by 1 precision \
     unit, or when none falls by the time the trial point is the state itself, every number \
     unmoved in its double, at a cap on steps only if the caller sets one, or on a signal, and \
     answers its least state and which atoms it left frozen";

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

/// Each number marked as its group is.
fn numbered(sizes: &[usize], marks: &[bool]) -> Vec<bool> {
    sizes
        .iter()
        .zip(marks)
        .flat_map(|(n, m)| std::iter::repeat_n(*m, *n))
        .collect()
}

/// The Levenberg path in precision units: the kept rows from each one's first nonzero.
struct Path {
    n: usize,
    kept: Vec<usize>,
    rows: Vec<(usize, Vec<f64>)>,
    pulls: Vec<f64>,
    start: f64,
    longest: f64,
}

/// A point, its length, and `|L^-1 p|^2`.
type Point = (Vec<f64>, f64, f64);

impl Path {
    fn of(linear: &Linear, deltas: &[f64]) -> Path {
        let n = deltas.len();
        let kept: Vec<usize> = (0..n).filter(|i| linear.fisher[*i][*i] > 0.0).collect();
        let rows: Vec<(usize, Vec<f64>)> = kept
            .iter()
            .enumerate()
            .map(|(a, i)| {
                let row = &linear.fisher[*i];
                let first = (0..a).find(|b| row[kept[*b]] != 0.0).unwrap_or(a);
                let scaled = (first..=a)
                    .map(|b| deltas[*i] * row[kept[b]] * deltas[kept[b]])
                    .collect();
                (first, scaled)
            })
            .collect();
        let pulls = kept
            .iter()
            .map(|i| linear.gradient[*i] * deltas[*i])
            .collect();
        let top = rows
            .iter()
            .fold(0.0, |m: f64, (_, r)| m.max(*r.last().expect("a diagonal")));
        let mut path = Path {
            n,
            kept,
            rows,
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

    /// `p(mu)` as `MODEL` states, `None` where `S + mu I` does not factor.
    fn solve(&self, mu: f64) -> Option<Point> {
        let m = self.kept.len();
        let mut factor: Vec<Vec<f64>> = Vec::with_capacity(m);
        for (i, (first, row)) in self.rows.iter().enumerate() {
            let mut l = vec![0.0; row.len()];
            for j in *first..i {
                let (fj, lj) = (self.rows[j].0, &factor[j]);
                let lo = fj.max(*first);
                let dot: f64 = l[lo - first..j - first]
                    .iter()
                    .zip(&lj[lo - fj..j - fj])
                    .map(|(a, b)| a * b)
                    .sum();
                l[j - first] = (row[j - first] - dot) / lj[j - fj];
            }
            let sum: f64 = l[..i - first].iter().map(|v| v * v).sum();
            let pivot = row[i - first] + mu - sum;
            if pivot.is_nan() || pivot <= 0.0 {
                return None;
            }
            l[i - first] = pivot.sqrt();
            factor.push(l);
        }
        let forward = |b: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0; m];
            for i in 0..m {
                let (first, l) = (self.rows[i].0, &factor[i]);
                let dot: f64 = l[..i - first]
                    .iter()
                    .zip(&y[first..i])
                    .map(|(a, b)| a * b)
                    .sum();
                y[i] = (b[i] - dot) / l[i - first];
            }
            y
        };
        let minus: Vec<f64> = self.pulls.iter().map(|g| -g).collect();
        let mut p = forward(&minus);
        for i in (0..m).rev() {
            let (first, l) = (self.rows[i].0, &factor[i]);
            p[i] /= l[i - first];
            let pi = p[i];
            for (k, l) in (first..i).zip(l) {
                p[k] -= l * pi;
            }
        }
        let length = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        let q = forward(&p).iter().map(|v| v * v).sum();
        Some((p, length, q))
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
        let (p, _, _) = self
            .solve(mu)
            .expect("a damping past one that factors factors");
        for (i, v) in self.kept.iter().zip(p) {
            out[*i] = v;
        }
        (mu, out)
    }
}
