// Concern: SOLVE: projected Levenberg-Marquardt in precision units, its step halved till the cost falls | Non-concern: a number's meaning, schedules | IO: (Problem, start, precisions) -> Descent

use sva_engine::Abandon;

pub const MODEL: &str = "projected Levenberg-Marquardt with the Fisher information as its \
     curvature (Fisher scoring), in precision units y_i = x_i/delta_i, where D = I is \
     Marquardt's scaling: g the cost's gradient and A its Fisher matrix in y, A = Q diag(l) Q^T \
     by cyclic Jacobi rotations (Golub and Van Loan, Matrix Computations, 4th ed., sec. 8.5), a \
     pair skipped where |a_pq| <= 2^-53 sqrt(|a_pp a_qq|), until a sweep rotates none or \
     leaves the off-diagonal sum of squares no smaller, and every \
     l_k <= P eps l_max cut (P the numbers, eps = 2^-52; their sec. 5.4.1 numerical rank); the \
     Levenberg path p(r) = -sum over the kept k of q_k (q_k^T g)/(l_k + mu), mu >= 0 the least \
     double with |p| <= r (bisection on the doubles' order, at most 64 halvings), each point \
     projected onto the feasible set nearest in y (the problem's own exact projection; Kanzow, \
     Yamashita and Fukushima 2004, J. Comput. Appl. Math. 172(2) 375-397, doi \
     10.1016/j.cam.2004.02.013); the Gauss-Newton point r = |p(0)| first, then r halved until \
     the true cost falls, the first that falls taken (a search choice: plain backtracking, \
     halving as bisection halves its bracket); it ends when a taken step moves no number by 1 \
     precision unit (the least move that counts), or when none falls by the time the trial \
     point is the state itself, every number unmoved in its double, at a cap on steps only if \
     the caller sets one, or on a signal, and answers its least state";

/// The cost's gradient and Fisher matrix at a state, in written coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub gradient: Vec<f64>,
    pub fisher: Vec<Vec<f64>>,
}

/// A cost over numbers: its value, its linearisation, the nearest feasible point in weighted
/// squares; it knows no schedule.
pub trait Problem {
    fn cost(&self, x: &[f64]) -> f64;
    fn linear(&self, x: &[f64]) -> Linear;
    /// `None` where nothing is feasible.
    fn project(&self, x: &[f64], weights: &[f64]) -> Option<Vec<f64>>;
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
    let ended = loop {
        if limits.steps.is_some_and(|cap| steps.len() >= cap) {
            break Ended::Cap;
        }
        if limits.stopped.abandoned().await {
            break Ended::Signal;
        }
        let path = Path::of(&problem.linear(&x), deltas);
        let mut evaluations = 0;
        let mut radius = path.longest();
        let taken = loop {
            if radius.is_nan() || radius <= 0.0 {
                break None;
            }
            evaluations += 1;
            let moved: Vec<f64> = x
                .iter()
                .zip(path.at(radius))
                .zip(deltas)
                .map(|((x, p), d)| x + d * p)
                .collect();
            let held = problem
                .project(&moved, &weights)
                .expect("a feasible set holds a point");
            let moved = held
                .iter()
                .zip(&x)
                .zip(deltas)
                .fold(0.0, |m: f64, ((a, b), d)| m.max((a - b).abs() / d));
            let c = problem.cost(&held);
            if c < cost {
                break Some((c, radius, held, moved));
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
                break Ended::Converged;
            }
            Some((c, radius, held, moved)) => {
                steps.push(Step {
                    cost: c,
                    radius: Some(radius),
                    moved,
                    evaluations,
                });
                x = held;
                cost = c;
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
    })
}

/// Each number's Fisher diagonal `A_ii` in written coordinates at `x`, what `sqrt(12/A_ii)`
/// precisions read (bits.rs).
pub fn curvatures(problem: &dyn Problem, x: &[f64]) -> Vec<f64> {
    let linear = problem.linear(x);
    (0..x.len()).map(|i| linear.fisher[i][i]).collect()
}

/// The Levenberg path in precision units over the kept eigenvectors.
struct Path {
    values: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    pulls: Vec<f64>,
}

impl Path {
    fn of(linear: &Linear, deltas: &[f64]) -> Path {
        let n = deltas.len();
        let scaled: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| deltas[i] * linear.fisher[i][j] * deltas[j])
                    .collect()
            })
            .collect();
        let gradient: Vec<f64> = linear
            .gradient
            .iter()
            .zip(deltas)
            .map(|(g, d)| g * d)
            .collect();
        let (values, vectors) = eigen(scaled);
        let top = values.iter().fold(0.0, |m: f64, v| m.max(*v));
        let cut = n as f64 * f64::EPSILON * top;
        let kept: Vec<usize> = (0..n).filter(|k| top > 0.0 && values[*k] > cut).collect();
        let vectors: Vec<Vec<f64>> = kept
            .iter()
            .map(|k| (0..n).map(|i| vectors[i][*k]).collect())
            .collect();
        let pulls = vectors
            .iter()
            .map(|q| q.iter().zip(&gradient).map(|(q, g)| q * g).sum())
            .collect();
        Path {
            values: kept.iter().map(|k| values[*k]).collect(),
            vectors,
            pulls,
        }
    }

    fn length(&self, mu: f64) -> f64 {
        self.values
            .iter()
            .zip(&self.pulls)
            .map(|(l, c)| (c / (l + mu)).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn longest(&self) -> f64 {
        self.length(0.0)
    }

    /// The point at `mu`, the least double whose length is at most `radius`.
    fn at(&self, radius: f64) -> Vec<f64> {
        let mu = match self.longest() <= radius {
            true => 0.0,
            false => {
                let norm = self.pulls.iter().map(|c| c * c).sum::<f64>().sqrt();
                let (mut lo, mut hi) = (0u64, (norm / radius).to_bits());
                while self.length(f64::from_bits(hi)) > radius {
                    hi = hi.saturating_mul(2).min(f64::MAX.to_bits());
                }
                while hi - lo > 1 {
                    let mid = lo + (hi - lo) / 2;
                    match self.length(f64::from_bits(mid)) <= radius {
                        true => hi = mid,
                        false => lo = mid,
                    }
                }
                f64::from_bits(hi)
            }
        };
        let n = self.vectors.first().map_or(0, Vec::len);
        let mut p = vec![0.0; n];
        for ((q, c), l) in self.vectors.iter().zip(&self.pulls).zip(&self.values) {
            let w = -c / (l + mu);
            for (p, q) in p.iter_mut().zip(q) {
                *p += w * q;
            }
        }
        p
    }
}

/// A symmetric matrix's eigenvalues and eigenvectors (column `k` of the second), by cyclic
/// Jacobi rotations as `MODEL` states.
fn eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    let u = f64::EPSILON / 2.0;
    let off = |a: &[Vec<f64>]| -> f64 {
        (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum()
    };
    let mut last = f64::INFINITY;
    loop {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p][q];
                if apq.abs() <= u * (a[p][p] * a[q][q]).abs().sqrt() {
                    continue;
                }
                rotated = true;
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + libm::hypot(1.0, theta));
                let c = 1.0 / libm::hypot(1.0, t);
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (upper, lower) = a.split_at_mut(q);
                for (apk, aqk) in upper[p].iter_mut().zip(lower[0].iter_mut()) {
                    (*apk, *aqk) = (c * *apk - s * *aqk, s * *apk + c * *aqk);
                }
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        let now = off(&a);
        if !rotated || now >= last {
            break;
        }
        last = now;
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}
