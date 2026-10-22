// Concern: one proximal Adam step: Adam on the distance, then each priced number's threshold at price x weight x Adam's step | Non-concern: the gradient, the weights | IO: (State, gradient) -> State

use super::state::{Role, State};

pub const LEARNING: f64 = 0.15;
const GAIN_LEARNING: f64 = 0.05;
const MOST_STEPS: f64 = 30.0;
const B1: f64 = 0.9;
const B2: f64 = 0.999;
/// A tenth of a group's median gradient scale floors each member's, so a number at rest is
/// given no unbounded step, and no unbounded threshold.
const FLOOR: f64 = 0.1;

#[derive(Clone)]
pub struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
    n: u64,
    group: Vec<usize>,
    groups: usize,
}

impl Adam {
    pub fn of(s: &State) -> Adam {
        let mut keys: Vec<(u8, usize)> = Vec::new();
        let group = s
            .roles
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let kind = match r {
                    Role::Centre { .. } => 0,
                    Role::Dev { .. } => 1,
                    Role::Own { root: true, .. } => 2,
                    Role::Own { .. } => 3,
                };
                let key = (kind, s.var_of[i]);
                match keys.iter().position(|k| *k == key) {
                    Some(g) => g,
                    None => {
                        keys.push(key);
                        keys.len() - 1
                    }
                }
            })
            .collect();
        Adam {
            m: vec![0.0; s.x.len()],
            v: vec![0.0; s.x.len()],
            n: 0,
            group,
            groups: keys.len(),
        }
    }
}

impl Adam {
    /// Each new coordinate's moments and group copied from the one it copies.
    pub fn extend(&mut self, pairs: &[(usize, usize)]) {
        for (_, from) in pairs {
            self.m.push(self.m[*from]);
            self.v.push(self.v[*from]);
            self.group.push(self.group[*from]);
        }
    }
}

pub struct Priced<'a> {
    pub lambda: f64,
    pub weights: &'a [f64],
    pub scales: &'a [f64],
    pub held: Option<&'a [bool]>,
}

fn median(mut x: Vec<f64>) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    x.sort_by(f64::total_cmp);
    let n = x.len();
    Some(match n % 2 {
        1 => x[n / 2],
        _ => 0.5 * (x[n / 2 - 1] + x[n / 2]),
    })
}

pub fn step(s: &mut State, adam: &mut Adam, grad: &[f64], p: &Priced) {
    adam.n += 1;
    let nn = adam.n as i32;
    let (c1, c2) = (
        1.0 - libm::pow(B1, f64::from(nn)),
        1.0 - libm::pow(B2, f64::from(nn)),
    );
    let mut scale_of: Vec<Vec<f64>> = vec![Vec::new(); adam.groups];
    for (i, g) in grad.iter().enumerate() {
        let g = if g.is_finite() { *g } else { 0.0 };
        adam.m[i] = B1 * adam.m[i] + (1.0 - B1) * g;
        adam.v[i] = B2 * adam.v[i] + (1.0 - B2) * g * g;
        let sv = (adam.v[i] / c2).sqrt();
        if sv > 0.0 {
            scale_of[adam.group[i]].push(sv);
        }
    }
    let floors: Vec<f64> = scale_of
        .into_iter()
        .map(|x| median(x).map_or(1e-30, |m| FLOOR * m))
        .collect();
    for i in 0..s.x.len() {
        let var = &s.vars[s.var_of[i]];
        let mh = adam.m[i] / c1;
        let sv = (adam.v[i] / c2).sqrt().max(floors[adam.group[i]]);
        let root = matches!(s.roles[i], Role::Own { root: true, .. });
        let step = match root {
            true => {
                let sc = p.scales[i];
                (GAIN_LEARNING * sc / (sv + 1e-30)).min(1e3 * sc)
            }
            false => {
                let lr = LEARNING * var.res;
                (lr / (sv + 1e-8)).min(lr * MOST_STEPS)
            }
        };
        let mut x = s.x[i] - step * mh;
        match p.held {
            Some(held) if held[i] => x = 0.0,
            Some(_) => {}
            None if p.lambda > 0.0 && p.weights[i] > 0.0 => {
                let cut = p.lambda * p.weights[i] * step;
                x = match root {
                    true => x - cut,
                    false => x.signum() * (x.abs() - cut).max(0.0),
                };
            }
            None => {}
        }
        x = match &s.roles[i] {
            Role::Dev { .. } => x,
            _ => {
                let (lo, hi) = s.bounds(i);
                x.clamp(lo, hi)
            }
        };
        s.x[i] = if x.is_finite() { x } else { 0.0 };
    }
}

/// The numbers a debias holds at zero: every priced one now at zero.
pub fn zeros(s: &State, weights: &[f64]) -> Vec<bool> {
    (0..s.x.len())
        .map(|i| weights[i] > 0.0 && s.x[i] == 0.0)
        .collect()
}
