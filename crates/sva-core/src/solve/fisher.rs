// Concern: the objective's gradient and Fisher matrix at a state by forward tangents through the ear and the stage; seeds shared by colouring | Non-concern: the search (lm.rs) | IO: (atoms) -> Linear

use std::f64::consts::LN_2;

use sva_engine::internal::{Ear, Frame};
use sva_formula::C64;

use super::atoms::{Atom, Clip};
use super::bits::Priced;
use super::dual::Dual;
use super::lm::{Linear, Problem};
use super::objective::Objective;
use super::spectrum::Model;

pub const MODEL: &str = "the residual in nats f = sum_i d_i^2/2, d_i = (m_i - e_i)/sigma_i, \
     sigma_i^2 = N0 rho (objective.rs), differentiated in forward mode, one seed per number: \
     each bin's tangent (spectrum.rs), then the ear's excitation with the lower skirts' p_l \
     moving with X, V's slope 10 log10(e)/(E_T + E), the modulation stage, linear from rest, \
     applied to the tangent of V, a magnitude element through d|y| = Re(conj(y) dy)/|y| (0 at \
     y = 0); the gradient g_j = sum_i d_i r_ij and the Fisher matrix (Gauss-Newton's, the \
     variance held) A = sum_i r_i r_i^T, r_ij = dm_i/dx_j/sigma_i; an atom reaches a frame when \
     any window of the frame weighs a sample whose cell meets [on, end), atoms reaching no \
     common frame share a colour (Curtis, Powell and Reid 1974, IMA J. Appl. Math. 13(1) \
     117-119, doi 10.1093/imamat/13.1.117; intervals coloured greedily by first frame, which is \
     optimal), and the seeds are the passes, one per colour and number slot, each frame's \
     tangents those of the one atom of that colour reaching it, split by atom before the \
     stage, whose memory reaches every later frame; the tangents carry no bound: they propose \
     steps and curvatures, and every step is taken on the true cost";

/// Each atom's reach `(first, last)` in frames (empty where `first > last`), its colour, and
/// the seed passes: per colour its atoms' most numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Colouring {
    pub reach: Vec<(i64, i64)>,
    pub colours: Vec<usize>,
    pub passes: usize,
}

pub fn colouring(ear: &Ear, clip: &Clip, atoms: &[Atom]) -> Colouring {
    let rate = f64::from(clip.rate);
    let (first, last) = ear.frames(clip.len);
    let reach: Vec<(i64, i64)> = atoms
        .iter()
        .map(|a| {
            let lo = libm::floor(a.on * rate - 0.5) as i64;
            let hi = libm::ceil(a.end() * rate + 0.5) as i64;
            let met = |m: i64| {
                let (f, l) = ear.reach(m);
                f <= hi && l >= lo
            };
            let from = (first..=last).find(|m| met(*m));
            let to = (first..=last).rev().find(|m| met(*m));
            match (from, to) {
                (Some(f), Some(t)) => (f, t),
                _ => (1, 0),
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..atoms.len()).collect();
    order.sort_by_key(|a| (reach[*a].0, *a));
    let mut ends: Vec<i64> = Vec::new();
    let mut colours = vec![0; atoms.len()];
    for a in order {
        let (from, to) = reach[a];
        let free = ends.iter().position(|end| *end < from);
        let c = match free {
            Some(c) => c,
            None => {
                ends.push(i64::MIN);
                ends.len() - 1
            }
        };
        if from <= to {
            ends[c] = to;
        }
        colours[a] = c;
    }
    let passes = (0..ends.len())
        .map(|c| {
            (0..atoms.len())
                .filter(|a| colours[*a] == c)
                .map(|a| atoms[a].values().len())
                .max()
                .unwrap_or(0)
        })
        .sum();
    Colouring {
        reach,
        colours,
        passes,
    }
}

/// The cost `f` in nats at a state, its `Linear`, and the colouring's passes.
pub struct Linearised {
    pub cost: f64,
    pub linear: Linear,
    pub passes: usize,
}

/// Composes, per bin, what `per` answers each range's frame with by number.
fn composed(
    ear: &Ear,
    m: i64,
    numbers: usize,
    per: &mut dyn FnMut(&Frame) -> Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new(); numbers];
    ear.compose(m, &mut |frame| {
        let each = per(frame);
        let bins = frame.hz().len();
        for (j, o) in out.iter_mut().enumerate() {
            match each.get(j).filter(|e| !e.is_empty()) {
                Some(e) => o.extend_from_slice(e),
                None => o.extend(std::iter::repeat_n(0.0, bins)),
            }
        }
        vec![(); bins]
    });
    out
}

/// `atoms`' cost and its linearisation, partials and harmonics summed in samples where
/// `coherent`.
pub fn linearise(objective: &Objective, atoms: &[Atom], coherent: bool) -> Linearised {
    let (ear, clip) = (objective.ear(), objective.clip());
    let model = Model::new(atoms, clip, coherent);
    let colouring = colouring(ear, &clip, atoms);
    let numbers: Vec<usize> = atoms.iter().map(|a| a.values().len()).collect();
    let offsets: Vec<usize> = numbers
        .iter()
        .scan(0, |at, n| {
            let here = *at;
            *at += n;
            Some(here)
        })
        .collect();
    let columns: usize = numbers.iter().sum();
    let (first, last) = ear.frames(clip.len);
    let frames = (last - first + 1) as usize;
    let channels = ear.thresholds().len();
    let mut levels = vec![vec![0.0; frames]; channels];
    let mut d_levels = vec![vec![vec![0.0; frames]; channels]; columns];
    let colours = colouring.colours.iter().max().map_or(0, |c| c + 1);
    for m in first..=last {
        let at = (m - first) as usize;
        let power = ear.compose(m, &mut |frame| model.bins(frame).power);
        for (c, v) in ear.levels(&ear.excite(m, &power)).into_iter().enumerate() {
            levels[c][at] = v;
        }
        for colour in 0..colours {
            let reaching: Vec<usize> = (0..atoms.len())
                .filter(|a| {
                    let (f, t) = colouring.reach[*a];
                    colouring.colours[*a] == colour && f <= m && m <= t
                })
                .collect();
            assert!(reaching.len() <= 1, "a colour's atoms share no frame");
            let Some(&a) = reaching.first() else {
                continue;
            };
            let d_power = composed(ear, m, numbers[a], &mut |frame| model.tangents(frame, a));
            for (j, d) in d_power.iter().enumerate() {
                let moved: Vec<Dual> = power
                    .iter()
                    .zip(d)
                    .map(|(v, d)| Dual::new(*v, *d))
                    .collect();
                let moved = ear.excite(m, &moved);
                for (c, v) in ear.levels(&moved).into_iter().enumerate() {
                    d_levels[offsets[a] + j][c][at] = v.tangent;
                }
            }
        }
    }
    let outputs = ear.filtered(&levels);
    let expected = objective.elements(&outputs);
    let cost = objective.residual_bits(&expected) * LN_2;
    let d_outputs: Vec<Vec<Vec<Vec<C64>>>> = d_levels.iter().map(|d| ear.filtered(d)).collect();
    let mut gradient = vec![0.0; columns];
    let mut fisher = vec![vec![0.0; columns]; columns];
    let mut rows: Vec<(usize, f64)> = Vec::with_capacity(columns);
    for (c, channel) in objective.heard().channels.iter().enumerate() {
        for (k, band) in channel.bands.iter().enumerate() {
            let sd = (objective.n0() * band.rate_hz).sqrt();
            for (e, clipped) in band.elements.iter().enumerate() {
                let y = outputs[c][k][e];
                let d = (expected[c][k][e] - clipped) / sd;
                rows.clear();
                for (p, column) in d_outputs.iter().enumerate() {
                    let r = band.band.element_slope(y, column[c][k][e]) / sd;
                    if r != 0.0 {
                        gradient[p] += d * r;
                        rows.push((p, r));
                    }
                }
                for (p, r) in &rows {
                    for (q, s) in &rows {
                        fisher[*p][*q] += r * s;
                    }
                }
            }
        }
    }
    Linearised {
        cost,
        linear: Linear { gradient, fisher },
        passes: colouring.passes,
    }
}

/// A fixed structure's numbers as one vector in `values()`' order atom by atom: its cost in
/// nats, its linearisation, its projection atom by atom (atoms.rs).
pub struct Fit<'a> {
    objective: &'a Objective,
    atoms: Vec<Atom>,
    priced: Vec<Priced>,
    coherent: bool,
}

impl<'a> Fit<'a> {
    pub fn new(
        objective: &'a Objective,
        atoms: Vec<Atom>,
        priced: Vec<Priced>,
        coherent: bool,
    ) -> Fit<'a> {
        assert_eq!(atoms.len(), priced.len(), "one pricing per atom");
        Fit {
            objective,
            atoms,
            priced,
            coherent,
        }
    }

    pub fn numbers(&self) -> Vec<f64> {
        self.atoms.iter().flat_map(|a| a.values()).collect()
    }

    pub fn deltas(&self) -> Vec<f64> {
        self.priced
            .iter()
            .flat_map(|p| p.deltas.iter().copied())
            .collect()
    }

    pub fn atoms_at(&self, x: &[f64]) -> Vec<Atom> {
        let mut at = 0;
        self.atoms
            .iter()
            .map(|a| {
                let n = a.values().len();
                let held = a.lifted(&x[at..at + n]);
                at += n;
                held
            })
            .collect()
    }

    pub fn linearise(&self, x: &[f64]) -> Linearised {
        linearise(self.objective, &self.atoms_at(x), self.coherent)
    }
}

impl Problem for Fit<'_> {
    fn cost(&self, x: &[f64]) -> f64 {
        self.objective.bits(&self.atoms_at(x), self.coherent) * LN_2
    }

    fn linear(&self, x: &[f64]) -> Linear {
        self.linearise(x).linear
    }

    fn project(&self, x: &[f64], weights: &[f64]) -> Option<Vec<f64>> {
        let mut at = 0;
        let mut out = Vec::with_capacity(x.len());
        for (atom, priced) in self.atoms_at(x).iter().zip(&self.priced) {
            let n = atom.values().len();
            let held = priced.feasible.project(atom, &weights[at..at + n])?;
            out.extend(held.values());
            at += n;
        }
        Some(out)
    }
}
