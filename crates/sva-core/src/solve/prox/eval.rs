// Concern: one state's d' and gradient: each line's bins summed coherently per phase draw, through the reading's stages and adjoint | Non-concern: a step (step.rs) | IO: (Fit, State) -> Evaluated

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use sva_formula::C64;
use sva_samples::measure::internal::{Ear, Grid, Internal};

use super::draws::{DRAWS, phase};
use super::lines::{Adjoint, Bins};
use super::played::{self, Lowered, ROOT_SEED, Renders};
use super::state::{Slot, State};
use super::{BITS_PER_JND2, write};
use crate::cli_error::CliError;

pub struct Fit {
    pub reference: Internal,
    /// The reference's third-octave mean squares per frame, `[m * THIRDS + b]`.
    pub thirds: Vec<f64>,
    pub ear: Arc<Ear>,
    pub threads: NonZeroUsize,
    /// The text of every file the composition's model reads, by path.
    pub texts: BTreeMap<String, String>,
    pub renders: Renders,
}

/// A residual over the phase draws.
#[derive(Clone, Copy, Debug)]
pub struct Residual {
    /// Its description length: each element's `d^2` coded at its weight, the draws' mean.
    pub bits: f64,
    /// That mean's standard error from the draws' spread.
    pub error: f64,
    /// The root of `d'^2`'s mean over the draws, every element unweighted.
    pub d_prime: f64,
    /// The root mean square `d` over the weighted cells.
    pub per_cell: f64,
}

pub struct Evaluated {
    pub residual: Residual,
    pub grad: Option<Vec<f64>>,
}

impl Fit {
    pub fn grid(&self) -> Grid {
        self.ear.grid
    }

    fn lowered(&self, s: &State, threads: NonZeroUsize) -> Result<Lowered, CliError> {
        let text = |p: &str| self.texts.get(p).cloned();
        played::lowered(s, &text, &self.ear.grid, &self.renders, threads)
    }
}

/// Each element's weight: its set's independent samples per frame.
pub fn weights(fit: &Fit) -> Vec<f64> {
    let per_set = sva_samples::measure::internal::weights(fit.ear.grid.step_secs());
    let set = fit.reference.frames * sva_samples::measure::internal::BARKS;
    (0..fit.reference.elements.len())
        .map(|e| per_set[e / set])
        .collect()
}

/// The residual of each draw's reading against the reference.
pub fn residual(fit: &Fit, w: &[f64], draws: &[Vec<f64>]) -> Residual {
    let k = draws.len() as f64;
    let (mut weighted, mut plain) = (Vec::new(), 0.0);
    for elements in draws {
        let (mut a, mut b) = (0.0, 0.0);
        for ((x, y), w) in elements.iter().zip(&fit.reference.elements).zip(w) {
            let d = x - y;
            a += w * d * d;
            b += d * d;
        }
        weighted.push(BITS_PER_JND2 * a);
        plain += b;
    }
    let bits = weighted.iter().sum::<f64>() / k;
    let var = match k > 1.0 {
        true => {
            weighted
                .iter()
                .map(|x| (x - bits) * (x - bits))
                .sum::<f64>()
                / (k - 1.0)
        }
        false => 0.0,
    };
    Residual {
        bits,
        error: (var / k).sqrt(),
        d_prime: (plain / k).sqrt(),
        per_cell: (bits / BITS_PER_JND2 / w.iter().sum::<f64>().max(f64::MIN_POSITIVE)).sqrt(),
    }
}

/// A line's bin at a phase offset `(cos, sin)`.
fn at(b: &Bins, j: usize, (c, s): (f64, f64), k: usize) -> C64 {
    let tone = b.cos[j].scale(c) - b.quad[j].scale(s);
    match b.noise.len() {
        0 => tone,
        n => tone + b.noise[k % n][j],
    }
}

/// Power per bin with each line coherent and offset by its draw's phase.
fn power(g: &Grid, reads: &[Vec<Bins>], offsets: &[(f64, f64)], k: usize) -> Vec<f64> {
    let (bins, scale) = (g.bins(), g.scale());
    let mut p = vec![0.0; g.frames * bins];
    let mut sum = vec![C64::new(0.0, 0.0); g.frames * bins];
    for (read, o) in reads.iter().zip(offsets) {
        for b in read {
            for j in 0..bins {
                sum[b.m * bins + j] = sum[b.m * bins + j] + at(b, j, *o, k);
            }
        }
    }
    for (p, x) in p.iter_mut().zip(&sum) {
        *p = x.norm_sqr() * scale;
    }
    p
}

/// Each line's offset in draw `k`: none where its node reads no `ph`.
fn offsets(phased: &[bool], k: usize) -> Vec<(f64, f64)> {
    (0..phased.len())
        .map(|l| match phased[l] {
            true => {
                let (sn, c) = libm::sincos(phase(l, k));
                (c, sn)
            }
            false => (1.0, 0.0),
        })
        .collect()
}

/// A slot, and how far its played value moves a unit of its coordinate.
fn sloped(s: &State, slots: impl Iterator<Item = Slot>) -> Vec<(Slot, f64)> {
    slots.map(|slot| (slot, played::slope(s, slot))).collect()
}

fn landed(root: &[(Slot, f64)], own: &[(Slot, f64)], local: &[(u32, f64)], out: &mut [f64]) {
    for (k, d) in local {
        let (slot, slope) = match *k >= ROOT_SEED {
            true => root[(*k - ROOT_SEED) as usize],
            false => own[*k as usize],
        };
        let g = d * slope;
        match slot {
            Slot::Fused { centre, dev } => {
                out[centre] += g;
                out[dev] += g;
            }
            Slot::Own(o) => out[o] += g,
            Slot::Fixed(_) => {}
        }
    }
}

pub fn evaluate(
    fit: &Fit,
    s: &State,
    grad: bool,
    threads: NonZeroUsize,
) -> Result<Evaluated, CliError> {
    let g = fit.grid();
    let low = fit.lowered(s, threads)?;
    let tasks: Vec<usize> = (0..low.lines.len()).collect();
    let reads = sva_engine::each(threads, tasks, |i| low.lines[i].line.bins(&g, &low.master));
    let draws = DRAWS;
    let phased: Vec<bool> = (0..s.model.lines.len())
        .map(|l| write::line_reads(s, l, "ph"))
        .collect();
    let each = sva_engine::each(threads, (0..draws).collect(), |k| {
        let o = offsets(&phased, k);
        let f = fit.ear.forward(&power(&g, &reads, &o, k));
        (o, f)
    });
    let w = weights(fit);
    let elements: Vec<Vec<f64>> = each.iter().map(|(_, f)| f.elements.clone()).collect();
    let res = residual(fit, &w, &elements);
    if !grad {
        return Ok(Evaluated {
            residual: res,
            grad: None,
        });
    }
    let dps = sva_engine::each(threads, (0..draws).collect(), |k| {
        let f = &each[k].1;
        let unit: Vec<f64> = f
            .elements
            .iter()
            .zip(&fit.reference.elements)
            .zip(&w)
            .map(|((a, b), w)| 2.0 * BITS_PER_JND2 * w * (a - b) / draws as f64)
            .collect();
        fit.ear.backward(f, &unit)
    });
    let (bins, scale) = (g.bins(), g.scale());
    let totals: Vec<Vec<C64>> = sva_engine::each(threads, (0..draws).collect(), |k| {
        let mut sums = vec![C64::new(0.0, 0.0); g.frames * bins];
        for (read, o) in reads.iter().zip(&each[k].0) {
            for b in read {
                for j in 0..bins {
                    sums[b.m * bins + j] = sums[b.m * bins + j] + at(b, j, *o, k);
                }
            }
        }
        sums.iter()
            .zip(&dps[k])
            .map(|(x, d)| x.scale(2.0 * scale * d))
            .collect()
    });
    let tasks: Vec<usize> = (0..low.lines.len()).collect();
    let parts = sva_engine::each(threads, tasks, |l| {
        let mut frame = vec![usize::MAX; g.frames];
        for (k, b) in reads[l].iter().enumerate() {
            frame[b.m] = k;
        }
        let mut cos = vec![C64::new(0.0, 0.0); reads[l].len() * bins];
        let mut quad = cos.clone();
        let per = reads[l].len() * bins;
        let realisations = reads[l].first().map_or(0, |b| b.noise.len()).max(1);
        let mut noise = vec![C64::new(0.0, 0.0); realisations * per];
        for k in 0..draws {
            let (c, sn) = each[k].0[l];
            for (i, b) in reads[l].iter().enumerate() {
                for j in 0..bins {
                    let y = totals[k][b.m * bins + j];
                    cos[i * bins + j] = cos[i * bins + j] + y.scale(c);
                    quad[i * bins + j] = quad[i * bins + j] - y.scale(sn);
                    let at = (k % realisations) * per + i * bins + j;
                    noise[at] = noise[at] + y;
                }
            }
        }
        let read = |v: &Vec<C64>, m: usize, j: usize| v[frame[m] * bins + j];
        let adj = Adjoint {
            cos: &|m, j| read(&cos, m, j),
            quad: &|m, j| read(&quad, m, j),
            noise: &|r, m, j| noise[r * per + frame[m] * bins + j],
        };
        let mut local = Vec::new();
        low.lines[l].line.back(&g, &low.master, &adj, &mut local);
        local
    });
    let root = sloped(s, s.root_slots().into_iter().map(|(_, slot)| slot));
    let mut out = vec![0.0; s.x.len()];
    for (l, local) in parts.iter().enumerate() {
        let own = sloped(s, low.lines[l].slots.iter().copied());
        landed(&root, &own, local, &mut out);
    }
    Ok(Evaluated {
        residual: res,
        grad: Some(out),
    })
}

/// Each line's least-squares gain on the reference's third-octave mean squares at its gain 1.
pub fn scales(fit: &Fit, s: &State) -> Result<Vec<f64>, CliError> {
    let mut unit = s.clone();
    for root in s.root.iter().flatten() {
        unit.x[*root] = 1.0;
    }
    let g = fit.grid();
    let low = fit.lowered(&unit, fit.threads)?;
    let mut out = vec![0.0; s.x.len()];
    for (i, root) in s.root.iter().enumerate() {
        let Some(root) = root else {
            continue;
        };
        let read = low.lines[i].line.bins(&g, &low.master);
        let own = fit
            .ear
            .thirds(&power(&g, std::slice::from_ref(&read), &[(1.0, 0.0)], 0));
        let (mut num, mut den) = (0.0, 0.0);
        for (e1, r) in own.iter().zip(&fit.thirds) {
            num += r * e1;
            den += e1 * e1;
        }
        out[*root] = match den > 0.0 {
            true => (num / den).max(1e-18),
            false => 1e-18,
        };
    }
    Ok(out)
}
