// Concern: one state's d' and gradient: each line's bins summed coherently per phase draw, through the reading's stages and adjoint | Non-concern: a step (step.rs) | IO: (Fit, State) -> Evaluated

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use sva_formula::C64;
use sva_samples::measure::internal::{Ear, Grid, Internal};

use super::lines::{Adjoint, Bins};
use super::played::{self, Lowered, ROOT_SEED, Renders};
use super::state::{Slot, State};
use super::write;
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

pub struct Evaluated {
    /// `d'` as the root of its square's mean over the phase draws.
    pub distance: f64,
    /// That root's standard error from the draws' spread.
    pub error: f64,
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

/// How many draws of the lines' phases the expected loss is read over.
pub const DRAWS: usize = 16;

/// Line `line`'s phase offset in draw `draw`: `2 pi` times splitmix64 of `line << 32 | draw`
/// over `2^64`, the same every solve and target.
pub fn phase(line: usize, draw: usize) -> f64 {
    let mut z = ((line as u64) << 32 | draw as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    std::f64::consts::TAU * (z >> 11) as f64 / (1u64 << 53) as f64
}

/// The standard error of `sqrt(mean(squares))` over the draws, by the delta method.
pub fn draws_error(squares: &[f64]) -> f64 {
    let k = squares.len() as f64;
    let mean = squares.iter().sum::<f64>() / k;
    if k < 2.0 || mean <= 0.0 {
        return 0.0;
    }
    let var = squares.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (k - 1.0);
    (var / k).sqrt() / (2.0 * mean.sqrt())
}

/// A line's bin at a phase offset `(cos, sin)`.
fn at(b: &Bins, j: usize, (c, s): (f64, f64)) -> C64 {
    let tone = b.cos[j].scale(c) - b.quad[j].scale(s);
    match b.noise.is_empty() {
        true => tone,
        false => tone + b.noise[j],
    }
}

/// Power per bin with each line coherent and offset by its draw's phase.
fn power(g: &Grid, reads: &[Vec<Bins>], offsets: &[(f64, f64)]) -> Vec<f64> {
    let (bins, scale) = (g.bins(), g.scale());
    let mut p = vec![0.0; g.frames * bins];
    let mut sum = vec![C64::new(0.0, 0.0); g.frames * bins];
    for (read, o) in reads.iter().zip(offsets) {
        for b in read {
            for j in 0..bins {
                sum[b.m * bins + j] = sum[b.m * bins + j] + at(b, j, *o);
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

fn landed(s: &State, slots: &[Slot], local: &[(u32, f64)], out: &mut [f64]) {
    let root = s.root_slots();
    for (k, d) in local {
        let slot = match *k >= ROOT_SEED {
            true => root[(*k - ROOT_SEED) as usize].1,
            false => slots[*k as usize],
        };
        let g = d * played::slope(s, slot);
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
        .map(|l| write::phased(s, l))
        .collect();
    let each = sva_engine::each(threads, (0..draws).collect(), |k| {
        let o = offsets(&phased, k);
        let f = fit.ear.forward(&power(&g, &reads, &o));
        let r: Vec<f64> = f
            .elements
            .iter()
            .zip(&fit.reference.elements)
            .map(|(a, b)| a - b)
            .collect();
        (o, f, r)
    });
    let squares: Vec<f64> = each
        .iter()
        .map(|(_, _, r)| r.iter().map(|x| x * x).sum::<f64>())
        .collect();
    let distance = (squares.iter().sum::<f64>() / draws as f64).sqrt();
    let error = draws_error(&squares);
    if !grad {
        return Ok(Evaluated {
            distance,
            error,
            grad: None,
        });
    }
    let dps = sva_engine::each(threads, (0..draws).collect(), |k| {
        let (_, f, r) = &each[k];
        let unit: Vec<f64> = r
            .iter()
            .map(|x| match distance > 0.0 {
                true => x / (draws as f64 * distance),
                false => 0.0,
            })
            .collect();
        fit.ear.backward(f, &unit)
    });
    let (bins, scale) = (g.bins(), g.scale());
    let mut sums = vec![C64::new(0.0, 0.0); g.frames * bins];
    let totals: Vec<Vec<C64>> = (0..draws)
        .map(|k| {
            sums.iter_mut().for_each(|x| *x = C64::new(0.0, 0.0));
            for (read, o) in reads.iter().zip(&each[k].0) {
                for b in read {
                    for j in 0..bins {
                        sums[b.m * bins + j] = sums[b.m * bins + j] + at(b, j, *o);
                    }
                }
            }
            sums.iter()
                .zip(&dps[k])
                .map(|(x, d)| x.scale(2.0 * scale * d))
                .collect()
        })
        .collect();
    let tasks: Vec<usize> = (0..low.lines.len()).collect();
    let parts = sva_engine::each(threads, tasks, |l| {
        let mut frame = vec![usize::MAX; g.frames];
        for (k, b) in reads[l].iter().enumerate() {
            frame[b.m] = k;
        }
        let mut cos = vec![C64::new(0.0, 0.0); reads[l].len() * bins];
        let mut quad = cos.clone();
        let mut noise = cos.clone();
        for k in 0..draws {
            let (c, sn) = each[k].0[l];
            for (i, b) in reads[l].iter().enumerate() {
                for j in 0..bins {
                    let y = totals[k][b.m * bins + j];
                    cos[i * bins + j] = cos[i * bins + j] + y.scale(c);
                    quad[i * bins + j] = quad[i * bins + j] - y.scale(sn);
                    noise[i * bins + j] = noise[i * bins + j] + y;
                }
            }
        }
        let read = |v: &Vec<C64>, m: usize, j: usize| v[frame[m] * bins + j];
        let adj = Adjoint {
            cos: &|m, j| read(&cos, m, j),
            quad: &|m, j| read(&quad, m, j),
            noise: &|m, j| read(&noise, m, j),
        };
        let mut local = Vec::new();
        low.lines[l].line.back(&g, &low.master, &adj, &mut local);
        local
    });
    let mut out = vec![0.0; s.x.len()];
    for (l, local) in parts.iter().enumerate() {
        landed(s, &low.lines[l].slots, local, &mut out);
    }
    Ok(Evaluated {
        distance,
        error,
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
            .thirds(&power(&g, std::slice::from_ref(&read), &[(1.0, 0.0)]));
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
