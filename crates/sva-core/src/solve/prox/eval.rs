// Concern: one state's d' and gradient: each line's bins, their power summed across lines, the reading's stages and adjoint | Non-concern: what a step does (step.rs) | IO: (Fit, State) -> Evaluated

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use sva_formula::C64;
use sva_samples::measure::internal::{Ear, Grid, Internal};

use super::played::{self, Lowered, ROOT_SEED, Renders};
use super::state::{Slot, State};
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
    pub distance: f64,
    pub grad: Option<Vec<f64>>,
}

type Bins = Vec<(usize, Vec<C64>)>;

impl Fit {
    pub fn grid(&self) -> Grid {
        self.ear.grid
    }

    fn lowered(&self, s: &State, threads: NonZeroUsize) -> Result<Lowered, CliError> {
        let text = |p: &str| self.texts.get(p).cloned();
        played::lowered(s, &text, &self.ear.grid, &self.renders, threads)
    }
}

/// Power per bin summed over lines: the phases between lines are unknown, so their cross terms
/// average out; within a line its terms add coherently.
fn power(g: &Grid, reads: &[Bins]) -> Vec<f64> {
    let (bins, scale) = (g.bins(), g.scale());
    let mut p = vec![0.0; g.frames * bins];
    for read in reads {
        for (m, x) in read {
            for (j, v) in x.iter().enumerate() {
                p[m * bins + j] += v.norm_sqr() * scale;
            }
        }
    }
    p
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
    let f = fit.ear.forward(&power(&g, &reads));
    let residual: Vec<f64> = f
        .elements
        .iter()
        .zip(&fit.reference.elements)
        .map(|(a, b)| a - b)
        .collect();
    let distance = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    if !grad {
        return Ok(Evaluated {
            distance,
            grad: None,
        });
    }
    let unit: Vec<f64> = residual
        .iter()
        .map(|r| if distance > 0.0 { r / distance } else { 0.0 })
        .collect();
    let dp = fit.ear.backward(&f, &unit);
    let (bins, scale) = (g.bins(), g.scale());
    let tasks: Vec<usize> = (0..low.lines.len()).collect();
    let parts = sva_engine::each(threads, tasks, |l| {
        let mut at = vec![usize::MAX; g.frames];
        for (k, (m, _)) in reads[l].iter().enumerate() {
            at[*m] = k;
        }
        let gamma = |m: usize, j: usize| -> C64 {
            reads[l][at[m]].1[j].scale(2.0 * scale * dp[m * bins + j])
        };
        let mut local = Vec::new();
        low.lines[l].line.back(&g, &low.master, &gamma, &mut local);
        local
    });
    let mut out = vec![0.0; s.x.len()];
    for (l, local) in parts.iter().enumerate() {
        landed(s, &low.lines[l].slots, local, &mut out);
    }
    Ok(Evaluated {
        distance,
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
        let own = fit.ear.thirds(&power(&g, std::slice::from_ref(&read)));
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
