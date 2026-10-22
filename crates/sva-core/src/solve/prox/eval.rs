// Concern: one state's distance and gradient: every line's transform in the reading's bins, the axes on a tape, adjoints back | Non-concern: what a step does (step.rs) | IO: (Fit, State) -> Evaluated

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use sva_formula::C64;
use sva_samples::measure::heard::BANDS;

use super::played::{self, Lowered, ROOT_SEED, Renders};
use super::state::{Slot, State};
use crate::cli_error::CliError;
use crate::heard::distance::{self, Axes, Reference, Side};
use crate::heard::ear::Ear;
use crate::heard::lines::{Carrier, Grid, Read};
use crate::heard::tape::{Arr, Tape};

pub struct Fit {
    pub reference: Reference,
    pub ear: Arc<Ear>,
    pub threads: NonZeroUsize,
    pub grid: Grid,
    /// The text of every file the composition's model reads, by path.
    pub texts: BTreeMap<String, String>,
    pub renders: Renders,
}

pub struct Evaluated {
    pub distance: f64,
    pub axes: Vec<Axes>,
    pub grad: Option<Vec<f64>>,
}

impl Fit {
    fn lowered(&self, s: &State, threads: NonZeroUsize) -> Result<Lowered, CliError> {
        let ms_len = self.ms_len();
        let spans: Vec<(usize, usize)> = (0..self.reference.peaks.len())
            .filter(|n| self.reference.rises[*n].is_some())
            .map(|n| {
                let (a, b) = self.reference.rise_span(n);
                (a.min(ms_len), (b + 2).min(ms_len))
            })
            .collect();
        let text = |p: &str| self.texts.get(p).cloned();
        played::lowered(s, &text, &self.grid, &self.renders, &spans, threads)
    }

    fn ms_len(&self) -> usize {
        self.reference.frames * self.ear.hop_ms + 2 * self.ear.hop_ms
    }

    fn band_of(&self) -> Vec<usize> {
        sva_samples::measure::heard::bins(self.grid.n, self.grid.sr).0
    }
}

/// Each frame's bins: its lines' windowed samples in one transform, their noise beside.
fn total(g: &Grid, reads: &[Read]) -> Vec<Vec<C64>> {
    let mut buf = vec![vec![0.0; g.window]; g.frames];
    let mut noise = vec![vec![C64::new(0.0, 0.0); g.bins()]; g.frames];
    for r in reads {
        for (m, x) in &r.samples {
            buf[*m].iter_mut().zip(x).for_each(|(b, v)| *b += v);
        }
        for (m, bins) in &r.noise {
            noise[*m]
                .iter_mut()
                .zip(bins)
                .for_each(|(b, v)| *b = *b + *v);
        }
    }
    buf.iter()
        .zip(noise)
        .map(|(x, n)| {
            let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
            re[g.left..g.left + g.window].copy_from_slice(x);
            sva_samples::fft::fft(&mut re, &mut im);
            (0..g.bins())
                .map(|j| C64::new(re[j], im[j]) + n[j])
                .collect()
        })
        .collect()
}

/// Each frame's `dL/dx_k` from its bins' `gamma`: `w_k Re(sum_j gamma_j e^{2 pi i j (left + k)/n})`.
fn samples_back(g: &Grid, gamma: &dyn Fn(usize, usize) -> C64) -> Vec<Vec<f64>> {
    let w = sva_samples::measure::heard::hann_periodic(g.window);
    (0..g.frames)
        .map(|m| {
            let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
            for j in 0..g.bins() {
                let y = gamma(m, j);
                re[j] = y.re;
                im[j] = -y.im;
            }
            sva_samples::fft::fft(&mut re, &mut im);
            (0..g.window).map(|k| w[k] * re[g.left + k]).collect()
        })
        .collect()
}

fn band_power(g: &Grid, band: &[usize], x: &[Vec<C64>]) -> Vec<f64> {
    let scale = g.scale();
    let mut p = vec![0.0; BANDS * g.frames];
    for (m, bins) in x.iter().enumerate() {
        for (j, v) in bins.iter().enumerate() {
            p[band[j] * g.frames + m] += v.norm_sqr() * scale;
        }
    }
    p
}

/// Each coordinate's share of a local gradient, through its played value's slope.
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
    let g = fit.grid;
    let r = &fit.reference;
    let low = fit.lowered(s, threads)?;
    let tasks: Vec<usize> = (0..low.lines.len()).collect();
    let reads = sva_engine::each(threads, tasks, |i| low.lines[i].line.bins(&g, &low.master));
    let x = total(&g, &reads);
    let band = fit.band_of();
    let notes = r.peaks.len();
    let mut hz = Vec::new();
    let mut means = Vec::new();
    let mut rows: Vec<Vec<Option<usize>>> = Vec::new();
    for (l, read) in reads.iter().enumerate() {
        let mut own = Vec::new();
        for (k, (f, per)) in read.tones.iter().enumerate() {
            let tone = matches!(low.lines[l].line.terms[k].carrier, Carrier::Tone { .. });
            if !tone || per.is_empty() || notes == 0 {
                own.push(None);
                continue;
            }
            own.push(Some(hz.len()));
            hz.push(*f);
            let mut row = vec![0.0; notes];
            for (n, seg) in r.notes().enumerate() {
                let w = 1.0 / (seg.m1 - seg.m0) as f64;
                row[n] = per
                    .iter()
                    .filter(|(m, _)| *m >= seg.m0 && *m < seg.m1)
                    .map(|(_, p)| p * w)
                    .sum();
            }
            means.extend(row);
        }
        rows.push(own);
    }
    let ms_len = fit.ms_len();
    let mut ms = vec![0.0; ms_len];
    for l in &low.lines {
        for (j, p) in l.line.ms_power() {
            if j < ms_len {
                ms[j] += p;
            }
        }
    }
    let mut tape = Tape::new();
    let pw = tape.leaf(Arr::new(BANDS, g.frames, band_power(&g, &band, &x)), grad);
    let lp = (!hz.is_empty()).then(|| tape.leaf(Arr::new(hz.len(), notes, means), grad));
    let pms = tape.leaf(Arr::row(ms), grad);
    let rises = (0..notes)
        .map(|n| {
            let (a, b) = r.rise_span(n);
            distance::rise(&mut tape, pms, a, b)
        })
        .collect();
    let side = Side {
        power: pw,
        line_hz: hz,
        line_power: lp,
        rises,
    };
    let d = distance::distance(&mut tape, r, &side);
    let value = tape.scalar_of(d.total);
    if !grad {
        return Ok(Evaluated {
            distance: value,
            axes: d.axes,
            grad: None,
        });
    }
    let one = Arr::scalar(1.0);
    let back = tape.backward(&[(d.total, &one)]);
    let zeros = |a: &Arr| Arr::zeros(a.r, a.c);
    let gp = back.of(pw).cloned().unwrap_or_else(|| zeros(tape.val(pw)));
    let gm = back
        .of(pms)
        .cloned()
        .unwrap_or_else(|| zeros(tape.val(pms)));
    let gl = lp.map(|lp| back.of(lp).cloned().unwrap_or_else(|| zeros(tape.val(lp))));
    let scale = g.scale();
    let segs: Vec<(usize, usize)> = r.notes().map(|s| (s.m0, s.m1)).collect();
    let gamma = |m: usize, j: usize| x[m][j].scale(2.0 * scale * gp.v[band[j] * g.frames + m]);
    let xbar = samples_back(&g, &gamma);
    let tasks: Vec<usize> = (0..low.lines.len()).collect();
    let parts = sva_engine::each(threads, tasks, |l| {
        let own = |k: usize, m: usize| -> f64 {
            let (Some(gl), Some(row)) = (gl.as_ref(), rows[l][k]) else {
                return 0.0;
            };
            segs.iter()
                .enumerate()
                .filter(|(_, (m0, m1))| m >= *m0 && m < *m1)
                .map(|(n, (m0, m1))| gl.v[row * notes + n] / (m1 - m0) as f64)
                .sum()
        };
        let ms_adj = |j: usize| gm.v.get(j).copied().unwrap_or(0.0);
        let mut local = Vec::new();
        low.lines[l]
            .line
            .back(&g, &low.master, &xbar, &gamma, &own, &ms_adj, &mut local);
        local
    });
    let mut out = vec![0.0; s.x.len()];
    for (l, local) in parts.iter().enumerate() {
        landed(s, &low.lines[l].slots, local, &mut out);
    }
    Ok(Evaluated {
        distance: value,
        axes: d.axes,
        grad: Some(out),
    })
}

/// Each line's least-squares power against the reference, at its gain 1: a gain's own scale.
pub fn scales(fit: &Fit, s: &State) -> Result<Vec<f64>, CliError> {
    let mut unit = s.clone();
    for root in s.root.iter().flatten() {
        unit.x[*root] = 1.0;
    }
    let g = fit.grid;
    let low = fit.lowered(&unit, fit.threads)?;
    let band = fit.band_of();
    let mut out = vec![0.0; s.x.len()];
    for (i, root) in s.root.iter().enumerate() {
        let Some(root) = root else {
            continue;
        };
        let read = low.lines[i].line.bins(&g, &low.master);
        let p = band_power(&g, &band, &total(&g, std::slice::from_ref(&read)));
        let (mut num, mut den) = (0.0, 0.0);
        for b in 0..BANDS {
            for m in 0..g.frames {
                let e1 = p[b * g.frames + m];
                num += fit.reference.power(b, m) * e1;
                den += e1 * e1;
            }
        }
        out[*root] = match den > 0.0 {
            true => (num / den).max(1e-18),
            false => 1e-18,
        };
    }
    Ok(out)
}
