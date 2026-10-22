// Concern: one state's distance and every coordinate's gradient: each score line on its own tape, the axes on one more | Non-concern: what a step does with them (step.rs) | IO: (Fit, State) -> Evaluated

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::state::{Slot, State};
use crate::cli_error::CliError;
use crate::heard::distance::{self, Axes, Reference, Side};
use crate::heard::ear::Ear;
use crate::heard::power::{self, Played};
use crate::heard::tape::{Arr, Sparse, Tape, V};

pub struct Fit {
    pub reference: Reference,
    pub ear: Arc<Ear>,
    pub threads: NonZeroUsize,
}

pub struct Evaluated {
    pub distance: f64,
    pub axes: Vec<Axes>,
    pub grad: Option<Vec<f64>>,
}

struct Lined {
    tape: Tape,
    leaves: Vec<(V, Slot)>,
    played: Played,
    /// Each line's mean power per note segment, `[lines, notes]`.
    means: Option<V>,
}

fn line(fit: &Fit, s: &State, i: usize) -> Result<Option<Lined>, CliError> {
    let mut tape = Tape::new();
    let mut leaves = Vec::new();
    let root = s.scope(&mut tape, &s.root_slots(), &mut leaves);
    let own = s.scope(&mut tape, &s.slots(i), &mut leaves);
    let frames = fit.reference.frames;
    let Some(played) = power::played(&mut tape, &s.model, &fit.ear, i, &root, &own, frames)? else {
        return Ok(None);
    };
    let notes = fit.reference.peaks.len();
    let means = match (played.line_power, notes) {
        (Some(lp), n) if n > 0 => {
            let (k, f) = (tape.val(lp).r, tape.val(lp).c);
            let mut entries = Vec::new();
            for row in 0..k {
                entries.extend(fit.reference.note_means(row, played.grid.m_lo, f));
            }
            Some(tape.linear(
                Arc::new(Sparse {
                    r: k,
                    c: n,
                    entries,
                }),
                lp,
            ))
        }
        _ => None,
    };
    Ok(Some(Lined {
        tape,
        leaves,
        played,
        means,
    }))
}

pub fn evaluate(
    fit: &Fit,
    s: &State,
    grad: bool,
    threads: NonZeroUsize,
) -> Result<Evaluated, CliError> {
    let lines: Vec<usize> = (0..s.model.lines.len()).collect();
    let lined = sva_engine::each(threads, lines, |i| line(fit, s, i));
    let lined: Vec<Lined> = lined
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    let r = &fit.reference;
    let m = r.frames;
    let ms_len = m * fit.ear.hop_ms + 2 * fit.ear.hop_ms;
    let notes = r.peaks.len();
    let mut power = vec![0.0; 48 * m];
    let mut ms = vec![0.0; ms_len];
    let mut hz = Vec::new();
    let mut means = Vec::new();
    let mut rows = Vec::with_capacity(lined.len());
    for l in &lined {
        let p = l.tape.val(l.played.power);
        let g = l.played.grid;
        for b in 0..48 {
            for f in 0..g.frames {
                power[b * m + g.m_lo + f] += p.v[b * p.c + f];
            }
        }
        let pm = l.tape.val(l.played.power_ms);
        for j in 0..g.ms {
            let at = g.g0 + j as i64;
            if at >= 0 && (at as usize) < ms_len {
                ms[at as usize] += pm.v[j];
            }
        }
        rows.push(hz.len());
        if let Some(mv) = l.means {
            hz.extend_from_slice(&l.played.line_hz);
            means.extend_from_slice(&l.tape.val(mv).v);
        }
    }
    let mut tape = Tape::new();
    let pw = tape.leaf(Arr::new(48, m, power), grad);
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
    let g = tape.backward(&[(d.total, &one)]);
    let zeros = |a: &Arr| Arr::zeros(a.r, a.c);
    let gp = g.of(pw).cloned().unwrap_or_else(|| zeros(tape.val(pw)));
    let gm = g.of(pms).cloned().unwrap_or_else(|| zeros(tape.val(pms)));
    let gl = lp.map(|lp| g.of(lp).cloned().unwrap_or_else(|| zeros(tape.val(lp))));
    let tasks: Vec<(Lined, usize)> = lined.into_iter().zip(rows).collect();
    let parts = sva_engine::each(threads, tasks, |(l, row)| {
        let gr = l.played.grid;
        let mut seed_p = Arr::zeros(48, gr.frames);
        for b in 0..48 {
            for f in 0..gr.frames {
                seed_p.v[b * gr.frames + f] = gp.v[b * m + gr.m_lo + f];
            }
        }
        let mut seed_m = Arr::zeros(1, gr.ms);
        for j in 0..gr.ms {
            let at = gr.g0 + j as i64;
            if at >= 0 && (at as usize) < gm.c {
                seed_m.v[j] = gm.v[at as usize];
            }
        }
        let mut seeds = vec![(l.played.power, &seed_p), (l.played.power_ms, &seed_m)];
        let seed_l;
        if let (Some(mv), Some(gl)) = (l.means, gl.as_ref()) {
            let k = l.tape.val(mv).r;
            seed_l = Arr::new(k, notes, gl.v[row * notes..(row + k) * notes].to_vec());
            seeds.push((mv, &seed_l));
        }
        let grads = l.tape.backward(&seeds);
        l.leaves
            .iter()
            .map(|(leaf, slot)| (*slot, grads.scalar(*leaf)))
            .collect::<Vec<_>>()
    });
    let mut out = vec![0.0; s.x.len()];
    for part in parts {
        for (slot, g) in part {
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
    Ok(Evaluated {
        distance: value,
        axes: d.axes,
        grad: Some(out),
    })
}

/// Each line's least-squares power against the reference, at its gain 1: a gain's own scale.
pub fn scales(fit: &Fit, s: &State) -> Result<Vec<f64>, CliError> {
    let mut out = vec![0.0; s.x.len()];
    let r = &fit.reference;
    for (i, root) in s.root.iter().enumerate() {
        let Some(root) = root else {
            continue;
        };
        let mut unit = s.clone();
        unit.x[*root] = 1.0;
        let Some(l) = line(fit, &unit, i)? else {
            continue;
        };
        let p = l.tape.val(l.played.power);
        let g = l.played.grid;
        let (mut num, mut den) = (0.0, 0.0);
        for b in 0..48 {
            for f in 0..g.frames {
                let e1 = p.v[b * p.c + f];
                let reference = r.power(b, g.m_lo + f);
                num += reference * e1;
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
