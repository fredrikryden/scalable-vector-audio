// Concern: the judge proxy's axes per segment in JND and the one distance a fit lowers, on the tape | Non-concern: where either side's pattern came from | IO: (Reference, Side) -> distance, axes

use std::sync::Arc;

use sva_samples::measure::heard::{BANDS, Heard};

use super::ear::{Ear, INTEGRATION_SECS};
use super::tape::{Arr, Conv, Pad, Sparse, Tape, V};

const EPS: f64 = 0.003;
const SABS: f64 = 0.01;
const SOFT_T: f64 = 0.5;
const ATTACK_WEIGHT: f64 = 0.3;
const PITCH_WEIGHT: f64 = 0.5;
const SHORT_NOTE_SECS: f64 = 0.1;
const LEAST_FRAMES: usize = 3;
const DECAY_RANGE_PHON: f64 = 30.0;
const BEATING_RANGE_PHON: f64 = 20.0;
const BEATING_LEAST_FRAMES: usize = 20;
const SAVGOL_MOST: usize = 51;
const PEAK_CENTS: f64 = 30.0;

pub struct Segment {
    pub a: f64,
    pub b: f64,
    pub m0: usize,
    pub m1: usize,
    pub gain: f64,
    pub note: Option<usize>,
    /// N' of the reference over this segment, `[48, m1 - m0]`.
    nref: Arr,
}

pub struct Peaks {
    pub hz: Vec<f64>,
    pub amplitude: Vec<f64>,
    pub cents_per_jnd: f64,
}

pub struct Reference {
    pub ear: Arc<Ear>,
    pub start_secs: f64,
    pub scale: f64,
    pub frames: usize,
    pub segments: Vec<Segment>,
    /// Per note segment (`segments[1..]`), the reference's peaks.
    pub peaks: Vec<Peaks>,
    pub rises: Vec<Option<f64>>,
    pub duration_secs: f64,
    raw: Vec<f64>,
}

/// What the model side hands the distance, per frame and per note segment.
pub struct Side {
    /// Power `[48, frames]`, full-scale units.
    pub power: V,
    /// Each line's frequency, and its mean power per note segment `[lines, notes]`.
    pub line_hz: Vec<f64>,
    pub line_power: Option<V>,
    /// Each note segment's rise, where the model starts it from quiet.
    pub rises: Vec<Option<V>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Axes {
    pub at_secs: f64,
    pub balance: f64,
    pub sharpness: f64,
    pub decay: Option<f64>,
    pub beating: Option<f64>,
    pub pitch: Option<f64>,
    pub attack: Option<f64>,
}

impl Axes {
    pub fn largest(&self) -> f64 {
        [
            Some(self.balance),
            Some(self.sharpness),
            self.decay,
            self.beating,
            self.pitch,
            self.attack,
        ]
        .into_iter()
        .flatten()
        .fold(0.0, f64::max)
    }
}

pub struct Distance {
    pub total: V,
    pub axes: Vec<Axes>,
    /// What the total sums: half the whole clip's terms, the notes' soft maximum, the
    /// weighted largest pitch error and the weighted mean attack error.
    pub terms: [f64; 4],
}

impl Reference {
    pub fn of(heard: &Heard, ear: Arc<Ear>) -> Reference {
        let scale = Ear::power_scale(heard.rms);
        let m = heard.frames;
        let hop = ear.hop_ms as f64 / 1000.0;
        let mut spans = vec![(0.0, heard.duration_secs, heard.rms, None)];
        for (i, n) in heard.notes.iter().enumerate() {
            let (a, b) = (n.at_secs - heard.start_secs, n.next_secs - heard.start_secs);
            spans.push((a, b, n.rms, Some(i)));
        }
        let mut segments = Vec::new();
        let mut tape = Tape::new();
        for (a, b, rms, note) in spans {
            let m0 = libm::round(a / hop) as usize;
            let m1 = (libm::round(b / hop) as usize).min(m);
            if m1 < m0 + LEAST_FRAMES {
                continue;
            }
            let gain = (heard.rms / (rms + 1e-12)) * (heard.rms / (rms + 1e-12));
            let cols: Vec<f64> = (0..BANDS)
                .flat_map(|band| {
                    heard.power[band * m + m0..band * m + m1]
                        .iter()
                        .map(|p| p * scale * gain)
                })
                .collect();
            let e = tape.constant(Arr::new(BANDS, m1 - m0, cols));
            let (np, _) = ear.loudness(&mut tape, e);
            segments.push(Segment {
                a,
                b,
                m0,
                m1,
                gain,
                note,
                nref: tape.val(np).clone(),
            });
        }
        let mut peaks = Vec::new();
        let mut rises = Vec::new();
        for s in segments.iter().filter(|s| s.note.is_some()) {
            let n = &heard.notes[s.note.expect("a note segment")];
            peaks.push(Peaks {
                hz: n.peaks.iter().map(|p| p.hz).collect(),
                amplitude: n.peaks.iter().map(|p| p.amplitude).collect(),
                cents_per_jnd: if s.b - s.a < SHORT_NOTE_SECS {
                    10.0
                } else {
                    5.0
                },
            });
            rises.push(n.rise_secs.filter(|_| n.new));
        }
        Reference {
            ear,
            start_secs: heard.start_secs,
            scale,
            frames: m,
            segments,
            peaks,
            rises,
            duration_secs: heard.duration_secs,
            raw: heard.power.clone(),
        }
    }

    /// The reference's own power in band `b` at frame `m`, full-scale units.
    pub fn power(&self, b: usize, m: usize) -> f64 {
        self.raw[b * self.frames + m]
    }

    pub fn notes(&self) -> impl Iterator<Item = &Segment> {
        self.segments.iter().filter(|s| s.note.is_some())
    }

    /// Weights from each line's power per frame to its mean per note segment.
    pub fn note_means(&self, line: usize, m_lo: usize, frames: usize) -> Vec<(u32, u32, f64)> {
        let mut out = Vec::new();
        for (s, seg) in self.notes().enumerate() {
            let (lo, hi) = (seg.m0.max(m_lo), seg.m1.min(m_lo + frames));
            let w = 1.0 / (seg.m1 - seg.m0) as f64;
            for m in lo..hi {
                out.push((
                    (line * self.peaks.len() + s) as u32,
                    (line * frames + m - m_lo) as u32,
                    w,
                ));
            }
        }
        out
    }

    /// The rms envelope's ms span each note segment's rise is read over.
    pub fn rise_span(&self, note: usize) -> (usize, usize) {
        let seg = self.notes().nth(note).expect("a note segment");
        let a = (seg.a * 1000.0) as usize;
        let b = ((seg.b.min(seg.a + 0.1)) * 1000.0) as usize;
        (a, b)
    }
}

fn savgol(wl: usize) -> Vec<f64> {
    let m = (wl / 2) as f64;
    let den = (2.0 * m + 1.0) * (2.0 * m - 1.0) * (2.0 * m + 3.0);
    (0..wl)
        .map(|i| {
            let j = i as f64 - m;
            (3.0 * (3.0 * m * m + 3.0 * m - 1.0) - 15.0 * j * j) / den
        })
        .collect()
}

fn integrator(frames: usize) -> Arc<Conv> {
    let len = (6.0 * INTEGRATION_SECS / 0.01) as usize + 1;
    let mut k: Vec<f64> = (0..len)
        .map(|j| sva_formula::exp(-(j as f64) * 0.01 / INTEGRATION_SECS))
        .collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|x| *x /= s);
    k.reverse();
    Arc::new(Conv {
        k,
        stride: 1,
        off: -(len as i64 - 1),
        out: frames,
        pad: Pad::Edge,
    })
}

/// `10 log2(x + e)`.
fn phon(tape: &mut Tape, x: V, e: f64) -> V {
    let y = tape.offset(x, e);
    let l = tape.log2(y);
    tape.scale(l, 10.0)
}

fn row_max(a: &Arr) -> Vec<f64> {
    (0..a.r)
        .map(|i| {
            a.v[i * a.c..(i + 1) * a.c]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

/// The least-squares slope per second of `l` over columns `[from, from + n)`, 10 ms apart.
fn slope(tape: &mut Tape, l: V, from: usize, n: usize) -> V {
    let mean = (n - 1) as f64 / 2.0;
    let tt: Vec<f64> = (0..n).map(|i| (i as f64 - mean) * 0.01).collect();
    let ss: f64 = tt.iter().map(|x| x * x).sum();
    let part = tape.cols(l, from, from + n);
    let w = tape.constant(Arr::row(tt.iter().map(|x| x / ss).collect()));
    let p = tape.mul(part, w);
    tape.sum(p)
}

/// Frames from the loudest on, until 30 phon under it: `(start, count)`.
fn decay_run(l: &[f64]) -> (usize, usize) {
    let p = l
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, v)| {
            if *v > bv { (i, *v) } else { (bi, bv) }
        })
        .0;
    let from = (p + 2).min(l.len());
    let n = l[from..]
        .iter()
        .position(|v| *v < l[p] - DECAY_RANGE_PHON)
        .unwrap_or(l.len() - from);
    (from, n)
}

struct Terms {
    total: V,
    axes: Axes,
}

fn segment_terms(tape: &mut Tape, r: &Reference, seg: &Segment, model: V) -> Terms {
    let ear = &r.ear;
    let w = seg.m1 - seg.m0;
    let cut = tape.cols(model, seg.m0, seg.m1);
    let e = tape.scale(cut, r.scale * seg.gain);
    let (np, n24) = ear.loudness(tape, e);
    let nref = tape.constant(seg.nref.clone());
    let nref24 = tape.matl(Arc::clone(&ear.pool), nref);
    let dph = {
        let a = tape.offset(nref24, EPS);
        let b = tape.offset(n24, EPS);
        let q = tape.div(b, a);
        let l = tape.log2(q);
        tape.scale(l, 10.0)
    };
    let wts = tape.max(nref24, n24);
    let wts = tape.detach(wts);
    let ad = tape.sabs(dph, SABS);
    let num = tape.mul(wts, ad);
    let num = tape.sum(num);
    let den = tape.sum(wts);
    let den = tape.offset(den, 1e-9);
    let bal = tape.div(num, den);

    let sw = tape.constant(Arr::column(ear.sharp.clone()));
    let sharp = |tape: &mut Tape, n: V| -> (V, V) {
        let tot = tape.sum_cols(n);
        let tot = tape.offset(tot, 1e-9);
        let wn = tape.mul(n, sw);
        let s = tape.sum_cols(wn);
        (tape.div(s, tot), tot)
    };
    let (sa, ta) = sharp(tape, nref);
    let (sb, tb) = sharp(tape, np);
    let wt = tape.max(ta, tb);
    let wt = tape.detach(wt);
    let x = tape.mul(sb, wt);
    let x = tape.sum(x);
    let y = tape.mul(sa, wt);
    let y = tape.sum(y);
    let q = tape.div(x, y);
    let q = tape.offset(q, -1.0);
    let shp = tape.scale(q, 100.0);
    let shp = tape.sabs(shp, SABS);
    let sharpness = tape.scale(shp, 0.2);

    let smooth = integrator(w);
    let la = {
        let s = tape.sum_cols(nref24);
        let s = tape.scale(s, 0.5);
        let s = tape.conv(Arc::clone(&smooth), s);
        phon(tape, s, 1e-4)
    };
    let lb = {
        let s = tape.sum_cols(n24);
        let s = tape.scale(s, 0.5);
        let s = tape.conv(smooth, s);
        phon(tape, s, 1e-4)
    };
    let (fa, na) = decay_run(&tape.val(la).v);
    let mut decay = None;
    if na >= 5 {
        let sa_ = slope(tape, la, fa, na);
        let (fb, nb) = decay_run(&tape.val(lb).v);
        let sb_ = if nb >= 5 {
            slope(tape, lb, fb, nb)
        } else {
            sa_
        };
        let sav = tape.scalar_of(sa_);
        if sav < -5.0 {
            let d = tape.sub(sb_, sa_);
            let d = tape.sabs(d, SABS);
            let jnd = (0.15 * sav.abs()).max(1.0 / (seg.b - seg.a).max(0.05));
            decay = Some(tape.scale(d, 1.0 / jnd));
        }
    }

    let mut beating = None;
    if w >= BEATING_LEAST_FRAMES {
        let wl = (w / 2 * 2 - 1).min(SAVGOL_MOST);
        let lr = phon(tape, nref24, EPS);
        let act: Vec<f64> = {
            let l = tape.val(lr);
            let top = row_max(l);
            l.v.iter()
                .enumerate()
                .map(|(i, v)| f64::from(u8::from(*v > top[i / l.c] - BEATING_RANGE_PHON)))
                .collect()
        };
        let act = tape.constant(Arr::new(BANDS / 2, w, act));
        let counts: Vec<f64> = {
            let a = tape.val(act);
            (0..a.r)
                .map(|i| a.v[i * a.c..(i + 1) * a.c].iter().sum())
                .collect()
        };
        let mut fl = Vec::new();
        for n in [nref24, n24] {
            let l = if n == nref24 { lr } else { phon(tape, n, EPS) };
            let t = match wl >= 5 {
                true => tape.conv(
                    Arc::new(Conv {
                        k: savgol(wl),
                        stride: 1,
                        off: -((wl / 2) as i64),
                        out: w,
                        pad: Pad::Edge,
                    }),
                    l,
                ),
                false => {
                    let s = tape.sum_rows(l);
                    let m = tape.scale(s, 1.0 / w as f64);
                    tape.detach(m)
                }
            };
            let d = tape.sub(l, t);
            let d2 = tape.square(d);
            let d2 = tape.mul(d2, act);
            let s = tape.sum_rows(d2);
            let c = tape.constant(Arr::column(counts.iter().map(|c| c + 1e-9).collect()));
            let ms = tape.div(s, c);
            let ms = tape.offset(ms, 1e-6);
            let rms = tape.sqrt(ms);
            let wb: Vec<f64> = {
                let nv = tape.val(n);
                (0..nv.r)
                    .map(|i| {
                        let mean = nv.v[i * nv.c..(i + 1) * nv.c].iter().sum::<f64>() / w as f64;
                        mean * f64::from(u8::from(counts[i] >= BEATING_LEAST_FRAMES as f64))
                    })
                    .collect()
            };
            let total: f64 = wb.iter().sum::<f64>() + 1e-9;
            let wb = tape.constant(Arr::column(wb.iter().map(|x| x / total).collect()));
            let p = tape.mul(rms, wb);
            fl.push(tape.sum(p));
        }
        let d = tape.sub(fl[1], fl[0]);
        let d = tape.sabs(d, SABS);
        beating = Some(tape.scale(d, 1.0 / 0.3));
    }

    let mut total = tape.add(bal, sharpness);
    if let Some(d) = decay {
        total = tape.add(total, d);
    }
    if let Some(b) = beating {
        let half = tape.scale(b, 0.5);
        total = tape.add(total, half);
    }
    Terms {
        total,
        axes: Axes {
            at_secs: seg.a,
            balance: tape.scalar_of(bal),
            sharpness: tape.scalar_of(sharpness),
            decay: decay.map(|d| tape.scalar_of(d)),
            beating: beating.map(|b| tape.scalar_of(b)),
            pitch: None,
            attack: None,
        },
    }
}

/// Each note's pitch error in JND: its reference peaks against the model's lines near them.
fn pitch(tape: &mut Tape, r: &Reference, side: &Side) -> Vec<Option<V>> {
    let notes = r.peaks.len();
    let Some(lp) = side.line_power else {
        return vec![None; notes];
    };
    let mut out = Vec::new();
    for (s, pk) in r.peaks.iter().enumerate() {
        let np = pk.hz.len();
        if np == 0 || pk.amplitude.iter().all(|a| *a <= 0.0) {
            out.push(None);
            continue;
        }
        let (mut wn, mut wc) = (Vec::new(), Vec::new());
        for (l, hz) in side.line_hz.iter().enumerate() {
            if hz.is_nan() || *hz <= 0.0 {
                continue;
            }
            for (p, ph) in pk.hz.iter().enumerate() {
                let c = 1200.0 * libm::log2(hz / ph);
                let g = sva_formula::exp(-(c / PEAK_CENTS) * (c / PEAK_CENTS));
                if g < 1e-12 {
                    continue;
                }
                let at = (l * notes + s) as u32;
                wn.push((p as u32, at, g));
                wc.push((p as u32, at, g * c.abs()));
            }
        }
        let shape = |entries| {
            Arc::new(Sparse {
                r: 1,
                c: np,
                entries,
            })
        };
        let num = tape.linear(shape(wn), lp);
        let cen = tape.linear(shape(wc), lp);
        let wsq: f64 = pk.amplitude.iter().map(|a| a * a).sum();
        let rel_ref: Vec<f64> = pk.amplitude.iter().map(|a| a * a / (wsq + 1e-30)).collect();
        let total = tape.sum(num);
        let total = tape.offset(total, 1e-30);
        let rel = tape.div(num, total);
        let rel = tape.offset(rel, 1e-12);
        let rref = tape.constant(Arr::row(rel_ref.iter().map(|x| x + 1e-12).collect()));
        let q = tape.div(rel, rref);
        let q = tape.log2(q);
        let db = tape.scale(q, 10.0 * std::f64::consts::LOG10_2);
        let z = tape.offset(db, 10.0);
        let z = tape.scale(z, 0.5);
        let ez = tape.exp(z);
        let one = tape.offset(ez, 1.0);
        let hundred = tape.num(100.0);
        let miss = tape.div(hundred, one);
        let most = tape.val(num).v.iter().copied().fold(0.0, f64::max);
        let den = tape.offset(num, 1e-6 * most + 1e-20);
        let spread = tape.div(cen, den);
        let err = tape.add(spread, miss);
        let err = tape.clamp(err, f64::NEG_INFINITY, 100.0);
        let wsum: f64 = pk.amplitude.iter().sum();
        let w = tape.constant(Arr::row(
            pk.amplitude
                .iter()
                .map(|a| a / pk.cents_per_jnd / (wsum + 1e-30))
                .collect(),
        ));
        let e = tape.mul(err, w);
        out.push(Some(tape.sum(e)));
    }
    out
}

pub fn distance(tape: &mut Tape, r: &Reference, side: &Side) -> Distance {
    let mut whole = None;
    let mut notes = Vec::new();
    let mut axes = Vec::new();
    for seg in &r.segments {
        let t = segment_terms(tape, r, seg, side.power);
        axes.push(t.axes);
        match seg.note {
            None => whole = Some(t.total),
            Some(_) => notes.push(t.total),
        }
    }
    let soft = match notes.is_empty() {
        true => tape.num(0.0),
        false => {
            let v = tape.concat(notes);
            let x = tape.scale(v, 1.0 / SOFT_T);
            let e = tape.exp(x);
            let m = tape.mean(e);
            let l = tape.ln(m);
            tape.scale(l, SOFT_T)
        }
    };
    let whole = whole.unwrap_or_else(|| tape.num(0.0));
    let half = tape.scale(whole, 0.5);
    let mut terms = [tape.scalar_of(half), tape.scalar_of(soft), 0.0, 0.0];
    let mut total = tape.add(half, soft);
    let offset = usize::from(r.segments.first().is_some_and(|s| s.note.is_none()));

    let pitches = pitch(tape, r, side);
    let mut most: Option<V> = None;
    for (s, p) in pitches.iter().enumerate() {
        if let Some(p) = p {
            axes[offset + s].pitch = heard_pitch(tape, r, side, s);
            most = Some(match most {
                Some(m) => tape.max(m, *p),
                None => *p,
            });
        }
    }
    if let Some(m) = most {
        let w = tape.scale(m, PITCH_WEIGHT);
        terms[2] = tape.scalar_of(w);
        total = tape.add(total, w);
    }

    let mut attack = Vec::new();
    for (s, (ra, rb)) in r.rises.iter().zip(&side.rises).enumerate() {
        if let (Some(ra), Some(rb)) = (ra, rb) {
            let q = tape.scale(*rb, 1.0 / ra);
            let q = tape.offset(q, -1.0);
            let q = tape.sabs(q, SABS);
            let j = tape.scale(q, 4.0);
            axes[offset + s].attack = Some(tape.scalar_of(j));
            attack.push(j);
        }
    }
    if !attack.is_empty() {
        let v = tape.concat(attack);
        let m = tape.mean(v);
        let w = tape.scale(m, ATTACK_WEIGHT);
        terms[3] = tape.scalar_of(w);
        total = tape.add(total, w);
    }
    Distance { total, axes, terms }
}

/// A rise off a model's power per millisecond: the 10-90 % crossings of its 2 ms rms,
/// interpolated, where it starts 15 dB under its first peak.
pub fn rise(tape: &mut Tape, power_ms: V, from: usize, to: usize) -> Option<V> {
    let n = tape.val(power_ms).c;
    let to = to.min(n.saturating_sub(1));
    if to < from + 8 {
        return None;
    }
    let two = Arc::new(Conv {
        k: vec![0.5, 0.5],
        stride: 1,
        off: from as i64,
        out: to - from,
        pad: Pad::Edge,
    });
    let ms = tape.conv(two, power_ms);
    let ms = tape.clamp(ms, 0.0, f64::INFINITY);
    let e = tape.sqrt(ms);
    let v = tape.val(e).v.clone();
    let pk = v.iter().copied().fold(0.0, f64::max);
    if pk.is_nan() || pk <= 0.0 || v[0] > pk * libm::pow(10.0, -15.0 / 20.0) {
        return None;
    }
    let p = v
        .iter()
        .position(|x| *x == pk)
        .expect("the peak is in the envelope");
    let on = v[..=p].iter().rposition(|x| *x <= 0.1 * pk).unwrap_or(0);
    let top = on
        + v[on..=p]
            .iter()
            .position(|x| *x >= 0.9 * pk)
            .expect("the peak itself is at 90 % of the peak");
    let pkv = tape.cols(e, p, p + 1);
    let crossing = |tape: &mut Tape, i: usize, frac: f64| -> V {
        let (a, b) = (tape.cols(e, i, i + 1), tape.cols(e, i + 1, i + 2));
        let level = tape.scale(pkv, frac);
        let num = tape.sub(level, a);
        let den = tape.sub(b, a);
        let tiny = tape.num(1e-12 * pk);
        let den = tape.max(den, tiny);
        let u = tape.div(num, den);
        let u = tape.clamp(u, 0.0, 1.0);
        tape.offset(u, i as f64)
    };
    let t10 = crossing(tape, on.min(p.saturating_sub(1)), 0.1);
    let t90 = crossing(tape, top.saturating_sub(1).max(on), 0.9);
    let d = tape.sub(t90, t10);
    let d = tape.clamp(d, 1.0 / 48.0, f64::INFINITY);
    Some(tape.scale(d, 0.001))
}

/// A note's pitch as the judge reads a chord: each reference peak's distance to the model's
/// nearest strong line under the ceiling, capped at 100 cents, the amplitude-weighted median.
fn heard_pitch(tape: &Tape, r: &Reference, side: &Side, note: usize) -> Option<f64> {
    const MODEL_PEAKS: usize = 20;
    const CAP_CENTS: f64 = 100.0;
    let lp = tape.val(side.line_power?);
    let pk = &r.peaks[note];
    let mut lines: Vec<(f64, f64)> = side
        .line_hz
        .iter()
        .enumerate()
        .map(|(l, hz)| (*hz, lp.v[l * lp.c + note]))
        .filter(|(hz, p)| {
            *hz > 30.0 && *hz < sva_samples::measure::heard::PEAK_CEILING_HZ && *p > 0.0
        })
        .collect();
    lines.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top = lines.first()?.1;
    lines.retain(|(_, p)| *p >= top * 0.01);
    lines.truncate(MODEL_PEAKS);
    let mut c: Vec<(f64, f64)> = pk
        .hz
        .iter()
        .zip(&pk.amplitude)
        .map(|(hz, a)| {
            let near = lines
                .iter()
                .map(|(m, _)| (1200.0 * libm::log2(hz / m)).abs())
                .fold(CAP_CENTS, f64::min);
            (near, *a)
        })
        .collect();
    if c.is_empty() {
        return None;
    }
    c.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = c.iter().map(|x| x.1).sum();
    let mut held = 0.0;
    let at = c
        .iter()
        .position(|(_, w)| {
            held += w;
            held >= total / 2.0
        })
        .unwrap_or(c.len() - 1);
    Some(c[at].0 / pk.cents_per_jnd)
}

/// A measured side: its own pattern, and for each of the reference's notes the measured note
/// nearest it in onset, its peaks its lines and its rise its own.
pub fn measured(tape: &mut Tape, heard: &Heard, r: &Reference) -> Side {
    let m = r.frames;
    let mut power = vec![0.0; BANDS * m];
    for b in 0..BANDS {
        for f in 0..m.min(heard.frames) {
            power[b * m + f] = heard.power[b * heard.frames + f];
        }
    }
    let power = tape.constant(Arr::new(BANDS, m, power));
    let notes = r.peaks.len();
    let (mut line_hz, mut cells, mut rises) = (Vec::new(), Vec::new(), Vec::new());
    for (s, seg) in r.notes().enumerate() {
        let at = r.start_secs + seg.a;
        let near = heard
            .notes
            .iter()
            .min_by(|x, y| (x.at_secs - at).abs().total_cmp(&(y.at_secs - at).abs()));
        rises.push(near.and_then(|n| n.rise_secs.map(|x| tape.num(x))));
        for p in near.map_or(&[][..], |n| &n.peaks[..]) {
            line_hz.push(p.hz);
            let mut row = vec![0.0; notes];
            row[s] = p.amplitude * p.amplitude;
            cells.extend(row);
        }
    }
    let line_power =
        (!line_hz.is_empty()).then(|| tape.constant(Arr::new(line_hz.len(), notes, cells)));
    Side {
        power,
        line_hz,
        line_power,
        rises,
    }
}
