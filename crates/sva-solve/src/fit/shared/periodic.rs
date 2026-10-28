// Concern: the periodic level: its factor on a render, its rate, phase and depths off a residual, its price, an add attached | Non-concern: the other kinds | IO: (Problem, Reading, Sound) -> Scored

use std::f64::consts::{LOG2_E, TAU};

use super::super::adds::golden;
use super::super::problem::{Draft, Problem, Reading};
use super::super::render::{Held, change, db};
use super::super::written::{Written, rewrite, write};
use super::{Variable, members};
use crate::grid::Grid;
use crate::optimize::code::{Number, Price};
use crate::optimize::{self, Added, Change, Edit, Ledger, Scored};
use crate::sources::{Link, Quantity, SharedKind, Sound};

/// A level in dB moving as `depth sin(2 pi rate t + phase)`, its numbers rate and phase, each
/// referencing source's own its depth.
pub(super) struct Periodic;

/// The power factor per frame a periodic level at `(rate, phase)` puts on a source's level at
/// `depth`, as `MODEL` states: its mean under the frame's kernel.
fn factor(p: &Problem, numbers: &[f64], depth: f64) -> Vec<f64> {
    let (rate, phase) = (numbers[0], numbers[1]);
    p.times
        .iter()
        .map(|t| {
            p.kernel
                .iter()
                .map(|(at, share)| {
                    let turn = TAU * rate * (t + at) + phase;
                    share * libm::pow(10.0, depth * libm::sin(turn) / 10.0)
                })
                .sum::<f64>()
        })
        .collect()
}

/// The share of a frame's power each time from its centre holds, at the analysis hop: the
/// squared window over each hop the frame pools, as `MODEL` states.
pub(super) fn kernel(grid: &Grid, window: &[f64]) -> Vec<(f64, f64)> {
    let pooled = (grid.hop / grid.step).max(1);
    let step = grid.step as f64;
    let middle = (pooled - 1) as f64 / 2.0;
    let mut shares: Vec<(i64, f64)> = Vec::new();
    for j in 0..pooled {
        for (k, w) in window.iter().enumerate() {
            let at = (j as f64 - middle) * step + k as f64 - (grid.long / 2) as f64;
            let cell = libm::round(at / step) as i64;
            match shares.binary_search_by_key(&cell, |(c, _)| *c) {
                Ok(i) => shares[i].1 += w * w,
                Err(i) => shares.insert(i, (cell, w * w)),
            }
        }
    }
    let total: f64 = shares.iter().map(|(_, v)| v).sum();
    shares
        .iter()
        .map(|(c, v)| (*c as f64 * step / grid.rate, v / total))
        .collect()
}

/// Per frame its time beside the level in dB by which `back` holds `base`'s power: their
/// weighted correlation over `base`'s weighted energy, frames where it is not positive left out.
fn series(p: &Problem, back: &[Vec<f64>], base: &[Vec<f64>]) -> Vec<(f64, f64)> {
    (0..p.frames())
        .filter_map(|f| {
            let (energy, pull) = p.moments(f, &base[f], &back[f]);
            (energy > 0.0 && pull > 0.0).then(|| (p.times[f], db(pull / energy)))
        })
        .collect()
}

/// The least-squares coefficients of each row's terms for its value, by the normal equations
/// and Gauss-Jordan elimination; `None` where they are singular.
fn least_squares(rows: &[(Vec<f64>, f64)]) -> Option<Vec<f64>> {
    let n = rows.first()?.0.len();
    let mut a = vec![vec![0.0; n + 1]; n];
    for (x, y) in rows {
        for i in 0..n {
            for j in 0..n {
                a[i][j] += x[i] * x[j];
            }
            a[i][n] += x[i] * y;
        }
    }
    for c in 0..n {
        let pivot = (c..n).max_by(|i, j| a[*i][c].abs().total_cmp(&a[*j][c].abs()))?;
        if a[pivot][c] == 0.0 {
            return None;
        }
        a.swap(c, pivot);
        let held = a[c].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r != c {
                let k = row[c] / held[c];
                for (x, y) in row[c..].iter_mut().zip(&held[c..]) {
                    *x -= k * y;
                }
            }
        }
    }
    let out: Vec<f64> = (0..n).map(|i| a[i][n] / a[i][i]).collect();
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// The peak of the modulation spectrum of `series`, as `MODEL` states, beside its rate step.
fn modulation_peak(p: &Problem, series: &[Vec<(f64, f64)>]) -> Option<(f64, f64)> {
    let span = p.times.last()? - p.times.first()?;
    let mut detrended = Vec::new();
    for s in series {
        let rows: Vec<(Vec<f64>, f64)> = s.iter().map(|(t, y)| (vec![1.0, *t], *y)).collect();
        let line = least_squares(&rows)?;
        detrended.push(
            s.iter()
                .map(|(t, y)| (*t, y - line[0] - line[1] * t))
                .collect::<Vec<_>>(),
        );
    }
    let step = 1.0 / span;
    let top = 1.0 / (2.0 * p.grid.frame_secs());
    let power = |rate: f64| -> f64 {
        detrended
            .iter()
            .map(|s| {
                let (re, im) = s.iter().fold((0.0, 0.0), |(re, im), (t, e)| {
                    let (sin, cos) = libm::sincos(TAU * rate * t);
                    (re + e * cos, im + e * sin)
                });
                re * re + im * im
            })
            .sum()
    };
    let rates = (1..).map(|k| k as f64 * step).take_while(|r| *r <= top);
    rates
        .map(|r| (r, power(r)))
        .max_by(|x, y| x.1.total_cmp(&y.1))
        .map(|(r, _)| (r, step))
}

/// A periodic level's phase at `rate` and each of `series`' depths, as `MODEL` states.
fn periodic_at(p: &Problem, series: &[Vec<(f64, f64)>], rate: f64) -> Option<(f64, Vec<f64>)> {
    let attenuation: f64 = p
        .kernel
        .iter()
        .map(|(at, share)| share * libm::cos(TAU * rate * at))
        .sum();
    let mut pairs = Vec::new();
    for s in series {
        let rows: Vec<(Vec<f64>, f64)> = s
            .iter()
            .map(|(t, y)| {
                let (sin, cos) = libm::sincos(TAU * rate * t);
                (vec![1.0, *t, sin, cos], *y)
            })
            .collect();
        let c = least_squares(&rows)?;
        pairs.push((c[2], c[3]));
    }
    let (xx, xy, yy) = pairs.iter().fold((0.0, 0.0, 0.0), |(xx, xy, yy), (a, b)| {
        (xx + a * a, xy + a * b, yy + b * b)
    });
    let mut phase = 0.5 * libm::atan2(2.0 * xy, xx - yy);
    let mut depths: Vec<f64> = pairs
        .iter()
        .map(|(a, b)| (a * libm::cos(phase) + b * libm::sin(phase)) / attenuation)
        .collect();
    if depths.iter().sum::<f64>() < 0.0 {
        phase += std::f64::consts::PI;
        depths.iter_mut().for_each(|d| *d = -*d);
    }
    Some((phase.rem_euclid(TAU), depths))
}

fn depth_step(written: &Written) -> f64 {
    written
        .source
        .numbers()
        .iter()
        .zip(&written.price.numbers)
        .find(|((q, _), _)| *q == Quantity::Depth)
        .map_or(f64::INFINITY, |(_, n)| n.delta)
}

/// An add attached to a periodic level beside what it sets.
type Attached = (Held, Vec<(usize, Held)>, optimize::Shared);

impl Variable for Periodic {
    fn transform(&self, p: &Problem, numbers: &[f64], own: &[f64], render: &mut [Vec<f64>]) {
        for (row, m) in render.iter_mut().zip(factor(p, numbers, own[0])) {
            row.iter_mut().for_each(|u| *u *= m);
        }
    }

    /// As `MODEL` states.
    fn price(&self, p: &Problem, uses: &[(&Written, &Link)]) -> Price {
        let phase = uses
            .iter()
            .map(|(w, l)| depth_step(w) / l.own[0].abs())
            .fold(f64::INFINITY, f64::min);
        Price {
            alternatives: 1,
            counts: Vec::new(),
            numbers: vec![
                Number {
                    range: 1.0 / (2.0 * p.grid.frame_secs()),
                    delta: phase / (TAU * p.clip.secs()),
                },
                Number {
                    range: TAU,
                    delta: phase,
                },
            ],
        }
    }

    /// As `MODEL` states.
    fn attach(
        &self,
        (p, kept, ledger, reading): (&Problem, &[Held], &Ledger, &Reading),
        key: u64,
        (sound, step): (&Sound, f64),
        s: usize,
    ) -> Option<Scored<Problem>> {
        let residual = &reading.residual;
        let old = members(kept, s);
        let mut back = residual.clone();
        for a in &old {
            for (r, u) in back
                .iter_mut()
                .zip(kept[*a].render(p, &reading.shared).iter())
            {
                r.iter_mut().zip(u).for_each(|(r, u)| *r += u);
            }
        }
        let templates = p.templates(sound);
        let free: Vec<Vec<f64>> = templates
            .iter()
            .map(|t| {
                (0..p.frames())
                    .map(|f| {
                        let (energy, pull) = p.moments(f, t.at(f), &back[f]);
                        match energy > 0.0 {
                            true => (pull / energy).max(0.0),
                            false => 0.0,
                        }
                    })
                    .collect()
            })
            .collect();
        let frames = p.frames().max(1) as f64;
        let shape: Vec<f64> = free
            .iter()
            .map(|g| g.iter().sum::<f64>() / frames)
            .collect();
        let mut all = vec![series(p, &back, &p.summed(&templates, &|c, _| shape[c]))];
        for a in &old {
            let held = &kept[*a];
            let level = |c: usize, f: usize| libm::pow(10.0, held.written.levels[c][f] / 10.0);
            all.push(series(p, &back, &p.summed(&held.templates, &level)));
        }
        let (start, rate_step) = modulation_peak(p, &all)?;
        let build = |rate: f64| -> Option<(f64, Attached, f64)> {
            let (phase, depths) = periodic_at(p, &all, rate)?;
            let mut after = reading.shared.clone();
            after[s].numbers = vec![rate, phase];
            let link = |d: f64| Link {
                to: s,
                kind: SharedKind::Periodic,
                own: vec![d],
            };
            let links = vec![link(depths[0])];
            let factor = factor(p, &after[s].numbers, depths[0]);
            let fitted: Vec<Vec<f64>> = free
                .iter()
                .map(|g| g.iter().zip(&factor).map(|(g, m)| g / m).collect())
                .collect();
            let written = write(p, (sound, step), key, &fitted, &links);
            let new = Held::new(p, templates.clone(), written);
            let others: Vec<(usize, Held)> = old
                .iter()
                .zip(&depths[1..])
                .map(|(a, d)| {
                    let held = &kept[*a];
                    let links: Vec<Link> = held
                        .links()
                        .iter()
                        .filter(|l| l.to != s)
                        .cloned()
                        .chain(std::iter::once(link(*d)))
                        .collect();
                    let levels = held.written.source.levels.clone();
                    let at = (held.sound(), held.written.step);
                    let written = rewrite(p, at, held.key(), levels, &links);
                    (*a, held.relinked(written))
                })
                .collect();
            let mut delta = new.render(p, &after).into_owned();
            for (a, held) in &others {
                let now = held.render(p, &after);
                let before = kept[*a].render(p, &reading.shared);
                for ((u, x), y) in delta.iter_mut().zip(now.iter()).zip(before.iter()) {
                    for ((u, x), y) in u.iter_mut().zip(x).zip(y) {
                        *u += x - y;
                    }
                }
            }
            let missed = 0.5 * LOG2_E * change(p, residual, &delta);
            let held = std::iter::once(&new.written).chain(others.iter().map(|(_, h)| &h.written));
            let price = super::price(p, held, SharedKind::Periodic, s);
            let variable = optimize::Shared {
                kind: SharedKind::Periodic as usize,
                numbers: vec![rate, phase],
            };
            let uses = [s];
            let used: Vec<Vec<usize>> = others
                .iter()
                .map(|(_, h)| h.links().iter().map(|l| l.to).collect())
                .collect();
            let edit = Change {
                added: Some((&new.written.price, &uses)),
                removed: None,
                replaced: others
                    .iter()
                    .zip(&used)
                    .map(|((a, h), u)| (*a, &h.written.price, u.as_slice()))
                    .collect(),
                shared: vec![(s, &price)],
            };
            let change = ledger.changed(&edit, missed);
            let precision = price.numbers[0].delta;
            change
                .is_finite()
                .then_some((change, (new, others, variable), precision))
        };
        let (_, _, precision) = build(start)?;
        let score = |rate: f64| build(rate).map_or(f64::INFINITY, |(v, ..)| v);
        let range = (start - rate_step, start + rate_step);
        let (rate, _) = golden(&score, range, precision, (start, score(start)));
        let (change, (new, others, variable), _) = build(rate)?;
        Some(Scored {
            change,
            edit: Edit::Add(Added {
                draft: Draft::Made(Box::new(new)),
                shared: vec![(s, variable)],
                others,
            }),
        })
    }
}
