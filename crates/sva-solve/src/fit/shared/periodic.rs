// Concern: the periodic level: its factor, its rate, phase and shape off a residual, its price, an add attached | Non-concern: the other kinds | IO: (Problem, Reading, Proposal) -> Scored

use std::f64::consts::{LOG2_E, PI, TAU};

use super::super::adds::golden;
use super::super::problem::{Draft, Problem, Reading};
use super::super::render::{Held, Proposal, change, db};
use super::super::written::{Written, rewrite, write};
use super::{Variable, members};
use crate::optimize::code::{Count, Number, Price};
use crate::optimize::{self, Added, Change, Edit, Ledger, Scored};
use crate::sources::{Link, Quantity, SharedKind};

/// A level in dB moving as `depth` times its shape, `sin(2 pi rate t + phase)` plus each
/// harmonic `k >= 2`'s `a_k sin(2 pi k rate t + phi_k)`: its numbers rate, phase, then each
/// harmonic's `a_k` and `phi_k`, each referencing source's own its depth.
pub(super) struct Periodic;

/// The shape `numbers` give at `t`.
fn shape(numbers: &[f64], t: f64) -> f64 {
    let (rate, phase) = (numbers[0], numbers[1]);
    let turn = TAU * rate * t;
    let harmonics = numbers[2..].chunks(2).enumerate().map(|(i, h)| {
        let k = (i + 2) as f64;
        h[0] * libm::sin(k * turn + h[1])
    });
    libm::sin(turn + phase) + harmonics.sum::<f64>()
}

/// The power factor per frame a periodic level at `numbers` puts on a source's level at
/// `depth`, as `MODEL` states: its mean under the frame's kernel.
fn factor(p: &Problem, numbers: &[f64], depth: f64) -> Vec<f64> {
    (0..p.frames())
        .map(|f| factor_at(p, numbers, depth, f))
        .collect()
}

fn factor_at(p: &Problem, numbers: &[f64], depth: f64, f: usize) -> f64 {
    p.view(f)
        .kernel
        .iter()
        .map(|(at, share)| {
            let level = depth * shape(numbers, p.times[f] + at);
            share * libm::pow(10.0, level / 10.0)
        })
        .sum()
}

/// The frames any of `home`'s columns is read in, in order.
fn read(home: &[Vec<usize>]) -> Vec<usize> {
    let mut frames: Vec<usize> = home.concat();
    frames.sort_unstable();
    frames.dedup();
    frames
}

/// Per frame of `frames` the level in dB by which `back` holds `base`'s power: their weighted
/// correlation over `base`'s weighted energy, frames where it is not positive left out.
fn series(
    p: &Problem,
    frames: &[usize],
    back: &[Vec<f64>],
    base: &[Vec<f64>],
) -> Vec<(usize, f64)> {
    frames
        .iter()
        .filter_map(|f| {
            let (energy, pull) = p.moments(*f, &base[*f], &back[*f]);
            (energy > 0.0 && pull > 0.0).then(|| (*f, db(pull / energy)))
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

/// The highest rate a periodic level may move at whose sources' columns are read in `homes`'
/// frames, as `MODEL` states: the frame Nyquist of the coarsest view among them, a correctness
/// limit.
fn nyquist(p: &Problem, homes: &[Vec<usize>]) -> f64 {
    homes
        .iter()
        .flatten()
        .map(|f| 1.0 / (2.0 * p.view(*f).grid.frame_secs()))
        .fold(f64::INFINITY, f64::min)
}

/// The peak of the modulation spectrum of `series` under `top`, as `MODEL` states, beside its
/// rate step.
fn modulation_peak(p: &Problem, series: &[Vec<(usize, f64)>], top: f64) -> Option<(f64, f64)> {
    let mut detrended = Vec::new();
    for s in series {
        let rows: Vec<(Vec<f64>, f64)> = s
            .iter()
            .map(|(f, y)| (vec![1.0, p.times[*f]], *y))
            .collect();
        let line = least_squares(&rows)?;
        detrended.push(
            s.iter()
                .map(|(f, y)| (p.times[*f], y - line[0] - line[1] * p.times[*f]))
                .collect::<Vec<_>>(),
        );
    }
    let step = 1.0 / p.span();
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
    let rates = (1..).map(|k| k as f64 * step).take_while(|r| *r < top);
    rates
        .map(|r| (r, power(r)))
        .max_by(|x, y| x.1.total_cmp(&y.1))
        .map(|(r, _)| (r, step))
}

/// The rank-1 factor of `rows` by alternating least squares, as `MODEL` states: each row's
/// weight beside the shared row; `None` where it is nought.
fn rank_one(rows: &[Vec<f64>]) -> Option<(Vec<f64>, Vec<f64>)> {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut shared = rows
        .iter()
        .max_by(|a, b| dot(a, a).total_cmp(&dot(b, b)))?
        .clone();
    let mut weights;
    let mut held = f64::INFINITY;
    loop {
        let norm = dot(&shared, &shared);
        if norm == 0.0 {
            return None;
        }
        weights = rows
            .iter()
            .map(|r| dot(r, &shared) / norm)
            .collect::<Vec<f64>>();
        let total = dot(&weights, &weights);
        if total == 0.0 {
            return None;
        }
        shared = (0..shared.len())
            .map(|j| {
                rows.iter()
                    .zip(&weights)
                    .map(|(r, w)| w * r[j])
                    .sum::<f64>()
                    / total
            })
            .collect();
        let misfit: f64 = rows
            .iter()
            .zip(&weights)
            .map(|(r, w)| {
                r.iter()
                    .zip(&shared)
                    .map(|(x, u)| (x - w * u).powi(2))
                    .sum::<f64>()
            })
            .sum();
        if misfit >= held {
            break;
        }
        held = misfit;
    }
    Some((weights, shared))
}

/// A frame's level in dB a periodic level at `numbers` puts on a source at `depth`, as
/// `MODEL` states: the dB of `factor`'s.
fn exact_db(p: &Problem, numbers: &[f64], depth: f64, f: usize) -> f64 {
    db(factor_at(p, numbers, depth, f))
}

/// One linear fit of `series` at `rate`: each source's line, its depth and the shared numbers,
/// as `MODEL` states.
type Linear = (Vec<[f64; 2]>, Vec<f64>, Vec<f64>);

fn linear_at(
    p: &Problem,
    attenuation: &[Vec<f64>],
    series: &[Vec<(usize, f64)>],
    rate: f64,
) -> Option<Linear> {
    let mut lines = Vec::new();
    let mut pairs = Vec::new();
    for s in series {
        let rows: Vec<(Vec<f64>, f64)> = s
            .iter()
            .map(|(f, y)| {
                let (t, held) = (p.times[*f], &attenuation[p.of[*f]]);
                let mut row = vec![1.0, t];
                for (k, a) in held.iter().enumerate() {
                    let (sin, cos) = libm::sincos(TAU * (k + 1) as f64 * rate * t);
                    row.extend([a * sin, a * cos]);
                }
                (row, *y)
            })
            .collect();
        let fitted = least_squares(&rows)?;
        lines.push([fitted[0], fitted[1]]);
        pairs.push(fitted[2..].to_vec());
    }
    let (mut depths, shared) = rank_one(&pairs)?;
    let fundamental = libm::hypot(shared[0], shared[1]);
    if fundamental == 0.0 {
        return None;
    }
    depths.iter_mut().for_each(|d| *d *= fundamental);
    let turned = match depths.iter().sum::<f64>() < 0.0 {
        true => {
            depths.iter_mut().for_each(|d| *d = -*d);
            PI
        }
        false => 0.0,
    };
    let mut numbers = vec![rate];
    for (k, h) in shared.chunks(2).enumerate() {
        let (sin, cos) = (h[0] / fundamental, h[1] / fundamental);
        if k > 0 {
            numbers.push(libm::hypot(sin, cos));
        }
        numbers.push((libm::atan2(cos, sin) + turned).rem_euclid(TAU));
    }
    Some((lines, depths, numbers))
}

/// A periodic level's numbers at `rate` with `harmonics` harmonics, the fundamental's among
/// them, and each of `series`' depths, as `MODEL` states: the linear fit, and where `exact`,
/// each frame's value then less the gap between the level's exact dB and its linear one,
/// refitted while the misfit to the exact dB falls.
fn periodic_at(
    p: &Problem,
    series: &[Vec<(usize, f64)>],
    (rate, harmonics): (f64, usize),
    exact: bool,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let attenuation: Vec<Vec<f64>> = p
        .views
        .iter()
        .map(|v| {
            (1..=harmonics)
                .map(|k| {
                    let k = k as f64;
                    v.kernel
                        .iter()
                        .map(|(at, share)| share * libm::cos(TAU * k * rate * at))
                        .sum()
                })
                .collect()
        })
        .collect();
    let linear_db = |numbers: &[f64], depth: f64, f: usize| {
        let (turn, held) = (TAU * rate * p.times[f], &attenuation[p.of[f]]);
        let harmonics = numbers[2..]
            .chunks(2)
            .enumerate()
            .map(|(i, h)| held[i + 1] * h[0] * libm::sin((i + 2) as f64 * turn + h[1]));
        depth * (held[0] * libm::sin(turn + numbers[1]) + harmonics.sum::<f64>())
    };
    if !exact {
        return linear_at(p, &attenuation, series, rate).map(|(_, d, n)| (n, d));
    }
    let mut read = series.to_vec();
    let mut best = None;
    let mut held = f64::INFINITY;
    while let Some((lines, depths, numbers)) = linear_at(p, &attenuation, &read, rate) {
        let exact: Vec<Vec<f64>> = series
            .iter()
            .zip(&depths)
            .map(|(s, d)| {
                s.iter()
                    .map(|(f, _)| exact_db(p, &numbers, *d, *f))
                    .collect()
            })
            .collect();
        let mut misfit = 0.0;
        for ((s, line), e) in series.iter().zip(&lines).zip(&exact) {
            for ((f, y), e) in s.iter().zip(e) {
                let at = line[0] + line[1] * p.times[*f] + e;
                misfit += (y - at) * (y - at);
            }
        }
        if misfit >= held {
            break;
        }
        held = misfit;
        for (((r, s), depth), e) in read.iter_mut().zip(series).zip(&depths).zip(&exact) {
            for (((_, v), (f, y)), e) in r.iter_mut().zip(s).zip(e) {
                *v = y - (e - linear_db(&numbers, *depth, *f));
            }
        }
        best = Some((numbers, depths));
    }
    best
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
    fn price(&self, p: &Problem, numbers: &[f64], uses: &[(&Written, &Link)]) -> Price {
        let (mut phase, mut amplitude, mut homes) = (f64::INFINITY, f64::INFINITY, Vec::new());
        for (w, l) in uses {
            let depth = l.own[0].abs();
            phase = phase.min(depth_step(w) / depth);
            let columns = p.columns(&Proposal::of(&w.source));
            let lowest = columns.floors.iter().copied().fold(f64::INFINITY, f64::min);
            amplitude = amplitude.min(2.0 * (p.top_db - lowest) / depth);
            homes.extend(columns.home);
        }
        let harmonics: Vec<&[f64]> = numbers[2..].chunks(2).collect();
        let fastest = harmonics
            .iter()
            .enumerate()
            .map(|(i, h)| (i + 2) as f64 * h[0])
            .fold(1.0, f64::max);
        let mut out = vec![
            Number {
                range: nyquist(p, &homes),
                delta: phase / (TAU * p.clip.secs() * fastest),
            },
            Number {
                range: TAU,
                delta: phase,
            },
        ];
        for h in &harmonics {
            out.push(Number {
                range: amplitude,
                delta: phase,
            });
            out.push(Number {
                range: TAU,
                delta: phase / h[0],
            });
        }
        Price {
            alternatives: 1,
            counts: vec![Count::From1(1 + harmonics.len() as u64)],
            numbers: out,
        }
    }

    /// As `MODEL` states.
    fn attach(
        &self,
        (p, kept, ledger, reading): (&Problem, &[Held], &Ledger, &Reading),
        key: u64,
        (proposal, step): (&Proposal, f64),
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
        let templates = p.templates(proposal);
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
        let home = p.columns(proposal).home;
        let shape: Vec<f64> = free
            .iter()
            .zip(&home)
            .map(|(g, h)| h.iter().map(|f| g[*f]).sum::<f64>() / h.len().max(1) as f64)
            .collect();
        let base = p.summed(&templates, &|c, _| shape[c]);
        let mut all = vec![series(p, &read(&home), &back, &base)];
        let mut homes = home;
        for a in &old {
            let held = &kept[*a];
            let level = |c: usize, f: usize| libm::pow(10.0, held.written.levels[c][f] / 10.0);
            let home = p.columns(&held.proposal()).home;
            all.push(series(
                p,
                &read(&home),
                &back,
                &p.summed(&held.templates, &level),
            ));
            homes.extend(home);
        }
        let top = nyquist(p, &homes);
        let (start, rate_step) = modulation_peak(p, &all, top)?;
        let build = |at: (f64, usize), exact: bool| -> Option<(f64, Attached, f64)> {
            let (numbers, depths) = periodic_at(p, &all, at, exact)?;
            let mut after = reading.shared.clone();
            after[s].numbers = numbers.clone();
            let link = |d: f64| Link {
                to: s,
                kind: SharedKind::Periodic,
                own: vec![d],
            };
            let links = vec![link(depths[0])];
            let factor = factor(p, &numbers, depths[0]);
            let fitted: Vec<Vec<f64>> = free
                .iter()
                .map(|g| g.iter().zip(&factor).map(|(g, m)| g / m).collect())
                .collect();
            let written = write(p, (proposal, step), key, &fitted, &links);
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
                    let at = (&held.proposal(), held.written.step);
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
            let variable = optimize::Shared {
                kind: SharedKind::Periodic as usize,
                numbers,
            };
            let held = std::iter::once(&new.written).chain(others.iter().map(|(_, h)| &h.written));
            let price = super::price(p, held, &variable, s);
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
        // Harmonic `harmonics` stays under the frame Nyquist at every rate searched.
        let refit = |harmonics: usize, from: f64| {
            let range = (
                start - rate_step,
                (start + rate_step).min(top / harmonics as f64),
            );
            let (_, _, precision) = build((from, harmonics), false)?;
            let score = |rate: f64| match harmonics as f64 * rate < top {
                true => build((rate, harmonics), false).map_or(f64::INFINITY, |b| b.0),
                false => f64::INFINITY,
            };
            let (rate, _) = golden(&score, range, precision, (from, score(from)));
            (harmonics as f64 * rate < top)
                .then(|| build((rate, harmonics), true).map(|b| (rate, b)))
                .flatten()
        };
        let (mut rate, mut best) = refit(1, start)?;
        for harmonics in 2.. {
            if harmonics as f64 * rate >= top {
                break;
            }
            match refit(harmonics, rate) {
                Some((at, more)) if more.0 < best.0 => (rate, best) = (at, more),
                _ => break,
            }
        }
        let (change, (new, others, variable), _) = best;
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
