// Concern: a fit's shared variables: a periodic level's factor, rate, phase, depths, price, an add attached to one | Non-concern: their bookkeeping (optimize/) | IO: (Problem, Reading, Kind) -> Scored

use std::f64::consts::{LOG2_E, TAU};

use super::super::atoms::{self, Link, Quantity, SharedKind};
use super::super::grid::Grid;
use super::super::optimize::code::{Number, Price};
use super::super::optimize::{self, Added, Change, Edit, Ledger, Scored};
use super::{Draft, Held, Kind, Problem, Reading, State, Template, Written};
use super::{change, db, golden, rewrite, rewritten, write};

/// The one kind of shared variable, its numbers its rate and phase.
pub(super) const PERIODIC: usize = SharedKind::Periodic as usize;

impl Problem {
    /// The power factor per frame `links` put on an atom's level, as `MODEL` states, each
    /// link indexing `shared`.
    pub(super) fn factor(&self, links: &[Link], shared: &[optimize::Shared]) -> Vec<f64> {
        self.times
            .iter()
            .map(|t| {
                links
                    .iter()
                    .inspect(|link| {
                        assert_eq!(link.kind as usize, shared[link.to].kind, "a link's kind");
                    })
                    .map(|link| match link.kind {
                        SharedKind::Periodic => {
                            let (rate, phase) =
                                (shared[link.to].numbers[0], shared[link.to].numbers[1]);
                            self.kernel
                                .iter()
                                .map(|(at, share)| {
                                    let turn = TAU * rate * (t + at) + phase;
                                    share * libm::pow(10.0, link.own[0] * libm::sin(turn) / 10.0)
                                })
                                .sum::<f64>()
                        }
                    })
                    .product()
            })
            .collect()
    }
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

/// The shared variables `state`'s atoms reference, by index, as their nodes write them.
pub(super) fn active(state: &State) -> Vec<(usize, atoms::Shared)> {
    let shared = state.shared();
    (0..shared.len())
        .filter(|s| {
            state
                .kept()
                .iter()
                .any(|h| h.links().iter().any(|l| l.to == *s))
        })
        .map(|s| {
            let node = atoms::Shared {
                kind: SharedKind::ALL[shared[s].kind],
                numbers: shared[s].numbers.clone(),
            };
            (s, node)
        })
        .collect()
}

fn members(kept: &[Held], s: usize) -> Vec<usize> {
    (0..kept.len())
        .filter(|a| kept[*a].links().iter().any(|l| l.to == s))
        .collect()
}

/// Per frame its time beside the level in dB by which `back` holds `base`'s power: their
/// weighted correlation over `base`'s weighted energy, frames where it is not positive left out.
fn series(p: &Problem, back: &[Vec<f64>], base: &dyn Fn(usize) -> Vec<f64>) -> Vec<(f64, f64)> {
    (0..p.frames())
        .filter_map(|f| {
            let (energy, pull) = p.moments(f, &base(f), &back[f]);
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
        .atom
        .numbers()
        .iter()
        .zip(&written.price.numbers)
        .find(|((q, _), _)| *q == Quantity::Depth)
        .map_or(f64::INFINITY, |(_, n)| n.delta)
}

/// Periodic level `s`'s price, `held` the atoms referencing it, as `MODEL` states.
pub(super) fn shared_price<'a>(
    p: &Problem,
    held: impl Iterator<Item = &'a Held>,
    s: usize,
) -> Price {
    let phase = held
        .flat_map(|h| {
            h.links()
                .iter()
                .filter(|l| l.to == s)
                .map(|l| depth_step(&h.written) / l.own[0].abs())
        })
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

/// An add attached to periodic level `s` beside what it sets.
type Attached = (Held, Vec<(usize, Held)>, optimize::Shared);

/// `kind` keyed `key` added attached to periodic level `s`, the atoms attached to it already
/// set beside it, as `MODEL` states; `None` where the frames hold no such level.
pub(super) fn attach(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    key: u64,
    kind: &Kind,
    s: usize,
) -> Option<Scored<Problem>> {
    let residual = &reading.residual;
    let old = members(kept, s);
    let mut back = residual.clone();
    for a in &old {
        for (f, r) in back.iter_mut().enumerate() {
            r.iter_mut()
                .zip(kept[*a].model(p, f))
                .for_each(|(r, u)| *r += u);
        }
    }
    let templates = p.templates(kind);
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
    let summed = |columns: &[Template], gain: &dyn Fn(usize, usize) -> f64, f: usize| {
        let mut u = vec![0.0; p.range.len()];
        for (c, t) in columns.iter().enumerate() {
            let g = gain(c, f);
            u.iter_mut().zip(t.at(f)).for_each(|(u, t)| *u += g * t);
        }
        u
    };
    let mut all = vec![series(p, &back, &|f| {
        summed(&templates, &|c, _| shape[c], f)
    })];
    for a in &old {
        let held = &kept[*a];
        let level = |c: usize, f: usize| libm::pow(10.0, held.written.levels[c][f] / 10.0);
        all.push(series(p, &back, &|f| summed(&held.templates, &level, f)));
    }
    let (start, step) = modulation_peak(p, &all)?;
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
        let factor = p.factor(&links, &after);
        let fitted: Vec<Vec<f64>> = free
            .iter()
            .map(|g| g.iter().zip(&factor).map(|(g, m)| g / m).collect())
            .collect();
        let new = Held {
            kind: kind.clone(),
            templates: templates.clone(),
            written: write(p, kind, key, &fitted, (&links, &after)),
        };
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
                let trajectories = held.written.atom.levels().into_iter().cloned().collect();
                let key = held.written.atom.key;
                let written = rewrite(p, &held.kind, key, trajectories, (&links, &after));
                (*a, rewritten(held, written))
            })
            .collect();
        let delta = |f: usize| {
            let mut u = new.model(p, f);
            for (a, held) in &others {
                let after = held.model(p, f);
                let before = kept[*a].model(p, f);
                for ((u, x), y) in u.iter_mut().zip(after).zip(before) {
                    *u += x - y;
                }
            }
            u
        };
        let missed = 0.5 * LOG2_E * change(p, residual, &delta);
        let held = std::iter::once(&new).chain(others.iter().map(|(_, h)| h));
        let variable = optimize::Shared {
            kind: PERIODIC,
            numbers: vec![rate, phase],
        };
        let price = shared_price(p, held, s);
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
    let range = (start - step, start + step);
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

/// The DL change replacing kept atom `a` by `after`, or removing it where `None`, each shared
/// variable it references repriced.
pub(super) fn swapped(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    a: usize,
    after: Option<&Held>,
    missed: f64,
) -> f64 {
    let rest = || {
        let others = kept.iter().enumerate().filter(move |(i, _)| *i != a);
        others.map(|(_, h)| h).chain(after)
    };
    let prices: Vec<(usize, Price)> = kept[a]
        .links()
        .iter()
        .map(|l| (l.to, shared_price(p, rest(), l.to)))
        .collect();
    let uses: Vec<usize> =
        after.map_or_else(Vec::new, |h| h.links().iter().map(|l| l.to).collect());
    let change = Change {
        added: None,
        removed: after.is_none().then_some(a),
        replaced: after.map_or_else(Vec::new, |h| vec![(a, &h.written.price, &uses[..])]),
        shared: prices.iter().map(|(s, price)| (*s, price)).collect(),
    };
    ledger.changed(&change, missed)
}
