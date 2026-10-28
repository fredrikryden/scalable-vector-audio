// Concern: an add: a sound refined against the residual, gliding where its tracks glide, scored in closed form, attached | Non-concern: proposing it (proposals.rs) | IO: (Problem, Proposal) -> Scored

use std::f64::consts::LOG2_E;

use sva_engine::each;

use super::peaks::{moves, searched};
use super::problem::{Draft, Problem, Reading};
use super::proposals::stretched;
use super::render::{Columns, Held, Proposal, Template, glided, hzs};
use super::shared::{self, linked};
use super::written::{Written, frequency_steps, write};
use crate::optimize::{self, Added, Edit, Ledger, Scored};
use crate::sources::{Link, Sound};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Glide {
    pub(super) rate: f64,
    pub(super) step: f64,
}

/// The least of `score` over `[lo, hi]` by golden section (Kiefer 1953) till the bracket is
/// narrower than `step`, beside its value, `at` and its value `held` kept where none beats it.
pub(super) fn golden(
    score: &dyn Fn(f64) -> f64,
    (lo, hi): (f64, f64),
    step: f64,
    held: (f64, f64),
) -> (f64, f64) {
    let ratio = (libm::sqrt(5.0) - 1.0) / 2.0;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - ratio * (b - a);
    let mut d = a + ratio * (b - a);
    let (mut sc, mut sd) = (score(c), score(d));
    let mut best = held;
    while b - a > step && step > 0.0 {
        for (x, v) in [(c, sc), (d, sd)] {
            if v < best.1 {
                best = (x, v);
            }
        }
        if sc < sd {
            (b, d, sd) = (d, c, sc);
            c = b - ratio * (b - a);
            sc = score(c);
        } else {
            (a, c, sc) = (c, d, sd);
            d = a + ratio * (b - a);
            sd = score(d);
        }
    }
    for (x, v) in [(c, sc), (d, sd)] {
        if v < best.1 {
            best = (x, v);
        }
    }
    best
}

/// One number's standard deviation under the weights, its proposal at it less and plus `h`
/// being `below` and `above`: `(sum w (du/dx)^2)^-1/2`, `u` its expected power through `links`.
pub(super) fn spread(
    p: &Problem,
    (below, above): (&Proposal, &Proposal),
    h: f64,
    gains: &[Vec<f64>],
    (links, shared): (&[Link], &[optimize::Shared]),
) -> f64 {
    let (lo, hi): (Vec<Template>, Vec<Template>) = (p.templates(below), p.templates(above));
    let mut du: Vec<Vec<f64>> = (0..p.frames())
        .map(|f| {
            (0..p.range.len())
                .map(|b| {
                    lo.iter()
                        .zip(&hi)
                        .zip(gains)
                        .map(|((l, u), g)| g[f] * (u.at(f)[b] - l.at(f)[b]))
                        .sum::<f64>()
                        / (2.0 * h)
                })
                .collect()
        })
        .collect();
    linked(p, links, shared, &mut du);
    let information: f64 = du
        .iter()
        .zip(&p.weight)
        .map(|(du, w)| du.iter().zip(w).map(|(du, w)| w * du * du).sum::<f64>())
        .sum();
    1.0 / libm::sqrt(information)
}

pub(super) fn refined(
    p: &Problem,
    ledger: &Ledger,
    residual: &[Vec<f64>],
    (proposal, step): (Proposal, f64),
    key: u64,
) -> (Proposal, f64) {
    let bin = p.grid.rate / p.grid.n as f64;
    let score = |s: &Proposal| scored(p, ledger, residual, (s, step), key).0;
    let (best, written) = scored(p, ledger, residual, (&proposal, step), key);
    let glide = proposal.sound.glide();
    let partial_at = |hz: f64| Proposal::one(Sound::Partial { hz, glide });
    // Harmonic `c` at `hz` refined as a partial: its Hz, score and standard deviation.
    let partial = |c: usize, hz: f64, held: f64| {
        let h = bin * libm::cbrt(f64::EPSILON);
        let near = (&partial_at(hz - h), &partial_at(hz + h));
        let sigma = spread(p, near, h, &written.gains[c..=c], (&[], &[]));
        let columns = p.columns(&partial_at(hz));
        let written_step = frequency_steps(p, &columns, &written.levels[c..=c]).0;
        let at = |h: f64| score(&partial_at(h));
        let range = (hz - bin / 2.0, hz + bin / 2.0);
        let (to, v) = golden(&at, range, written_step.min(sigma), (hz, held));
        (to, v, sigma)
    };
    match proposal.sound {
        Sound::Partial { hz, .. } => {
            let (hz, v, _) = partial(0, hz, best);
            (partial_at(hz), v)
        }
        Sound::Harmonic { .. } => {
            let hzs = hzs(&proposal);
            let refined = each(p.threads, (0..proposal.columns).collect(), |c| {
                let held = score(&partial_at(hzs[c]));
                let (to, _, sigma) = partial(c, hzs[c], held);
                let k = (c + 1) as f64;
                let deviation = 2.0 * to * sigma / (k * k);
                (deviation.is_finite() && deviation > 0.0)
                    .then(|| (c + 1, to, 1.0 / (deviation * deviation)))
            });
            let points: Vec<(usize, f64, f64)> = refined.into_iter().flatten().collect();
            if points.len() < 2 {
                return (proposal, best);
            }
            let (hz, stretch) = stretched(&points);
            let fitted = Proposal {
                sound: Sound::Harmonic { hz, stretch, glide },
                columns: proposal.columns,
            };
            let v = score(&fitted);
            match v < best {
                true => (fitted, v),
                false => (proposal, best),
            }
        }
        Sound::Noise { .. } | Sound::Shaped { .. } => (proposal, best),
    }
}

fn start(track: &[(f64, f64)], rate: f64) -> Option<f64> {
    (!track.is_empty()).then(|| {
        sva_formula::exp(track.iter().map(|(t, y)| y - rate * t).sum::<f64>() / track.len() as f64)
    })
}

pub(super) fn rate_floor(p: &Problem, columns: &Columns, levels: &[Vec<f64>]) -> f64 {
    let span = p.times.last().copied().unwrap_or(0.0) - p.times.first().copied().unwrap_or(0.0);
    frequency_steps(p, columns, levels).0 / columns.starts[0] / span
}

/// The glide `tracks` of `(t, ln hz)` give, as `MODEL` states: one slope of ln Hz on time by
/// least squares, each track its own intercept, written at the coarser of its standard error
/// and `floor`; 0 where it is within that of 0 or too few points hold it.
pub(super) fn slope(tracks: &[Vec<(f64, f64)>], floor: f64) -> Glide {
    type Track<'a> = (&'a [(f64, f64)], (f64, f64));
    let means = |t: &[(f64, f64)]| {
        let n = t.len() as f64;
        (
            t.iter().map(|x| x.0).sum::<f64>() / n,
            t.iter().map(|x| x.1).sum::<f64>() / n,
        )
    };
    let held: Vec<Track> = tracks
        .iter()
        .filter(|t| t.len() >= 2)
        .map(|t| (t.as_slice(), means(t)))
        .collect();
    let (mut sxx, mut sxy, mut count) = (0.0, 0.0, 0);
    for (t, (mt, my)) in &held {
        for (x, y) in t.iter() {
            sxx += (x - mt) * (x - mt);
            sxy += (x - mt) * (y - my);
        }
        count += t.len();
    }
    let dof = count as f64 - held.len() as f64 - 1.0;
    if dof < 1.0 || sxx <= 0.0 {
        return Glide::default();
    }
    let rate = sxy / sxx;
    let rss: f64 = held
        .iter()
        .map(|(t, (mt, my))| {
            t.iter()
                .map(|(x, y)| {
                    let e = y - my - rate * (x - mt);
                    e * e
                })
                .sum::<f64>()
        })
        .sum();
    let step = libm::sqrt(rss / dof / sxx).max(floor);
    match step > 0.0 && rate.abs() > step {
        true => Glide { rate, step },
        false => Glide::default(),
    }
}

/// The fixed `proposal` gliding where its peak tracks glide, beside the glide's precision, as
/// `MODEL` states.
fn gliding(
    p: &Problem,
    ledger: &Ledger,
    residual: &[Vec<f64>],
    mean: &[f64],
    proposal: &Proposal,
    key: u64,
) -> Option<(Proposal, f64)> {
    if matches!(proposal.sound, Sound::Noise { .. } | Sound::Shaped { .. }) {
        return None;
    }
    let tracks: Vec<Vec<(f64, f64)>> = hzs(proposal)
        .iter()
        .map(|hz| searched(p, residual, mean, *hz))
        .collect();
    let (_, written) = scored(p, ledger, residual, (proposal, 0.0), key);
    let glide = slope(
        &tracks,
        rate_floor(p, &p.columns(proposal), &written.levels),
    );
    if glide.rate == 0.0 || !moves(p, &glided(proposal, glide.rate, p.middle())) {
        return None;
    }
    let starts: Vec<(usize, f64, f64)> = tracks
        .iter()
        .enumerate()
        .filter_map(|(c, t)| start(t, glide.rate).map(|s| (c + 1, s, 1.0)))
        .collect();
    let moved = glided(proposal, glide.rate, p.middle());
    let sound = match (&moved.sound, &starts[..]) {
        (Sound::Partial { .. }, [(_, hz, _), ..]) => Sound::Partial {
            hz: *hz,
            glide: glide.rate,
        },
        (Sound::Harmonic { .. }, [_, _, ..]) => {
            let (hz, stretch) = stretched(&starts);
            Sound::Harmonic {
                hz,
                stretch,
                glide: glide.rate,
            }
        }
        _ => moved.sound,
    };
    let columns = moved.columns;
    Some((Proposal { sound, columns }, glide.step))
}

fn scored(
    p: &Problem,
    ledger: &Ledger,
    residual: &[Vec<f64>],
    (proposal, step): (&Proposal, f64),
    key: u64,
) -> (f64, Written) {
    let (missed, written) = adding(p, residual, proposal, step, key);
    (ledger.added(&written.price, missed), written)
}

/// `proposal` keyed `key`, its glide at `step`, written against `residual` beside the change in
/// the missed bits adding it makes, in closed form, as `MODEL` states.
pub(super) fn adding(
    p: &Problem,
    residual: &[Vec<f64>],
    proposal: &Proposal,
    step: f64,
    key: u64,
) -> (f64, Written) {
    let templates = p.templates(proposal);
    // Each column's weighted energy and correlation with the residual, frame by frame.
    let moments: Vec<Vec<(f64, f64)>> = templates
        .iter()
        .map(|t| {
            (0..p.frames())
                .map(|f| p.moments(f, t.at(f), &residual[f]))
                .collect()
        })
        .collect();
    let fitted: Vec<Vec<f64>> = moments
        .iter()
        .map(|m| {
            m.iter()
                .map(|(energy, pull)| match *energy > 0.0 {
                    true => (pull / energy).max(0.0),
                    false => 0.0,
                })
                .collect()
        })
        .collect();
    let written = write(p, (proposal, step), key, &fitted, &[]);
    // One column's drop in closed form, sum w (2 r g t - g^2 t^2) = 2 g pull - g^2 energy.
    let drop: f64 = match &moments[..] {
        [m] => m
            .iter()
            .zip(&written.gains[0])
            .map(|((energy, pull), g)| 2.0 * g * pull - g * g * energy)
            .sum(),
        _ => (0..p.frames())
            .map(|f| {
                let mut u = vec![0.0; p.range.len()];
                for (t, g) in templates.iter().zip(&written.gains) {
                    for (u, t) in u.iter_mut().zip(t.at(f)) {
                        *u += g[f] * t;
                    }
                }
                u.iter()
                    .zip(&residual[f])
                    .zip(&p.weight[f])
                    .map(|((u, r), w)| w * (2.0 * r * u - u * u))
                    .sum::<f64>()
            })
            .sum(),
    };
    (-0.5 * LOG2_E * drop, written)
}

/// The fixed `proposal` refined, or its gliding variant where that is lower, alone and attached
/// to each shared variable it may.
pub(super) fn add(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    key: u64,
    proposal: Proposal,
) -> Vec<Scored<Problem>> {
    let residual = &reading.residual;
    let (fixed, change) = refined(p, ledger, residual, (proposal, 0.0), key);
    let (proposal, step, change) = match gliding(p, ledger, residual, &reading.mean, &fixed, key) {
        Some((glide, step)) => {
            let (glide, glided) = refined(p, ledger, residual, (glide, step), key);
            match glided < change && p.apart(reading, &glide, None) {
                true => (glide, step, glided),
                false => (fixed, 0.0, change),
            }
        }
        None => (fixed, 0.0, change),
    };
    if !p.apart(reading, &proposal, None) {
        return Vec::new();
    }
    let attached = reading
        .targets
        .iter()
        .filter_map(|s| shared::attach(p, kept, ledger, reading, key, (&proposal, step), *s));
    let alone = Scored {
        change,
        edit: Edit::Add(Added::alone(Draft::Proposed(proposal.clone(), step))),
    };
    std::iter::once(alone).chain(attached).collect()
}
