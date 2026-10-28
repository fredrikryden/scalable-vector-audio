// Concern: the adds a residual proposes: a partial per peak, each voice a sieve finds, a noise per band, a shaped noise | Non-concern: refining or scoring them (adds.rs) | IO: (residual) -> Sounds

use super::peaks::qifft;
use super::problem::Problem;
use super::render::{Held, db};
use crate::analysis::{Members, Track, floor_db, sieved, span_hz};
use crate::sources::{Sound, Source, Trajectory};

/// The bins `power` peaks at over their floor, each a track spanning the clip beside its bin,
/// as `MODEL` states.
fn peaks(p: &Problem, power: &[f64]) -> Vec<(Track, usize)> {
    let span = (0.0, p.clip.secs());
    let mut out = Vec::new();
    for i in 1..power.len().saturating_sub(1) {
        if power[i] <= power[i - 1] || power[i] < power[i + 1] {
            continue;
        }
        let (hz, level) = qifft(p, power, i);
        if level > floor_db(hz, p.spl_db) {
            out.push((Track { hz, span }, i));
        }
    }
    out
}

/// `(f0, B)` by least squares of `(f_k / k)^2` on `k^2` over `members`, `B` at least 0.
fn stiffness(tracks: &[Track], members: &Members) -> (f64, f64) {
    let points: Vec<(usize, f64, f64)> = members
        .iter()
        .map(|(t, k)| (*k, tracks[*t].hz, 1.0))
        .collect();
    stretched(&points)
}

/// `(f0, B)` by least squares of `(f_k / k)^2` on `k^2` over `(k, f_k, weight)`, `B` at least 0.
pub(super) fn stretched(points: &[(usize, f64, f64)]) -> (f64, f64) {
    let points: Vec<(f64, f64, f64)> = points
        .iter()
        .map(|(k, hz, w)| {
            let k = *k as f64;
            (k * k, (hz / k) * (hz / k), *w)
        })
        .collect();
    let total: f64 = points.iter().map(|p| p.2).sum();
    let mean = |pick: fn(&(f64, f64, f64)) -> f64| {
        points.iter().map(|p| p.2 * pick(p)).sum::<f64>() / total
    };
    let (mx, my) = (mean(|p| p.0), mean(|p| p.1));
    let spread: f64 = points.iter().map(|p| p.2 * (p.0 - mx) * (p.0 - mx)).sum();
    let slope = match spread > 0.0 {
        true => {
            points
                .iter()
                .map(|p| p.2 * (p.0 - mx) * (p.1 - my))
                .sum::<f64>()
                / spread
        }
        false => 0.0,
    };
    let (a, c) = match slope > 0.0 {
        true => (my - slope * mx, slope),
        false => (my, 0.0),
    };
    match a > 0.0 {
        true => (libm::sqrt(a), c / a),
        false => (libm::sqrt(my), 0.0),
    }
}

/// The members at the stretched Hz of `(f0, B)` among `pool`: the nearest track within half an
/// analysis bin of each harmonic's, as `MODEL` states.
fn assigned(tracks: &[Track], pool: &[usize], (f0, b): (f64, f64), p: &Problem) -> Members {
    let (bin, top) = (p.grid.rate / p.grid.n as f64, span_hz(p.grid.rate).1);
    let mut out = Vec::new();
    for k in 1.. {
        let kf = k as f64;
        let hz = f0 * kf * libm::sqrt(1.0 + b * kf * kf);
        if hz > top {
            break;
        }
        let near = pool
            .iter()
            .map(|t| (*t, (tracks[*t].hz - hz).abs()))
            .filter(|(_, miss)| 2.0 * miss <= bin)
            .min_by(|x, y| x.1.total_cmp(&y.1).then(x.0.cmp(&y.0)));
        if let Some((t, _)) = near {
            out.push((t, k));
        }
    }
    out
}

/// Every voice the sieve finds among `tracks`, its stretch fitted, as `MODEL` states.
fn voices(p: &Problem, tracks: &[Track]) -> Vec<Sound> {
    let bin = p.grid.rate / p.grid.n as f64;
    let mut pool: Vec<usize> = (0..tracks.len()).collect();
    let mut out = Vec::new();
    while let Some((_, found)) = sieved(tracks, &pool, bin, span_hz(p.grid.rate), p.threads) {
        let mut members = found;
        let mut seen: Vec<Members> = Vec::new();
        let mut fitted = stiffness(tracks, &members);
        loop {
            let next = assigned(tracks, &pool, fitted, p);
            if next.is_empty() || next == members || seen.contains(&next) {
                break;
            }
            seen.push(std::mem::replace(&mut members, next));
            fitted = stiffness(tracks, &members);
        }
        pool.retain(|t| !members.iter().any(|(m, _)| m == t));
        let count = members.iter().map(|(_, k)| *k).max().unwrap_or(0);
        let voice = |count: usize| Sound::Harmonic {
            hz: fitted.0,
            stretch: fitted.1,
            count,
            glide: 0.0,
        };
        let level = Trajectory {
            base: 0.0,
            knots: vec![(p.clip.secs(), 0.0)],
        };
        let probe = Source::new(0, 0.0, voice(count.max(1)), vec![level; count.max(1)]);
        if count >= 2 && probe.writes(&p.clip) {
            out.push(voice(count));
        }
    }
    out
}

pub(super) fn averaged(p: &Problem, residual: &[Vec<f64>]) -> Vec<f64> {
    (0..p.range.len())
        .map(|b| residual.iter().map(|r| r[b]).sum::<f64>() / residual.len() as f64)
        .collect()
}

pub(super) fn candidates(p: &Problem, kept: &[Held], residual: &[Vec<f64>]) -> Vec<Sound> {
    let mean = averaged(p, residual);
    let found = peaks(p, &mean);
    let mut out: Vec<Sound> = found
        .iter()
        .map(|(t, _)| Sound::Partial {
            hz: t.hz,
            glide: 0.0,
        })
        .collect();
    let (_, tonal) = p.lobes.floor(&mean, p.looks * p.frames() as f64);
    let tracks: Vec<Track> = found
        .into_iter()
        .filter(|(_, i)| tonal.binary_search(i).is_ok())
        .map(|(t, _)| t)
        .collect();
    out.extend(voices(p, &tracks));
    for (lo, hi) in p.grid.bands() {
        let held = kept.iter().any(|h| *h.sound() == Sound::Noise { lo, hi });
        if !held {
            out.push(Sound::Noise { lo, hi });
        }
    }
    out.extend(shaped(p, residual));
    out
}

/// The shaped noise `residual` proposes, as `MODEL` states, `None` where every gain is 0.
fn shaped(p: &Problem, residual: &[Vec<f64>]) -> Option<Sound> {
    let bands = p.grid.bands();
    // Each band's weighted energy and correlation with the residual, frame by frame.
    let moments: Vec<Vec<(f64, f64)>> = bands
        .iter()
        .map(|band| {
            let t = &p.grid.noise_bins(*band)[p.range.clone()];
            (0..p.frames())
                .map(|f| p.moments(f, t, &residual[f]))
                .collect()
        })
        .collect();
    let ratio = |(e, c): (f64, f64)| match e > 0.0 {
        true => (c / e).max(0.0),
        false => 0.0,
    };
    // `sum w (u^2 - 2 r u)` over the bands, no cross terms, `u` the gains times the envelope.
    let misfit = |gains: &[f64], envelope: &[f64]| -> f64 {
        moments
            .iter()
            .zip(gains)
            .map(|(m, g)| {
                m.iter()
                    .zip(envelope)
                    .map(|((e, c), v)| g * g * v * v * e - 2.0 * g * v * c)
                    .sum::<f64>()
            })
            .sum()
    };
    let mut envelope = vec![1.0; p.frames()];
    let mut gains = vec![0.0; bands.len()];
    let mut held = f64::INFINITY;
    loop {
        gains = moments
            .iter()
            .map(|m| {
                ratio(
                    m.iter()
                        .zip(&envelope)
                        .fold((0.0, 0.0), |(e, c), ((me, mc), v)| {
                            (e + v * v * me, c + v * mc)
                        }),
                )
            })
            .collect();
        envelope = (0..p.frames())
            .map(|f| {
                ratio(
                    moments
                        .iter()
                        .zip(&gains)
                        .fold((0.0, 0.0), |(e, c), (m, g)| {
                            (e + g * g * m[f].0, c + g * m[f].1)
                        }),
                )
            })
            .collect();
        let now = misfit(&gains, &envelope);
        if now >= held {
            break;
        }
        held = now;
    }
    let top = gains.iter().copied().fold(0.0, f64::max);
    (top > 0.0).then(|| Sound::Shaped {
        bands: bands
            .into_iter()
            .zip(gains)
            .filter(|(_, g)| *g > 0.0)
            .map(|((lo, hi), g)| (lo, hi, db(g / top)))
            .collect(),
    })
}
