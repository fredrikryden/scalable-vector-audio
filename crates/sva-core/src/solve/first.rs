// Concern: one sound's coarse fit: kept peaks, one voice, one noise per band, one NNLS per frame, level knots | Non-concern: onsets, the pyramid | IO: (samples, rate, spl) -> First

use sva_engine::hearing::{CAM_LAST, cam_hz, level_jnd_db};
use sva_engine::{default_threads, each};

use super::add::{Spectrogram, Track, floor_db, sieved, span_hz};
use super::atoms::{Atom, Clip, Sound, Trajectory};
use super::grid::Grid;
use super::pyramid::decimated;

pub const MODEL: &str = "one sound, fitted coarsely, as the smallest thumbnail of an image \
     pyramid holds a picture: its pitch, its colour over a handful of bands, its rise and \
     decay and its tone against its noise, no more than the coarse grid (grid.rs: octave \
     bands, 48 ms frames) resolves; the clip read at its rate over the largest whole factor \
     that keeps the feasible set's top (Cam 38.9) under the Nyquist; the sound spans the \
     clip's first audible frame to its last, and so does every atom, only its level moving; \
     the clip's power averaged over those frames is K looks, K the span over half the window, \
     Hann windows half a window apart being near independent (Welch 1967), so a bin's mean \
     has variance P^2 / K; each local maximum of it over the threshold is a peak, its Hz and \
     level by quadratic interpolation of its dB and its neighbours' (QIFFT; Smith and Serra \
     1987), its power at most its main lobe's; a peak is kept only where its power in its \
     band (its amplitude through the window's kernel, sidelobes included) exceeds the band's \
     noise, the median of the band's averaged bins times their count, by more than its own \
     standard error, its power over root K, and, louder peaks first, where its bin's power \
     exceeds the louder kept peaks' kernels there by more than its standard error, so a \
     sidelobe is no peak; one harmonic voice where the sieve (add.rs) groups \
     the kept peaks, its members at n f0 for every n up to its highest, each member's power \
     its peak's, or the averaged power's largest bin within a main lobe of n f0 where no kept \
     peak is a member, fixed, so the voice is one column whose gain is its level; each kept \
     peak used once, as a member or a partial; at most one partial per band, its loudest kept \
     peak not a member, in a band holding no member, as two sines in one band are one column \
     on the coarse grid; a partial and a noise in one band are one column too, so a band \
     holding a member or a partial holds no noise, the tone having out-powered the band's \
     noise; every other band one noise atom exactly its width, kept where the averaged \
     grid's fit gives it more than its band's averaged power over root K, so a band whose \
     power the tones' leakage explains holds none; each frame's gains by non-negative least \
     squares of the frame's band powers on those columns, each band weighted by \
     1/(P_B + F_B), P_B the band's averaged power and F_B the threshold's band power: a \
     power's spread grows with its mean, so a quiet band counts for more than in an absolute \
     fit, yet a band only a tone's leakage fills does not pull the tone's gain as relative \
     weights would; the columns scaled to unit weighted norm (Jacobi) before the solve, as \
     the weights span decades; a ridge of one \
     rounding unit of the largest diagonal makes the normal matrix positive definite, so the \
     minimiser is unique; Lawson and Hanson's active set (1974, ch. 23) solves it, stopping at \
     the first step whose cost does not fall, the rounding floor; a gain's level is 10 log10 \
     of it, held at the threshold where lower (a voice's at its members' mean threshold); an \
     atom never over its threshold is dropped; each level's frame centres, with its first \
     held back to the onset and its last on to the end, are thinned by \
     Ramer-Douglas-Peucker (1972-73) at the level JND at the atom's peak sensation level \
     (Jesteadt, Wier and Green 1977, hearing.rs), the largest dB gap from a chord; the fit's \
     grid is the written atoms through the same templates, each at its level at the frame's \
     centre";

/// One sound's coarse fit: the clip it read, its atoms, and the clip's and the atoms' coarse
/// grids, each band's power spread flat over its bins.
pub struct First {
    pub clip: Clip,
    pub atoms: Vec<Atom>,
    pub clip_grid: Grid,
    pub fit_grid: Grid,
}

/// The rate `rate` is read at, as `MODEL` states.
pub fn rate_for(rate: u32) -> u32 {
    let need = 2.0 * cam_hz(CAM_LAST);
    let most = (f64::from(rate) / need) as u32;
    (1..=most.max(1))
        .rev()
        .find(|d| rate.is_multiple_of(*d))
        .map_or(rate, |d| rate / d)
}

/// What one column stands for.
enum Role {
    Voice,
    Partial(f64),
    Noise(usize),
}

pub fn fit(samples: &[f64], rate: u32, spl_db: f64) -> First {
    let to = rate_for(rate);
    let (samples, _) = decimated(samples, rate, to);
    let clip = Clip {
        rate: to,
        len: samples.len(),
    };
    let grid = Grid::of(&Spectrogram::of(&samples, to), samples.len());
    let atoms = atoms(&grid, &clip, spl_db);
    let fit_grid = heard(&atoms, &grid);
    First {
        clip,
        atoms,
        clip_grid: grid.broad(),
        fit_grid,
    }
}

/// The frames from the first whose band map is over its threshold anywhere to the last.
fn audible(grid: &Grid, spl_db: f64) -> Option<(usize, usize)> {
    let floors = grid.band_floors(spl_db);
    let heard: Vec<bool> = grid
        .band_db_map()
        .iter()
        .map(|row| row.iter().zip(&floors).any(|(d, f)| d > f))
        .collect();
    Some((
        heard.iter().position(|h| *h)?,
        heard.iter().rposition(|h| *h)?,
    ))
}

fn db(power: f64) -> f64 {
    10.0 * libm::log10(power)
}

/// A sine's level in dBFS whose peak bin holds `power`.
fn sine_db(grid: &Grid, power: f64) -> f64 {
    20.0 * libm::log10(2.0 * libm::sqrt(power) / grid.sum_w)
}

/// The band of `bands` whose bins hold bin `b`.
fn band_of(grid: &Grid, bands: &[(f64, f64)], b: usize) -> Option<usize> {
    bands
        .iter()
        .position(|band| grid.bins_of(*band).contains(&b))
}

/// The kept peaks of the averaged power `mean` over `looks` looks, each beside its band,
/// furthest over their threshold first, as `MODEL` states.
fn peaks(
    grid: &Grid,
    mean: &[f64],
    span: (f64, f64),
    spl_db: f64,
    looks: f64,
) -> Vec<(Track, usize)> {
    let bands = grid.bands();
    let noise: Vec<f64> = bands
        .iter()
        .map(|band| {
            let mut bins = mean[grid.bins_of(*band)].to_vec();
            bins.sort_by(f64::total_cmp);
            bins[bins.len() / 2] * bins.len() as f64
        })
        .collect();
    let (lo, hi) = span_hz(grid.rate);
    let se = 1.0 / libm::sqrt(looks);
    let mut candidates: Vec<(f64, Track, usize, usize)> = Vec::new();
    for b in 1..mean.len() - 1 {
        let at = grid.hz(b);
        if at < lo || at > hi || mean[b] <= mean[b - 1] || mean[b] < mean[b + 1] {
            continue;
        }
        let Some(band) = band_of(grid, &bands, b) else {
            continue;
        };
        let (hz, peak) = qifft(grid, mean, b);
        let level = sine_db(grid, peak);
        let over = level - floor_db(hz, spl_db);
        let power = libm::pow(10.0, level / 10.0) * grid.sine_bands(hz)[band];
        if over <= 0.0 || power - noise[band] <= power * se {
            continue;
        }
        let track = Track {
            hz,
            span,
            levels: (level, level),
        };
        candidates.push((over, track, band, b));
    }
    candidates.sort_by(|x, y| {
        (y.1.levels.0)
            .total_cmp(&x.1.levels.0)
            .then(x.1.hz.total_cmp(&y.1.hz))
    });
    let mut out: Vec<(f64, Track, usize)> = Vec::new();
    for (over, track, band, b) in candidates {
        let leak: f64 = out
            .iter()
            .map(|(_, t, _)| libm::pow(10.0, t.levels.0 / 10.0) * grid.sine(t.hz, b))
            .sum();
        if mean[b] - leak > mean[b] * se {
            out.push((over, track, band));
        }
    }
    out.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.hz.total_cmp(&y.1.hz)));
    out.into_iter().map(|(_, t, b)| (t, b)).collect()
}

/// The bins either side of a peak QIFFT's parabola reads: the three-point stencil.
const STENCIL: usize = 1;

/// The Hz and peak-bin power of the peak at bin `b` of `power`, by QIFFT over `STENCIL`, the
/// power at most its main lobe's, the bins `grid.lobe()` either side of `b`.
fn qifft(grid: &Grid, power: &[f64], b: usize) -> (f64, f64) {
    let lobe = grid.lobe();
    let dbs = |p: f64| db(p.max(f64::MIN_POSITIVE));
    let (a, c, e) = (
        dbs(power[b - STENCIL]),
        dbs(power[b]),
        dbs(power[b + STENCIL]),
    );
    let curve = a - 2.0 * c + e;
    let p = match curve < 0.0 {
        true => 0.5 * (a - e) / curve,
        false => 0.0,
    };
    let hz = grid.hz(b) + p * grid.rate / grid.n as f64;
    let lobe_power: f64 = power[b.saturating_sub(lobe)..(b + lobe + 1).min(power.len())]
        .iter()
        .map(|p| p.max(0.0))
        .sum();
    (
        hz,
        libm::pow(10.0, (c - 0.25 * (a - e) * p) / 10.0).min(lobe_power),
    )
}

/// A voice's f0, each member's level in dBFS, the kept peaks it uses and their bands.
struct Voice {
    f0: f64,
    levels: Vec<f64>,
    used: Vec<usize>,
    bands: Vec<usize>,
}

/// The voice the sieve finds among `tracks`: its f0, each member's level in dBFS and the
/// bands its kept members stand in, as `MODEL` states.
fn voice(grid: &Grid, clip: &Clip, mean: &[f64], tracks: &[(Track, usize)]) -> Option<Voice> {
    let plain: Vec<Track> = tracks
        .iter()
        .map(|(t, _)| Track {
            hz: t.hz,
            span: t.span,
            levels: t.levels,
        })
        .collect();
    let pool: Vec<usize> = (0..plain.len()).collect();
    let (f0, members) = sieved(&plain, &pool, grid.rate / grid.n as f64, span_hz(grid.rate))?;
    let count = members.iter().map(|(_, n)| *n).max()?;
    let probe = Atom::harmonic(
        0,
        0.0,
        Trajectory::steady(f0),
        vec![0.0; count - 1],
        Trajectory {
            base: 0.0,
            knots: vec![(clip.secs(), 0.0)],
        },
    );
    if !probe.writes(clip) {
        return None;
    }
    let lobe = grid.lobe();
    let levels: Vec<f64> = (1..=count)
        .map(|n| match members.iter().find(|(_, m)| *m == n) {
            Some((t, _)) => tracks[*t].0.levels.0,
            None => {
                let centre = libm::round(grid.bin_of(f0 * n as f64)) as usize;
                let reach = centre.saturating_sub(lobe)..(centre + lobe + 1).min(mean.len());
                let most = mean[reach].iter().copied().fold(0.0, f64::max);
                sine_db(grid, most.max(f64::MIN_POSITIVE))
            }
        })
        .collect();
    let used: Vec<usize> = members.iter().map(|(t, _)| *t).collect();
    let bands = used.iter().map(|t| tracks[*t].1).collect();
    Some(Voice {
        f0,
        levels,
        used,
        bands,
    })
}

/// The atoms of the one sound `grid` holds, as `MODEL` states.
fn atoms(grid: &Grid, clip: &Clip, spl_db: f64) -> Vec<Atom> {
    let Some((first, last)) = audible(grid, spl_db) else {
        return Vec::new();
    };
    let on = match first {
        0 => 0.0,
        f => grid.frame_span(f).0,
    };
    let end = match last + 1 == grid.power.len() {
        true => clip.secs(),
        false => grid.frame_span(last).1.min(clip.secs()),
    };
    let frames = &grid.power[first..=last];
    let bins = grid.bins();
    let mean: Vec<f64> = (0..bins)
        .map(|b| frames.iter().map(|f| f[b]).sum::<f64>() / frames.len() as f64)
        .collect();
    let looks = ((end - on) * grid.rate / (grid.long as f64 / 2.0)).max(1.0);
    let tracks = peaks(grid, &mean, (on, end), spl_db, looks);
    let voice = voice(grid, clip, &mean, &tracks);
    let bands = grid.bands();
    let mut tonal = vec![false; bands.len()];
    let mut roles: Vec<Role> = Vec::new();
    let mut columns: Vec<Vec<f64>> = Vec::new();
    if let Some(Voice {
        f0,
        levels,
        bands: held,
        ..
    }) = &voice
    {
        let centre = levels.iter().sum::<f64>() / levels.len() as f64;
        let mut column = vec![0.0; bands.len()];
        for (k, l) in levels.iter().enumerate() {
            let gain = libm::pow(10.0, (l - centre) / 10.0);
            for (c, t) in column.iter_mut().zip(grid.sine_bands(f0 * (k + 1) as f64)) {
                *c += gain * t;
            }
        }
        held.iter().for_each(|b| tonal[*b] = true);
        roles.push(Role::Voice);
        columns.push(column);
    }
    let used = voice.as_ref().map_or(Vec::new(), |v| v.used.clone());
    let voiced = tonal.clone();
    for (band, held) in voiced.iter().enumerate() {
        let loudest = tracks
            .iter()
            .enumerate()
            .filter(|(t, (_, b))| *b == band && !used.contains(t))
            .max_by(|x, y| {
                x.1.0
                    .levels
                    .0
                    .total_cmp(&y.1.0.levels.0)
                    .then(y.0.cmp(&x.0))
            });
        if let (false, Some((_, (track, _)))) = (*held, loudest) {
            tonal[band] = true;
            roles.push(Role::Partial(track.hz));
            columns.push(grid.sine_bands(track.hz));
        }
    }
    for (band, range) in bands.iter().enumerate().filter(|(b, _)| !tonal[*b]) {
        roles.push(Role::Noise(band));
        columns.push(grid.noise_bands(*range));
    }
    let observed: Vec<Vec<f64>> = frames.iter().map(|f| grid.pooled(f)).collect();
    let averaged: Vec<f64> = (0..bands.len())
        .map(|b| observed.iter().map(|o| o[b]).sum::<f64>() / observed.len() as f64)
        .collect();
    let floors: Vec<f64> = grid
        .band_floors(spl_db)
        .iter()
        .map(|f| grid.band_power(*f))
        .collect();
    let weight: Vec<f64> = averaged
        .iter()
        .zip(&floors)
        .map(|(p, f)| 1.0 / (p + f))
        .collect();
    let projected = |columns: &[Vec<f64>], frame: &[f64]| -> Vec<f64> {
        columns
            .iter()
            .map(|c| (0..frame.len()).map(|b| weight[b] * c[b] * frame[b]).sum())
            .collect()
    };
    let gains = scaled(&normal(&columns, &weight), &projected(&columns, &averaged));
    let keep: Vec<bool> = roles
        .iter()
        .zip(&columns)
        .zip(&gains)
        .map(|((role, column), gain)| match role {
            Role::Noise(b) => gain * column[*b] > averaged[*b] / libm::sqrt(looks),
            _ => true,
        })
        .collect();
    let (roles, columns): (Vec<Role>, Vec<Vec<f64>>) = roles
        .into_iter()
        .zip(columns)
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(rc, _)| rc)
        .unzip();
    let normal = normal(&columns, &weight);
    let gains: Vec<Vec<f64>> = each(default_threads(), observed, |frame| {
        scaled(&normal, &projected(&columns, &frame))
    });
    let times: Vec<f64> = (first..=last).map(|f| grid.centre_secs(f)).collect();
    let column = |j: usize| -> Vec<f64> { gains.iter().map(|g| g[j]).collect() };
    let mut out = Vec::new();
    for (j, role) in roles.iter().enumerate() {
        let (floor, make): (f64, Box<dyn Fn(Trajectory) -> Atom>) = match role {
            Role::Voice => {
                let Voice { f0, levels, .. } = voice.as_ref().expect("a voice");
                let centre = levels.iter().sum::<f64>() / levels.len() as f64;
                let offsets: Vec<f64> = levels[..levels.len() - 1]
                    .iter()
                    .map(|l| l - centre)
                    .collect();
                let floor = (1..=levels.len())
                    .map(|n| floor_db(f0 * n as f64, spl_db))
                    .sum::<f64>()
                    / levels.len() as f64;
                let f0 = *f0;
                (
                    floor,
                    Box::new(move |level| {
                        Atom::harmonic(0, on, Trajectory::steady(f0), offsets.clone(), level)
                    }),
                )
            }
            Role::Partial(hz) => (
                floor_db(*hz, spl_db),
                Box::new(move |level| Atom::partial(0, on, Trajectory::steady(*hz), level)),
            ),
            Role::Noise(b) => {
                let band = bands[*b];
                (
                    floor_db(libm::sqrt(band.0 * band.1), spl_db),
                    Box::new(move |level| Atom::noise(0, on, band, level)),
                )
            }
        };
        let levels: Vec<f64> = column(j).iter().map(|a| db(*a)).collect();
        let peak = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if peak <= floor {
            continue;
        }
        let held: Vec<f64> = levels.iter().map(|l| l.max(floor)).collect();
        let level = thinned((on, end), &times, &held, level_jnd_db(peak - floor));
        out.push(Atom {
            key: out.len() as u64,
            ..make(level)
        });
    }
    out
}

/// A level from `on` to `end` through `levels` at `times`, the first held back to `on` and
/// the last on to `end`, thinned by Ramer-Douglas-Peucker at `tolerance` dB.
fn thinned((on, end): (f64, f64), times: &[f64], levels: &[f64], tolerance: f64) -> Trajectory {
    let mut points = vec![(on, levels[0])];
    points.extend(
        times
            .iter()
            .zip(levels)
            .filter(|(t, _)| on < **t && **t < end)
            .map(|(t, l)| (*t, *l)),
    );
    points.push((end, levels[levels.len() - 1]));
    let mut keep = vec![false; points.len()];
    let last = points.len() - 1;
    (keep[0], keep[last]) = (true, true);
    let mut open = vec![(0, last)];
    while let Some((i, j)) = open.pop() {
        let ((ti, li), (tj, lj)) = (points[i], points[j]);
        let gap = |k: usize| {
            let (t, l) = points[k];
            (l - (li + (lj - li) * (t - ti) / (tj - ti))).abs()
        };
        let Some(k) = (i + 1..j).max_by(|x, y| gap(*x).total_cmp(&gap(*y)).then(y.cmp(x))) else {
            continue;
        };
        if gap(k) > tolerance {
            keep[k] = true;
            open.push((i, k));
            open.push((k, j));
        }
    }
    let kept: Vec<(f64, f64)> = points
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect();
    let base = kept[0].1;
    Trajectory {
        base,
        knots: kept[1..].iter().map(|(t, l)| (*t, l - base)).collect(),
    }
}

/// `T^T W T` over `columns`.
fn normal(columns: &[Vec<f64>], weight: &[f64]) -> Vec<Vec<f64>> {
    let m = columns.len();
    let mut g = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i..m {
            let sum: f64 = (0..weight.len())
                .map(|k| weight[k] * columns[i][k] * columns[j][k])
                .sum();
            (g[i][j], g[j][i]) = (sum, sum);
        }
    }
    g
}

/// `nnls` of `g` and `h` with each column scaled to unit weighted norm (Jacobi), so bands
/// whose weights differ by many decades do not leave the normal matrix singular in floating
/// point, then ridged and scaled back.
fn scaled(g: &[Vec<f64>], h: &[f64]) -> Vec<f64> {
    let d: Vec<f64> = (0..h.len())
        .map(|i| match g[i][i] > 0.0 {
            true => 1.0 / libm::sqrt(g[i][i]),
            false => 0.0,
        })
        .collect();
    let unit: Vec<Vec<f64>> = (0..h.len())
        .map(|i| (0..h.len()).map(|j| d[i] * g[i][j] * d[j]).collect())
        .collect();
    let h: Vec<f64> = h.iter().zip(&d).map(|(v, s)| v * s).collect();
    nnls(&ridged(unit), &h)
        .iter()
        .zip(&d)
        .map(|(x, s)| x * s)
        .collect()
}

/// `g` with one rounding unit of its largest diagonal added to its diagonal: no number of the
/// data's, only the arithmetic's, so every eigenvalue is positive and the minimiser unique,
/// while each product `g x` already carries a rounding error that large, so the answer moves
/// no further than rounding moves it.
fn ridged(mut g: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let ridge = f64::EPSILON * (0..g.len()).map(|i| g[i][i]).fold(0.0, f64::max);
    for (i, row) in g.iter_mut().enumerate() {
        row[i] += ridge;
    }
    g
}

/// `argmin 1/2 x^T G x - h^T x` over `x >= 0`, by Lawson and Hanson's active set, as `MODEL`
/// states.
fn nnls(g: &[Vec<f64>], h: &[f64]) -> Vec<f64> {
    let m = h.len();
    let cost = |x: &[f64]| {
        (0..m)
            .map(|i| x[i] * (0.5 * (0..m).map(|j| g[i][j] * x[j]).sum::<f64>() - h[i]))
            .sum::<f64>()
    };
    let mut x = vec![0.0; m];
    let mut passive = vec![false; m];
    let mut least = 0.0;
    loop {
        let w: Vec<f64> = (0..m)
            .map(|i| h[i] - (0..m).map(|j| g[i][j] * x[j]).sum::<f64>())
            .collect();
        let Some(enter) = (0..m)
            .filter(|i| !passive[*i] && w[*i] > 0.0)
            .max_by(|a, b| w[*a].total_cmp(&w[*b]).then(b.cmp(a)))
        else {
            break;
        };
        let mut next = x.clone();
        let mut set = passive.clone();
        set[enter] = true;
        loop {
            let p: Vec<usize> = (0..m).filter(|i| set[*i]).collect();
            let Some(z) = solved(g, h, &p) else {
                set = passive.clone();
                next = x.clone();
                break;
            };
            if z.iter().all(|v| *v > 0.0) {
                for (i, v) in p.iter().zip(&z) {
                    next[*i] = *v;
                }
                break;
            }
            let (step, out) = p
                .iter()
                .zip(&z)
                .filter(|(_, v)| **v <= 0.0)
                .map(|(i, v)| (next[*i] / (next[*i] - v), *i))
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                .expect("a non-positive entry");
            for (i, v) in p.iter().zip(&z) {
                next[*i] += step * (v - next[*i]);
                if *i == out || next[*i] <= 0.0 {
                    next[*i] = 0.0;
                    set[*i] = false;
                }
            }
        }
        let c = cost(&next);
        if c >= least {
            break;
        }
        (x, passive, least) = (next, set, c);
    }
    x
}

/// `G_PP z = h_P` by Cholesky over the indices `p`, `None` where `G_PP` is not positive
/// definite in floating point.
fn solved(g: &[Vec<f64>], h: &[f64], p: &[usize]) -> Option<Vec<f64>> {
    let k = p.len();
    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let s = g[p[i]][p[j]] - (0..j).map(|q| l[i][q] * l[j][q]).sum::<f64>();
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i][i] = libm::sqrt(s);
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = vec![0.0; k];
    for i in 0..k {
        y[i] = (h[p[i]] - (0..i).map(|q| l[i][q] * y[q]).sum::<f64>()) / l[i][i];
    }
    for i in (0..k).rev() {
        y[i] = (y[i] - (i + 1..k).map(|q| l[q][i] * y[q]).sum::<f64>()) / l[i][i];
    }
    Some(y)
}

/// The level in dB `level` holds at `t`, from `on`.
pub fn level_at(level: &Trajectory, on: f64, t: f64) -> f64 {
    let mut from = (on, level.base);
    for (at, d) in &level.knots {
        let to = (*at, level.base + d);
        if t <= to.0 {
            return match to.0 > from.0 {
                true => from.1 + (to.1 - from.1) * (t - from.0) / (to.0 - from.0),
                false => to.1,
            };
        }
        from = to;
    }
    from.1
}

/// The coarse grid `atoms` put in each of `like`'s frames through its templates, each atom's
/// level at the frame's centre, each band's power spread flat over its bins.
pub fn heard(atoms: &[Atom], like: &Grid) -> Grid {
    let bands = like.bands().len();
    let templates: Vec<Vec<(f64, Vec<f64>)>> = atoms
        .iter()
        .map(|atom| match &atom.sound {
            Sound::Partial { hz } => vec![(0.0, like.sine_bands(hz.base))],
            Sound::Harmonic { hz, offsets } => {
                let last = -offsets.iter().sum::<f64>();
                offsets
                    .iter()
                    .chain([&last])
                    .enumerate()
                    .map(|(k, o)| (*o, like.sine_bands(hz.base * (k + 1) as f64)))
                    .collect()
            }
            Sound::Noise { lo, hi } => vec![(0.0, like.noise_bands((*lo, *hi)))],
        })
        .collect();
    let power = (0..like.power.len())
        .map(|f| {
            let t = like.centre_secs(f);
            let mut frame = vec![0.0; bands];
            for (atom, parts) in atoms.iter().zip(&templates) {
                if t < atom.on || t > atom.end() {
                    continue;
                }
                let level = level_at(&atom.level, atom.on, t);
                for (offset, template) in parts {
                    let gain = libm::pow(10.0, (level + offset) / 10.0);
                    for (p, v) in frame.iter_mut().zip(template) {
                        *p += gain * v;
                    }
                }
            }
            like.spread(&frame)
        })
        .collect();
    like.with(power)
}

#[cfg(test)]
mod tests {
    use super::{nnls, ridged};

    /// Three overlapping columns, one gain held at nought by the bound: the ridge leaves the
    /// answer where it was to rounding.
    #[test]
    fn the_ridge_does_not_change_the_answer() {
        let g = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 1.0],
            vec![0.5, 1.0, 2.0],
        ];
        let h = vec![3.0, 2.0, -1.0];
        let (plain, ridge) = (nnls(&g, &h), nnls(&ridged(g.clone()), &h));
        assert_eq!(plain[2], 0.0);
        for (p, r) in plain.iter().zip(&ridge) {
            assert!(
                (p - r).abs() <= 8.0 * f64::EPSILON * p.abs().max(1.0),
                "{p} {r}"
            );
        }
    }
}
