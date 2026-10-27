// Concern: one sound's coarse fit: kept peaks, one voice, one noise per band, levels read off band envelopes, level knots | Non-concern: onsets, the pyramid | IO: (samples, rate, spl) -> First

use sva_engine::hearing::{CAM_LAST, cam_hz, level_jnd_db};

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
     1987), its power at most its main lobe's; a peak is kept only where its power inside \
     its main lobe (its amplitude through the window's kernel over the lobe's bins) exceeds \
     the noise inside that lobe, the median of its band's averaged bins times the lobe's \
     width in bins, by more than its own standard error, its power over root K, and, louder \
     peaks first, where its bin's power exceeds the louder kept peaks' kernels there by more \
     than its standard error, so a sidelobe is no peak; a band may hold several partials, the \
     lobe test already refusing a band's noise and a sine's sidelobes; one harmonic voice \
     where the sieve (add.rs) groups the kept peaks, its members at n f0 for every n up to \
     its highest, each member's shape its peak's level, or the averaged power's largest bin \
     within a main lobe of n f0 where no kept peak is a member; each kept peak used once, as \
     a member or a partial; a band holding a member or a partial holds no noise, the tones \
     having out-powered its noise; no level is fitted, each is read off the clip's band \
     envelopes, each frame's power in a band over the atom's unit-level power there: a \
     band's tones, its partials and the voice's harmonics, share its envelope in proportion \
     to their peaks' powers there, so each frame shifts them all by the dB of the envelope \
     over their summed power, a partial at its peak's level plus that shift; the voice's \
     level the mean of that shift over the bands holding its kept members, each such band's \
     harmonics offset by its time-mean difference from that mean; leakage between bands is \
     ignored at the coarse level, every band read as its own atoms' alone; every other band, louder first, one noise atom \
     exactly its width, its level its envelope over its unit power there, kept where its \
     averaged power exceeds the time-mean leakage of the atoms read before it by more than \
     its averaged power over root K, so a band whose power the tones' leakage explains holds \
     none; a level is held at the threshold where lower (a voice's at its members' mean \
     threshold); an atom never over its threshold is dropped; each level's frame centres, \
     with its first held back to the onset and its last on to the end, are thinned by \
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
    let lobe = grid.lobe();
    let noise: Vec<f64> = bands
        .iter()
        .map(|band| {
            let mut bins = mean[grid.bins_of(*band)].to_vec();
            bins.sort_by(f64::total_cmp);
            bins[bins.len() / 2] * (2 * lobe + 1) as f64
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
        let power = libm::pow(10.0, level / 10.0)
            * (b.saturating_sub(lobe)..(b + lobe + 1).min(mean.len()))
                .map(|k| grid.sine(hz, k))
                .sum::<f64>();
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

/// One atom read off the clip's band envelopes: its threshold, its level in dB at each frame,
/// its unit-level power in each band and how it is written.
struct Read {
    floor: f64,
    levels: Vec<f64>,
    template: Vec<f64>,
    make: Box<dyn Fn(Trajectory) -> Atom>,
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
    let observed: Vec<Vec<f64>> = frames.iter().map(|f| grid.pooled(f)).collect();
    let envelope = |band: usize, gain: f64| -> Vec<f64> {
        observed
            .iter()
            .map(|o| db(o[band].max(f64::MIN_POSITIVE) / gain))
            .collect()
    };
    let used = voice.as_ref().map_or(Vec::new(), |v| v.used.clone());
    let partials: Vec<&(Track, usize)> = tracks
        .iter()
        .enumerate()
        .filter(|(t, _)| !used.contains(t))
        .map(|(_, t)| t)
        .collect();
    let mut members: Vec<usize> = voice.as_ref().map_or(Vec::new(), |v| v.bands.clone());
    members.sort_unstable();
    members.dedup();
    let mut tonal = vec![false; bands.len()];
    members
        .iter()
        .chain(partials.iter().map(|(_, b)| b))
        .for_each(|b| tonal[*b] = true);
    let shaped: Vec<f64> = (0..bands.len())
        .map(|b| {
            voice.as_ref().map_or(0.0, |v| {
                v.levels
                    .iter()
                    .enumerate()
                    .map(|(k, s)| {
                        libm::pow(10.0, s / 10.0) * grid.sine_bands(v.f0 * (k + 1) as f64)[b]
                    })
                    .sum()
            })
        })
        .collect();
    let shift: Vec<Option<Vec<f64>>> = (0..bands.len())
        .map(|b| {
            let held = shaped[b]
                + partials
                    .iter()
                    .filter(|(_, at)| *at == b)
                    .map(|(t, _)| libm::pow(10.0, t.levels.0 / 10.0) * grid.sine_bands(t.hz)[b])
                    .sum::<f64>();
            tonal[b].then(|| envelope(b, held))
        })
        .collect();
    let mut reads: Vec<Read> = Vec::new();
    if let Some(Voice {
        f0, levels: shape, ..
    }) = &voice
    {
        let f0 = *f0;
        let harmonic = |k: usize| f0 * (k + 1) as f64;
        let shifts: Vec<&Vec<f64>> = members.iter().filter_map(|b| shift[*b].as_ref()).collect();
        let level: Vec<f64> = (0..observed.len())
            .map(|f| shifts.iter().map(|s| s[f]).sum::<f64>() / shifts.len() as f64)
            .collect();
        let offset: Vec<f64> = shifts
            .iter()
            .map(|s| s.iter().zip(&level).map(|(a, v)| a - v).sum::<f64>() / level.len() as f64)
            .collect();
        let own: Vec<f64> = shape
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let bin = libm::round(grid.bin_of(harmonic(k))) as usize;
                let band = band_of(grid, &bands, bin);
                s + members
                    .iter()
                    .position(|m| Some(*m) == band)
                    .map_or(0.0, |i| offset[i])
            })
            .collect();
        let centre = own.iter().sum::<f64>() / own.len() as f64;
        let offsets: Vec<f64> = own.iter().map(|o| o - centre).collect();
        let mut template = vec![0.0; bands.len()];
        for (k, o) in offsets.iter().enumerate() {
            for (t, v) in template.iter_mut().zip(grid.sine_bands(harmonic(k))) {
                *t += libm::pow(10.0, o / 10.0) * v;
            }
        }
        let floor = (0..shape.len())
            .map(|k| floor_db(harmonic(k), spl_db))
            .sum::<f64>()
            / shape.len() as f64;
        let written = offsets[..offsets.len() - 1].to_vec();
        reads.push(Read {
            floor,
            levels: level.iter().map(|l| l + centre).collect(),
            template,
            make: Box::new(move |level| {
                Atom::harmonic(0, on, Trajectory::steady(f0), written.clone(), level)
            }),
        });
    }
    for (track, band) in partials {
        let hz = track.hz;
        let shift = shift[*band].as_ref().expect("a partial's band is tonal");
        reads.push(Read {
            floor: floor_db(hz, spl_db),
            levels: shift.iter().map(|r| track.levels.0 + r).collect(),
            template: grid.sine_bands(hz),
            make: Box::new(move |level| Atom::partial(0, on, Trajectory::steady(hz), level)),
        });
    }
    let averaged: Vec<f64> = (0..bands.len())
        .map(|b| observed.iter().map(|o| o[b]).sum::<f64>() / observed.len() as f64)
        .collect();
    let mut open: Vec<usize> = (0..bands.len()).filter(|b| !tonal[*b]).collect();
    open.sort_by(|x, y| averaged[*y].total_cmp(&averaged[*x]).then(x.cmp(y)));
    for band in open {
        let leak: f64 = reads
            .iter()
            .map(|r| {
                let power = r
                    .levels
                    .iter()
                    .map(|l| libm::pow(10.0, l / 10.0))
                    .sum::<f64>()
                    / r.levels.len() as f64;
                power * r.template[band]
            })
            .sum();
        if averaged[band] - leak <= averaged[band] / libm::sqrt(looks) {
            continue;
        }
        let range = bands[band];
        let template = grid.noise_bands(range);
        reads.push(Read {
            floor: floor_db(libm::sqrt(range.0 * range.1), spl_db),
            levels: envelope(band, template[band]),
            template,
            make: Box::new(move |level| Atom::noise(0, on, range, level)),
        });
    }
    let times: Vec<f64> = (first..=last).map(|f| grid.centre_secs(f)).collect();
    let mut out = Vec::new();
    for Read {
        floor,
        levels,
        make,
        ..
    } in reads
    {
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
