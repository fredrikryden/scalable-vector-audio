// Concern: one sound's first fit: its peaks, one harmonic voice, flat noise bands, one weighted NNLS per frame, level knots | Non-concern: onsets, the pyramid | IO: (samples, rate, spl) -> First

use sva_engine::hearing::{CAM_LAST, cam, cam_hz, level_jnd_db};
use sva_engine::{default_threads, each};

use super::add::{Spectrogram, Track, floor_db, sieved, span_hz};
use super::atoms::{Atom, Clip, Sound, Trajectory};
use super::grid::Grid;
use super::pyramid::decimated;

pub const MODEL: &str = "one sound, fitted once: the clip read at its rate over the largest \
     whole factor that keeps the feasible set's top (Cam 38.9) under the Nyquist, its grid \
     (grid.rs); the sound spans the clip's first audible frame to its last, and so does every \
     atom, only its level moving; partials from the peaks of the clip's power averaged over \
     those frames: each local maximum over the threshold as a sine's level, its Hz and level by \
     quadratic interpolation of its dB and its neighbours' (QIFFT; Smith and Serra 1987), its \
     power at most its main lobe's, kept where a sine's lobe there explains more of its peak \
     bin than a flat floor does (a least-squares fit of the lobe and a constant, both \
     non-negative, over the bins of the noise band below that holds it, the floor that band \
     would fit), which a ripple of the averaged noise does not; one harmonic voice where the sieve \
     (add.rs) groups those peaks, its members at n f0 for every n up to its highest, each peak \
     used once, as a member or a partial, each partial's Hz then read again by QIFFT off the \
     averaged power less every other column at its gain in the same fit of the averaged \
     power, at the local maximum within a bin of where it was read, so a neighbour's lobe \
     does not pull it; noise bands flat in Hz, whole Cams merged upward \
     until each holds at least the bins a sine's main lobe covers, so no band is narrower than \
     a partial's template, the last short remainder joining the band below it; each frame's \
     gains by non-negative least squares of the frame's bin powers on those templates \
     (grid.rs), each bin weighted by 1/(P_b + F_b), P_b the clip's averaged power and F_b \
     the threshold's peak-bin power: a periodogram's spread grows with its mean, so a quiet \
     bin counts for more than in an absolute fit, yet a lobe's skirts, where its template is \
     near nought and a frame's leakage (an onset, the clip's edge) is not, do not outweigh \
     its peak by their power ratio squared as relative weights would; read off the averaged \
     power rather than the frame's, as one frame's noise bin is exponentially distributed and \
     weighting by its own draw biases a band low; a ridge of one rounding unit of the largest \
     diagonal makes the normal matrix positive definite, so the minimiser is unique, and moves \
     it by no more than the rounding the matrix products already carry; Lawson and Hanson's \
     active set (1974, ch. 23) solves it, stopping at the first step whose cost does not fall, \
     the rounding floor; a gain's level is 10 log10 of it, held at the threshold where lower; \
     an atom never over its threshold is dropped; a harmonic's level is its members' mean dB \
     per frame and each offset its member's time mean less it; each level's frame centres, \
     with its first held back to the onset and its last on to the end, are thinned by \
     Ramer-Douglas-Peucker (1972-73) at the level JND at the atom's peak sensation level \
     (Jesteadt, Wier and Green 1977, hearing.rs), the largest dB gap from a chord; the fit's \
     grid is the written atoms through the same templates, each at its level at the frame's \
     centre";

/// One sound's first fit: the clip it read, its atoms, and the clip's and the atoms' grids.
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

/// A template over bins `start..start + values.len()`.
struct Column {
    start: usize,
    values: Vec<f64>,
}

impl Column {
    fn sine(grid: &Grid, hz: f64) -> Column {
        let lobe = grid.lobe_of(hz);
        Column {
            start: lobe.first().map_or(0, |(b, _)| *b),
            values: lobe.iter().map(|(_, p)| *p).collect(),
        }
    }

    fn band(grid: &Grid, band: (f64, f64)) -> Column {
        let bins = grid.bins_of(band);
        Column {
            start: bins.start,
            values: vec![grid.band(bins.len()); bins.len()],
        }
    }

    fn bins(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.values.len()
    }
}

/// What one column stands for.
enum Role {
    Member,
    Partial(f64),
    Noise((f64, f64)),
}

pub fn fit(samples: &[f64], rate: u32, spl_db: f64) -> First {
    let to = rate_for(rate);
    let (samples, _) = decimated(samples, rate, to);
    let clip = Clip {
        rate: to,
        len: samples.len(),
    };
    let clip_grid = Grid::of(&Spectrogram::of(&samples, to));
    let atoms = atoms(&clip_grid, &clip, spl_db);
    let fit_grid = heard(&atoms, &clip_grid);
    First {
        clip,
        atoms,
        clip_grid,
        fit_grid,
    }
}

/// The frames from the first whose Cam map is over its threshold anywhere to the last.
fn audible(grid: &Grid, spl_db: f64) -> Option<(usize, usize)> {
    let floors = grid.cam_floors(spl_db);
    let heard: Vec<bool> = grid
        .cam_db()
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

/// The partials the averaged power `mean` proposes, furthest over their threshold first, as
/// `MODEL` states.
fn peaks(
    grid: &Grid,
    mean: &[f64],
    span: (f64, f64),
    spl_db: f64,
    bands: &[(f64, f64)],
) -> Vec<Track> {
    let (lo, hi) = span_hz(grid.rate);
    let bins = mean.len();
    let mut out: Vec<(f64, Track)> = Vec::new();
    for b in 1..bins - 1 {
        let at = grid.hz(b);
        if at < lo || at > hi || mean[b] <= mean[b - 1] || mean[b] < mean[b + 1] {
            continue;
        }
        let (hz, peak) = qifft(grid, mean, b);
        let over = sine_db(grid, peak) - floor_db(hz, spl_db);
        let Some(band) = bands.iter().find(|band| grid.bins_of(**band).contains(&b)) else {
            continue;
        };
        if over <= 0.0 || !lobe_over_floor(grid, mean, hz, b, grid.bins_of(*band)) {
            continue;
        }
        let level = sine_db(grid, peak);
        out.push((
            over,
            Track {
                hz,
                span,
                levels: (level, level),
            },
        ));
    }
    out.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.hz.total_cmp(&y.1.hz)));
    out.into_iter().map(|(_, t)| t).collect()
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

/// Each partial's Hz read again by QIFFT off `mean` less every other column at `gains`, so a
/// neighbour's lobe does not pull it: at the local maximum within `STENCIL` of the bin it was
/// read at, as QIFFT's offset lies within half a bin of its peak bin, so taking a neighbour
/// off moves that bin by one at most; the stencil read about each candidate reaches twice
/// that.
fn reread(grid: &Grid, mean: &[f64], columns: &mut [Column], roles: &mut [Role], gains: &[f64]) {
    let bins = mean.len();
    let fitted: Vec<(usize, f64)> = roles
        .iter()
        .enumerate()
        .filter_map(|(j, r)| match r {
            Role::Partial(hz) => Some((j, *hz)),
            _ => None,
        })
        .collect();
    for (j, hz) in fitted {
        let centre = libm::round(grid.bin_of(hz)) as usize;
        let reach = centre.saturating_sub(2 * STENCIL)..(centre + 2 * STENCIL + 1).min(bins);
        let mut rest = mean.to_vec();
        for (i, c) in columns.iter().enumerate().filter(|(i, _)| *i != j) {
            for (b, v) in c.bins().zip(&c.values).filter(|(b, _)| reach.contains(b)) {
                rest[b] -= gains[i] * v;
            }
        }
        let peak = (centre.saturating_sub(STENCIL).max(STENCIL)
            ..=(centre + STENCIL).min(bins - 1 - STENCIL))
            .filter(|b| rest[*b] > rest[b - STENCIL] && rest[*b] >= rest[b + STENCIL])
            .max_by(|x, y| rest[*x].total_cmp(&rest[*y]).then(y.cmp(x)));
        if let Some(b) = peak {
            let (hz, _) = qifft(grid, &rest, b);
            roles[j] = Role::Partial(hz);
            columns[j] = Column::sine(grid, hz);
        }
    }
}

/// Whether a sine's lobe at `hz` explains more of peak bin `b` of `mean` than the flat floor
/// of the noise band over `bins` that holds it, as `MODEL` states.
fn lobe_over_floor(
    grid: &Grid,
    mean: &[f64],
    hz: f64,
    b: usize,
    bins: std::ops::Range<usize>,
) -> bool {
    let lobe = grid.lobe_of(hz);
    let template = |k: usize| lobe.iter().find(|(j, _)| *j == k).map_or(0.0, |(_, p)| *p);
    let count = bins.len() as f64;
    let (mut tt, mut t1, mut mt, mut m1) = (0.0, 0.0, 0.0, 0.0);
    for k in bins {
        let t = template(k);
        (tt, t1, mt, m1) = (tt + t * t, t1 + t, mt + mean[k] * t, m1 + mean[k]);
    }
    let det = tt * count - t1 * t1;
    let (mut a, mut c) = ((mt * count - t1 * m1) / det, (tt * m1 - t1 * mt) / det);
    if c < 0.0 || !det.is_finite() || det <= 0.0 {
        (a, c) = (mt / tt, 0.0);
    }
    a > 0.0 && a * template(b) > c
}

/// Whole Cams merged upward until each holds a main lobe's bins, as `MODEL` states.
fn bands(grid: &Grid) -> Vec<(f64, f64)> {
    let least = 2 * grid.lobe() + 1;
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut open: Option<(f64, f64)> = None;
    for (lo, hi) in grid.cams() {
        let band = open.map_or((lo, hi), |(l, _)| (l, hi));
        match grid.bins_of(band).len() >= least {
            true => {
                out.push(band);
                open = None;
            }
            false => open = Some(band),
        }
    }
    if let Some((_, hi)) = open {
        match out.last_mut() {
            Some(last) => last.1 = hi,
            None => out.push(open.expect("open")),
        }
    }
    out
}

/// The atoms of the one sound `grid` holds, as `MODEL` states.
fn atoms(grid: &Grid, clip: &Clip, spl_db: f64) -> Vec<Atom> {
    let Some((first, last)) = audible(grid, spl_db) else {
        return Vec::new();
    };
    let on = grid.frame_span(first).0;
    let end = grid.frame_span(last).1.min(clip.secs());
    let frames = &grid.power[first..=last];
    let bins = grid.bins();
    let mean: Vec<f64> = (0..bins)
        .map(|b| frames.iter().map(|f| f[b]).sum::<f64>() / frames.len() as f64)
        .collect();
    let bands = bands(grid);
    let tracks = peaks(grid, &mean, (on, end), spl_db, &bands);
    let feasible = span_hz(grid.rate);
    let pool: Vec<usize> = (0..tracks.len()).collect();
    let voice = sieved(&tracks, &pool, grid.rate / grid.n as f64, feasible)
        .map(|(f0, members)| {
            (
                f0,
                members.iter().map(|(_, n)| *n).max().expect("a member"),
                members,
            )
        })
        .filter(|(f0, count, _)| {
            Atom::harmonic(
                0,
                0.0,
                Trajectory::steady(*f0),
                vec![0.0; count - 1],
                Trajectory {
                    base: 0.0,
                    knots: vec![(clip.secs(), 0.0)],
                },
            )
            .writes(clip)
        });
    let mut roles: Vec<Role> = Vec::new();
    let mut columns: Vec<Column> = Vec::new();
    if let Some((f0, count, _)) = &voice {
        for n in 1..=*count {
            roles.push(Role::Member);
            columns.push(Column::sine(grid, f0 * n as f64));
        }
    }
    for (t, track) in tracks.iter().enumerate() {
        if voice
            .as_ref()
            .is_some_and(|(_, _, members)| members.iter().any(|(q, _)| *q == t))
        {
            continue;
        }
        roles.push(Role::Partial(track.hz));
        columns.push(Column::sine(grid, track.hz));
    }
    for band in bands {
        roles.push(Role::Noise(band));
        columns.push(Column::band(grid, band));
    }
    let weight: Vec<f64> = (0..bins)
        .map(|b| {
            let floor = grid.floor_power(floor_db(grid.hz(b), spl_db));
            1.0 / (mean[b] + floor)
        })
        .collect();
    let projected = |columns: &[Column], frame: &[f64]| -> Vec<f64> {
        columns
            .iter()
            .map(|c| {
                c.bins()
                    .zip(&c.values)
                    .map(|(b, t)| weight[b] * t * frame[b])
                    .sum()
            })
            .collect()
    };
    let averaged = nnls(
        &ridged(normal(&columns, &weight)),
        &projected(&columns, &mean),
    );
    reread(grid, &mean, &mut columns, &mut roles, &averaged);
    let normal = ridged(normal(&columns, &weight));
    let gains: Vec<Vec<f64>> = each(default_threads(), frames.to_vec(), |frame| {
        nnls(&normal, &projected(&columns, &frame))
    });
    let times: Vec<f64> = (first..=last).map(|f| grid.centre_secs(f)).collect();
    let column = |j: usize| -> Vec<f64> { gains.iter().map(|g| g[j]).collect() };
    let mut out = Vec::new();
    let mut key = 0;
    if let Some((f0, count, _)) = voice {
        let members: Vec<(f64, Vec<f64>)> = (0..count)
            .map(|j| (floor_db(f0 * (j + 1) as f64, spl_db), column(j)))
            .collect();
        if let Some(atom) = harmonic(f0, &members, (on, end), &times) {
            out.push(Atom { key, ..atom });
            key += 1;
        }
    }
    for (j, role) in roles.iter().enumerate() {
        let (floor, make): (f64, Box<dyn Fn(Trajectory) -> Atom>) = match role {
            Role::Member => continue,
            Role::Partial(hz) => (
                floor_db(*hz, spl_db),
                Box::new(move |level| Atom::partial(0, on, Trajectory::steady(*hz), level)),
            ),
            Role::Noise((lo, hi)) => (
                floor_db(cam_hz((cam(*lo) + cam(*hi)) / 2.0), spl_db),
                Box::new(move |level| Atom::noise(0, on, (*lo, *hi), level)),
            ),
        };
        let levels: Vec<f64> = column(j).iter().map(|a| db(*a)).collect();
        let peak = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if peak <= floor {
            continue;
        }
        let held: Vec<f64> = levels.iter().map(|l| l.max(floor)).collect();
        let level = thinned((on, end), &times, &held, level_jnd_db(peak - floor));
        out.push(Atom { key, ..make(level) });
        key += 1;
    }
    out
}

/// The harmonic voice at `f0` whose members' thresholds and gains per frame are `members`,
/// as `MODEL` states; `None` where none is ever over its threshold.
fn harmonic(f0: f64, members: &[(f64, Vec<f64>)], span: (f64, f64), times: &[f64]) -> Option<Atom> {
    let peak = members
        .iter()
        .flat_map(|(floor, g)| g.iter().map(move |a| db(*a) - floor))
        .fold(f64::NEG_INFINITY, f64::max);
    if peak <= 0.0 {
        return None;
    }
    let held: Vec<Vec<f64>> = members
        .iter()
        .map(|(floor, g)| g.iter().map(|a| db(*a).max(*floor)).collect())
        .collect();
    let count = held.len() as f64;
    let mean: Vec<f64> = (0..times.len())
        .map(|t| held.iter().map(|m| m[t]).sum::<f64>() / count)
        .collect();
    let offsets: Vec<f64> = held[..held.len() - 1]
        .iter()
        .map(|m| m.iter().zip(&mean).map(|(d, l)| d - l).sum::<f64>() / times.len() as f64)
        .collect();
    let level = thinned(span, times, &mean, level_jnd_db(peak));
    Some(Atom::harmonic(
        0,
        span.0,
        Trajectory::steady(f0),
        offsets,
        level,
    ))
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
fn normal(columns: &[Column], weight: &[f64]) -> Vec<Vec<f64>> {
    let m = columns.len();
    let mut g = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i..m {
            let (a, b) = (&columns[i], &columns[j]);
            let (lo, hi) = (a.start.max(b.start), a.bins().end.min(b.bins().end));
            let sum: f64 = (lo..hi)
                .map(|k| weight[k] * a.values[k - a.start] * b.values[k - b.start])
                .sum();
            (g[i][j], g[j][i]) = (sum, sum);
        }
    }
    g
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

/// The grid `atoms` put in each of `like`'s frames through its templates, each atom's level at
/// the frame's centre.
pub fn heard(atoms: &[Atom], like: &Grid) -> Grid {
    let bins = like.bins();
    let power = (0..like.power.len())
        .map(|f| {
            let t = like.centre_secs(f);
            let mut frame = vec![0.0; bins];
            for atom in atoms.iter().filter(|a| a.on <= t && t <= a.end()) {
                let level = level_at(&atom.level, atom.on, t);
                let mut add = |column: Column, db: f64| {
                    let gain = libm::pow(10.0, db / 10.0);
                    for (b, v) in column.bins().zip(&column.values) {
                        frame[b] += gain * v;
                    }
                };
                match &atom.sound {
                    Sound::Partial { hz } => add(Column::sine(like, hz.base), level),
                    Sound::Harmonic { hz, offsets } => {
                        let last = -offsets.iter().sum::<f64>();
                        for (k, o) in offsets.iter().chain([&last]).enumerate() {
                            add(Column::sine(like, hz.base * (k + 1) as f64), level + o);
                        }
                    }
                    Sound::Noise { lo, hi } => add(Column::band(like, (*lo, *hi)), level),
                }
            }
            frame
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
