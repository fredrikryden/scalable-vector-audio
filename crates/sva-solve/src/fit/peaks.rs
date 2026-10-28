// Concern: where the residual peaks: local floor, tonal lobes, QIFFT maxima, a peak's basin and track, lines too near to add | Non-concern: what a peak proposes | IO: (power, Hz) -> peaks, tracks

use std::ops::Range;

use sva_engine::hearing::{cam, cam_hz};

use super::problem::{Problem, View};
use super::render::Proposal;
use super::render::db;
use crate::analysis::floor_db;
use crate::grid::Grid;
use crate::sources::Sound;

/// The local noise floor and the tonal lobes over the feasible bins, as `MODEL` states: each
/// bin's median window, the main lobe's half-width in bins, and the test's ratio.
pub(super) struct Lobes {
    windows: Vec<Range<usize>>,
    pub(super) half: usize,
    ratio: f64,
}

impl Lobes {
    pub(super) fn half(grid: &Grid) -> usize {
        (2 * grid.n).div_ceil(grid.long)
    }

    pub(super) fn of(grid: &Grid, range: &Range<usize>) -> Lobes {
        let half = Lobes::half(grid);
        let count = range.len();
        let windows = (0..count)
            .map(|i| {
                let c = cam(grid.hz(range.start + i));
                let at = |cams: f64| grid.bin_of(cam_hz(cams)) - range.start as f64;
                let lo =
                    (libm::ceil(at(c - 0.5)).max(0.0) as usize).min(i.saturating_sub(2 * half));
                let hi = (libm::floor(at(c + 0.5)) as usize + 1).max(i + 2 * half + 1);
                lo..hi.min(count)
            })
            .collect();
        Lobes {
            windows,
            half,
            ratio: 2.0 * libm::log(count.max(1) as f64),
        }
    }

    /// `spectrum`'s noise floor per bin at `looks` looks a bin, and the tonal peaks, as
    /// `MODEL` states: on each side of the bin's own lobe the median over its window outside
    /// the tonal lobes over the median's share of the mean, the higher side the floor.
    pub(super) fn floor(&self, spectrum: &[f64], looks: f64) -> (Vec<f64>, Vec<usize>) {
        let share = libm::pow(1.0 - 1.0 / (9.0 * looks), 3.0);
        let median = |bins: Range<usize>, barred: &[bool]| -> Option<f64> {
            let mut held: Vec<f64> = bins.filter(|j| !barred[*j]).map(|j| spectrum[j]).collect();
            let middle = held.len() / 2;
            (!held.is_empty())
                .then(|| *held.select_nth_unstable_by(middle, f64::total_cmp).1 / share)
        };
        let medians = |barred: &[bool], side: fn(f64, f64) -> f64| -> Vec<f64> {
            self.windows
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let left = median(w.start..i.saturating_sub(self.half).max(w.start), barred);
                    let right = median((i + self.half + 1).min(w.end)..w.end, barred);
                    match (left, right) {
                        (Some(l), Some(r)) => side(l, r),
                        (Some(v), None) | (None, Some(v)) => v,
                        (None, None) => {
                            w.clone().map(|j| spectrum[j]).fold(f64::INFINITY, f64::min)
                        }
                    }
                })
                .collect()
        };
        let tonal = |floor: &[f64]| -> Vec<usize> {
            (1..spectrum.len().saturating_sub(1))
                .filter(|i| {
                    let (a, c, e) = (spectrum[i - 1], spectrum[*i], spectrum[i + 1]);
                    let near = (i + 1).saturating_sub(2 * self.half)
                        ..(i + 2 * self.half).min(spectrum.len());
                    c > a
                        && c >= e
                        && c > self.ratio * floor[*i]
                        && near.clone().all(|j| spectrum[j] <= c)
                })
                .collect()
        };
        let first = medians(&vec![false; spectrum.len()], f64::min);
        let floor = medians(&self.inside(&tonal(&first), spectrum.len()), f64::max);
        let peaks = tonal(&floor);
        (floor, peaks)
    }

    fn inside(&self, peaks: &[usize], count: usize) -> Vec<bool> {
        let mut out = vec![false; count];
        for i in peaks {
            let span = i.saturating_sub(self.half)..(i + self.half + 1).min(count);
            out[span].iter_mut().for_each(|b| *b = true);
        }
        out
    }
}

pub(super) fn qifft(view: &View, power: &[f64], i: usize) -> (f64, f64) {
    let dbs = |v: f64| db(v.max(f64::MIN_POSITIVE));
    let (a, c, e) = (dbs(power[i - 1]), dbs(power[i]), dbs(power[i + 1]));
    let curve = a - 2.0 * c + e;
    let shift = match curve < 0.0 {
        true => 0.5 * (a - e) / curve,
        false => 0.0,
    };
    let hz = view.grid.hz(view.range.start + i) + shift * view.bin_hz();
    let peak = libm::pow(10.0, (c - 0.25 * (a - e) * shift) / 10.0);
    (
        hz,
        20.0 * libm::log10(2.0 * libm::sqrt(peak) / view.grid.sum_w),
    )
}

fn maxima(p: &Problem, view: &View, power: &[f64], bins: Range<usize>) -> Vec<(f64, f64)> {
    let bins = bins.start.max(1)..bins.end.min(power.len().saturating_sub(1));
    bins.filter(|i| power[*i] > power[i - 1] && power[*i] >= power[i + 1])
        .map(|i| (qifft(view, power, i), power[i]))
        .filter(|((hz, level), _)| *level > floor_db(*hz, p.spl_db))
        .map(|((hz, _), power)| (hz, power))
        .collect()
}

fn bins_of(view: &View, (lo, hi): (f64, f64)) -> Range<usize> {
    let local = |hz: f64| view.grid.bin_of(hz) - view.range.start as f64;
    let first = libm::ceil(local(lo)).max(0.0) as usize;
    let last = libm::floor(local(hi)).max(-1.0) + 1.0;
    first..(last.max(0.0) as usize).min(view.range.len())
}

fn lobe_hz(p: &Problem, hz: f64) -> f64 {
    let view = p.at_hz(hz);
    view.lobes.half as f64 * view.bin_hz()
}

pub(super) fn lines(p: &Problem, proposal: &Proposal) -> Vec<Vec<f64>> {
    match proposal.sound {
        Sound::Noise { .. } | Sound::Shaped { .. } => Vec::new(),
        Sound::Partial { .. } | Sound::Harmonic { .. } => p.columns(proposal).paths,
    }
}

pub(super) fn moves(p: &Problem, proposal: &Proposal) -> bool {
    lines(p, proposal).iter().any(|path| {
        let (lo, hi) = path
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), hz| {
                (lo.min(*hz), hi.max(*hz))
            });
        hi - lo >= lobe_hz(p, lo)
    })
}

/// Whether lines of `a` and `b` lie too near for their powers to add, as `MODEL` states.
pub(super) fn collide(p: &Problem, a: &[Vec<f64>], b: &[Vec<f64>]) -> bool {
    a.iter().any(|x| {
        b.iter().any(|y| {
            x.iter()
                .zip(y)
                .any(|(x, y)| (x - y).abs() < lobe_hz(p, *x).max(lobe_hz(p, *y)))
        })
    })
}

/// The peak nearest each frame's Hz on `path` in Cam within a main lobe of it, as
/// `(t, ln hz)`, in frames whose view reads it.
pub(super) fn pathed(p: &Problem, power: &[Vec<f64>], path: &[f64]) -> Vec<(f64, f64)> {
    path.iter()
        .enumerate()
        .filter(|(f, hz)| p.view(*f).holds(**hz))
        .filter_map(|(f, hz)| {
            let (view, lobe, target) = (p.view(f), lobe_hz(p, *hz), cam(*hz));
            maxima(p, view, &power[f], bins_of(view, (hz - lobe, hz + lobe)))
                .into_iter()
                .map(|(h, _)| h)
                .min_by(|a, b| {
                    (cam(*a) - target)
                        .abs()
                        .total_cmp(&(cam(*b) - target).abs())
                })
                .map(|h| (p.times[f], libm::log(h)))
        })
        .collect()
}

/// The basin of `view`'s `mean` about the bin nearest `hz`: from the local maximum it climbs to,
/// down to the local minimum on each side.
fn basin(view: &View, mean: &[f64], hz: f64) -> Range<usize> {
    let bin = libm::round(view.grid.bin_of(hz)) as usize;
    let mut i = bin.clamp(view.range.start, view.range.end - 1) - view.range.start;
    loop {
        let up = [i.checked_sub(1), (i + 1 < mean.len()).then_some(i + 1)]
            .into_iter()
            .flatten()
            .filter(|j| mean[*j] > mean[i])
            .max_by(|a, b| mean[*a].total_cmp(&mean[*b]).then(b.cmp(a)));
        match up {
            Some(j) => i = j,
            None => break,
        }
    }
    let (mut lo, mut hi) = (i, i);
    while lo > 0 && mean[lo - 1] < mean[lo] {
        lo -= 1;
    }
    while hi + 1 < mean.len() && mean[hi + 1] < mean[hi] {
        hi += 1;
    }
    lo..hi + 1
}

/// The peak track a candidate at `hz` holds through `power`, as `MODEL` states: in each frame of
/// the view reading `hz` the highest local maximum over its floor in that basin, `(t, ln hz)`.
pub(super) fn searched(
    p: &Problem,
    power: &[Vec<f64>],
    mean: &[Vec<f64>],
    hz: f64,
) -> Vec<(f64, f64)> {
    let v = p.reading(hz);
    let view = &p.views[v];
    let bins = basin(view, &mean[v], hz);
    view.frames
        .clone()
        .filter_map(|f| {
            maxima(p, view, &power[f], bins.clone())
                .into_iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(h, _)| (p.times[f], libm::log(h)))
        })
        .collect()
}
