// Concern: one envelope per ERB band beside its impulse floor, or its knots within a dB bound | Non-concern: the discrete spectrum (spectrum.rs), what a band means | IO: (planes, rate, prune) -> bands

use sva_formula::filter::Shape;

use crate::biquad::{Coeffs, State, design};

/// Glasberg and Moore's auditory filter width in Hz.
pub fn erb_hz(hz: f64) -> f64 {
    24.7 * (4.37 * hz / 1000.0 + 1.0)
}

/// One Cam is one filter width along the cochlea.
pub fn cam(hz: f64) -> f64 {
    21.4 * (1.0 + 4.37 * hz / 1000.0).log10()
}

pub fn hz_at_cam(cam: f64) -> f64 {
    (10f64.powf(cam / 21.4) - 1.0) * 1000.0 / 4.37
}

/// 1 to 40 Cam is 26 Hz to 16.7 kHz.
pub const FIRST_CAM: f64 = 1.0;
pub const BAND_COUNT: usize = 40;

/// Under the 2-3 ms gap-detection limit, a hundredth of the samples it came from.
pub const DECIMATED_HZ: f64 = 2000.0;

/// Sixteen time constants is past every band's peak.
fn floor_secs(erb: f64) -> f64 {
    (16.0 / erb).clamp(0.02, 2.0)
}

/// A filter cannot report an onset faster than its own, so a caller subtracts this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandFloor {
    pub peak: f64,
    pub time_to_peak_secs: f64,
    pub rise_10_90_secs: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BandTrack {
    pub centre_hz: f64,
    pub cam: f64,
    pub erb_hz: f64,
    pub q: f64,
    pub peak: f64,
    pub time_to_peak_secs: Option<f64>,
    pub rise_10_90_secs: Option<f64>,
    pub floor: BandFloor,
    pub rms: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bands {
    pub rate_hz: f64,
    pub start_secs: f64,
    pub bands: Vec<BandTrack>,
}

/// `hz: None` is the band's own ERB, capped at a quarter of the decimated rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Smoothing {
    pub hz: Option<f64>,
    pub poles: usize,
}

const OWN_WIDTH: Smoothing = Smoothing { hz: None, poles: 1 };

/// Squaring rectifies; the root at decimation makes the track an RMS.
struct Chain {
    pass: Coeffs,
    smooth: Coeffs,
    a: State,
    b: State,
    energy: Vec<State>,
}

impl Chain {
    fn new(centre_hz: f64, sr: f64, smoothing: Smoothing) -> Chain {
        let erb = erb_hz(centre_hz);
        let hz = smoothing.hz.unwrap_or(erb.min(DECIMATED_HZ / 4.0));
        Chain {
            pass: design(Shape::Bandpass, centre_hz, centre_hz / erb, 0.0, sr),
            smooth: design(Shape::OnePole, hz, 0.0, 0.0, sr),
            a: State::default(),
            b: State::default(),
            energy: vec![State::default(); smoothing.poles],
        }
    }

    #[inline]
    fn step(&mut self, x: f64) -> f64 {
        let banded = self.b.step(&self.pass, self.a.step(&self.pass, x));
        self.energy
            .iter_mut()
            .fold(banded * banded, |e, pole| pole.step(&self.smooth, e))
    }
}

fn trace(
    centre_hz: f64,
    sr: f64,
    smoothing: Smoothing,
    (hop, n, width): (usize, usize, usize),
    at: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let mut chains: Vec<Chain> = (0..width)
        .map(|_| Chain::new(centre_hz, sr, smoothing))
        .collect();
    let mut out = Vec::with_capacity(n / hop + 1);
    for i in 0..n {
        let mut each = chains
            .iter_mut()
            .enumerate()
            .map(|(c, chain)| chain.step(at(c, i)));
        let first = each.next().expect("a channel");
        let energy = each.fold(first, |sum, e| sum + e) / width as f64;
        if i % hop == 0 {
            out.push(energy.max(0.0).sqrt());
        }
    }
    out
}

/// Interpolated between the frames straddling it.
fn crossing(track: &[f64], until: usize, level: f64, rate: f64) -> Option<f64> {
    let hit = track[..=until].iter().position(|&v| v >= level)?;
    if hit == 0 {
        return Some(0.0);
    }
    let (lo, hi) = (track[hit - 1], track[hit]);
    let frac = match hi > lo {
        true => (level - lo) / (hi - lo),
        false => 0.0,
    };
    Some((hit as f64 - 1.0 + frac) / rate)
}

fn stats(track: &[f64], rate: f64) -> (f64, Option<f64>, Option<f64>) {
    let peak = track.iter().copied().fold(0.0, f64::max);
    if peak <= 0.0 {
        return (peak, None, None);
    }
    let at = track
        .iter()
        .position(|&v| v == peak)
        .expect("the peak is one of the frames");
    let t10 = crossing(track, at, 0.1 * peak, rate);
    let t90 = crossing(track, at, 0.9 * peak, rate);
    let rise = t10.zip(t90).map(|(a, b)| (b - a).max(0.0));
    (peak, Some(at as f64 / rate), rise)
}

fn floor_of(centre_hz: f64, sr: f64, smoothing: Smoothing, hop: usize, rate: f64) -> BandFloor {
    let n = (floor_secs(erb_hz(centre_hz)) * sr).round().max(4.0) as usize;
    let track = trace(centre_hz, sr, smoothing, (hop, n, 1), |_, i| {
        f64::from(u8::from(i == 0))
    });
    let (peak, at, rise) = stats(&track, rate);
    BandFloor {
        peak,
        time_to_peak_secs: at.unwrap_or(0.0),
        rise_10_90_secs: rise,
    }
}

pub const HIGHEST_CENTRE_OF_RATE: f64 = 0.45;

/// A centre the design cannot place is left out, never clamped onto the last it could.
pub(crate) fn centres(sr: f64) -> Vec<f64> {
    (0..BAND_COUNT)
        .map(|k| hz_at_cam(FIRST_CAM + k as f64))
        .filter(|hz| *hz < HIGHEST_CENTRE_OF_RATE * sr)
        .collect()
}

pub fn least_rate_hz() -> f64 {
    hz_at_cam(FIRST_CAM) / HIGHEST_CENTRE_OF_RATE
}

pub fn analyze(planes: &[&[f64]], sr: f64, start_secs: f64) -> Bands {
    smoothed(planes, sr, start_secs, OWN_WIDTH)
}

pub const PRUNED_MODEL: &str = "each band's track as `bands` traces it, read at every source \
                                sample instead of every 2 kHz one, in dB, 20 log10 of its \
                                rms, any level under `floor_db` (the band's peak plus the \
                                asked floor) cut to it. Its knots are a swing filter's \
                                (Elmeleegy, H. et al. (2009), Online piece-wise linear \
                                approximation of numerical streams with precision \
                                guarantees, Proc. VLDB Endow. 2(1) 145-156): from each knot \
                                the next is the last sample some line from it stays within \
                                eps_db - g/2 of every sample since, its value that line's, \
                                rounded to a grid of g dB, g the largest power of ten, at \
                                most 1, not over eps_db / 5, so the polyline through the \
                                stored knots is within eps_db of the cut track at every \
                                source sample. `max_error_db` is the largest distance \
                                between the two, evaluated there, the knots joined \
                                linearly in dB. Peak, time to peak and rise are read off \
                                every source sample, the impulse floor too";

/// The fewest decimals a pruned knot is written to: the grid's step is at most a fifth of
/// the bound.
fn grid_per_db(eps_db: f64) -> f64 {
    let decimals = (5.0 / eps_db).log10().ceil().max(0.0) as usize;
    (0..decimals).fold(1.0, |per, _| per * 10.0)
}

/// Under this a bound is finer than the tens of dB a level spans hold in an f64, so no
/// evaluated error could certify it: a correctness limit, not a cost one.
pub const LEAST_PRUNE_DB: f64 = 1e-6;

/// `eps_db` over `LEAST_PRUNE_DB`; `floor_db` under 0, against each band's own peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prune {
    pub eps_db: f64,
    pub floor_db: f64,
}

/// A source sample's index from the read's start, and the level there in dB.
pub type Knot = (usize, f64);

#[derive(Clone, Debug, PartialEq)]
pub struct PrunedBand {
    pub centre_hz: f64,
    pub cam: f64,
    pub erb_hz: f64,
    pub q: f64,
    pub peak: f64,
    pub time_to_peak_secs: Option<f64>,
    pub rise_10_90_secs: Option<f64>,
    pub floor: BandFloor,
    /// `None` where the band never sounds, and holds no knot.
    pub floor_db: Option<f64>,
    pub max_error_db: f64,
    pub knots: Vec<Knot>,
}

/// `rate_hz` is the source rate a knot's index counts at.
#[derive(Clone, Debug, PartialEq)]
pub struct PrunedBands {
    pub rate_hz: f64,
    pub start_secs: f64,
    pub prune: Prune,
    pub bands: Vec<PrunedBand>,
}

pub fn pruned(planes: &[&[f64]], sr: f64, start_secs: f64, prune: Prune) -> PrunedBands {
    let shape = (crate::measure::held(planes), planes.len());
    PrunedBands {
        rate_hz: sr,
        start_secs,
        prune,
        bands: centres(sr)
            .into_iter()
            .map(|centre_hz| {
                let (n, width) = shape;
                let track = trace(centre_hz, sr, OWN_WIDTH, (1, n, width), |c, i| planes[c][i]);
                let (peak, at, rise) = stats(&track, sr);
                let floor_db = (peak > 0.0).then(|| 20.0 * peak.log10() + prune.floor_db);
                let levels = |floor: f64| track.iter().map(move |v| (20.0 * v.log10()).max(floor));
                let (knots, max_error_db) = match floor_db {
                    Some(floor) => {
                        let knots = swing(levels(floor), prune.eps_db);
                        let worst = worst(levels(floor), &knots);
                        (knots, worst)
                    }
                    None => (Vec::new(), 0.0),
                };
                let erb = erb_hz(centre_hz);
                PrunedBand {
                    centre_hz,
                    cam: cam(centre_hz),
                    erb_hz: erb,
                    q: centre_hz / erb,
                    peak,
                    time_to_peak_secs: at,
                    rise_10_90_secs: rise,
                    floor: floor_of(centre_hz, sr, OWN_WIDTH, 1, sr),
                    floor_db,
                    max_error_db,
                    knots,
                }
            })
            .collect(),
    }
}

/// As `PRUNED_MODEL` states. Each closed run's line holds every sample within `within`, and
/// rounding its end to the grid moves it by at most half a step anywhere along the run.
fn swing(levels: impl Iterator<Item = f64>, eps_db: f64) -> Vec<Knot> {
    let per = grid_per_db(eps_db);
    let on_grid = |db: f64| (db * per).round() / per;
    let within = (eps_db - 0.5 / per) * (1.0 - 1e-9);
    let mut levels = levels.enumerate();
    let Some((_, first)) = levels.next() else {
        return Vec::new();
    };
    let mut anchor = (0, on_grid(first));
    let mut knots = vec![anchor];
    let (mut last, mut lo, mut hi) = ((0, first), f64::NEG_INFINITY, f64::INFINITY);
    let close = |anchor: Knot, (at, db): Knot, lo: f64, hi: f64| {
        let run = (at - anchor.0) as f64;
        let slope = ((db - anchor.1) / run).clamp(lo, hi);
        (at, on_grid(anchor.1 + slope * run))
    };
    for (at, db) in levels {
        let run = (at - anchor.0) as f64;
        let (low, high) = (
            (db - within - anchor.1) / run,
            (db + within - anchor.1) / run,
        );
        if low.max(lo) > high.min(hi) {
            anchor = close(anchor, last, lo, hi);
            knots.push(anchor);
            let run = (at - anchor.0) as f64;
            (lo, hi) = (
                (db - within - anchor.1) / run,
                (db + within - anchor.1) / run,
            );
        } else {
            (lo, hi) = (low.max(lo), high.min(hi));
        }
        last = (at, db);
    }
    if last.0 > anchor.0 {
        knots.push(close(anchor, last, lo, hi));
    }
    knots
}

/// The polyline through `knots`, linear in dB, against every level it stands for.
fn worst(levels: impl Iterator<Item = f64>, knots: &[Knot]) -> f64 {
    let mut joins = knots.windows(2).peekable();
    let mut most = 0f64;
    for (at, db) in levels.enumerate() {
        let line = match joins.peek() {
            None => knots[0].1,
            Some(&[a, b]) => a.1 + (b.1 - a.1) * (at - a.0) as f64 / (b.0 - a.0) as f64,
            Some(_) => unreachable!("windows of two"),
        };
        most = most.max((line - db).abs());
        if joins.peek().is_some_and(|pair| at == pair[1].0) {
            joins.next();
        }
    }
    most
}

pub fn smoothed(planes: &[&[f64]], sr: f64, start_secs: f64, smoothing: Smoothing) -> Bands {
    let shape = (crate::measure::held(planes), planes.len());
    let hop = (sr / DECIMATED_HZ).round().max(1.0) as usize;
    let rate = sr / hop as f64;
    Bands {
        rate_hz: rate,
        start_secs,
        bands: centres(sr)
            .into_iter()
            .map(|centre_hz| {
                let (n, width) = shape;
                let track = trace(centre_hz, sr, smoothing, (hop, n, width), |c, i| {
                    planes[c][i]
                });
                let (peak, at, rise) = stats(&track, rate);
                let erb = erb_hz(centre_hz);
                BandTrack {
                    centre_hz,
                    cam: cam(centre_hz),
                    erb_hz: erb,
                    q: centre_hz / erb,
                    peak,
                    time_to_peak_secs: at,
                    rise_10_90_secs: rise,
                    floor: floor_of(centre_hz, sr, smoothing, hop, rate),
                    rms: track,
                }
            })
            .collect(),
    }
}
