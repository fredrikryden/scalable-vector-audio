// Concern: fits one recording: its clip at the fit's rate and unit RMS, the options, the run, the Fit it leaves | Non-concern: the problem (problem.rs), JSON | IO: (samples, rate, Options) -> Fit

mod adds;
mod adjusts;
mod peaks;
mod problem;
mod progress;
mod proposals;
mod render;
mod resume;
mod shared;
mod written;

use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;

use sva_core::{Abandon, CliError};
use sva_engine::hearing::{CAM_FIRST, CAM_LAST, cam_hz};
use sva_engine::{default_threads, each, fft};

use crate::analysis::Spectrogram;
use crate::grid::Grid;
use crate::optimize::{self, Edit, End, Pace, TOLERANCE};
use crate::sources::{self, Clip, Source};

pub use problem::MODEL;
pub use progress::{Action, Geometry, Line, Tweak};
pub use resume::Resume;

use problem::{Problem, State, feasible};
use progress::{first_line, line};

/// A full-scale sine at 103.95 dB SPL by default (SMPTE RP 200: -20 dBFS pink noise at 83 dB
/// SPL, a sine's RMS 3.01 dB under its peak).
pub const SPL_DB: f64 = 103.95;

pub fn mono(planes: &[Vec<f32>]) -> Vec<f64> {
    let len = planes.iter().map(Vec::len).max().unwrap_or(0);
    let width = planes.len().max(1) as f64;
    (0..len)
        .map(|n| {
            planes
                .iter()
                .map(|p| f64::from(p.get(n).copied().unwrap_or(0.0)))
                .sum::<f64>()
                / width
        })
        .collect()
}

/// The rate `rate` is read at, as `MODEL` states.
fn rate_for(rate: u32) -> u32 {
    let need = 2.0 * cam_hz(CAM_LAST);
    let most = (f64::from(rate) / need) as u32;
    (1..=most.max(1))
        .rev()
        .find(|d| rate.is_multiple_of(*d))
        .map_or(rate, |d| rate / d)
}

/// `samples` read at `to`, a whole factor under `rate`, by keeping the bins under its Nyquist.
fn decimated(samples: &[f64], rate: u32, to: u32) -> Vec<f64> {
    let d = (rate / to) as usize;
    if d == 1 || samples.is_empty() {
        return samples.to_vec();
    }
    let n = samples.len().div_ceil(d) * d;
    let mut x = samples.to_vec();
    x.resize(n, 0.0);
    let (re, im) = fft::dft(&x, &vec![0.0; n]);
    let m = n / d;
    let (mut kept_re, mut kept_im) = (vec![0.0; m], vec![0.0; m]);
    for k in 0..m {
        let from = match (2 * k < m, 2 * (m - k) < m) {
            (true, _) => Some(k),
            (_, true) => Some(n - (m - k)),
            _ => None,
        };
        if let Some(j) = from {
            (kept_re[k], kept_im[k]) = (re[j], im[j]);
        }
    }
    let (y, _) = fft::idft(&kept_re, &kept_im);
    y.iter().map(|v| v / d as f64).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub tolerance: f64,
    pub from: Option<Resume>,
    /// The threads the fit computes on; its answers are the same at any count.
    pub threads: NonZeroUsize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            tolerance: TOLERANCE,
            from: None,
            threads: default_threads(),
        }
    }
}

/// A fit: the clip it read, its sources in unit-RMS space, the master gain bringing them back,
/// the SPL that space is heard at, its DL's parts, the pass it resumed from and its last, this
/// run's adds and adjusts, these by `Tweak::ALL`, its threshold, why it ended, and both
/// coarse grids, each band's power spread flat.
pub struct Fit {
    pub clip: Clip,
    pub sources: Vec<Source>,
    pub shared: Vec<(usize, sources::Shared)>,
    pub gain_db: f64,
    pub spl_db: f64,
    pub source_bits: f64,
    pub residual_bits: f64,
    pub from_pass: usize,
    pub passes: usize,
    pub adds: usize,
    pub adjusts: usize,
    pub tweaks: [usize; 4],
    pub tolerance: f64,
    pub ended: End,
    pub clip_grid: Grid,
    pub fit_grid: Grid,
}

struct Paced<'a> {
    threads: NonZeroUsize,
    stop: &'a dyn Abandon,
}

impl Pace for Paced<'_> {
    fn width(&self) -> NonZeroUsize {
        self.threads
    }

    fn each<T: Send, R: Send>(&self, tasks: Vec<T>, run: impl Fn(T) -> R + Sync) -> Vec<R> {
        each(self.threads, tasks, run)
    }

    fn stopped(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        self.stop.abandoned()
    }
}

fn refused(clip: &[f64]) -> Option<CliError> {
    match (clip.is_empty(), clip.iter().all(|s| s.is_finite())) {
        (true, _) => Some(CliError::usage(
            "solve.empty_clip",
            "the recording holds no sample, so there is nothing to fit",
        )),
        (_, false) => Some(CliError::usage(
            "solve.not_finite",
            "the recording holds a sample that is not a finite number",
        )),
        _ => None,
    }
}

/// One sound's fit of `samples` under `options`, its start and each pass heard by `hear` as
/// they end, `stop` asked before each pass and between its moves, as `MODEL` states.
pub async fn fit(
    samples: &[f64],
    rate: u32,
    spl_db: f64,
    options: &Options,
    hear: &mut dyn FnMut(&Line),
    stop: &dyn Abandon,
    clock: &dyn Fn() -> f64,
) -> Result<Fit, CliError> {
    if let Some(refused) = refused(samples) {
        return Err(refused);
    }
    let to = rate_for(rate);
    let samples = decimated(samples, rate, to);
    let clip = Clip {
        rate: to,
        len: samples.len(),
    };
    let square = samples.iter().map(|s| s * s).sum::<f64>() / samples.len().max(1) as f64;
    let rms = match square > 0.0 {
        true => libm::sqrt(square),
        false => 1.0,
    };
    let unit: Vec<f64> = samples.iter().map(|s| s / rms).collect();
    let gain_db = 20.0 * libm::log10(rms);
    let heard_at = spl_db + gain_db;
    let grid = Grid::of(
        &Spectrogram::of(&unit, clip.rate),
        unit.len(),
        options.threads,
    );
    if feasible(&grid).is_empty() {
        return Err(CliError::usage(
            "solve.rate_too_low",
            format!(
                "at {rate} Hz no analysis bin lies inside the feasible set from {} Hz (Cam {}), \
                 so the clip holds nothing a fit hears; a correctness limit, with no override",
                cam_hz(CAM_FIRST),
                CAM_FIRST
            ),
        ));
    }
    let p = Problem::of(grid, clip, heard_at, gain_db, options.threads);
    let start_bits = State::of(&p, Vec::new(), Vec::new(), 0, 0).dl();
    let (state, from_pass) = match &options.from {
        Some(from) => (resume::resumed(&p, from, gain_db)?, from.passes),
        None => (State::of(&p, Vec::new(), Vec::new(), 0, 0), 0),
    };
    let mut first = first_line(&p, &state, start_bits, gain_db);
    first.seconds = clock();
    hear(&first);
    let (mut adds, mut adjusts) = (0, 0);
    let mut tweaks = [0; 4];
    let mut heard = |before: &State, next: &State, began: f64| {
        match next.moved() {
            Some(Edit::Replace(_, (), tweak) | Edit::Remove(_, tweak)) => {
                adjusts += 1;
                tweaks[Tweak::ALL
                    .iter()
                    .position(|t| *t == tweak)
                    .expect("a tweak")] += 1;
            }
            _ => adds += 1,
        }
        let mut heard = line(&p, next, before, start_bits);
        heard.seconds = clock();
        heard.pass_secs = heard.seconds - began;
        hear(&heard);
    };
    let pace = Paced {
        threads: options.threads,
        stop,
    };
    let (state, ended) =
        optimize::run(&p, state, options.tolerance, &pace, clock, &mut heard).await;
    let model = p.full(&p.model(state.kept(), state.shared()));
    Ok(Fit {
        clip,
        sources: render::sources(state.kept()),
        shared: shared::active(&state),
        gain_db,
        spl_db: heard_at,
        source_bits: state.thing_bits(),
        residual_bits: state.missed_bits(),
        from_pass,
        passes: state.passes(),
        adds,
        adjusts,
        tweaks,
        tolerance: options.tolerance,
        ended,
        clip_grid: p.grid.broad(),
        fit_grid: p.grid.with(model).broad(),
    })
}
