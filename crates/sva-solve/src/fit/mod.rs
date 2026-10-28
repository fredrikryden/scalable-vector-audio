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
use sva_engine::hearing::{CAM_FIRST, CAM_LAST, WINDOW_SECS, WINDOW_SPLITS_HZ, cam_hz};
use sva_engine::{default_threads, each, fft};

use crate::analysis::Spectrogram;
use crate::grid::{Grid, Grids};
use crate::optimize::{self, Edit, End, Pace, TOLERANCE};
use crate::sources::{self, Clip, Source};

pub use problem::MODEL;
pub use progress::{Action, Geometry, Line, Tweak};
pub use resume::Resume;

use problem::{Problem, State};
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

/// An analysis band: a Hann window `secs` long over the Hz under `below_hz` no band before it
/// reads, the last, which names no edge, on to the top.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    pub below_hz: Option<f64>,
    pub secs: f64,
}

/// The hearing model's ladder (hearing.rs).
pub fn cited() -> Vec<Window> {
    WINDOW_SECS
        .iter()
        .enumerate()
        .map(|(i, secs)| Window {
            below_hz: WINDOW_SPLITS_HZ.get(i).copied(),
            secs: *secs,
        })
        .collect()
}

fn only() -> Vec<Window> {
    vec![Window {
        below_hz: None,
        secs: WINDOW_SECS[0],
    }]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub tolerance: f64,
    pub from: Option<Resume>,
    /// The analysis bands, low to high; `None` one band at 64 ms, which reads a shorter clip too.
    pub windows: Option<Vec<Window>>,
    /// The threads the fit computes on; its answers are the same at any count.
    pub threads: NonZeroUsize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            tolerance: TOLERANCE,
            from: None,
            windows: None,
            threads: default_threads(),
        }
    }
}

/// A fit: the clip it read, its sources in unit-RMS space, the master gain bringing them back,
/// the SPL that space is heard at, its DL's parts, the pass it resumed from and its last, this
/// run's adjusts by `Tweak::ALL` (its adds the rest of its passes), its threshold, why it ended, and both
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
    pub tweaks: [usize; 4],
    pub tolerance: f64,
    pub ended: End,
    pub clip_grid: Grids,
    pub fit_grid: Grids,
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

/// Why `windows` mean nothing for `clip`, as `Window` states.
fn unmeant(windows: &[Window], clip: &Clip) -> Option<CliError> {
    let refuse = |code, message: String| Some(CliError::usage(code, message));
    if windows.is_empty() {
        return refuse("solve.no_window", "`windows` names no band".to_string());
    }
    let rate = f64::from(clip.rate);
    let short = windows
        .iter()
        .find(|w| !w.secs.is_finite() || libm::round(w.secs * rate) < 2.0);
    if let Some(w) = short {
        return refuse(
            "solve.window_not_positive",
            format!(
                "a window is a finite length holding two samples or more at the fit's {rate} \
                 Hz, not {}s; a correctness limit, with no override",
                w.secs
            ),
        );
    }
    let edges: Vec<f64> = windows
        .iter()
        .map(|w| w.below_hz.unwrap_or(f64::INFINITY))
        .collect();
    let rising = std::iter::once(&0.0)
        .chain(&edges)
        .zip(&edges)
        .all(|(a, b)| a < b);
    let (last, before) = edges.split_last().expect("a band");
    if !rising || last.is_finite() || before.iter().any(|e| e.is_infinite()) {
        return refuse(
            "solve.windows_out_of_order",
            format!(
                "the bands' edges {edges:?} Hz do not rise from 0, each band but the last naming \
                 its own and the last none"
            ),
        );
    }
    let long = windows.iter().find(|w| w.secs > clip.secs());
    long.and_then(|w| {
        refuse(
            "solve.window_too_long",
            format!(
                "a window of {}s is longer than the {}s clip, so no frame of it lies inside; \
                 a correctness limit, with no override",
                w.secs,
                clip.secs()
            ),
        )
    })
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
    let square = samples.iter().map(|s| s * s).sum::<f64>() / samples.len() as f64;
    let rms = match square > 0.0 {
        true => libm::sqrt(square),
        false => 1.0,
    };
    let unit: Vec<f64> = samples.iter().map(|s| s / rms).collect();
    let gain_db = 20.0 * libm::log10(rms);
    let heard_at = spl_db + gain_db;
    if let Some(refused) = options.windows.as_deref().and_then(|w| unmeant(w, &clip)) {
        return Err(refused);
    }
    let windows = options.windows.clone().unwrap_or_else(only);
    let grids = windows
        .iter()
        .map(|w| {
            let spectrogram = Spectrogram::of(&unit, clip.rate, w.secs);
            let grid = Grid::of(&spectrogram, unit.len(), options.threads);
            (grid, w.below_hz.unwrap_or(f64::INFINITY))
        })
        .collect();
    let Some(p) = Problem::of(grids, clip, heard_at, gain_db, options.threads) else {
        return Err(CliError::usage(
            "solve.rate_too_low",
            format!(
                "at {rate} Hz no analysis bin lies inside the feasible set from {} Hz (Cam {}), \
                 so the clip holds nothing a fit hears; a correctness limit, with no override",
                cam_hz(CAM_FIRST),
                CAM_FIRST
            ),
        ));
    };
    let start_bits = State::of(&p, Vec::new(), Vec::new(), 0, 0).dl();
    let (state, from_pass) = match &options.from {
        Some(from) => (resume::resumed(&p, from, gain_db)?, from.passes),
        None => (State::of(&p, Vec::new(), Vec::new(), 0, 0), 0),
    };
    let mut first = first_line(&p, &state, start_bits, gain_db);
    first.seconds = clock();
    hear(&first);
    let mut tweaks = [0; 4];
    let mut heard = |before: &State, next: &State, began: f64| {
        if let Some(Edit::Replace(_, (), tweak) | Edit::Remove(_, tweak)) = next.moved() {
            tweaks[Tweak::ALL
                .iter()
                .position(|t| *t == tweak)
                .expect("a tweak")] += 1;
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
    let model = p.model(state.kept(), state.shared());
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
        tweaks,
        tolerance: options.tolerance,
        ended,
        clip_grid: Grids(
            p.views
                .iter()
                .map(|v| v.grid.only(v.band).broad())
                .collect(),
        ),
        fit_grid: Grids(
            p.views
                .iter()
                .map(|v| v.grid.with(p.full(v, &model)).broad())
                .collect(),
        ),
    })
}
