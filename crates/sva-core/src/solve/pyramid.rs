// Concern: PYRAMID: levels coarse to fine, each refining the last level's atoms then growing them; stages, progress | Non-concern: births, JSON (report.rs) | IO: (samples, rate, levels) -> stages

use std::cell::Cell;
use std::f64::consts::{FRAC_1_SQRT_2, LN_2};
use std::num::NonZeroUsize;
use std::pin::Pin;

use sva_engine::{Abandon, fft};

use super::add::{self, COHERENT, Heard, State, length, priced, priced_and_pinned};
use super::atoms::{Atom, Clip, files};
use super::bits;
use super::calibrate::n0;
use super::lm::{Descent, Ended};
use super::objective::Objective;

pub const MODEL: &str = "levels at scales s = r^-(L-1-k), k = 0 the coarsest, up to s = 1, r = \
     1/sqrt(2) (exempt), the caller's coarsest n of them run; each level reads the clip at \
     rate/D, D the rate's largest whole factor no larger than s (the clip padded with silence to \
     a whole D, its DFT's bins from the new Nyquist up dropped, every D-th sample kept, the share \
     of the clip's energy the dropped bins held by Parseval reported per stage as cut_db beside \
     its rate_hz), renders its atoms at that rate, as the engine renders them at any rate, and \
     reads both through its own ear at s, N0 calibrated there at that rate (calibrate.rs), prices \
     each atom again there and drops one heard nowhere, pins its frequencies and edges where they \
     are when their Fisher precision there is no finer than ADD's analysis bin (add.rs), so a \
     level too coarse to resolve one leaves it where the analysis put it, then refines the \
     state's atoms by LM and grows them by ADD (add.rs): in rounds until one keeps no birth at \
     the coarsest level, by one round at each finer unless the caller asks rounds at every level, \
     every atom unfrozen at a level's start; numbers are Hz, s and dB, so a level's state starts \
     the next unchanged; each level is a stage whose length is its residual bits plus B(x) at the \
     finest scale run, every stage's atoms priced at one precision set, the latest stage's: an \
     atom at the precisions that scale reads for the latest stage's atom of its key, else at its \
     own stage's (at its level where that scale hears it nowhere), so precisions read again per \
     stage move no stage's length and the stages share one measure, priced again as each \
     finishes; the last stage no longer than every earlier the one chosen; a signal, asked before \
     every descent's step, ends the run with the stage it interrupts, measured as one that \
     finished; a run asked coarse reads every level it runs by the coarse model (coarse.rs); each \
     stage sounds the clip's excitation map and its atoms' through its level's ear as a noise \
     vocoder (the ear's vocode.rs), the one operator for both";

/// The ratio between levels' scales, exempt from the no-tuning rule.
pub const RATIO: f64 = FRAC_1_SQRT_2;

/// Levels by default, exempt: s = 2.83, 2, 1.41 and 1.
pub const LEVELS: usize = 4;

/// A full-scale sine at 103.95 dB SPL by default (SMPTE RP 200: -20 dBFS pink noise at 83 dB
/// SPL, a sine's RMS 3.01 dB under its peak).
pub const SPL_DB: f64 = 103.95;

/// The planes' mean, the one channel the ear reads, padded with silence to the longest.
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

/// `levels` scales, coarsest first, the last 1.
pub fn scales(levels: usize) -> Vec<f64> {
    (0..levels)
        .map(|k| libm::pow(RATIO, -((levels - 1 - k) as f64)))
        .collect()
}

/// Each run level's rate and scale, coarsest first, as `MODEL` states.
pub fn read_at(rate: u32, levels: usize, coarsest: usize) -> Vec<(u32, f64)> {
    assert!(
        (1..=levels).contains(&coarsest),
        "a run reads from 1 to all its levels"
    );
    scales(levels)[..coarsest]
        .iter()
        .enumerate()
        .map(|(k, s)| (rate_at(rate, *s, levels - 1 - k), *s))
        .collect()
}

/// `rate` over its largest whole factor no larger than `s = RATIO^-e`, within the `e + 1`
/// ulps `s` is rounded by.
fn rate_at(rate: u32, s: f64, e: usize) -> u32 {
    let most = s * (1.0 + (e + 1) as f64 * f64::EPSILON);
    (1..=(libm::floor(most) as u32).max(1))
        .rev()
        .find(|d| rate.is_multiple_of(*d))
        .map_or(rate, |d| rate / d)
}

/// `samples` read at `to`, as `MODEL` states, beside the energy share cut.
pub fn decimated(samples: &[f64], rate: u32, to: u32) -> (Vec<f64>, f64) {
    let d = (rate / to) as usize;
    if d == 1 || samples.is_empty() {
        return (samples.to_vec(), 0.0);
    }
    let n = samples.len().div_ceil(d) * d;
    let mut x = samples.to_vec();
    x.resize(n, 0.0);
    let (re, im) = fft::dft(&x, &vec![0.0; n]);
    let m = n / d;
    let (mut kept_re, mut kept_im) = (vec![0.0; m], vec![0.0; m]);
    let energy = |j: usize| re[j] * re[j] + im[j] * im[j];
    let all: f64 = (0..n).map(energy).sum();
    let mut kept = 0.0;
    for k in 0..m {
        let from = match (2 * k < m, 2 * (m - k) < m) {
            (true, _) => Some(k),
            (_, true) => Some(n - (m - k)),
            _ => None,
        };
        if let Some(j) = from {
            (kept_re[k], kept_im[k]) = (re[j], im[j]);
            kept += energy(j);
        }
    }
    let (y, _) = fft::idft(&kept_re, &kept_im);
    let cut = match all > 0.0 {
        true => ((all - kept) / all).max(0.0),
        false => 0.0,
    };
    (y.iter().map(|v| v / d as f64).collect(), cut)
}

/// A level's clip, samples, energy share cut and ADD's analysis.
struct Read {
    clip: Clip,
    samples: Vec<f64>,
    cut: f64,
    heard: Heard,
}

impl Read {
    fn of(samples: &[f64], rate: u32, to: u32) -> Read {
        let (samples, cut) = decimated(samples, rate, to);
        Read {
            clip: Clip {
                rate: to,
                len: samples.len(),
            },
            heard: Heard::of(&samples, to),
            cut,
            samples,
        }
    }
}

pub fn label(level: usize, levels: usize) -> String {
    format!("level {level} of {levels}")
}

/// What the run is doing, as the page names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Reading,
    Ranking,
    Descending,
    Measuring,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Action::Reading => "reading_reference",
            Action::Ranking => "ranking_proposals",
            Action::Descending => "descending",
            Action::Measuring => "measuring",
        }
    }
}

/// A count an activity is at, and its unit.
pub type Count = (usize, &'static str);

/// One progress line.
pub enum Event<'a> {
    Activity {
        action: Action,
        stage: &'a str,
        edit: Option<&'a str>,
        count: Option<Count>,
        part: (usize, usize),
        comparisons: u64,
        seconds: f64,
    },
    /// A descent's last step.
    Step {
        stage: &'a str,
        steps: usize,
        residual_bits: f64,
        numbers: usize,
        bits: f64,
        ended: Ended,
    },
    Finished(&'a Stage),
}

/// One level's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub label: String,
    pub scale: f64,
    /// The rate read at and the energy share that cut.
    pub rate: u32,
    pub cut: f64,
    pub atoms: Vec<Atom>,
    pub residual_bits: f64,
    pub bits: f64,
    pub length_bits: f64,
    pub least: bool,
    pub stopped: bool,
    pub seconds: f64,
    pub edits: Vec<String>,
    pub descents: Vec<(usize, Ended)>,
    pub files: Vec<(String, String)>,
    /// The clip's and the atoms' excitation, vocoded at `rate`.
    pub vocoded: [Vec<f64>; 2],
}

impl Stage {
    pub fn numbers(&self) -> usize {
        self.atoms.iter().map(|a| a.values().len()).sum()
    }
}

/// A caller's signal, kept once it has come, so a stage knows it was stopped.
pub struct Latch<'a> {
    signal: &'a dyn Abandon,
    came: Cell<bool>,
}

impl Abandon for Latch<'_> {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(async move {
            if !self.came.get() && self.signal.abandoned().await {
                self.came.set(true);
            }
            self.came.get()
        })
    }
}

/// What a run reports through, how it is stopped, and its clock in seconds.
pub struct Run<'a> {
    pub spl_db: f64,
    pub stopped: Latch<'a>,
    hear: &'a mut dyn FnMut(Event),
    now: &'a dyn Fn() -> f64,
    started: f64,
    comparisons: u64,
    stage: String,
    part: (usize, usize),
    pub edits: Vec<String>,
    descents: Vec<(usize, Ended)>,
    pub proposed: usize,
}

impl Run<'_> {
    pub fn activity(&mut self, action: Action, edit: Option<&str>, count: Option<Count>) {
        let seconds = (self.now)() - self.started;
        (self.hear)(Event::Activity {
            action,
            stage: &self.stage,
            edit,
            count,
            part: self.part,
            comparisons: self.comparisons,
            seconds,
        });
    }

    /// `evaluations` more evaluations of `objective`'s every element.
    pub fn evaluated(&mut self, objective: &Objective, evaluations: usize) {
        self.comparisons += objective.count() * evaluations as u64;
    }

    /// A descent ended: its evaluations and linearisations counted, its last step heard.
    pub fn descended(&mut self, objective: &Objective, descent: &Descent, bits: f64) {
        let evaluations: usize = descent.steps.iter().map(|s| s.evaluations + 1).sum();
        self.evaluated(objective, evaluations + 1);
        self.descents.push((descent.steps.len(), descent.ended));
        (self.hear)(Event::Step {
            stage: &self.stage,
            steps: descent.steps.len(),
            residual_bits: descent.cost / LN_2,
            numbers: descent.x.len(),
            bits,
            ended: descent.ended,
        });
    }
}

/// What a run is asked: its levels and how many of the coarsest it runs, whether by the coarse
/// model, the clip's full-scale sine in dB SPL, the threads N0 is fitted on, whether every
/// level grows in rounds (else the coarsest alone, a cost limit).
pub struct Asked {
    pub levels: usize,
    pub coarsest: usize,
    pub coarse: bool,
    pub spl_db: f64,
    pub threads: NonZeroUsize,
    pub every_level: bool,
}

/// One finished level as the stages compare it: its residual bits at the finest scale run
/// and each atom's prices there.
struct Measured {
    residual_bits: f64,
    priced: Vec<bits::Priced>,
}

/// Every stage's `length_bits` and `least` again, each atom priced as `last`'s atom of its key
/// is where `last` holds one of its numbers, else as its own stage priced it.
fn repriced(stages: &mut [Stage], measured: &[Measured]) {
    let Some((last, _)) = stages.split_last() else {
        return;
    };
    let finest = measured.last().expect("one measure per stage");
    let held: Vec<(u64, usize, bits::Priced)> = last
        .atoms
        .iter()
        .zip(&finest.priced)
        .map(|(a, p)| (a.key, a.values().len(), p.clone()))
        .collect();
    let mut least = f64::INFINITY;
    for (stage, own) in stages.iter_mut().zip(measured) {
        let priced: Vec<bits::Priced> = stage
            .atoms
            .iter()
            .zip(&own.priced)
            .map(|(a, p)| {
                held.iter()
                    .find(|(key, numbers, _)| *key == a.key && *numbers == a.values().len())
                    .map_or_else(|| p.clone(), |(_, _, p)| p.clone())
            })
            .collect();
        stage.length_bits = own.residual_bits + bits::bits(&stage.atoms, &priced);
        stage.least = stage.length_bits <= least;
        least = least.min(stage.length_bits);
    }
}

/// One stage per level, each heard as it finishes, as `MODEL` states.
pub async fn pyramid(
    samples: &[f64],
    rate: u32,
    asked: &Asked,
    hear: &mut dyn FnMut(Event),
    stopped: &dyn Abandon,
    now: &dyn Fn() -> f64,
) -> Vec<Stage> {
    let clip = Clip {
        rate,
        len: samples.len(),
    };
    let levels = asked.levels;
    let spl = asked.spl_db;
    let mut run = Run {
        spl_db: spl,
        stopped: Latch {
            signal: stopped,
            came: Cell::new(false),
        },
        hear,
        now,
        started: now(),
        comparisons: 0,
        stage: label(1, levels),
        part: (0, levels),
        edits: Vec::new(),
        descents: Vec::new(),
        proposed: 0,
    };
    run.activity(Action::Reading, None, None);
    let run_levels = read_at(rate, levels, asked.coarsest);
    let read = |k: usize| Read::of(samples, rate, run_levels[k].0);
    let at = |read: &Read, s: f64| {
        let n0 = n0(read.clip.rate, s, spl, asked.threads);
        let objective = Objective::new(&read.samples, read.clip, spl, s, n0);
        match asked.coarse {
            true => objective.coarse(),
            false => objective,
        }
    };
    let last = run_levels.len() - 1;
    let finest_read = read(last);
    let finest = at(&finest_read, run_levels[last].1);
    let mut state = State {
        atoms: Vec::new(),
        priced: Vec::new(),
        pinned: Vec::new(),
        frozen: Vec::new(),
        length: f64::INFINITY,
    };
    let mut stages: Vec<Stage> = Vec::new();
    let mut measured: Vec<Measured> = Vec::new();
    for (k, (_, s)) in run_levels.iter().copied().enumerate() {
        let began = now();
        run.stage = label(k + 1, levels);
        run.part = (k, levels);
        run.edits.clear();
        run.proposed = 0;
        run.descents.clear();
        run.activity(Action::Reading, None, None);
        let own = (k != last).then(|| read(k));
        let level_read = own.as_ref().unwrap_or(&finest_read);
        let level = (k != last).then(|| at(level_read, s));
        let objective = level.as_ref().unwrap_or(&finest);
        let heard = &level_read.heard;
        let (mut atoms, mut prices, mut pinned) = (Vec::new(), Vec::new(), Vec::new());
        for atom in state.atoms {
            if let Some((p, pins)) = priced_and_pinned(objective, &atom, spl, heard.bin_hz()) {
                atoms.push(atom);
                prices.push(p);
                pinned.push(pins);
            }
        }
        state = State {
            length: length(objective, &atoms, &prices),
            frozen: vec![false; atoms.len()],
            atoms,
            priced: prices,
            pinned,
        };
        state = add::add(
            objective,
            heard,
            state,
            k == 0 || asked.every_level,
            &mut run,
        )
        .await;
        run.activity(Action::Measuring, None, Some((1, "evaluation")));
        let residual_bits = objective.bits(&state.atoms, COHERENT);
        let bits = bits::bits(&state.atoms, &state.priced);
        let common: Vec<bits::Priced> = state
            .atoms
            .iter()
            .zip(&state.priced)
            .map(|(a, p)| priced(&finest, a, spl).unwrap_or_else(|| p.clone()))
            .collect();
        measured.push(Measured {
            residual_bits: finest.bits(&state.atoms, COHERENT),
            priced: common,
        });
        run.evaluated(objective, 1);
        run.evaluated(&finest, 1);
        let ear = objective.ear();
        let len = level_read.clip.len;
        let fit = objective.excitations(&state.atoms, COHERENT, ear.frames(len));
        let vocoded = [
            ear.vocoded(&ear.excitations(&[&level_read.samples]), len),
            ear.vocoded(&fit, len),
        ];
        let stopped = run.stopped.abandoned().await;
        stages.push(Stage {
            label: run.stage.clone(),
            scale: s,
            rate: level_read.clip.rate,
            cut: level_read.cut,
            atoms: state.atoms.clone(),
            residual_bits,
            bits,
            length_bits: f64::INFINITY,
            least: false,
            stopped,
            seconds: now() - began,
            edits: run.edits.clone(),
            descents: run.descents.clone(),
            files: files(&state.atoms, &clip),
            vocoded,
        });
        repriced(&mut stages, &measured);
        (run.hear)(Event::Finished(stages.last().expect("a stage")));
        if stopped {
            break;
        }
    }
    stages
}

/// The last stage shorter than every earlier.
pub fn chosen(stages: &[Stage]) -> Option<&Stage> {
    stages.iter().rfind(|s| s.least)
}
