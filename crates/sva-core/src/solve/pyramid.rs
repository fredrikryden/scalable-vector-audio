// Concern: PYRAMID: levels coarse to fine, each refining the last level's atoms then growing them; stages, progress | Non-concern: births, JSON (report.rs) | IO: (samples, rate, levels) -> stages

use std::f64::consts::{FRAC_1_SQRT_2, LN_2};
use std::num::NonZeroUsize;

use super::add::{self, COHERENT, Spectrogram, State, edit, length, priced, priced_and_pinned};
use super::atoms::{Atom, Clip, files};
use super::bits;
use super::calibrate::n0;
use super::fisher::Fit;
use super::lm::{Descent, Ended, Limits, descend};
use super::objective::Objective;

pub const MODEL: &str = "levels at scales s = r^-(L-1-k), k = 0 the coarsest, up to s = 1, r = \
     1/sqrt(2) (exempt); each level reads the clip once through its own ear at s, N0 \
     calibrated there (calibrate.rs), prices each atom again there and drops one heard \
     nowhere, pins its frequencies and edges where they are when their Fisher precision there \
     is no finer than ADD's analysis bin (add.rs), so a level too coarse to resolve one leaves \
     it where the analysis put it, refines each atom in birth order with every other held (block coordinate \
     descent: each call lowers the cost, so a sweep does), then grows the state by ADD \
     (add.rs); numbers are Hz, s and dB, so a \
     level's state starts the next unchanged; each level is a stage whose length is its \
     residual bits plus B(x) at s = 1, every atom priced there (or at its level where s = 1 \
     hears it nowhere), so the stages share one measure, the last stage no longer than every \
     earlier the one chosen";

/// The ratio between levels' scales, exempt from the no-tuning rule.
pub const RATIO: f64 = FRAC_1_SQRT_2;

/// Levels by default, exempt: s = 2.83, 2, 1.41 and 1; solve_lm.rs recovers a frequency at s = 2
/// but not at 4.
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
}

impl Stage {
    pub fn numbers(&self) -> usize {
        self.atoms.iter().map(|a| a.values().len()).sum()
    }
}

/// What a run reports through, how it is stopped, and its clock in seconds.
pub struct Run<'a> {
    pub spl_db: f64,
    pub stopped: &'a dyn Fn() -> bool,
    hear: &'a mut dyn FnMut(Event),
    now: &'a dyn Fn() -> f64,
    started: f64,
    comparisons: u64,
    stage: String,
    part: (usize, usize),
    pub edits: Vec<String>,
    descents: Vec<(usize, Ended)>,
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

/// What a run is asked: its levels, the clip's full-scale sine in dB SPL, the threads N0 is
/// fitted on.
pub struct Asked {
    pub levels: usize,
    pub spl_db: f64,
    pub threads: NonZeroUsize,
}

/// Each atom refined in birth order with every other held.
fn refine(objective: &Objective, mut state: State, run: &mut Run) -> State {
    for i in 0..state.atoms.len() {
        if (run.stopped)() {
            break;
        }
        run.activity(
            Action::Descending,
            Some(&edit(&state.atoms[i])),
            Some((i + 1, "atom")),
        );
        let mut others = state.atoms.clone();
        let atom = others.remove(i);
        let fit = Fit::new(
            objective,
            vec![atom],
            vec![state.priced[i].clone()],
            COHERENT,
        )
        .holding(others)
        .pinning(&state.pinned[i]);
        let start = fit.numbers();
        let stopped = run.stopped;
        let Some(descent) = descend(
            &fit,
            &start,
            &fit.deltas(),
            &Limits {
                steps: None,
                stopped,
            },
        ) else {
            continue;
        };
        run.descended(objective, &descent, bits::bits(&state.atoms, &state.priced));
        state.atoms[i] = fit.atoms_at(&descent.x).remove(0);
    }
    state.length = length(objective, &state.atoms, &state.priced);
    state
}

/// The pyramid over `samples` at `rate`: one stage per level, each heard as it finishes; a
/// stop ends it after the stage it interrupts.
pub fn pyramid(
    samples: &[f64],
    rate: u32,
    asked: &Asked,
    hear: &mut dyn FnMut(Event),
    stopped: &dyn Fn() -> bool,
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
        stopped,
        hear,
        now,
        started: now(),
        comparisons: 0,
        stage: label(1, levels),
        part: (0, levels),
        edits: Vec::new(),
        descents: Vec::new(),
    };
    run.activity(Action::Reading, None, None);
    let at = |s: f64| Objective::new(samples, clip, spl, s, n0(rate, s, spl, asked.threads));
    let finest = at(1.0);
    let heard = Spectrogram::of(samples, rate);
    let mut state = State {
        atoms: Vec::new(),
        priced: Vec::new(),
        pinned: Vec::new(),
        length: f64::INFINITY,
    };
    let mut stages: Vec<Stage> = Vec::new();
    let mut least = f64::INFINITY;
    for (k, s) in scales(levels).into_iter().enumerate() {
        let began = now();
        run.stage = label(k + 1, levels);
        run.part = (k, levels);
        run.edits.clear();
        run.descents.clear();
        run.activity(Action::Reading, None, None);
        let level = (s != 1.0).then(|| at(s));
        let objective = level.as_ref().unwrap_or(&finest);
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
            atoms,
            priced: prices,
            pinned,
        };
        state = refine(objective, state, &mut run);
        state = add::add(objective, &heard, state, &mut run);
        run.activity(Action::Measuring, None, Some((1, "evaluation")));
        let residual_bits = objective.bits(&state.atoms, COHERENT);
        let bits = bits::bits(&state.atoms, &state.priced);
        let common: Vec<bits::Priced> = state
            .atoms
            .iter()
            .zip(&state.priced)
            .map(|(a, p)| priced(&finest, a, spl).unwrap_or_else(|| p.clone()))
            .collect();
        let length_bits = length(&finest, &state.atoms, &common);
        run.evaluated(objective, 1);
        run.evaluated(&finest, 1);
        let is_least = length_bits <= least;
        least = least.min(length_bits);
        let stage = Stage {
            label: run.stage.clone(),
            scale: s,
            atoms: state.atoms.clone(),
            residual_bits,
            bits,
            length_bits,
            least: is_least,
            stopped: stopped(),
            seconds: now() - began,
            edits: run.edits.clone(),
            descents: run.descents.clone(),
            files: files(&state.atoms, &clip),
        };
        (run.hear)(Event::Finished(&stage));
        stages.push(stage);
        if stopped() {
            break;
        }
    }
    stages
}

/// The last stage shorter than every earlier.
pub fn chosen(stages: &[Stage]) -> Option<&Stage> {
    stages.iter().rfind(|s| s.least)
}
