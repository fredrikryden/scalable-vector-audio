// Concern: a pass as its progress line reports it: its move, the DL's parts, the residual's grid, the sources it leaves | Non-concern: the line's JSON (report.rs) | IO: (State, State) -> Line

use super::problem::{Problem, State};
use super::render::{Held, sources};
use super::shared::active;
use super::written::level_db;
use crate::analysis::floor_db;
use crate::optimize::Edit;
use crate::sources::{Clip, Shared, Source};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    Add,
    Adjust,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Action::Start => "start",
            Action::Add => "add",
            Action::Adjust => "adjust",
        }
    }
}

/// What an adjust changed, as `MODEL` states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tweak {
    Level,
    Remove,
    Frequency,
    Rethin,
}

impl Tweak {
    pub const ALL: [Tweak; 4] = [Tweak::Level, Tweak::Remove, Tweak::Frequency, Tweak::Rethin];

    pub fn name(self) -> &'static str {
        match self {
            Tweak::Level => "level",
            Tweak::Remove => "remove",
            Tweak::Frequency => "frequency",
            Tweak::Rethin => "rethin",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub rate_hz: u32,
    pub frame_secs: f64,
    pub first_centre_secs: f64,
    pub band_edges_hz: Vec<(f64, f64)>,
    pub band_floor_db: Vec<f64>,
    pub gain_db: f64,
}

/// One pass as a progress line reports it: its move, the DL's change and total, the residual's
/// band levels per frame (`None` where the fit covers the clip), each kept source's written
/// level per frame, and the sources it leaves on its clip.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    /// The pass, from 1; 0 for the start.
    pub pass: usize,
    pub action: Action,
    pub tweak: Option<Tweak>,
    pub source: Option<Source>,
    /// The source's level in dB before and after, as `level_db` states; `None` where it was
    /// not, or is no more.
    pub from_db: Option<f64>,
    pub to_db: Option<f64>,
    pub delta_bits: f64,
    pub source_bits: f64,
    pub residual_bits: f64,
    pub residual_db: Vec<Vec<Option<f64>>>,
    pub envelopes: Vec<(u64, Vec<Vec<f64>>)>,
    pub geometry: Option<Geometry>,
    pub seconds: f64,
    pub pass_secs: f64,
    /// The empty fit's DL.
    pub start_bits: f64,
    pub sources: Vec<Source>,
    pub shared: Vec<(usize, Shared)>,
    pub clip: Clip,
    /// The sources of the moves better than its own it skipped, their nodes refused.
    pub skipped: Vec<Source>,
}

pub(super) fn line(p: &Problem, state: &State, before: &State, start_bits: f64) -> Line {
    let grid = &p.grid;
    let residual_db = p
        .residual(state.kept(), state.shared())
        .iter()
        .map(|r| {
            let mut full = vec![0.0; grid.bins()];
            for (i, r) in r.iter().enumerate() {
                full[p.range.start + i] = r.max(0.0);
            }
            grid.pooled(&full)
                .iter()
                .map(|v| (*v > 0.0).then(|| grid.band_db(*v)))
                .collect()
        })
        .collect();
    let level = |h: &Held| Some(level_db(&h.written));
    let source = |h: &Held| Some(h.written.source.clone());
    let (action, tweak, source, from_db, to_db) = match state.moved() {
        None => (Action::Start, None, None, None, None),
        Some(Edit::Add(())) => {
            let added = state.kept().last().expect("an added source");
            (Action::Add, None, source(added), None, level(added))
        }
        Some(Edit::Replace(a, (), tweak)) => {
            let (from, to) = (&before.kept()[a], &state.kept()[a]);
            (
                Action::Adjust,
                Some(tweak),
                source(to),
                level(from),
                level(to),
            )
        }
        Some(Edit::Remove(a, tweak)) => {
            let from = &before.kept()[a];
            (Action::Adjust, Some(tweak), source(from), level(from), None)
        }
    };
    Line {
        pass: state.passes(),
        action,
        tweak,
        source,
        from_db,
        to_db,
        delta_bits: state.dl() - before.dl(),
        source_bits: state.thing_bits(),
        residual_bits: state.missed_bits(),
        residual_db,
        envelopes: state
            .kept()
            .iter()
            .map(|h| (h.key(), h.written.levels.clone()))
            .collect(),
        geometry: None,
        seconds: 0.0,
        pass_secs: 0.0,
        start_bits,
        sources: sources(state.kept()),
        shared: active(state),
        clip: p.clip,
        skipped: sources(state.skipped()),
    }
}
pub(super) fn first_line(p: &Problem, state: &State, start_bits: f64, gain_db: f64) -> Line {
    let mut first = line(p, state, state, start_bits);
    first.geometry = Some(Geometry {
        rate_hz: p.clip.rate,
        frame_secs: p.grid.frame_secs(),
        first_centre_secs: p.grid.centre_secs(0),
        band_edges_hz: p.grid.bands(),
        band_floor_db: p
            .grid
            .band_centres()
            .iter()
            .map(|hz| floor_db(*hz, p.spl_db))
            .collect(),
        gain_db,
    });
    first
}
