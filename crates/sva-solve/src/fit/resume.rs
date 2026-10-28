// Concern: the fit a composition holds, laid on this clip as the state a fit goes on from, or why it is no fit of it | Non-concern: reading the files (sources.rs) | IO: (files, passes) -> State

use sva_core::CliError;

use super::adds::rate_floor;
use super::problem::{Problem, State};
use super::render::{Held, Proposal};
use super::written::rewrite;
use crate::optimize::{self, Problem as _};
use crate::sources::{self, Shared, Sound, Source, birth};

/// The sources a fit resumes from, the passes that made them, and their master gain if written.
#[derive(Clone, Debug, PartialEq)]
pub struct Resume {
    pub sources: Vec<Source>,
    /// The shared nodes its sources reference, by index.
    pub shared: Vec<(usize, Shared)>,
    pub passes: usize,
    pub gain_db: Option<f64>,
}

impl Resume {
    /// The sources a composition's `files` hold after `passes` passes.
    pub fn of(files: &[(String, String)], passes: usize) -> Result<Resume, CliError> {
        let read = sources::read(files).ok_or_else(|| {
            CliError::usage(
                "solve.from_unread",
                format!(
                    "the composition to resume holds no `{}` of sources a fit wrote",
                    sources::SCORE
                ),
            )
        })?;
        let shared = sources::shared(files, &read).ok_or_else(|| {
            CliError::usage(
                "solve.from_unread",
                "the composition to resume lacks a shared node its sources reference",
            )
        })?;
        Ok(Resume {
            sources: read,
            shared,
            passes,
            gain_db: sources::gain_of(files),
        })
    }
}

/// Whether `sound` is one a fit writes: its Hz, stretch and bands positive.
fn written(sound: &Sound) -> bool {
    match sound {
        Sound::Partial { hz, .. } => *hz > 0.0,
        Sound::Harmonic { hz, stretch, .. } => *hz > 0.0 && *stretch >= 0.0,
        Sound::Noise { lo, hi } => 0.0 < *lo && lo < hi,
        Sound::Shaped { bands } => {
            !bands.is_empty() && bands.iter().all(|(lo, hi, _)| 0.0 < *lo && lo < hi)
        }
    }
}

/// The state `from` leaves on `p`, or why it is no fit of this clip.
pub(super) fn resumed(p: &Problem, from: &Resume, gain_db: f64) -> Result<State, CliError> {
    let unread = |why: &str| {
        CliError::usage(
            "solve.from_unread",
            format!("the composition to resume holds {why}, which no fit writes"),
        )
    };
    let mismatch = |why: String| {
        CliError::usage(
            "solve.from_mismatch",
            format!("the composition to resume was fitted to another clip: {why}"),
        )
    };
    if let Some(written) = from.gain_db.filter(|g| *g != gain_db) {
        return Err(mismatch(format!(
            "its master gain is {written} dB, this clip's {gain_db} dB"
        )));
    }
    let secs = p.clip.secs();
    let spares = p.spares();
    let top = from.shared.iter().map(|(to, _)| to + 1).max().unwrap_or(0);
    let mut shared: Vec<optimize::Shared> = (0..top).map(|_| spares[0].clone()).collect();
    for (to, node) in &from.shared {
        shared[*to].kind = node.kind as usize;
        shared[*to].numbers = node.numbers.clone();
    }
    let mut held = Vec::new();
    for source in &from.sources {
        if source.on != 0.0 || source.end() != secs {
            return Err(mismatch(format!(
                "source {} spans {}s to {}s, this clip 0s to {secs}s",
                source.key,
                source.on,
                source.end()
            )));
        }
        if from.sources.iter().filter(|a| a.key == source.key).count() > 1 {
            return Err(unread("two sources of one key"));
        }
        if !source.values().iter().all(|v| v.is_finite()) {
            return Err(unread("a number that is not finite"));
        }
        if !written(&source.sound) {
            return Err(unread("a source of a shape"));
        }
        let proposal = Proposal::of(source);
        let step = match source.sound.glide() != 0.0 {
            true => {
                let levels: Vec<Vec<f64>> = source.levels.iter().map(|t| p.at_frames(t)).collect();
                rate_floor(p, &p.columns(&proposal), &levels)
            }
            false => 0.0,
        };
        let at = (&proposal, step);
        let written = rewrite(p, at, source.key, source.levels.clone(), &source.links);
        held.push(Held::new(p, p.templates(&proposal), written));
    }
    Ok(State::of(
        p,
        held,
        shared,
        birth(&from.sources),
        from.passes,
    ))
}
