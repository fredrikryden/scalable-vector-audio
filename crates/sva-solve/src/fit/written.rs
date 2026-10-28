// Concern: a sound written as a source from its gains: its level curves thinned, its gains as written, its price | Non-concern: rendering it (render.rs) | IO: (Sound, gains or levels) -> Written

use sva_engine::hearing::level_jnd_db;

use super::problem::Problem;
use super::render::{Columns, Proposal, centre, db};
use crate::analysis::floor_db;
use crate::bits::{self, Look, Priced, frequency_delta, level_delta};
use crate::optimize::code::Price;
use crate::sources::{Feasible, Link, Quantity, Sound, Source, Trajectory};

/// A source written from its gains: the source, its glide's precision, each column's written
/// gain and level in dB per frame, and its price.
#[derive(Clone)]
pub(super) struct Written {
    pub(super) source: Source,
    pub(super) step: f64,
    pub(super) gains: Vec<Vec<f64>>,
    pub(super) levels: Vec<Vec<f64>>,
    pub(super) price: Price,
}

/// `proposal` written from `fitted`, each column's own gain per frame, as `MODEL` states.
pub(super) fn write(
    p: &Problem,
    (proposal, step): (&Proposal, f64),
    key: u64,
    fitted: &[Vec<f64>],
    links: &[Link],
) -> Written {
    let secs = p.clip.secs();
    let columns = p.columns(proposal);
    let trajectories: Vec<Trajectory> = fitted
        .iter()
        .zip(&columns.floors)
        .zip(&columns.home)
        .map(|((g, floor), home)| {
            let mut held: Vec<(f64, f64)> = home
                .iter()
                .map(|f| (p.times[*f], db(g[*f]).max(*floor)))
                .collect();
            held.sort_by(|a, b| a.0.total_cmp(&b.0));
            let peak = held.iter().map(|h| h.1).fold(f64::NEG_INFINITY, f64::max);
            thinned((0.0, secs), &held, level_jnd_db(peak - floor))
        })
        .collect();
    rewrite(p, (proposal, step), key, trajectories, links)
}

pub(super) fn rewrite(
    p: &Problem,
    (proposal, step): (&Proposal, f64),
    key: u64,
    trajectories: Vec<Trajectory>,
    links: &[Link],
) -> Written {
    let columns = p.columns(proposal);
    let levels: Vec<Vec<f64>> = trajectories.iter().map(|t| p.at_frames(t)).collect();
    let gains = levels
        .iter()
        .map(|l| l.iter().map(|l| libm::pow(10.0, l / 10.0)).collect())
        .collect();
    let source = Source {
        links: links.to_vec(),
        ..Source::new(key, 0.0, proposal.sound.clone(), trajectories)
    };
    let price = match &proposal.sound {
        Sound::Shaped { bands } => shaped_price(p, &source, bands, &levels[0], &columns),
        _ => priced(p, &source, &columns, &levels, step),
    };
    Written {
        source,
        step,
        gains,
        levels,
        price,
    }
}

impl Problem {
    pub(super) fn looks(
        &self,
        frames: &[usize],
        level: &[f64],
        floor: f64,
        hz: &[f64],
    ) -> Vec<Look> {
        frames
            .iter()
            .map(|f| Look {
                sl_db: level[*f] - floor,
                hz: hz[*f],
                secs: self.view(*f).grid.frame_secs(),
            })
            .collect()
    }

    fn finest(&self, frames: &[usize]) -> f64 {
        frames
            .iter()
            .map(|f| self.view(*f).grid.frame_secs())
            .fold(f64::INFINITY, f64::min)
    }
}

fn edge_step(p: &Problem, feasible: &Feasible) -> f64 {
    (feasible.hz.1 - feasible.hz.0) / (p.bands.len() + 1) as f64
}

/// A shaped noise's price at the precisions `MODEL` states, its envelope at `level` per frame.
fn shaped_price(
    p: &Problem,
    source: &Source,
    bands: &[(f64, f64, f64)],
    level: &[f64],
    columns: &Columns,
) -> Price {
    let (floor, home) = (columns.floors[0], &columns.home[0]);
    let hop = p.finest(home);
    let feasible = Feasible::of(&p.clip, (floor, p.top_db));
    let edge = edge_step(p, &feasible);
    let mut deltas = vec![hop];
    for (lo, hi, g) in bands {
        let heard: Vec<f64> = level.iter().map(|l| l + g).collect();
        let hz = vec![centre((*lo, *hi)); level.len()];
        let met: Vec<usize> = (0..p.frames())
            .filter(|f| p.view(*f).meets((*lo, *hi)))
            .collect();
        let at = p.looks(&met, &heard, floor_db(centre((*lo, *hi)), p.spl_db), &hz);
        deltas.extend([edge, edge, level_delta(&at)]);
    }
    let hz = vec![centre((bands[0].0, bands[0].1)); level.len()];
    let envelope = level_delta(&p.looks(home, level, floor, &hz));
    for (q, _) in &source.numbers()[deltas.len()..] {
        deltas.push(match q {
            Quantity::Time => hop,
            _ => envelope,
        });
    }
    bits::price(source, &Priced { feasible, deltas })
}

/// `source`'s price at the precisions `MODEL` states, harmonic `k` at `levels[k]` per frame
/// over its column's floor, its glide at `glide_step`.
fn priced(
    p: &Problem,
    source: &Source,
    columns: &Columns,
    levels: &[Vec<f64>],
    glide_step: f64,
) -> Price {
    let looks = |k: usize| {
        let at = (&levels[k], columns.floors[k], &columns.paths[k]);
        p.looks(&columns.home[k], at.0, at.1, at.2)
    };
    let hops: Vec<f64> = columns.home.iter().map(|h| p.finest(h)).collect();
    let lowest = columns.floors.iter().copied().fold(f64::INFINITY, f64::min);
    let feasible = Feasible::of(&p.clip, (lowest, p.top_db));
    let level_deltas: Vec<f64> = (0..columns.heard.len())
        .map(|k| level_delta(&looks(k)))
        .collect();
    let mut steps = None;
    let mut steps = || *steps.get_or_insert_with(|| frequency_steps(p, columns, levels));
    let mut level = 0;
    let mut deltas = Vec::new();
    for (q, _) in source.numbers() {
        if q == Quantity::Db {
            level += 1;
        }
        let member = level % columns.heard.len();
        deltas.push(match q {
            Quantity::Time => hops[member],
            Quantity::Db | Quantity::DbDeviation => level_deltas[member],
            Quantity::Hz => steps().0,
            Quantity::Stretch => steps().1,
            Quantity::Glide => glide_step,
            Quantity::Edge => edge_step(p, &feasible),
            Quantity::Depth => level_deltas[0],
        });
    }
    bits::price(source, &Priced { feasible, deltas })
}

/// The precisions a fundamental's Hz and its stretch are written at, as `MODEL` states,
/// harmonic `k` at `levels[k]` per frame over its column's floor.
pub(super) fn frequency_steps(p: &Problem, columns: &Columns, levels: &[Vec<f64>]) -> (f64, f64) {
    let hzs = &columns.heard;
    let looks = |k: usize| {
        let at = (&levels[k], columns.floors[k], &columns.paths[k]);
        p.looks(&columns.home[k], at.0, at.1, at.2)
    };
    let last = hzs.len() - 1;
    let (f0, h, top) = (hzs[0], hzs.len() as f64, hzs[last]);
    let moved = h * h * h * h * f0 * f0 / (2.0 * top);
    (
        frequency_delta(&looks(0)) * columns.starts[0] / f0,
        frequency_delta(&looks(last)) / moved,
    )
}

/// A level from `on` to `end` through `held`, each a time and level in time order, the first
/// held back to `on` and the last on to `end`, thinned by Ramer-Douglas-Peucker at `tolerance`
/// dB, its knots' times from `on`.
fn thinned((on, end): (f64, f64), held: &[(f64, f64)], tolerance: f64) -> Trajectory {
    let mut points = vec![(on, held[0].1)];
    points.extend(held.iter().filter(|(t, _)| on < *t && *t < end).copied());
    points.push((end, held[held.len() - 1].1));
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
        knots: kept[1..].iter().map(|(t, l)| (t - on, l - base)).collect(),
    }
}

/// The level in dB `level` holds at each of the ascending `times`, its knots' times from 0:
/// straight lines from `base` through each knot, held after the last, its knots walked once.
pub(super) fn levels_at(level: &Trajectory, times: &[f64]) -> Vec<f64> {
    let mut from = (0.0, level.base);
    let mut knots = level
        .knots
        .iter()
        .map(|(at, d)| (*at, level.base + d))
        .peekable();
    times
        .iter()
        .map(|t| {
            while let Some(to) = knots.peek().copied() {
                if *t <= to.0 {
                    return match to.0 > from.0 {
                        true => from.1 + (to.1 - from.1) * (t - from.0) / (to.0 - from.0),
                        false => to.1,
                    };
                }
                from = to;
                knots.next();
            }
            from.1
        })
        .collect()
}

pub(super) fn level_db(p: &Problem, written: &Written) -> f64 {
    let at = p.reported().frames.clone();
    let frames = at.len().max(1) as f64;
    let shape = match &written.source.sound {
        Sound::Shaped { bands } => bands.iter().map(|(_, _, g)| super::render::power(*g)).sum(),
        _ => 1.0,
    };
    let total: f64 = written.gains.iter().flat_map(|g| &g[at.clone()]).sum();
    db(shape * total / frames)
}
