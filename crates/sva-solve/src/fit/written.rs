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
    let floors = p.columns(proposal).floors;
    let trajectories: Vec<Trajectory> = fitted
        .iter()
        .zip(&floors)
        .map(|(g, floor)| {
            let held: Vec<f64> = g.iter().map(|g| db(*g).max(*floor)).collect();
            let peak = held.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            thinned((0.0, secs), &p.times, &held, level_jnd_db(peak - floor))
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
    let levels: Vec<Vec<f64>> = trajectories
        .iter()
        .map(|t| levels_at(t, &p.times))
        .collect();
    let gains = levels
        .iter()
        .map(|l| l.iter().map(|l| libm::pow(10.0, l / 10.0)).collect())
        .collect();
    let source = Source {
        links: links.to_vec(),
        ..Source::new(key, 0.0, proposal.sound.clone(), trajectories)
    };
    let price = match &proposal.sound {
        Sound::Shaped { bands } => shaped_price(p, &source, bands, &levels[0], columns.floors[0]),
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
    pub(super) fn looks(&self, level: &[f64], floor: f64, hz: &[f64]) -> Vec<Look> {
        level
            .iter()
            .zip(hz)
            .map(|(l, hz)| Look {
                sl_db: l - floor,
                hz: *hz,
            })
            .collect()
    }
}

fn edge_step(p: &Problem, feasible: &Feasible) -> f64 {
    (feasible.hz.1 - feasible.hz.0) / (p.grid.bands().len() + 1) as f64
}

/// A shaped noise's price at the precisions `MODEL` states, its envelope at `level` per frame
/// over `floor`.
fn shaped_price(
    p: &Problem,
    source: &Source,
    bands: &[(f64, f64, f64)],
    level: &[f64],
    floor: f64,
) -> Price {
    let hop = p.grid.frame_secs();
    let feasible = Feasible::of(&p.clip, (floor, p.top_db));
    let edge = edge_step(p, &feasible);
    let mut deltas = vec![hop];
    for (lo, hi, g) in bands {
        let heard: Vec<f64> = level.iter().map(|l| l + g).collect();
        let hz = vec![centre((*lo, *hi)); level.len()];
        let at = p.looks(&heard, floor_db(centre((*lo, *hi)), p.spl_db), &hz);
        deltas.extend([edge, edge, level_delta(hop, &at)]);
    }
    let hz = vec![centre((bands[0].0, bands[0].1)); level.len()];
    let envelope = level_delta(hop, &p.looks(level, floor, &hz));
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
    let hop = p.grid.frame_secs();
    let looks = |k: usize| p.looks(&levels[k], columns.floors[k], &columns.paths[k]);
    let lowest = columns.floors.iter().copied().fold(f64::INFINITY, f64::min);
    let feasible = Feasible::of(&p.clip, (lowest, p.top_db));
    let level_deltas: Vec<f64> = (0..columns.heard.len())
        .map(|k| level_delta(hop, &looks(k)))
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
            Quantity::Time => hop,
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
    let hop = p.grid.frame_secs();
    let hzs = &columns.heard;
    let looks = |k: usize| p.looks(&levels[k], columns.floors[k], &columns.paths[k]);
    let last = hzs.len() - 1;
    let (f0, h, top) = (hzs[0], hzs.len() as f64, hzs[last]);
    let moved = h * h * h * h * f0 * f0 / (2.0 * top);
    (
        frequency_delta(hop, &looks(0)) * columns.starts[0] / f0,
        frequency_delta(hop, &looks(last)) / moved,
    )
}

/// A level from `on` to `end` through `levels` at `times`, the first held back to `on` and
/// the last on to `end`, thinned by Ramer-Douglas-Peucker at `tolerance` dB, its knots' times
/// from `on`.
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

pub(super) fn level_db(written: &Written) -> f64 {
    let frames = written.gains[0].len().max(1) as f64;
    let shape = match &written.source.sound {
        Sound::Shaped { bands } => bands.iter().map(|(_, _, g)| super::render::power(*g)).sum(),
        _ => 1.0,
    };
    let total: f64 = written.gains.iter().flatten().sum();
    db(shape * total / frames)
}
