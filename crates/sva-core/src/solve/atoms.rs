// Concern: the three atom kinds solve writes: numbers, ranges, the feasible set and its projection, rows, nodes, keys | Non-concern: what the ear reads of one | IO: (Atom) -> numbers, row, node, Atom

use std::collections::HashMap;
use std::f64::consts::{LN_10, TAU};

use sva_ast::score::Item;
use sva_ast::{Expr, Literal};
use sva_engine::hearing::{CAM_FIRST, CAM_LAST, cam_hz};
use sva_formula::Num;

use super::dual::Dual;
use crate::json::num;

/// Every node writes these bits, so the model reads the same.
pub const NEPER_PER_DB: f64 = LN_10 / 20.0;

pub const DIRECTORY: &str = "atoms";

/// A full-scale sine (AES17).
pub const FULL_SCALE_DBFS: f64 = 0.0;

/// Its duration is the noise period.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clip {
    pub rate: u32,
    pub len: usize,
}

impl Clip {
    pub fn secs(&self) -> f64 {
        self.len as f64 / f64::from(self.rate)
    }
}

/// `base` at the atom's onset, then straight lines through each knot's time and deviation from
/// `base`, held after the last.
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory<N = f64> {
    pub base: N,
    pub knots: Vec<(N, N)>,
}

impl Trajectory {
    pub fn steady(base: f64) -> Trajectory {
        Trajectory {
            base,
            knots: Vec::new(),
        }
    }
}

impl<N: Num> Trajectory<N> {
    /// As a node folds `base + deviation`.
    pub fn values(&self) -> impl Iterator<Item = N> + '_ {
        std::iter::once(self.base).chain(self.knots.iter().map(|(_, d)| self.base + *d))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Sound<N = f64> {
    Partial {
        hz: Trajectory<N>,
    },
    /// Harmonic `k` turns `k` times as fast as `hz`; each but the last is at `offsets[k - 1]` dB,
    /// the last at minus their sum, so the level is the harmonics' mean.
    Harmonic {
        hz: Trajectory<N>,
        offsets: Vec<N>,
    },
    Noise {
        lo: N,
        hi: N,
    },
}

/// A level in dBFS whose last knot ends the atom; a noise's is its band's power, density times
/// `hi - lo`. `key` is its ordinal, every phase and the noise seed drawn from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Atom<N = f64> {
    pub key: u64,
    pub on: N,
    pub sound: Sound<N>,
    pub level: Trajectory<N>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantity {
    Time,
    Hz,
    HzDeviation,
    Db,
    DbDeviation,
    Offset,
    Edge,
}

/// Line `k`'s phase, the draw noise keyed `key` gives its own line `k`, so no number is written.
pub fn phase(key: u64, k: usize) -> f64 {
    TAU * sva_formula::draw_nearest(key, k as f64).expect("a line index is a whole step")
}

/// The ordinal a birth takes: one past the largest living, the first 0.
pub fn birth(atoms: &[Atom]) -> u64 {
    atoms.iter().map(|a| a.key + 1).max().unwrap_or(0)
}

impl<N: Num> Atom<N> {
    pub fn end(&self) -> N {
        self.level.knots.last().expect("a level ends on a knot").0
    }

    pub fn hz(&self) -> Option<&Trajectory<N>> {
        match &self.sound {
            Sound::Partial { hz } | Sound::Harmonic { hz, .. } => Some(hz),
            Sound::Noise { .. } => None,
        }
    }

    pub fn harmonics(&self) -> usize {
        match &self.sound {
            Sound::Harmonic { offsets, .. } => offsets.len() + 1,
            _ => 1,
        }
    }

    /// Every number but the key, in the order `with` reads.
    pub fn values(&self) -> Vec<N> {
        let mut out = vec![self.on];
        let knots = |out: &mut Vec<N>, t: &Trajectory<N>| {
            out.push(t.base);
            for (at, d) in &t.knots {
                out.push(*at);
                out.push(*d);
            }
        };
        match &self.sound {
            Sound::Partial { hz } => knots(&mut out, hz),
            Sound::Harmonic { hz, offsets } => {
                knots(&mut out, hz);
                out.extend(offsets);
            }
            Sound::Noise { lo, hi } => out.extend([*lo, *hi]),
        }
        knots(&mut out, &self.level);
        out
    }

    /// The same atom with `values` in `values()`' order.
    pub fn lifted<M: Num>(&self, values: &[M]) -> Atom<M> {
        assert_eq!(values.len(), self.values().len(), "one value per number");
        let mut next = values.iter().copied();
        let mut take = || next.next().expect("counted");
        let retake = |t: &Trajectory<N>, take: &mut dyn FnMut() -> M| Trajectory {
            base: take(),
            knots: t.knots.iter().map(|_| (take(), take())).collect(),
        };
        let on = take();
        let sound = match &self.sound {
            Sound::Partial { hz } => Sound::Partial {
                hz: retake(hz, &mut take),
            },
            Sound::Harmonic { hz, offsets } => Sound::Harmonic {
                hz: retake(hz, &mut take),
                offsets: offsets.iter().map(|_| take()).collect(),
            },
            Sound::Noise { .. } => Sound::Noise {
                lo: take(),
                hi: take(),
            },
        };
        let level = retake(&self.level, &mut take);
        Atom {
            key: self.key,
            on,
            sound,
            level,
        }
    }
}

impl Atom {
    pub fn partial(key: u64, on: f64, hz: Trajectory, level: Trajectory) -> Atom {
        Atom {
            key,
            on,
            sound: Sound::Partial { hz },
            level,
        }
    }

    /// `offsets` for every harmonic but the last.
    pub fn harmonic(
        key: u64,
        on: f64,
        hz: Trajectory,
        offsets: Vec<f64>,
        level: Trajectory,
    ) -> Atom {
        Atom {
            key,
            on,
            sound: Sound::Harmonic { hz, offsets },
            level,
        }
    }

    pub fn noise(key: u64, on: f64, (lo, hi): (f64, f64), level: Trajectory) -> Atom {
        Atom {
            key,
            on,
            sound: Sound::Noise { lo, hi },
            level,
        }
    }

    /// Each number beside its quantity, in `values()`' order.
    pub fn numbers(&self) -> Vec<(Quantity, f64)> {
        let mut out = vec![Quantity::Time];
        let knots = |out: &mut Vec<_>, t: &Trajectory, (base, dev)| {
            out.push(base);
            for _ in &t.knots {
                out.push(Quantity::Time);
                out.push(dev);
            }
        };
        match &self.sound {
            Sound::Partial { hz } => knots(&mut out, hz, (Quantity::Hz, Quantity::HzDeviation)),
            Sound::Harmonic { hz, offsets } => {
                knots(&mut out, hz, (Quantity::Hz, Quantity::HzDeviation));
                out.extend(offsets.iter().map(|_| Quantity::Offset));
            }
            Sound::Noise { .. } => out.extend([Quantity::Edge, Quantity::Edge]),
        }
        knots(&mut out, &self.level, (Quantity::Db, Quantity::DbDeviation));
        out.into_iter().zip(self.values()).collect()
    }

    /// Every number constant but number `i`, which moves at unit speed.
    pub fn seeded(&self, i: usize) -> Atom<Dual> {
        let values: Vec<Dual> = self
            .values()
            .iter()
            .enumerate()
            .map(|(j, v)| Dual::new(*v, f64::from(u8::from(i == j))))
            .collect();
        self.lifted(&values)
    }

    /// One node per kind and knot count.
    pub fn node(&self) -> String {
        let (f, l) = (
            self.hz().map_or(0, |t| t.knots.len()),
            self.level.knots.len(),
        );
        match &self.sound {
            Sound::Partial { .. } => format!("{DIRECTORY}/partial_f{f}_l{l}"),
            Sound::Harmonic { .. } => {
                format!("{DIRECTORY}/harmonic_h{}_f{f}_l{l}", self.harmonics())
            }
            Sound::Noise { .. } => format!("{DIRECTORY}/noise_l{l}"),
        }
    }

    pub fn node_text(&self, clip: &Clip) -> String {
        let l = self.level.knots.len();
        let end = format!("lt{l}");
        let neper = num(NEPER_PER_DB);
        let level = format!(
            "exp({neper}*{})",
            table("pwl", "level", &knot_names("lt", "ld", "level", l), None)
        );
        let turning = |f: usize| {
            table(
                "pwl_integral",
                "hz",
                &knot_names("ft", "fd", "hz", f),
                Some(&end),
            )
        };
        let line = |k: usize, phi: &str| format!("sin(2*pi*{k}*{phi} + 2*pi*draw({k}, seed=key))");
        let (what, body) = match &self.sound {
            Sound::Partial { hz } => (
                format!(
                    "one partial, its frequency through {} knots",
                    hz.knots.len()
                ),
                format!("{} * {level}", line(1, &turning(hz.knots.len()))),
            ),
            Sound::Harmonic { hz, offsets } => {
                let phi = turning(hz.knots.len());
                let written: Vec<String> = (1..=offsets.len()).map(|k| format!("o{k}")).collect();
                let mut lines: Vec<String> = written
                    .iter()
                    .enumerate()
                    .map(|(k, o)| format!("exp({neper}*{o})*{}", line(k + 1, &phi)))
                    .collect();
                let last = line(self.harmonics(), &phi);
                lines.push(match written.is_empty() {
                    true => last,
                    false => format!("exp({neper}*(-({})))*{last}", written.join(" + ")),
                });
                (
                    format!(
                        "{} harmonics of one fundamental through {} knots, the last at minus the \
                         others' offsets",
                        self.harmonics(),
                        hz.knots.len()
                    ),
                    format!("{level} * ({})", lines.join(" + ")),
                )
            }
            Sound::Noise { .. } => (
                "one band of noise flat in Hz, its level the band's power".to_string(),
                format!(
                    "noise(key, period={}s, lo=lo, hi=hi) * {level} / sqrt(hi - lo)",
                    num(clip.secs())
                ),
            ),
        };
        format!(
            "; Models: {what}, its level in dBFS through {l} knots, the last ending it | Neglects: \
             every other atom, each its own row | IO: (t) -> amplitude | Tags: atom\n\
             crop({body}, on, {end})\n"
        )
    }

    /// Played from the clip's start, its node one directory up.
    pub fn row(&self) -> String {
        let mut fields = vec![
            ("at".to_string(), "\"0s\"".to_string()),
            ("node".to_string(), format!("\"@../{}\"", self.node())),
            ("key".to_string(), self.key.to_string()),
            ("on".to_string(), num(self.on)),
        ];
        let knots = |fields: &mut Vec<(String, String)>,
                     t: &Trajectory,
                     (base, time, dev): (&str, &str, &str)| {
            fields.push((base.to_string(), num(t.base)));
            for (i, (at, d)) in t.knots.iter().enumerate() {
                fields.push((format!("{time}{}", i + 1), num(*at)));
                fields.push((format!("{dev}{}", i + 1), num(*d)));
            }
        };
        match &self.sound {
            Sound::Partial { hz } => knots(&mut fields, hz, ("hz", "ft", "fd")),
            Sound::Harmonic { hz, offsets } => {
                knots(&mut fields, hz, ("hz", "ft", "fd"));
                for (k, o) in offsets.iter().enumerate() {
                    fields.push((format!("o{}", k + 1), num(*o)));
                }
            }
            Sound::Noise { lo, hi } => {
                fields.push(("lo".to_string(), num(*lo)));
                fields.push(("hi".to_string(), num(*hi)));
            }
        }
        knots(&mut fields, &self.level, ("level", "lt", "ld"));
        let written: Vec<String> = fields
            .iter()
            .map(|(k, v)| format!("\"{k}\": {v}"))
            .collect();
        format!("{{{}}}", written.join(", "))
    }

    /// The atom a row `row` writes reads back, from its node's kind and counts and its numbers.
    fn read(node: &str, numbers: &HashMap<&str, f64>) -> Option<Atom> {
        let name = node.strip_prefix(&format!("../{DIRECTORY}/"))?;
        let mut counts = HashMap::new();
        let mut parts = name.split('_');
        let kind = parts.next()?;
        for part in parts {
            let (letter, count) = part.split_at_checked(1)?;
            counts.insert(letter, count.parse::<usize>().ok()?);
        }
        let get = |name: &str| numbers.get(name).copied();
        let trajectory = |base: &str, (time, dev): (&str, &str), count: usize| {
            Some(Trajectory {
                base: get(base)?,
                knots: (1..=count)
                    .map(|i| Some((get(&format!("{time}{i}"))?, get(&format!("{dev}{i}"))?)))
                    .collect::<Option<_>>()?,
            })
        };
        let key = get("key")?;
        let key =
            (key >= 0.0 && key.fract() == 0.0 && key < u64::MAX as f64).then_some(key as u64)?;
        let level = trajectory("level", ("lt", "ld"), *counts.get("l")?)?;
        let hz = || trajectory("hz", ("ft", "fd"), *counts.get("f")?);
        let sound = match kind {
            "partial" => Sound::Partial { hz: hz()? },
            "harmonic" => Sound::Harmonic {
                hz: hz()?,
                offsets: (1..*counts.get("h")?)
                    .map(|k| get(&format!("o{k}")))
                    .collect::<Option<_>>()?,
            },
            "noise" => Sound::Noise {
                lo: get("lo")?,
                hi: get("hi")?,
            },
            _ => return None,
        };
        let atom = Atom {
            key,
            on: get("on")?,
            sound,
            level,
        };
        (atom.values().len() + 1 == numbers.len()).then_some(atom)
    }
}

fn knot_names(time: &str, dev: &str, base: &str, count: usize) -> Vec<(String, String)> {
    (1..=count)
        .map(|k| (format!("{time}{k}"), format!("{base} + {dev}{k}")))
        .collect()
}

/// `hold` repeats the last value to there, as `pwl_integral` needs to keep turning.
fn table(name: &str, base: &str, knots: &[(String, String)], hold: Option<&str>) -> String {
    let mut args = vec!["t".to_string(), "on".to_string(), base.to_string()];
    for (at, value) in knots {
        args.push(at.clone());
        args.push(value.clone());
    }
    if let Some(hold) = hold {
        let (last_at, last) = knots
            .last()
            .map_or(("on".to_string(), base.to_string()), |(t, v)| {
                (t.clone(), v.clone())
            });
        args.push(match knots.is_empty() {
            true => hold.to_string(),
            false => format!("max({last_at}, {hold})"),
        });
        args.push(last);
    }
    format!("{name}({})", args.join(", "))
}

/// One row each in birth order, which is their keys' order.
pub fn score(atoms: &[Atom]) -> String {
    let mut out = String::from(
        "{\"annotation\": \"Models: the clip's atoms, one row each in birth order | Neglects: \
         what no atom explains | IO: (t) -> amplitude | Tags: score\"}\n",
    );
    let mut born: Vec<&Atom> = atoms.iter().collect();
    born.sort_by_key(|a| a.key);
    for atom in born {
        out.push_str(&atom.row());
        out.push('\n');
    }
    out
}

/// The atoms a score `score` wrote, `None` where it holds anything else.
pub fn read(text: &str) -> Option<Vec<Atom>> {
    let held = sva_ast::parse_score(text).ok()?;
    let mut out = Vec::new();
    for item in held.lines.iter().flat_map(|l| &l.items) {
        let Item::Event(event) = item else {
            return None;
        };
        let mut numbers = HashMap::new();
        for bind in &event.binds {
            let Expr::Lit(Literal::Num(v)) = &bind.value else {
                return None;
            };
            numbers.insert(bind.name.as_str(), *v);
        }
        out.push(Atom::read(&event.node.name, &numbers)?);
    }
    Some(out)
}

pub fn nodes(atoms: &[Atom], clip: &Clip) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = atoms
        .iter()
        .map(|a| (a.node(), a.node_text(clip)))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Hz the reading spans under the Nyquist, times in the clip, levels from the floor to full
/// scale; an atom at least `span`, a sample, long; a noise band at least `spacing`, its lines',
/// wide, so it always holds a line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feasible {
    pub hz: (f64, f64),
    pub secs: (f64, f64),
    pub db: (f64, f64),
    pub span: f64,
    pub spacing: f64,
}

/// As the node folds `-(o1 + o2 + ...)`'s sum.
fn summed(offsets: &[f64]) -> f64 {
    offsets.iter().fold(0.0, |s, o| s + o)
}

impl Feasible {
    /// `floor_db` is the atom's own (`spectrum::floor_db`).
    pub fn of(clip: &Clip, floor_db: f64) -> Feasible {
        Feasible {
            hz: (
                cam_hz(CAM_FIRST),
                cam_hz(CAM_LAST).min(f64::from(clip.rate) / 2.0),
            ),
            secs: (0.0, clip.secs()),
            db: (floor_db, FULL_SCALE_DBFS),
            span: 1.0 / f64::from(clip.rate),
            spacing: 1.0 / clip.secs(),
        }
    }

    pub fn hz_of(&self, atom: &Atom) -> (f64, f64) {
        (self.hz.0, self.hz.1 / atom.harmonics() as f64)
    }

    /// A deviation's range, and an offset's and their sum's: the width of its quantity's
    /// either way.
    pub fn offset(&self) -> f64 {
        self.db.1 - self.db.0
    }

    pub fn holds(&self, atom: &Atom) -> bool {
        let inside = |v: f64, (lo, hi): (f64, f64)| lo <= v && v <= hi;
        let ordered = |t: &Trajectory| {
            let mut last = atom.on;
            t.knots.iter().all(|(at, _)| {
                let held = last <= *at && *at <= self.secs.1;
                last = *at;
                held
            })
        };
        let traced = |t: &Trajectory, range| ordered(t) && t.values().all(|v| inside(v, range));
        let width = (-self.offset(), self.offset());
        let sound = match &atom.sound {
            Sound::Partial { hz } => traced(hz, self.hz_of(atom)),
            Sound::Harmonic { hz, offsets } => {
                traced(hz, self.hz_of(atom))
                    && offsets.iter().all(|o| inside(*o, width))
                    && inside(summed(offsets), width)
            }
            Sound::Noise { lo, hi } => {
                hi - lo >= self.spacing && inside(*lo, self.hz) && inside(*hi, self.hz)
            }
        };
        inside(atom.on, self.secs)
            && atom.end() - atom.on >= self.span
            && sound
            && traced(&atom.level, self.db)
    }

    /// Nearest in squared distance, number `i` of `numbers` weighing `weights[i] > 0`; `None`
    /// where nothing is feasible. Times and values share no constraint, so each is solved
    /// alone and exactly.
    pub fn project(&self, atom: &Atom, weights: &[f64]) -> Option<Atom> {
        let numbers = atom.numbers();
        assert_eq!(weights.len(), numbers.len(), "one weight per number");
        assert!(weights.iter().all(|w| *w > 0.0), "every weight positive");
        let (hz, db) = (self.hz_of(atom), self.db);
        let noise = matches!(atom.sound, Sound::Noise { .. });
        let short = self.secs.1 - self.secs.0 < self.span;
        if hz.0 > hz.1
            || db.0 > db.1
            || !db.0.is_finite()
            || short
            || (noise && hz.1 - hz.0 < self.spacing)
        {
            return None;
        }
        let mut values: Vec<f64> = numbers.iter().map(|(_, v)| *v).collect();
        let times: Vec<usize> = (0..numbers.len())
            .filter(|i| numbers[*i].0 == Quantity::Time)
            .collect();
        let level_from = numbers
            .iter()
            .position(|(q, _)| *q == Quantity::Db)
            .expect("a level");
        let (sound_times, level_times): (Vec<usize>, Vec<usize>) =
            times[1..].iter().partition(|i| **i < level_from);
        let points = |chain: &[usize]| -> Vec<(f64, f64)> {
            chain.iter().map(|i| (values[*i], weights[*i])).collect()
        };
        let (on, sound_at, level_at) = timed(
            (values[0], weights[0]),
            [&points(&sound_times), &points(&level_times)],
            self.secs,
            self.span,
        );
        values[0] = on;
        for (chain, held) in [(&sound_times, sound_at), (&level_times, level_at)] {
            for (i, v) in chain.iter().zip(held) {
                values[*i] = v;
            }
        }
        let trace = |values: &mut Vec<f64>, base: usize, range: (f64, f64), quantity| {
            let devs: Vec<usize> = (base + 1..numbers.len())
                .take_while(|i| {
                    matches!(numbers[*i].0, Quantity::Time) || numbers[*i].0 == quantity
                })
                .filter(|i| numbers[*i].0 == quantity)
                .collect();
            let (v, d) = chain(
                range,
                (values[base], weights[base]),
                &devs
                    .iter()
                    .map(|i| (values[*i], weights[*i]))
                    .collect::<Vec<_>>(),
            );
            values[base] = v;
            for (i, d) in devs.iter().zip(d) {
                values[*i] = d;
            }
        };
        match &atom.sound {
            Sound::Partial { .. } | Sound::Harmonic { .. } => {
                trace(&mut values, 1, hz, Quantity::HzDeviation);
            }
            Sound::Noise { .. } => {
                let lifted = (hz.0, hz.1 - self.spacing);
                let shifted = values[2] - self.spacing;
                let edges = expand(&pava(&[(values[1], weights[1]), (shifted, weights[2])]));
                let mut lo = edges[0].clamp(lifted.0, lifted.1);
                let held = edges[1].clamp(lifted.0, lifted.1);
                let hi = match held == shifted {
                    true => values[2],
                    false => (held + self.spacing).min(hz.1),
                };
                let mut hi = hi;
                while hi - lo < self.spacing {
                    match hi < hz.1 {
                        true => hi = hi.next_up(),
                        false => lo = lo.next_down().max(hz.0),
                    }
                }
                values[1] = lo;
                values[2] = hi;
            }
        }
        trace(&mut values, level_from, db, Quantity::DbDeviation);
        let offsets: Vec<usize> = (0..numbers.len())
            .filter(|i| numbers[*i].0 == Quantity::Offset)
            .collect();
        let held = summing(
            &offsets
                .iter()
                .map(|i| (values[*i], weights[*i]))
                .collect::<Vec<_>>(),
            self.offset(),
        );
        for (i, o) in offsets.iter().zip(held) {
            values[*i] = o;
        }
        Some(atom.lifted(&values))
    }
}

/// Given the root `r`, each chain is its isotonic fit clamped to `[r, secs]`, or, where the
/// last level time falls under `r + span`, that at `r + span` and the rest clamped to `[r, r +
/// span]`; a convex piecewise quadratic in `r`, minimised exactly piece by piece.
fn timed(
    (y, w): (f64, f64),
    [sound, level]: [&[(f64, f64)]; 2],
    (lo, hi): (f64, f64),
    span: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let (sound_blocks, level_blocks) = (pava(sound), pava(level));
    let (last, lead) = level.split_last().expect("a level ends on a knot");
    let lead_blocks = pava(lead);
    let last_mean = level_blocks.last().expect("a block").mean();
    let binds = |r: f64| last_mean < r + span;
    let top = hi - span;
    let slope = |r: f64| {
        let (mut a, mut b) = (w, w * y);
        let below = |blocks: &[Block], a: &mut f64, b: &mut f64| {
            for block in blocks.iter().filter(|block| block.mean() < r) {
                *a += block.weight;
                *b += block.weight * block.mean();
            }
        };
        below(&sound_blocks, &mut a, &mut b);
        match binds(r) {
            false => below(&level_blocks, &mut a, &mut b),
            true => {
                a += last.1;
                b += last.1 * (last.0 - span);
                below(&lead_blocks, &mut a, &mut b);
                for block in lead_blocks.iter().filter(|b| b.mean() > r + span) {
                    a += block.weight;
                    b += block.weight * (block.mean() - span);
                }
            }
        }
        (a, b)
    };
    let mut corners: Vec<f64> = sound_blocks
        .iter()
        .chain(&level_blocks)
        .map(Block::mean)
        .chain(lead_blocks.iter().flat_map(|b| [b.mean(), b.mean() - span]))
        .chain([last_mean - span])
        .filter(|c| lo < *c && *c < top)
        .chain([lo, top])
        .collect();
    corners.sort_by(f64::total_cmp);
    corners.dedup();
    let mut r = top;
    for pair in corners.windows(2) {
        let (a, b) = slope(pair[0] + (pair[1] - pair[0]) / 2.0);
        if a * pair[1] - b >= 0.0 {
            r = (b / a).clamp(pair[0], pair[1]);
            break;
        }
    }
    let clamped = |blocks: &[Block], (l, h): (f64, f64)| -> Vec<f64> {
        expand(blocks).iter().map(|v| v.clamp(l, h)).collect()
    };
    let sound_at = clamped(&sound_blocks, (r, hi));
    let mut level_at = match binds(r) {
        false => clamped(&level_blocks, (r, hi)),
        true => {
            let mut held = clamped(&lead_blocks, (r, r + span));
            held.push((r + span).min(hi));
            held
        }
    };
    let end = level_at.last_mut().expect("a level ends on a knot");
    while *end - r < span {
        match *end < hi {
            true => *end = end.next_up(),
            false => r = r.next_down(),
        }
    }
    (r, sound_at, level_at)
}

/// Offsets in `[-width, width]` and their sum too, nearest in weighted squares: given the
/// sum's multiplier `l`, each offset is `clamp(y_k - l/w_k)`, a sum falling in `l` and affine
/// between the points where a clamp starts or ends, so `l` is solved exactly piece by piece.
fn summing(points: &[(f64, f64)], width: f64) -> Vec<f64> {
    let at = |l: f64| -> Vec<f64> {
        points
            .iter()
            .map(|(y, w)| (y - l / w).clamp(-width, width))
            .collect()
    };
    let free = at(0.0);
    let total = summed(&free);
    if total.abs() <= width {
        return free;
    }
    let side = total.signum();
    let mut corners: Vec<f64> = points
        .iter()
        .flat_map(|(y, w)| [w * (y - width), w * (y + width)])
        .map(|c| c * side)
        .filter(|c| *c > 0.0)
        .chain([0.0])
        .collect();
    corners.sort_by(f64::total_cmp);
    corners.dedup();
    let solve = |(a, b): (f64, f64)| {
        let mid = side * (a + (b - a) / 2.0);
        let (mut pull, mut give) = (-side * width, 0.0);
        for (y, w) in points {
            match (y - mid / w).abs() < width {
                true => {
                    pull += y;
                    give += 1.0 / w;
                }
                false => pull += (y - mid / w).clamp(-width, width),
            }
        }
        let (p, q) = (side * a, side * b);
        (pull / give).clamp(p.min(q), p.max(q))
    };
    let l = corners
        .windows(2)
        .find(|pair| summed(&at(side * pair[1])) * side <= width)
        .map_or(side * corners[corners.len() - 1], |pair| {
            solve((pair[0], pair[1]))
        });
    let mut held = at(l);
    while summed(&held).abs() > width {
        let most = (0..held.len())
            .max_by(|a, b| (held[*a] * side).total_cmp(&(held[*b] * side)))
            .expect("an offset");
        held[most] = match side > 0.0 {
            true => held[most].next_down(),
            false => held[most].next_up(),
        };
    }
    held
}

fn expand(blocks: &[Block]) -> Vec<f64> {
    blocks
        .iter()
        .flat_map(|b| std::iter::repeat_n(b.mean(), b.count))
        .collect()
}

/// Its mean kept as itself, so a block of equal values holds their value to the bit.
#[derive(Clone, Copy)]
struct Block {
    mean: f64,
    weight: f64,
    count: usize,
}

impl Block {
    fn of((mean, weight): (f64, f64)) -> Block {
        Block {
            mean,
            weight,
            count: 1,
        }
    }

    fn mean(&self) -> f64 {
        self.mean
    }

    fn merged(&self, other: &Block) -> Block {
        let weight = self.weight + other.weight;
        Block {
            mean: self.mean + (other.mean - self.mean) * (other.weight / weight),
            weight,
            count: self.count + other.count,
        }
    }
}

fn pava(chain: &[(f64, f64)]) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::with_capacity(chain.len());
    for point in chain {
        let mut block = Block::of(*point);
        while let Some(last) = blocks.last()
            && last.mean() > block.mean()
        {
            block = last.merged(&block);
            blocks.pop();
        }
        blocks.push(block);
    }
    blocks
}

/// `v` and every `v + d_j` in `[lo, hi]`: given `v` each `d_j` is its own clip, which leaves a
/// convex piecewise quadratic in `v`, solved on the piece where its slope crosses zero.
fn chain((lo, hi): (f64, f64), (y, w): (f64, f64), devs: &[(f64, f64)]) -> (f64, Vec<f64>) {
    let slope = |v: f64| {
        devs.iter().fold(w * (v - y), |s, (e, wj)| {
            let at = v + e;
            s + wj * (at - at.clamp(lo, hi))
        })
    };
    let mut corners: Vec<f64> = devs
        .iter()
        .flat_map(|(e, _)| [lo - e, hi - e])
        .filter(|c| lo < *c && *c < hi)
        .chain([lo, hi])
        .collect();
    corners.sort_by(f64::total_cmp);
    let v = match corners.iter().position(|c| slope(*c) >= 0.0) {
        Some(0) => lo,
        None => hi,
        Some(k) => {
            let (a, b) = (corners[k - 1], corners[k]);
            let mid = a + (b - a) / 2.0;
            let (mut weight, mut pull) = (w, 0.0);
            for (e, wj) in devs {
                let at = mid + e;
                let target = match () {
                    () if at < lo => lo - e,
                    () if at > hi => hi - e,
                    () => continue,
                };
                weight += wj;
                pull += wj * (target - y);
            }
            (y + pull / weight).clamp(a, b)
        }
    };
    let deviations = devs
        .iter()
        .map(|(e, _)| match (lo..=hi).contains(&(v + e)) {
            true => *e,
            false => inward(v, (v + e).clamp(lo, hi) - v, (lo, hi)),
        })
        .collect();
    (v, deviations)
}

/// The fewest ulps that put `v + d`, as a node folds it, inside.
fn inward(v: f64, mut d: f64, (lo, hi): (f64, f64)) -> f64 {
    while v + d > hi {
        d = d.next_down();
    }
    while v + d < lo {
        d = d.next_up();
    }
    d
}
