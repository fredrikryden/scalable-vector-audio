// Concern: the three atom kinds solve writes: numbers, ranges, the feasible set and its projection, rows, nodes, keys | Non-concern: what the ear reads of one | IO: (Atom) -> numbers, row, node, Atom

use std::f64::consts::{LN_10, TAU};

use sva_engine::hearing::{CAM_FIRST, CAM_LAST, cam_hz};

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
pub struct Trajectory {
    pub base: f64,
    pub knots: Vec<(f64, f64)>,
}

impl Trajectory {
    pub fn steady(base: f64) -> Trajectory {
        Trajectory {
            base,
            knots: Vec::new(),
        }
    }

    /// As a node folds `base + deviation`.
    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        std::iter::once(self.base).chain(self.knots.iter().map(|(_, d)| self.base + d))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Sound {
    Partial {
        hz: Trajectory,
        phase: f64,
    },
    /// Harmonic `k` turns `k` times as fast as `hz`, at `offsets[k - 1]` dB and `phases[k - 1]`.
    Harmonic {
        hz: Trajectory,
        offsets: Vec<f64>,
        phases: Vec<f64>,
    },
    Noise {
        lo: f64,
        hi: f64,
        key: u64,
    },
}

/// A level in dBFS whose last knot ends the atom; a noise's is its band's power, density times
/// `hi - lo`.
#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub on: f64,
    pub sound: Sound,
    pub level: Trajectory,
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

/// The draw a noise keyed `ordinal` gives its own line `k`, so no new number is written.
pub fn phase(ordinal: u64, k: usize) -> f64 {
    TAU * sva_formula::draw_nearest(ordinal, k as f64).expect("a line index is a whole step")
}

impl Atom {
    pub fn partial(ordinal: u64, on: f64, hz: Trajectory, level: Trajectory) -> Atom {
        let phase = phase(ordinal, 1);
        Atom {
            on,
            sound: Sound::Partial { hz, phase },
            level,
        }
    }

    pub fn harmonic(
        ordinal: u64,
        on: f64,
        hz: Trajectory,
        offsets: Vec<f64>,
        level: Trajectory,
    ) -> Atom {
        let phases = (1..=offsets.len()).map(|k| phase(ordinal, k)).collect();
        Atom {
            on,
            sound: Sound::Harmonic {
                hz,
                offsets,
                phases,
            },
            level,
        }
    }

    pub fn noise(ordinal: u64, on: f64, (lo, hi): (f64, f64), level: Trajectory) -> Atom {
        Atom {
            on,
            sound: Sound::Noise {
                lo,
                hi,
                key: ordinal,
            },
            level,
        }
    }

    pub fn end(&self) -> f64 {
        self.level.knots.last().expect("a level ends on a knot").0
    }

    pub fn hz(&self) -> Option<&Trajectory> {
        match &self.sound {
            Sound::Partial { hz, .. } | Sound::Harmonic { hz, .. } => Some(hz),
            Sound::Noise { .. } => None,
        }
    }

    /// Every number but phases and key, in the order `with` reads.
    pub fn numbers(&self) -> Vec<(Quantity, f64)> {
        let mut out = vec![(Quantity::Time, self.on)];
        let knots = |out: &mut Vec<_>, t: &Trajectory, (base, dev)| {
            out.push((base, t.base));
            for (at, d) in &t.knots {
                out.push((Quantity::Time, *at));
                out.push((dev, *d));
            }
        };
        match &self.sound {
            Sound::Partial { hz, .. } => knots(&mut out, hz, (Quantity::Hz, Quantity::HzDeviation)),
            Sound::Harmonic { hz, offsets, .. } => {
                knots(&mut out, hz, (Quantity::Hz, Quantity::HzDeviation));
                out.extend(offsets.iter().map(|o| (Quantity::Offset, *o)));
            }
            Sound::Noise { lo, hi, .. } => {
                out.push((Quantity::Edge, *lo));
                out.push((Quantity::Edge, *hi));
            }
        }
        knots(&mut out, &self.level, (Quantity::Db, Quantity::DbDeviation));
        out
    }

    pub fn with(&self, values: &[f64]) -> Atom {
        assert_eq!(values.len(), self.numbers().len(), "one value per number");
        let mut next = values.iter().copied();
        let take = |next: &mut dyn Iterator<Item = f64>| next.next().expect("counted");
        let retake = |next: &mut dyn Iterator<Item = f64>, t: &mut Trajectory| {
            t.base = take(next);
            for knot in &mut t.knots {
                *knot = (take(next), take(next));
            }
        };
        let mut atom = self.clone();
        atom.on = take(&mut next);
        match &mut atom.sound {
            Sound::Partial { hz, .. } => retake(&mut next, hz),
            Sound::Harmonic { hz, offsets, .. } => {
                retake(&mut next, hz);
                offsets.iter_mut().for_each(|o| *o = take(&mut next));
            }
            Sound::Noise { lo, hi, .. } => {
                *lo = take(&mut next);
                *hi = take(&mut next);
            }
        }
        retake(&mut next, &mut atom.level);
        atom
    }

    /// One node per kind and knot count.
    pub fn node(&self) -> String {
        let (f, l) = (
            self.hz().map_or(0, |t| t.knots.len()),
            self.level.knots.len(),
        );
        match &self.sound {
            Sound::Partial { .. } => format!("{DIRECTORY}/partial_f{f}_l{l}"),
            Sound::Harmonic { offsets, .. } => {
                format!("{DIRECTORY}/harmonic_h{}_f{f}_l{l}", offsets.len())
            }
            Sound::Noise { .. } => format!("{DIRECTORY}/noise_l{l}"),
        }
    }

    pub fn node_text(&self, clip: &Clip) -> String {
        let l = self.level.knots.len();
        let end = format!("lt{l}");
        let level = format!(
            "exp({}*{})",
            num(NEPER_PER_DB),
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
        let (what, body) = match &self.sound {
            Sound::Partial { hz, .. } => (
                format!(
                    "one partial, its frequency through {} knots",
                    hz.knots.len()
                ),
                format!("sin(2*pi*{} + ph1) * {level}", turning(hz.knots.len())),
            ),
            Sound::Harmonic { hz, offsets, .. } => {
                let phi = turning(hz.knots.len());
                let lines: Vec<String> = (1..=offsets.len())
                    .map(|k| {
                        format!(
                            "exp({}*o{k})*sin(2*pi*{k}*{phi} + ph{k})",
                            num(NEPER_PER_DB)
                        )
                    })
                    .collect();
                (
                    format!(
                        "{} harmonics of one fundamental through {} knots",
                        offsets.len(),
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
            Sound::Partial { hz, phase } => {
                knots(&mut fields, hz, ("hz", "ft", "fd"));
                fields.push(("ph1".to_string(), num(*phase)));
            }
            Sound::Harmonic {
                hz,
                offsets,
                phases,
            } => {
                knots(&mut fields, hz, ("hz", "ft", "fd"));
                for (k, (o, ph)) in offsets.iter().zip(phases).enumerate() {
                    fields.push((format!("o{}", k + 1), num(*o)));
                    fields.push((format!("ph{}", k + 1), num(*ph)));
                }
            }
            Sound::Noise { lo, hi, key } => {
                fields.push(("key".to_string(), key.to_string()));
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

/// One row each in birth order, so a row's line is its ordinal.
pub fn score(atoms: &[Atom]) -> String {
    let mut out = String::from(
        "{\"annotation\": \"Models: the clip's atoms, one row each in birth order | Neglects: \
         what no atom explains | IO: (t) -> amplitude | Tags: score\"}\n",
    );
    for atom in atoms {
        out.push_str(&atom.row());
        out.push('\n');
    }
    out
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
/// scale; a noise band at least `spacing`, its lines', wide, so it always holds a line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feasible {
    pub hz: (f64, f64),
    pub secs: (f64, f64),
    pub db: (f64, f64),
    pub spacing: f64,
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
            spacing: 1.0 / clip.secs(),
        }
    }

    fn hz_of(&self, atom: &Atom) -> (f64, f64) {
        match &atom.sound {
            Sound::Harmonic { offsets, .. } => (self.hz.0, self.hz.1 / offsets.len() as f64),
            _ => self.hz,
        }
    }

    /// A deviation's range: the width of its quantity's either way.
    fn offset(&self) -> f64 {
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
        let sound = match &atom.sound {
            Sound::Partial { hz, .. } => traced(hz, self.hz_of(atom)),
            Sound::Harmonic { hz, offsets, .. } => {
                traced(hz, self.hz_of(atom))
                    && offsets
                        .iter()
                        .all(|o| inside(*o, (-self.offset(), self.offset())))
            }
            Sound::Noise { lo, hi, .. } => {
                hi - lo >= self.spacing && inside(*lo, self.hz) && inside(*hi, self.hz)
            }
        };
        inside(atom.on, self.secs) && sound && traced(&atom.level, self.db)
    }

    /// Nearest in squared distance, number `i` of `numbers` weighing `weights[i] > 0`; `None`
    /// where nothing is feasible. Times and values share no constraint, so each is solved
    /// alone and exactly: times by tree isotonic regression rooted at `on`, then clipped.
    pub fn project(&self, atom: &Atom, weights: &[f64]) -> Option<Atom> {
        let numbers = atom.numbers();
        assert_eq!(weights.len(), numbers.len(), "one weight per number");
        assert!(weights.iter().all(|w| *w > 0.0), "every weight positive");
        let (hz, db) = (self.hz_of(atom), self.db);
        let noise = matches!(atom.sound, Sound::Noise { .. });
        if hz.0 > hz.1 || db.0 > db.1 || !db.0.is_finite() || (noise && hz.1 - hz.0 < self.spacing)
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
        let chains = [sound_times, level_times];
        let solved = tree(
            (values[0], weights[0]),
            &chains
                .iter()
                .map(|c| c.iter().map(|i| (values[*i], weights[*i])).collect())
                .collect::<Vec<Vec<_>>>(),
        );
        values[0] = solved.0;
        for (chain, held) in chains.iter().zip(&solved.1) {
            for (i, v) in chain.iter().zip(held) {
                values[*i] = *v;
            }
        }
        for i in &times {
            values[*i] = values[*i].clamp(self.secs.0, self.secs.1);
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
        for (i, (q, _)) in numbers.iter().enumerate() {
            if *q == Quantity::Offset {
                values[i] = values[i].clamp(-self.offset(), self.offset());
            }
        }
        Some(atom.with(&values))
    }
}

/// A root under rising chains: each chain pooled alone, then the root's block absorbs the
/// lowest next block while its mean is under the root's.
fn tree(root: (f64, f64), chains: &[Vec<(f64, f64)>]) -> (f64, Vec<Vec<f64>>) {
    let pooled: Vec<Vec<Block>> = chains.iter().map(|c| pava(c)).collect();
    let mut next = vec![0usize; chains.len()];
    let mut held = Block::of(root);
    loop {
        let lowest = (0..chains.len())
            .filter(|c| next[*c] < pooled[*c].len())
            .min_by(|a, b| {
                pooled[*a][next[*a]]
                    .mean()
                    .total_cmp(&pooled[*b][next[*b]].mean())
            });
        match lowest {
            Some(c) if pooled[c][next[c]].mean() < held.mean() => {
                held = held.merged(&pooled[c][next[c]]);
                next[c] += 1;
            }
            _ => break,
        }
    }
    let root_value = held.mean();
    let values = pooled
        .iter()
        .zip(&next)
        .map(|(blocks, taken)| {
            let mut values = vec![root_value; blocks[..*taken].iter().map(|b| b.count).sum()];
            values.extend(expand(&blocks[*taken..]));
            values
        })
        .collect();
    (root_value, values)
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
