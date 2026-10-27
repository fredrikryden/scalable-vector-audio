// Concern: the four atom kinds solve writes: numbers, ranges, rows, nodes, keys, and their render through the engine | Non-concern: fitting them (fit.rs) | IO: (Atom) -> numbers, row, node, samples

use std::collections::HashMap;
use std::f64::consts::{LN_10, TAU};

use sva_ast::Composition;
use sva_ast::score::Item;
use sva_ast::{Expr, Literal};
use sva_engine::hearing::{CAM_FIRST, CAM_LAST, cam_hz};

use sva_engine::Tier;

use crate::json::num;
use crate::{Job, Output, Representation, asked, call, execute};

/// Every node writes these bits, so the model reads the same.
pub const NEPER_PER_DB: f64 = LN_10 / 20.0;

pub const DIRECTORY: &str = "atoms";

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
}

impl Trajectory {
    /// As a node folds `base + deviation`.
    pub fn values(&self) -> impl Iterator<Item = f64> + '_ {
        std::iter::once(self.base).chain(self.knots.iter().map(|(_, d)| self.base + *d))
    }
}

/// A partial's or harmonic's frequency at `t` is `hz(t) e^(glide (t - on))`: `glide` in nepers
/// per second, 0 a tone whose frequency only its knots move. A glide is written only on a steady
/// `hz`, whose phase then has the closed form `hz (e^(glide (t - on)) - 1) / glide`; on knots it
/// has none the engine writes, a correctness limit.
#[derive(Clone, Debug, PartialEq)]
pub enum Sound {
    Partial {
        hz: Trajectory,
        glide: f64,
    },
    /// Harmonic `k` turns `k sqrt(1 + stretch k^2)` times as fast as `hz` (a stiff string's
    /// partials; Fletcher 1964) at its own level, the first at the atom's, harmonic `k > 1` at
    /// `levels[k - 2]`.
    Harmonic {
        hz: Trajectory,
        stretch: f64,
        glide: f64,
        levels: Vec<Trajectory>,
    },
    Noise {
        lo: f64,
        hi: f64,
    },
    /// Noise over `bands`, each `(lo, hi, gain_db)`: band `k`'s power the atom's level plus
    /// its gain in dB, one level over time shared by every band (rank 1).
    Shaped {
        bands: Vec<(f64, f64, f64)>,
    },
}

/// A level in dBFS whose last knot ends the atom; a noise's is its band's power, density times
/// `hi - lo`. `key` is its ordinal, every phase and the noise seed drawn from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub key: u64,
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
    Stretch,
    Glide,
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

impl Atom {
    pub fn end(&self) -> f64 {
        self.level.knots.last().expect("a level ends on a knot").0
    }

    pub fn hz(&self) -> Option<&Trajectory> {
        match &self.sound {
            Sound::Partial { hz, .. } | Sound::Harmonic { hz, .. } => Some(hz),
            Sound::Noise { .. } | Sound::Shaped { .. } => None,
        }
    }

    /// Its kind by name.
    pub fn kind(&self) -> &'static str {
        match self.sound {
            Sound::Partial { .. } => "partial",
            Sound::Harmonic { .. } => "harmonic",
            Sound::Noise { .. } => "noise",
            Sound::Shaped { .. } => "shaped",
        }
    }

    /// Its glide in nepers per second, 0 where it has none.
    pub fn glide(&self) -> f64 {
        match self.sound {
            Sound::Partial { glide, .. } | Sound::Harmonic { glide, .. } => glide,
            Sound::Noise { .. } | Sound::Shaped { .. } => 0.0,
        }
    }

    /// This partial or harmonic gliding at `rate`, as `Sound` states; its `hz` steady.
    pub fn glided(mut self, rate: f64) -> Atom {
        match &mut self.sound {
            Sound::Partial { hz, glide } | Sound::Harmonic { hz, glide, .. } => {
                assert!(
                    rate == 0.0 || hz.knots.is_empty(),
                    "a glide is written on a steady frequency"
                );
                *glide = rate;
            }
            Sound::Noise { .. } | Sound::Shaped { .. } => {
                unreachable!("a noise has no frequency to glide")
            }
        }
        self
    }

    pub fn harmonics(&self) -> usize {
        match &self.sound {
            Sound::Harmonic { levels, .. } => levels.len() + 1,
            _ => 1,
        }
    }

    /// Every number but the key, in the order `with` reads.
    pub fn values(&self) -> Vec<f64> {
        let mut out = vec![self.on];
        let knots = |out: &mut Vec<f64>, t: &Trajectory| {
            out.push(t.base);
            for (at, d) in &t.knots {
                out.push(*at);
                out.push(*d);
            }
        };
        let glided = |out: &mut Vec<f64>, glide: f64| {
            if glide != 0.0 {
                out.push(glide);
            }
        };
        match &self.sound {
            Sound::Partial { hz, glide } => {
                knots(&mut out, hz);
                glided(&mut out, *glide);
            }
            Sound::Harmonic {
                hz,
                stretch,
                glide,
                levels,
            } => {
                knots(&mut out, hz);
                glided(&mut out, *glide);
                out.push(*stretch);
                for level in levels {
                    knots(&mut out, level);
                }
            }
            Sound::Noise { lo, hi } => out.extend([*lo, *hi]),
            Sound::Shaped { bands } => {
                for (lo, hi, gain) in bands {
                    out.extend([*lo, *hi, *gain]);
                }
            }
        }
        knots(&mut out, &self.level);
        out
    }

    pub fn partial(key: u64, on: f64, hz: Trajectory, level: Trajectory) -> Atom {
        Atom {
            key,
            on,
            sound: Sound::Partial { hz, glide: 0.0 },
            level,
        }
    }

    /// `levels` each harmonic's, the first's the atom's.
    pub fn harmonic(
        key: u64,
        on: f64,
        (hz, stretch): (Trajectory, f64),
        levels: Vec<Trajectory>,
    ) -> Atom {
        let mut levels = levels.into_iter();
        let level = levels.next().expect("a harmonic has a first");
        Atom {
            key,
            on,
            sound: Sound::Harmonic {
                hz,
                stretch,
                glide: 0.0,
                levels: levels.collect(),
            },
            level,
        }
    }

    /// Each harmonic's level, the first the atom's.
    pub fn levels(&self) -> Vec<&Trajectory> {
        let mut out = vec![&self.level];
        if let Sound::Harmonic { levels, .. } = &self.sound {
            out.extend(levels);
        }
        out
    }

    /// Harmonic `k`'s ratio to the fundamental, `k` where it is no harmonic.
    pub fn ratio(&self, k: usize) -> f64 {
        let k = k as f64;
        match &self.sound {
            Sound::Harmonic { stretch, .. } => k * libm::sqrt(1.0 + stretch * k * k),
            _ => k,
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

    /// `bands` each `(lo, hi, gain_db)`, as `Sound::Shaped` states.
    pub fn shaped(key: u64, on: f64, bands: Vec<(f64, f64, f64)>, level: Trajectory) -> Atom {
        Atom {
            key,
            on,
            sound: Sound::Shaped { bands },
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
        let glided = |out: &mut Vec<_>, glide: f64| {
            if glide != 0.0 {
                out.push(Quantity::Glide);
            }
        };
        match &self.sound {
            Sound::Partial { hz, glide } => {
                knots(&mut out, hz, (Quantity::Hz, Quantity::HzDeviation));
                glided(&mut out, *glide);
            }
            Sound::Harmonic {
                hz, glide, levels, ..
            } => {
                knots(&mut out, hz, (Quantity::Hz, Quantity::HzDeviation));
                glided(&mut out, *glide);
                out.push(Quantity::Stretch);
                for level in levels {
                    knots(&mut out, level, (Quantity::Db, Quantity::DbDeviation));
                }
            }
            Sound::Noise { .. } => out.extend([Quantity::Edge, Quantity::Edge]),
            Sound::Shaped { bands } => {
                for _ in bands {
                    out.extend([Quantity::Edge, Quantity::Edge, Quantity::DbDeviation]);
                }
            }
        }
        knots(&mut out, &self.level, (Quantity::Db, Quantity::DbDeviation));
        out.into_iter().zip(self.values()).collect()
    }

    /// One node per kind and knot count.
    pub fn node(&self) -> String {
        let (f, l) = (
            self.hz().map_or(0, |t| t.knots.len()),
            self.level.knots.len(),
        );
        let g = match self.glide() != 0.0 {
            true => "_g1",
            false => "",
        };
        match &self.sound {
            Sound::Partial { .. } => format!("{DIRECTORY}/partial_f{f}{g}_l{l}"),
            Sound::Harmonic { .. } => {
                let each: Vec<String> = self
                    .levels()
                    .iter()
                    .map(|t| t.knots.len().to_string())
                    .collect();
                format!(
                    "{DIRECTORY}/harmonic_h{}_f{f}{g}_l{}",
                    self.harmonics(),
                    each.join("-")
                )
            }
            Sound::Noise { .. } => format!("{DIRECTORY}/noise_l{l}"),
            Sound::Shaped { bands } => format!("{DIRECTORY}/shaped_b{}_l{l}", bands.len()),
        }
    }

    /// Whether the parser reads this atom's node: a harmonic of many harmonics nests past its
    /// bound, a correctness limit, not a cost one.
    pub fn writes(&self, clip: &Clip) -> bool {
        sva_ast::parse_file(&self.node_text(clip)).is_ok()
    }

    pub fn node_text(&self, clip: &Clip) -> String {
        let l = self.level.knots.len();
        let end = format!("lt{l}");
        let neper = num(NEPER_PER_DB);
        let levelled = |base: &str, time: &str, dev: &str, count: usize| {
            format!(
                "exp({neper}*{})",
                table("pwl", base, &knot_names(time, dev, base, count), None)
            )
        };
        let level = levelled("level", "lt", "ld", l);
        let glide = self.glide();
        let turning = |f: usize| match glide != 0.0 {
            true => "hz*(exp(glide*(t - on)) - 1)/glide".to_string(),
            false => table(
                "pwl_integral",
                "hz",
                &knot_names("ft", "fd", "hz", f),
                Some(&end),
            ),
        };
        let gliding = match glide != 0.0 {
            true => ", gliding at glide nepers per second",
            false => "",
        };
        let line = |k: usize, phi: &str| format!("sin(2*pi*{k}*{phi} + 2*pi*draw({k}, seed=key))");
        let stretched = |k: usize, phi: &str| {
            format!(
                "sin(2*pi*{k}*sqrt(1 + b*{})*{phi} + 2*pi*draw({k}, seed=key))",
                k * k
            )
        };
        let (what, body) = match &self.sound {
            Sound::Partial { hz, .. } => (
                format!(
                    "one partial, its frequency through {} knots{gliding}",
                    hz.knots.len()
                ),
                format!("{} * {level}", line(1, &turning(hz.knots.len()))),
            ),
            Sound::Harmonic { hz, levels, .. } => {
                let phi = turning(hz.knots.len());
                let mut lines = vec![format!("{level}*{}", stretched(1, &phi))];
                for (i, held) in levels.iter().enumerate() {
                    let k = i + 2;
                    let (base, time, dev) = (format!("l{k}"), format!("l{k}t"), format!("l{k}d"));
                    lines.push(format!(
                        "{}*{}",
                        levelled(&base, &time, &dev, held.knots.len()),
                        stretched(k, &phi)
                    ));
                }
                (
                    format!(
                        "{} harmonics of one fundamental through {} knots{gliding}, harmonic k \
                         at k sqrt(1 + b k^2) times it, each at its own level, the first's",
                        self.harmonics(),
                        hz.knots.len()
                    ),
                    lines.join(" + "),
                )
            }
            Sound::Noise { .. } => (
                "one band of noise flat in Hz, its level the band's power".to_string(),
                format!(
                    "noise(key, period={}s, lo=lo, hi=hi) * {level} / sqrt(hi - lo)",
                    num(clip.secs())
                ),
            ),
            Sound::Shaped { bands } => {
                let each: Vec<String> = (1..=bands.len())
                    .map(|k| {
                        format!(
                            "noise(key, period={}s, lo=lo{k}, hi=hi{k}) * exp({neper}*g{k}) / \
                             sqrt(hi{k} - lo{k})",
                            num(clip.secs())
                        )
                    })
                    .collect();
                (
                    format!(
                        "noise in {} bands flat in Hz, each its gain in dB over one level shared \
                         by every band, its level plus its gain the band's power",
                        bands.len()
                    ),
                    format!("({}) * {level}", each.join(" + ")),
                )
            }
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
        let glided = |fields: &mut Vec<(String, String)>, glide: f64| {
            if glide != 0.0 {
                fields.push(("glide".to_string(), num(glide)));
            }
        };
        match &self.sound {
            Sound::Partial { hz, glide } => {
                knots(&mut fields, hz, ("hz", "ft", "fd"));
                glided(&mut fields, *glide);
            }
            Sound::Harmonic {
                hz,
                stretch,
                glide,
                levels,
            } => {
                knots(&mut fields, hz, ("hz", "ft", "fd"));
                glided(&mut fields, *glide);
                fields.push(("b".to_string(), num(*stretch)));
                for (i, level) in levels.iter().enumerate() {
                    let k = i + 2;
                    let (base, time, dev) = (format!("l{k}"), format!("l{k}t"), format!("l{k}d"));
                    knots(&mut fields, level, (&base, &time, &dev));
                }
            }
            Sound::Noise { lo, hi } => {
                fields.push(("lo".to_string(), num(*lo)));
                fields.push(("hi".to_string(), num(*hi)));
            }
            Sound::Shaped { bands } => {
                for (k, (lo, hi, gain)) in bands.iter().enumerate() {
                    fields.push((format!("lo{}", k + 1), num(*lo)));
                    fields.push((format!("hi{}", k + 1), num(*hi)));
                    fields.push((format!("g{}", k + 1), num(*gain)));
                }
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
        let mut counts: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut parts = name.split('_');
        let kind = parts.next()?;
        for part in parts {
            let (letter, held) = part.split_at_checked(1)?;
            let each = held.split('-').map(|c| c.parse::<usize>().ok());
            counts.insert(letter, each.collect::<Option<_>>()?);
        }
        let one = |letter: &str| match counts.get(letter).map(Vec::as_slice) {
            Some([count]) => Some(*count),
            _ => None,
        };
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
        let held = counts.get("l")?;
        let level = trajectory("level", ("lt", "ld"), *held.first()?)?;
        let hz = || trajectory("hz", ("ft", "fd"), one("f")?);
        let glide = || match counts.get("g").map(Vec::as_slice) {
            None => Some(0.0),
            Some([1]) if one("f")? == 0 => get("glide").filter(|g| *g != 0.0),
            Some(_) => None,
        };
        if kind != "harmonic" && held.len() != 1 {
            return None;
        }
        let sound = match kind {
            "partial" => Sound::Partial {
                hz: hz()?,
                glide: glide()?,
            },
            "harmonic" if held.len() == one("h")? => Sound::Harmonic {
                hz: hz()?,
                stretch: get("b")?,
                glide: glide()?,
                levels: held[1..]
                    .iter()
                    .enumerate()
                    .map(|(i, count)| {
                        let k = i + 2;
                        let (time, dev) = (format!("l{k}t"), format!("l{k}d"));
                        trajectory(&format!("l{k}"), (&time, &dev), *count)
                    })
                    .collect::<Option<_>>()?,
            },
            "noise" => Sound::Noise {
                lo: get("lo")?,
                hi: get("hi")?,
            },
            "shaped" => Sound::Shaped {
                bands: (1..=one("b")?)
                    .map(|k| {
                        Some((
                            get(&format!("lo{k}"))?,
                            get(&format!("hi{k}"))?,
                            get(&format!("g{k}"))?,
                        ))
                    })
                    .collect::<Option<_>>()?,
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

/// The score every composition of atoms plays, and the master that plays it.
pub const SCORE: &str = "score/clip.jsonl";
pub const MASTER: &str = "; Models: the clip | Neglects: nothing | IO: (t) -> amplitude | \
     Tags: master\n@score/clip(t)\n";

/// Every file a composition of `atoms` holds: its nodes, its score and its master.
pub fn files(atoms: &[Atom], clip: &Clip) -> Vec<(String, String)> {
    let mut out = nodes(atoms, clip);
    out.push((SCORE.to_string(), score(atoms)));
    out.push(("master".to_string(), MASTER.to_string()));
    out
}

/// Every file a composition of `atoms` holds, its master scaling the score by one gain of
/// `gain_db` dB.
pub fn files_at(atoms: &[Atom], clip: &Clip, gain_db: f64) -> Vec<(String, String)> {
    let mut out = files(atoms, clip);
    out.retain(|(path, _)| path != "master");
    let master = format!(
        "; Models: the clip at one master gain in dB | Neglects: nothing | IO: (t) -> \
         amplitude | Tags: master\ngain = {}\nexp({}*gain) * @score/clip(t)\n",
        num(gain_db),
        num(NEPER_PER_DB)
    );
    out.push(("master".to_string(), master));
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

/// The ranges an atom's numbers are priced over (bits.rs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feasible {
    pub hz: (f64, f64),
    pub secs: (f64, f64),
    pub db: (f64, f64),
}

impl Feasible {
    pub fn of(clip: &Clip, db: (f64, f64)) -> Feasible {
        Feasible {
            hz: (
                cam_hz(CAM_FIRST),
                cam_hz(CAM_LAST).min(f64::from(clip.rate) / 2.0),
            ),
            secs: (0.0, clip.secs()),
            db,
        }
    }

    pub fn hz_of(&self, atom: &Atom) -> (f64, f64) {
        (self.hz.0, self.hz.1 / atom.ratio(atom.harmonics()))
    }

    /// A deviation's range: the width of its quantity's either way.
    pub fn offset(&self) -> f64 {
        self.db.1 - self.db.0
    }
}

/// The clip `files` write.
pub fn sounded(files: Vec<(String, String)>, clip: &Clip) -> Vec<f64> {
    let mut held = Composition::new();
    for (path, text) in files {
        held.insert(path, text);
    }
    let samples = [asked(&call("samples").expect("a call")).expect("a reading")];
    let target = format!("@master([0, {}s])", clip.secs());
    let job = Job {
        asked: &samples,
        rate: Some(clip.rate),
        ..Job::over(&held, &target)
    };
    let rendered = execute(job, &Tier::default()).expect("atoms' own nodes render");
    let answer = rendered
        .answer(&rendered.target, Representation::Samples)
        .expect("a render answers its samples");
    let Output::Samples(buffer) = answer.value else {
        unreachable!("samples answer samples")
    };
    buffer.plane(0).to_vec()
}
