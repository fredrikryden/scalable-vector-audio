// Concern: the four source kinds solve writes and the shared nodes they read: numbers, rows, nodes, keys, their render | Non-concern: fitting them (fit/) | IO: (Source) -> numbers, row, node, samples

use std::collections::HashMap;
use std::f64::consts::{LN_10, TAU};

use sva_ast::Composition;
use sva_ast::score::Item;
use sva_ast::{Expr, Literal};
use sva_core::json::num;
use sva_core::{CliError, Job, Output, Representation, Tier, asked, call, execute};
use sva_engine::hearing::{CAM_FIRST, CAM_LAST, cam_hz};

/// Every node writes these bits, so the model reads the same.
pub const NEPER_PER_DB: f64 = LN_10 / 20.0;

pub const DIRECTORY: &str = "sources";

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

/// A level in dB: `base` at the source's onset, then straight lines through each knot's time
/// from the onset and deviation from `base`, held after the last.
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    pub base: f64,
    pub knots: Vec<(f64, f64)>,
}

/// What a source is, its levels aside. A partial's or harmonic's frequency at `t` is
/// `hz e^(glide (t - on))`: `glide` in nepers per second, 0 a fixed tone, its phase then
/// `hz (e^(glide (t - on)) - 1) / glide` in closed form.
#[derive(Clone, Debug, PartialEq)]
pub enum Sound {
    Partial {
        hz: f64,
        glide: f64,
    },
    /// `count` harmonics, harmonic `k` turning `k sqrt(1 + stretch k^2)` times as fast as `hz`
    /// (a stiff string's partials; Fletcher 1964), each at its own level.
    Harmonic {
        hz: f64,
        stretch: f64,
        count: usize,
        glide: f64,
    },
    Noise {
        lo: f64,
        hi: f64,
    },
    /// Noise over `bands`, each `(lo, hi, gain_db)`: band `k`'s power the source's level plus
    /// its gain in dB, one level over time shared by every band (rank 1).
    Shaped {
        bands: Vec<(f64, f64, f64)>,
    },
}

impl Sound {
    /// How many levels it holds: a harmonic's one per harmonic, any other's one.
    pub fn columns(&self) -> usize {
        match self {
            Sound::Harmonic { count, .. } => *count,
            Sound::Partial { .. } | Sound::Noise { .. } | Sound::Shaped { .. } => 1,
        }
    }

    pub fn glide(&self) -> f64 {
        match self {
            Sound::Partial { glide, .. } | Sound::Harmonic { glide, .. } => *glide,
            Sound::Noise { .. } | Sound::Shaped { .. } => 0.0,
        }
    }
}

/// A kind of shared node: numbers several sources reference, each source its own beside them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedKind {
    /// A level in dB moving as `depth sin(2 pi rate t + phase)`, `depth` each source's own.
    Periodic,
}

/// Name, the letter naming a referencing node, its numbers, the referencing source's own.
type Named = (
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);
const SHARED: [Named; 1] = [("periodic", "p", &["rate", "phase"], &["depth"])];

impl SharedKind {
    pub const ALL: [SharedKind; 1] = [SharedKind::Periodic];

    fn named(self) -> Named {
        SHARED[self as usize]
    }

    pub fn name(self) -> &'static str {
        self.named().0
    }

    pub fn numbers(self) -> &'static [&'static str] {
        self.named().2
    }

    pub fn own(self) -> &'static [&'static str] {
        self.named().3
    }

    pub fn node(self, to: usize) -> String {
        format!("{DIRECTORY}/{}_{to}", self.name())
    }

    /// What a referencing source's node multiplies its amplitude by.
    fn factor(self, to: usize) -> String {
        match self {
            SharedKind::Periodic => {
                format!("exp({}*depth*@{}_{to}(t))", num(NEPER_PER_DB), self.name())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shared {
    pub kind: SharedKind,
    pub numbers: Vec<f64>,
}

impl Shared {
    pub fn node_text(&self) -> String {
        let (what, body) = match self.kind {
            SharedKind::Periodic => (
                "a level in dB moving periodically, its depth each referencing source's own",
                "sin(2*pi*rate*t + phase)",
            ),
        };
        let bound: Vec<String> = self
            .kind
            .numbers()
            .iter()
            .zip(&self.numbers)
            .map(|(n, v)| format!("{n} = {}\n", num(*v)))
            .collect();
        format!(
            "; Models: {what} | Neglects: the sources, each its own row | IO: (t) -> value | \
             Tags: shared\n{}{body}\n",
            bound.concat()
        )
    }
}

/// A source's reference to shared node `to` of its kind, beside the source's own numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    pub to: usize,
    pub kind: SharedKind,
    pub own: Vec<f64>,
}

/// One sound in the clip: its kind, its levels in dBFS, one per column (a noise's its band's
/// power, density times `hi - lo`), each one's last knot ending it. `key` is its ordinal, every
/// phase and the noise seed drawn from it; `links` the shared nodes it references, its amplitude
/// times each one's factor.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub key: u64,
    pub on: f64,
    pub sound: Sound,
    pub levels: Vec<Trajectory>,
    pub links: Vec<Link>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Quantity {
    Time,
    Hz,
    Db,
    DbDeviation,
    Stretch,
    Glide,
    Edge,
    /// A periodic level's depth in dB.
    Depth,
}

/// Line `k`'s phase, the draw noise keyed `key` gives its own line `k`, so no number is written.
pub fn phase(key: u64, k: usize) -> f64 {
    TAU * sva_formula::draw_nearest(key, k as f64).expect("a line index is a whole step")
}

/// The ordinal a birth takes: one past the largest living, the first 0.
pub fn birth(sources: &[Source]) -> u64 {
    sources.iter().map(|a| a.key + 1).max().unwrap_or(0)
}

impl Source {
    /// `sound` keyed `key` from `on` at `levels`, one per column, referencing nothing.
    pub fn new(key: u64, on: f64, sound: Sound, levels: Vec<Trajectory>) -> Source {
        assert_eq!(levels.len(), sound.columns(), "a level per column");
        Source {
            key,
            on,
            sound,
            levels,
            links: Vec::new(),
        }
    }

    pub fn end(&self) -> f64 {
        self.on
            + self.levels[0]
                .knots
                .last()
                .expect("a level ends on a knot")
                .0
    }

    pub fn hz(&self) -> Option<f64> {
        match &self.sound {
            Sound::Partial { hz, .. } | Sound::Harmonic { hz, .. } => Some(*hz),
            Sound::Noise { .. } | Sound::Shaped { .. } => None,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self.sound {
            Sound::Partial { .. } => "partial",
            Sound::Harmonic { .. } => "harmonic",
            Sound::Noise { .. } => "noise",
            Sound::Shaped { .. } => "shaped",
        }
    }

    pub fn harmonics(&self) -> usize {
        self.sound.columns()
    }

    /// Every number but the key, in the order `numbers` names them: a harmonic's levels after
    /// the first before its kind's numbers' end, its first level last.
    pub fn values(&self) -> Vec<f64> {
        self.numbers().into_iter().map(|(_, v)| v).collect()
    }

    /// Harmonic `k`'s ratio to the fundamental, `k` where it is no harmonic.
    pub fn ratio(&self, k: usize) -> f64 {
        let k = k as f64;
        match &self.sound {
            Sound::Harmonic { stretch, .. } => k * libm::sqrt(1.0 + stretch * k * k),
            _ => k,
        }
    }

    pub(crate) fn numbers(&self) -> Vec<(Quantity, f64)> {
        let mut out = vec![(Quantity::Time, self.on)];
        let knots = |out: &mut Vec<_>, t: &Trajectory, base: Quantity| {
            out.push((base, t.base));
            for (at, d) in &t.knots {
                out.push((Quantity::Time, *at));
                out.push((Quantity::DbDeviation, *d));
            }
        };
        let glided = |out: &mut Vec<_>, glide: f64| {
            if glide != 0.0 {
                out.push((Quantity::Glide, glide));
            }
        };
        match &self.sound {
            Sound::Partial { hz, glide } => {
                out.push((Quantity::Hz, *hz));
                glided(&mut out, *glide);
            }
            Sound::Harmonic {
                hz, stretch, glide, ..
            } => {
                out.push((Quantity::Hz, *hz));
                glided(&mut out, *glide);
                out.push((Quantity::Stretch, *stretch));
                for level in &self.levels[1..] {
                    knots(&mut out, level, Quantity::Db);
                }
            }
            Sound::Noise { lo, hi } => out.extend([(Quantity::Edge, *lo), (Quantity::Edge, *hi)]),
            Sound::Shaped { bands } => {
                for (lo, hi, gain) in bands {
                    out.extend([
                        (Quantity::Edge, *lo),
                        (Quantity::Edge, *hi),
                        (Quantity::DbDeviation, *gain),
                    ]);
                }
            }
        }
        knots(&mut out, &self.levels[0], Quantity::Db);
        for link in &self.links {
            out.extend(link.own.iter().map(|v| match link.kind {
                SharedKind::Periodic => (Quantity::Depth, *v),
            }));
        }
        out
    }

    /// One node per kind, knot counts and level tables, these by their hash.
    pub fn node(&self) -> String {
        let l = self.levels[0].knots.len();
        let g = match self.sound.glide() != 0.0 {
            true => "_g1",
            false => "",
        };
        let tables: Vec<String> = self.levels.iter().map(literal).collect();
        let key = fnv(&tables.join("\n"));
        let linked: String = self
            .links
            .iter()
            .map(|k| format!("_{}{}", k.kind.named().1, k.to))
            .collect();
        let key = format!("{linked}_n{key}");
        // `f0`: no frequency knot, which the name counts.
        match &self.sound {
            Sound::Partial { .. } => format!("{DIRECTORY}/partial_f0{g}_l{l}{key}"),
            Sound::Harmonic { count, .. } => {
                let each: Vec<String> = self
                    .levels
                    .iter()
                    .map(|t| t.knots.len().to_string())
                    .collect();
                format!(
                    "{DIRECTORY}/harmonic_h{count}_f0{g}_l{}{key}",
                    each.join("-")
                )
            }
            Sound::Noise { .. } => format!("{DIRECTORY}/noise_l{l}{key}"),
            Sound::Shaped { bands } => {
                format!("{DIRECTORY}/shaped_b{}_l{l}{key}", bands.len())
            }
        }
    }

    /// Whether the parser and engine read this source's node: a harmonic of many harmonics
    /// nests past its bound, a band holding no line at k over the clip's length is empty.
    pub fn writes(&self, clip: &Clip) -> bool {
        let holds = |lo: f64, hi: f64| {
            matches!(
                sva_formula::noise::band(clip.secs(), lo, Some(hi)),
                Some((first, sva_formula::Bound::Finite(last))) if first <= last
            )
        };
        let held = match &self.sound {
            Sound::Noise { lo, hi } => holds(*lo, *hi),
            Sound::Shaped { bands } => bands.iter().all(|(lo, hi, _)| holds(*lo, *hi)),
            Sound::Partial { .. } | Sound::Harmonic { .. } => true,
        };
        held && sva_ast::parse_file(&self.node_text(clip)).is_ok()
    }

    pub fn node_text(&self, clip: &Clip) -> String {
        let l = self.levels[0].knots.len();
        let end = format!("{}s", num(self.end() - self.on));
        let neper = num(NEPER_PER_DB);
        let levelled = |base: &str, held: &Trajectory| match held.knots.is_empty() {
            true => format!("exp({neper}*{base})"),
            false => format!("exp({neper}*({base} + {}))", literal(held)),
        };
        let level = levelled("level", &self.levels[0]);
        let glide = self.sound.glide();
        let phi = match glide != 0.0 {
            true => "hz*(exp(glide*(t - on)) - 1)/glide".to_string(),
            false => format!("pwl_integral(t - on, 0s, hz, {end}, hz)"),
        };
        let gliding = match glide != 0.0 {
            true => ", gliding at glide nepers per second",
            false => "",
        };
        let line = |k: usize| format!("sin(2*pi*{k}*{phi} + 2*pi*draw({k}, seed=key))");
        let stretched = |k: usize| {
            format!(
                "sin(2*pi*{k}*sqrt(1 + b*{})*{phi} + 2*pi*draw({k}, seed=key))",
                k * k
            )
        };
        let (what, body) = match &self.sound {
            Sound::Partial { .. } => (
                format!("one partial, its frequency through 0 knots{gliding}"),
                format!("{} * {level}", line(1)),
            ),
            Sound::Harmonic { count, .. } => {
                let mut lines = vec![format!("{level}*{}", stretched(1))];
                for (i, held) in self.levels[1..].iter().enumerate() {
                    let k = i + 2;
                    lines.push(format!(
                        "{}*{}",
                        levelled(&format!("l{k}"), held),
                        stretched(k)
                    ));
                }
                (
                    format!(
                        "{count} harmonics of one fundamental through 0 knots{gliding}, harmonic k \
                         at k sqrt(1 + b k^2) times it, each at its own level, the first's"
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
        let body = match self.links.is_empty() {
            true => body,
            false => {
                let factors: Vec<String> = self.links.iter().map(|k| k.kind.factor(k.to)).collect();
                format!("({body}) * {}", factors.join(" * "))
            }
        };
        let shared: String = self
            .links
            .iter()
            .map(|k| format!(", times the {} node {} it references", k.kind.name(), k.to))
            .collect();
        format!(
            "; Models: {what}, its level in dBFS through {l} knots, the last ending it{shared} | \
             Neglects: every other source, each its own row | IO: (t) -> amplitude | Tags: source\n\
             crop({body}, on, on + {end})\n"
        )
    }

    /// Played from the clip's start, its node one directory up, holding its levels' knots.
    pub fn row(&self) -> String {
        let mut fields = vec![
            ("at".to_string(), "\"0s\"".to_string()),
            ("node".to_string(), format!("\"@../{}\"", self.node())),
            ("key".to_string(), self.key.to_string()),
            ("on".to_string(), num(self.on)),
        ];
        let glided = |fields: &mut Vec<(String, String)>, glide: f64| {
            if glide != 0.0 {
                fields.push(("glide".to_string(), num(glide)));
            }
        };
        match &self.sound {
            Sound::Partial { hz, glide } => {
                fields.push(("hz".to_string(), num(*hz)));
                glided(&mut fields, *glide);
            }
            Sound::Harmonic {
                hz, stretch, glide, ..
            } => {
                fields.push(("hz".to_string(), num(*hz)));
                glided(&mut fields, *glide);
                fields.push(("b".to_string(), num(*stretch)));
                for (i, level) in self.levels[1..].iter().enumerate() {
                    fields.push((format!("l{}", i + 2), num(level.base)));
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
        fields.push(("level".to_string(), num(self.levels[0].base)));
        for link in &self.links {
            for (name, v) in link.kind.own().iter().zip(&link.own) {
                fields.push((name.to_string(), num(*v)));
            }
        }
        let written: Vec<String> = fields
            .iter()
            .map(|(k, v)| format!("\"{k}\": {v}"))
            .collect();
        format!("{{{}}}", written.join(", "))
    }

    /// The source a row writes reads back, its levels' knots from its node's text `text`;
    /// `None` for a shape no fit writes.
    fn read(node: &str, numbers: &HashMap<&str, f64>, text: &str) -> Option<Source> {
        let name = node.strip_prefix(&format!("../{DIRECTORY}/"))?;
        let mut counts: HashMap<&str, Vec<usize>> = HashMap::new();
        let mut parts = name.split('_');
        let kind = parts.next()?;
        for part in parts {
            let (letter, held) = part.split_at_checked(1)?;
            if letter == "n" {
                continue;
            }
            let each = held.split('-').map(|c| c.parse::<usize>().ok());
            counts.insert(letter, each.collect::<Option<_>>()?);
        }
        let one = |letter: &str| match counts.get(letter).map(Vec::as_slice) {
            Some([count]) => Some(*count),
            _ => None,
        };
        let get = |name: &str| numbers.get(name).copied();
        let key = get("key")?;
        let key =
            (key >= 0.0 && key.fract() == 0.0 && key < u64::MAX as f64).then_some(key as u64)?;
        let on = get("on")?;
        let held = counts.get("l")?;
        let mut tables = tables(text)?.into_iter();
        let mut levelled = |base: &str, count: usize| {
            let knots = match count {
                0 => Vec::new(),
                _ => tables.next().filter(|t| t.len() == count)?,
            };
            Some(Trajectory {
                base: get(base)?,
                knots,
            })
        };
        let first = levelled("level", *held.first()?)?;
        let hz = || (one("f")? == 0).then(|| get("hz"))?;
        let glide = || match counts.get("g").map(Vec::as_slice) {
            None => Some(0.0),
            Some([1]) => get("glide").filter(|g| *g != 0.0),
            Some(_) => None,
        };
        if kind != "harmonic" && held.len() != 1 {
            return None;
        }
        let mut levels = vec![first];
        let sound = match kind {
            "partial" => Sound::Partial {
                hz: hz()?,
                glide: glide()?,
            },
            "harmonic" if held.len() == one("h")? => {
                for (i, count) in held[1..].iter().enumerate() {
                    levels.push(levelled(&format!("l{}", i + 2), *count)?);
                }
                Sound::Harmonic {
                    hz: hz()?,
                    stretch: get("b")?,
                    count: held.len(),
                    glide: glide()?,
                }
            }
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
        if tables.next().is_some() {
            return None;
        }
        let mut links = Vec::new();
        for kind in SharedKind::ALL {
            let Some(to) = counts.get(kind.named().1) else {
                continue;
            };
            let [to] = to.as_slice() else {
                return None;
            };
            links.push(Link {
                to: *to,
                kind,
                own: kind.own().iter().map(|n| get(n)).collect::<Option<_>>()?,
            });
        }
        let source = Source {
            key,
            on,
            sound,
            levels,
            links,
        };
        let knots: usize = source.levels.iter().map(|t| t.knots.len()).sum();
        (source.values().len() + 1 == numbers.len() + 2 * knots).then_some(source)
    }
}

/// Each shared node `sources` reference, by index; `None` where one is missing.
pub fn shared(files: &[(String, String)], sources: &[Source]) -> Option<Vec<(usize, Shared)>> {
    let mut out: Vec<(usize, Shared)> = Vec::new();
    for link in sources.iter().flat_map(|a| &a.links) {
        if out.iter().any(|(to, _)| *to == link.to) {
            continue;
        }
        let path = link.kind.node(link.to);
        let (_, text) = files.iter().find(|(p, _)| *p == path)?;
        let bound: HashMap<&str, f64> = text
            .lines()
            .filter_map(|l| {
                let (name, value) = l.split_once(" = ")?;
                Some((name.trim(), value.trim().parse().ok()?))
            })
            .collect();
        let numbers = link.kind.numbers().iter().map(|n| bound.get(n).copied());
        let numbers = numbers.collect::<Option<_>>()?;
        out.push((
            link.to,
            Shared {
                kind: link.kind,
                numbers,
            },
        ));
    }
    Some(out)
}

/// A level's deviations from its base as a knot table of literals, which no token bound counts.
fn literal(level: &Trajectory) -> String {
    let mut args = vec!["t - on".to_string(), "0s".to_string(), "0".to_string()];
    for (at, d) in &level.knots {
        args.extend([format!("{}s", num(*at)), num(*d)]);
    }
    format!("pwl({})", args.join(", "))
}

/// Each table `literal` wrote into `text`, its knots after the first.
fn tables(text: &str) -> Option<Vec<Vec<(f64, f64)>>> {
    let opening = "pwl(t - on, ";
    text.match_indices(opening)
        .map(|(at, _)| {
            let rest = &text[at + opening.len()..];
            let values: Vec<f64> = rest[..rest.find(')')?]
                .split(", ")
                .enumerate()
                .map(|(i, v)| match i % 2 {
                    0 => v.strip_suffix('s')?.parse().ok(),
                    _ => v.parse().ok(),
                })
                .collect::<Option<_>>()?;
            match values.as_slice() {
                [first, zero, knots @ ..]
                    if *first == 0.0 && *zero == 0.0 && knots.len().is_multiple_of(2) =>
                {
                    Some(knots.chunks(2).map(|c| (c[0], c[1])).collect())
                }
                _ => None,
            }
        })
        .collect()
}

/// One row each in birth order.
pub fn score(sources: &[Source]) -> String {
    let mut out = String::from(
        "{\"annotation\": \"Models: the clip's sources, one row each in birth order | Neglects: \
         what no source explains | IO: (t) -> amplitude | Tags: score\"}\n",
    );
    let mut born: Vec<&Source> = sources.iter().collect();
    born.sort_by_key(|a| a.key);
    for source in born {
        out.push_str(&source.row());
        out.push('\n');
    }
    out
}

pub fn read(files: &[(String, String)]) -> Option<Vec<Source>> {
    let file = |path: &str| {
        files
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, t)| t.as_str())
    };
    let held = sva_ast::parse_score(file(SCORE)?).ok()?;
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
        let path = event.node.name.strip_prefix("../")?;
        out.push(Source::read(&event.node.name, &numbers, file(path)?)?);
    }
    Some(out)
}

pub const SCORE: &str = "score/clip.jsonl";

/// Every file a composition of `sources` holds: its nodes, the `shared` nodes they reference by
/// index, its score, and its master scaling the score by one gain of `gain_db` dB.
pub fn files_at(
    sources: &[Source],
    shared: &[(usize, Shared)],
    clip: &Clip,
    gain_db: f64,
) -> Vec<(String, String)> {
    let mut out = nodes(sources, clip);
    for (to, node) in shared {
        out.push((node.kind.node(*to), node.node_text()));
    }
    out.push((SCORE.to_string(), score(sources)));
    let master = format!(
        "; Models: the clip at one master gain in dB | Neglects: nothing | IO: (t) -> \
         amplitude | Tags: master\ngain = {}\nexp({}*gain) * @score/clip(t)\n",
        num(gain_db),
        num(NEPER_PER_DB)
    );
    out.push(("master".to_string(), master));
    out
}

pub fn gain_of(files: &[(String, String)]) -> Option<f64> {
    let (_, master) = files.iter().find(|(path, _)| path == "master")?;
    master
        .lines()
        .find_map(|l| l.strip_prefix("gain = "))
        .and_then(|v| v.trim().parse().ok())
}

pub fn nodes(sources: &[Source], clip: &Clip) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = sources
        .iter()
        .map(|a| (a.node(), a.node_text(clip)))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The ranges a source's numbers are priced over (bits.rs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Feasible {
    pub hz: (f64, f64),
    pub secs: (f64, f64),
    pub db: (f64, f64),
}

impl Feasible {
    pub(crate) fn of(clip: &Clip, db: (f64, f64)) -> Feasible {
        Feasible {
            hz: (
                cam_hz(CAM_FIRST),
                cam_hz(CAM_LAST).min(f64::from(clip.rate) / 2.0),
            ),
            secs: (0.0, clip.secs()),
            db,
        }
    }

    pub(crate) fn hz_of(&self, source: &Source) -> (f64, f64) {
        (self.hz.0, self.hz.1 / source.ratio(source.harmonics()))
    }

    pub(crate) fn offset(&self) -> f64 {
        self.db.1 - self.db.0
    }
}

pub fn sounded(files: Vec<(String, String)>, clip: &Clip) -> Result<Vec<f64>, CliError> {
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
    let rendered = execute(job, &Tier::default())?;
    let answer = rendered.answer(&rendered.target, Representation::Samples)?;
    let Output::Samples(buffer) = answer.value else {
        unreachable!("samples answer samples")
    };
    Ok(buffer.plane(0).to_vec())
}

/// FNV-1a, the same on every platform.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}
