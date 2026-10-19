// Concern: attributes a target's energy down its ref tree, one entry per node, a stretch at a time | Non-concern: rendering the buffers it reads | IO: (sources, deps, target) -> Tally -> entries

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::buffer::Buffer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Amplitude,
    Hertz,
    Q,
    Decibels,
    Seconds,
    Radians,
    Ratio,
}

impl Unit {
    pub fn name(self) -> &'static str {
        match self {
            Unit::Amplitude => "amplitude",
            Unit::Hertz => "hz",
            Unit::Q => "q",
            Unit::Decibels => "db",
            Unit::Seconds => "seconds",
            Unit::Radians => "radians",
            Unit::Ratio => "ratio",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalKind {
    Audio,
    Control(Unit),
}

impl SignalKind {
    pub fn is_audio(self) -> bool {
        self == SignalKind::Audio
    }

    pub fn unit(self) -> Unit {
        match self {
            SignalKind::Audio => Unit::Amplitude,
            SignalKind::Control(u) => u,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LedgerEntry {
    pub node: String,
    pub channel: Option<usize>,
    pub depth: usize,
    pub kind: SignalKind,
    pub rms: f64,
    pub peak: f64,
    pub min: f64,
    pub max: f64,
    /// The part of its reader's own energy this node accounts for, so one reader's refs add
    /// to one over a sum. `None` for a non-audio kind, a ref no addend isolates, and under a
    /// silent reader.
    pub share: Option<f64>,
    /// A sample outside [-1, 1], the full scale a `.wav` destination clamps to, so a peak of
    /// exactly 1.0 is not clipped.
    pub clipped: Option<bool>,
}

/// `read`: what a row gave its reader; `own`: a reader's own samples, which shares are of.
pub struct Sources<'a> {
    pub read: &'a BTreeMap<String, usize>,
    pub own: &'a BTreeMap<String, usize>,
    pub attributed: &'a BTreeSet<String>,
    pub widths: &'a [usize],
}

/// Each sum one running fold in sample order: the bits a sum over the whole buffer gives.
pub struct Tally {
    rows: Vec<Row>,
}

struct Row {
    node: String,
    depth: usize,
    kind: SignalKind,
    attributed: bool,
    read: Option<usize>,
    reader: Option<usize>,
    channels: Vec<Sums>,
}

#[derive(Clone, Copy)]
struct Sums {
    squares: f64,
    count: usize,
    peak: f64,
    min: f64,
    max: f64,
    product: f64,
    energy: f64,
}

impl Sums {
    fn new() -> Sums {
        // What an empty `sum` folds from, so a running fold is that sum's bits.
        let zero: f64 = std::iter::empty::<f64>().sum();
        Sums {
            squares: zero,
            count: 0,
            peak: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            product: zero,
            energy: zero,
        }
    }
}

impl Tally {
    pub fn new(
        sources: &Sources,
        deps: &BTreeMap<String, Vec<String>>,
        kinds: &BTreeMap<String, SignalKind>,
        target: &str,
        depth_limit: usize,
    ) -> Tally {
        let width = |path: &str| sources.read.get(path).map_or(1, |s| sources.widths[*s]);
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut queue: VecDeque<(&str, usize, &str)> = VecDeque::new();
        let mut rows = Vec::new();
        queue.push_back((target, 0, target));
        seen.insert(target);
        while let Some((path, depth, reader)) = queue.pop_front() {
            // `kinds` names what holds a buffer; a node composed into another holds none.
            rows.push(Row {
                node: path.to_string(),
                depth,
                kind: kinds.get(path).copied().unwrap_or(SignalKind::Audio),
                attributed: path == target || sources.attributed.contains(path),
                read: sources.read.get(path).copied(),
                reader: sources
                    .own
                    .get(reader)
                    .copied()
                    .filter(|s| sources.widths[*s] > 0),
                channels: vec![Sums::new(); width(path)],
            });
            if depth >= depth_limit {
                continue;
            }
            for child in deps.get(path).map(Vec::as_slice).unwrap_or(&[]) {
                if seen.insert(child.as_str()) {
                    let key = deps
                        .get_key_value(child.as_str())
                        .map(|(k, _)| k.as_str())
                        .unwrap_or(child.as_str());
                    queue.push_back((key, depth + 1, path));
                }
            }
        }
        Tally { rows }
    }

    pub fn add(&mut self, held: &[Option<&Buffer>], range: std::ops::Range<usize>) {
        let empty: &[f64] = &[];
        let slice = |source: Option<usize>, c: usize| -> Option<std::borrow::Cow<'_, [f64]>> {
            let buffer = held[source?]?;
            Some(buffer.plane_slice(c % buffer.width(), range.clone()))
        };
        for row in &mut self.rows {
            for (c, sums) in row.channels.iter_mut().enumerate() {
                let x = match row.read {
                    Some(source) => slice(Some(source), c).expect("a row's own source"),
                    None => std::borrow::Cow::Borrowed(empty),
                };
                sums.count += x.len();
                sums.squares = x.iter().map(|&v| v * v).fold(sums.squares, |a, b| a + b);
                sums.peak = x.iter().fold(sums.peak, |a, &v| a.max(v.abs()));
                sums.min = x.iter().fold(sums.min, |a, &v| a.min(v));
                sums.max = x.iter().fold(sums.max, |a, &v| a.max(v));
                if let Some(y) = slice(row.reader, c) {
                    sums.energy = y.iter().map(|&v| v * v).fold(sums.energy, |a, b| a + b);
                    let products = x.iter().zip(y.iter()).map(|(a, b)| a * b);
                    sums.product = products.fold(sums.product, |a, b| a + b);
                }
            }
        }
    }

    pub fn entries(&self) -> Vec<LedgerEntry> {
        let mut out = Vec::new();
        for row in &self.rows {
            let components = row.channels.len();
            let gated = row.kind.is_audio();
            for (c, sums) in row.channels.iter().enumerate() {
                let rms = match sums.count {
                    0 => 0.0,
                    count => (sums.squares / count as f64).sqrt(),
                };
                // Over a sum the parts are the whole: one reader's refs add to exactly one.
                let share = || match (row.reader, sums.energy > 0.0) {
                    (Some(_), true) => Some(sums.product / sums.energy),
                    (Some(_), false) => None,
                    (None, _) => Some(0.0),
                };
                out.push(LedgerEntry {
                    node: row.node.clone(),
                    channel: (components > 1).then_some(c),
                    depth: row.depth,
                    kind: row.kind,
                    rms,
                    peak: sums.peak,
                    min: sums.min,
                    max: sums.max,
                    share: (gated && row.attributed).then(share).flatten(),
                    clipped: gated.then_some(sums.peak > 1.0),
                });
            }
        }
        out
    }
}
