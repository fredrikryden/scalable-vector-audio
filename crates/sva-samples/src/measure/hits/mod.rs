// Concern: reads a clip's hits: onsets, spans and energy per band, each with its novelty's strength | Non-concern: what sounds (pitch), where strikes fall | IO: (planes, rate, start) -> Hits

pub(crate) mod spectra;
mod strikes;

pub use strikes::{Cue, Kind};

use strikes::BAND_EDGES_HZ;

use spectra::{Clip, db_of_power};

pub const MODEL: &str = "hits off spectra averaged over Hann frames: an onset is every peak of \
                         a band-wise log superflux, the largest within 4 hops and standing over \
                         its median within 150 ms by its strength, new where the power after it \
                         that was not there in the 60 ms before outweighs the power carried over, \
                         else a re-strike; a period most onsets sit on completes them with soft \
                         re-strikes. Each hit's end 30 dB under its peak, its attack and level \
                         off 5 ms levels, and its energy in its first 120 ms per superflux band, \
                         low, mid and high.";

const END_DB: f64 = 30.0;
const BAND_SECS: f64 = 0.12;
const BAND_SAMPLES_48K: f64 = 1024.0;
/// A band holding no power, as `Clip::levels` reads a span holding no frame.
const NONE_DB: f64 = -200.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    /// Placed at the sample its strike rises.
    pub at_secs: f64,
    /// Where it falls 30 dB under its peak, or the next hit.
    pub end_secs: f64,
    pub next_secs: f64,
    pub kind: Kind,
    pub cue: Cue,
    pub attack_secs: f64,
    pub level_db: f64,
    /// Energy in its first 120 ms, low, mid and high, split at `BAND_EDGES_HZ`.
    pub band_db: [f64; 3],
    /// The novelty peak over its median within 150 ms.
    pub strength: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hits {
    pub resolution_secs: f64,
    pub bands_hz: [f64; 2],
    pub hits: Vec<Hit>,
}

pub(crate) fn mono(planes: &[&[f64]]) -> Vec<f64> {
    let len = super::held(planes);
    let count = planes.len() as f64;
    (0..len)
        .map(|n| planes.iter().map(|p| p[n]).sum::<f64>() / count)
        .collect()
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64) -> Hits {
    let clip = Clip::new(&mono(planes), sr);
    let (strikes, resolution_secs) = strikes::strikes(&clip);
    let energy = Energy::of(&clip.x);
    let placed: Vec<f64> = strikes
        .iter()
        .map(|s| match s.cue {
            Cue::Crop => s.at,
            _ => energy.placed(&clip, s.at),
        })
        .chain([clip.secs()])
        .collect();
    let hits = strikes
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let next = strikes.get(i + 1).map_or(clip.secs(), |n| n.at);
            let h = hit(&clip, s, next);
            let (at, next) = (placed[i], placed[i + 1]);
            Hit {
                at_secs: at + start,
                end_secs: h.end_secs.min(next) + start,
                next_secs: next + start,
                attack_secs: (h.attack_secs + h.at_secs - at).max(0.0),
                ..h
            }
        })
        .collect();
    Hits {
        resolution_secs,
        bands_hz: [BAND_EDGES_HZ.0, BAND_EDGES_HZ.1],
        hits,
    }
}

const STRIKE_SPAN_SECS: f64 = 0.012;
const PLACE_REACH_SECS: f64 = 0.0107;

/// Running energy, for where within a hop a strike rises.
struct Energy {
    running: Vec<f64>,
}

impl Energy {
    fn of(x: &[f64]) -> Energy {
        let mut running = Vec::with_capacity(x.len() + 1);
        running.push(0.0);
        for v in x {
            running.push(running[running.len() - 1] + v * v);
        }
        Energy { running }
    }

    fn over(&self, from: usize, to: usize) -> f64 {
        let last = self.running.len() - 1;
        self.running[to.min(last)] - self.running[from.min(last)]
    }

    /// The sample within the novelty frame about `t` whose 12 ms after most outweighs the
    /// 12 ms before, the clip's mean power over that span under both: where the strike rises.
    fn placed(&self, clip: &Clip, t: f64) -> f64 {
        let reach = (STRIKE_SPAN_SECS * clip.sr).round().max(1.0) as usize;
        let len = self.running.len() - 1;
        let floor = self.over(0, len) / len.max(1) as f64 * reach as f64 * 1e-6;
        let around = (PLACE_REACH_SECS * clip.sr) as usize;
        let at = clip.at(t);
        let mut best = (f64::NEG_INFINITY, at);
        for n in at.saturating_sub(around)..(at + around).min(len) {
            let ratio = self.over(n, n + reach) / (self.over(n.saturating_sub(reach), n) + floor);
            if ratio > best.0 {
                best = (ratio, n);
            }
        }
        best.1 as f64 / clip.sr
    }
}

/// The 5 ms frames in `[a, b)` as (time, dB).
fn framed(clip: &Clip, a: f64, b: f64) -> Vec<(f64, f64)> {
    clip.level_t
        .iter()
        .zip(&clip.level_db)
        .filter(|(t, _)| **t >= a && **t < b)
        .map(|(t, v)| (*t, *v))
        .collect()
}

/// The power in `[a, b)` below, between and above the superflux's band edges, in dB.
fn band_db(clip: &Clip, (a, b): (usize, usize)) -> [f64; 3] {
    let n = spectra::sized(BAND_SAMPLES_48K, clip.sr);
    let Some(p) = clip.spectrum_of(a, b, n) else {
        return [NONE_DB; 3];
    };
    let hz = |k: usize| k as f64 * clip.sr / n as f64;
    let held = |lo: f64, hi: f64| {
        let power: f64 = (0..p.len())
            .filter(|&k| hz(k) >= lo && hz(k) < hi)
            .map(|k| p[k])
            .sum();
        match power > 0.0 {
            true => db_of_power(power),
            false => NONE_DB,
        }
    };
    [
        held(0.0, BAND_EDGES_HZ.0),
        held(BAND_EDGES_HZ.0, BAND_EDGES_HZ.1),
        held(BAND_EDGES_HZ.1, f64::INFINITY),
    ]
}

fn hit(clip: &Clip, s: &strikes::Strike, b: f64) -> Hit {
    let a = s.at;
    let frames = framed(clip, a, b);
    let peak = clip.loudest(a, b);
    let crest = match frames.len() > 1 {
        true => {
            frames
                .iter()
                .fold(
                    (a, f64::NEG_INFINITY),
                    |m, &(t, v)| if v > m.1 { (t, v) } else { m },
                )
                .0
        }
        false => a,
    };
    let end = frames
        .iter()
        .find(|(t, v)| *t > crest && *v < peak - END_DB)
        .map_or(b, |(t, _)| *t);
    Hit {
        at_secs: a,
        end_secs: end.min(b),
        next_secs: b,
        kind: s.kind,
        cue: s.cue,
        attack_secs: crest - a,
        level_db: peak,
        band_db: band_db(clip, (clip.at(a), clip.at(b.min(a + BAND_SECS)))),
        strength: s.strength,
    }
}
