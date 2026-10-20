// Concern: reads a clip as a score: events, each one's sound, hit and voices, new ones flagged | Non-concern: where strikes fall (strikes.rs), the sieve (voices.rs) | IO: (planes, rate, start) -> Score

mod spectra;
mod strikes;
mod voices;

pub use strikes::{Cue, Kind};
pub use voices::{Hit, Partial};

use spectra::{Clip, cents};
use voices::{Voice, coherence, hit, period, voices};

pub const MODEL: &str = "events off spectra averaged over Hann frames: an onset is a peak \
                         of a band-wise log superflux (an attack) or of a partial flux (a \
                         partial appearing or rising: a soft or legato attack), kept only where \
                         something new sounds after it: power new against the 60 ms before (a \
                         new note), or a level rise out of a dip with an air tick (a \
                         re-strike); a regular level wobble under 80 ms (beats, tremolo) or a \
                         glide landing within 300 cents of what sounded is none; one per 60 ms; \
                         a period most onsets sit on completes them with soft re-strikes. Each \
                         event's voices by a harmonic sieve on the power that is new (a voice \
                         new at it) and on all that sounds, a fundamental where its own peak \
                         is heard, each read again off the event's later part (a bend glides \
                         in), tonal or noisy by its loudest partials' coherence over 10 ms, a \
                         noisy voice taking the event's periodicity (McLeod's NSDF) where that \
                         lies within 60 cents of an octave or two of it; the event tonal, \
                         noisy-pitched or noisy by its share of power in prominent peaks and its \
                         periodicity, an unpitched or short hit a kick, snare or hat by band.";

const VOICE_SPAN_SECS: f64 = 0.3;
const SETTLE_SECS: f64 = 0.015;
const PRE_SECS: (f64, f64) = (0.07, 0.01);
const NEW_OVER: f64 = 1.5;
const SHORT_SECS: f64 = 0.03;
const TONAL_SHARE: (f64, f64) = (0.4, 0.15);
const TONAL_PERIOD: (f64, f64) = (0.7, 0.4);
const END_DB: f64 = 30.0;
const HIT_DECAY_DB: f64 = 20.0;
const HIT_SECS: f64 = 0.12;
const LATE_SHARE: f64 = 0.4;
const LATE_LEAST_SECS: f64 = 0.05;
const BEND_CENTS: f64 = 250.0;
const TONE_COHERENCE: f64 = 0.8;
const PERIOD_CENTS: f64 = 60.0;
const SAME_VOICE_CENTS: f64 = 50.0;
const PERIOD_LEAST_48K: f64 = 1024.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sound {
    Tonal,
    NoisyPitched,
    Noisy,
}

impl Sound {
    pub fn key(self) -> &'static str {
        match self {
            Sound::Tonal => "tonal",
            Sound::NoisyPitched => "noisy_pitched",
            Sound::Noisy => "noisy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Tonal,
    Noisy,
}

impl Tone {
    pub fn key(self) -> &'static str {
        match self {
            Tone::Tonal => "tonal",
            Tone::Noisy => "noisy",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Voiced {
    pub hz: f64,
    pub tone: Tone,
    /// Sounding newly at the event; every voice of a re-strike or a held start.
    pub new: bool,
    pub share: f64,
    pub db: f64,
    pub bend_cents: Option<f64>,
    pub coherence: Option<f64>,
    pub partials: Vec<Partial>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub at_secs: f64,
    /// Where it falls 30 dB under its peak, or the next event.
    pub end_secs: f64,
    pub next_secs: f64,
    pub kind: Kind,
    pub cue: Cue,
    pub sound: Sound,
    pub hit: Option<Hit>,
    pub attack_secs: f64,
    pub level_db: f64,
    pub period: Option<(f64, f64)>,
    pub voices: Vec<Voiced>,
    /// The loudest new voice, else the loudest.
    pub main: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub resolution_secs: f64,
    pub events: Vec<Event>,
}

fn mono(planes: &[&[f64]]) -> Vec<f64> {
    let len = super::held(planes);
    let count = planes.len() as f64;
    (0..len)
        .map(|n| planes.iter().map(|p| p[n]).sum::<f64>() / count)
        .collect()
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64) -> Score {
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
    let events = strikes
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let next = strikes.get(i + 1).map_or(clip.secs(), |n| n.at);
            let e = event(&clip, s, next)?;
            let (at, next) = (placed[i], placed[i + 1]);
            Some(Event {
                at_secs: at + start,
                end_secs: e.end_secs.min(next) + start,
                next_secs: next + start,
                attack_secs: (e.attack_secs + e.at_secs - at).max(0.0),
                ..e
            })
        })
        .collect();
    Score {
        resolution_secs,
        events,
    }
}

/// A voice's frequency where it lands, and the bend it glided in from, in cents.
type Settled = (f64, Option<f64>);

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

fn event(clip: &Clip, s: &strikes::Strike, b: f64) -> Option<Event> {
    let a = s.at;
    let span_end = b.min(a + VOICE_SPAN_SECS);
    let post = match b - a > SHORT_SECS {
        true => clip.spectrum(a + SETTLE_SECS, span_end)?,
        false => clip.spectrum(a, b)?,
    };
    let pre = clip.spectrum(a - PRE_SECS.0, a - PRE_SECS.1);
    let diff: Vec<f64> = match (&pre, s.kind) {
        (Some(pre), k) if k != Kind::Held => post
            .iter()
            .zip(pre)
            .map(|(p, q)| (p - NEW_OVER * q).max(0.0))
            .collect(),
        _ => post.clone(),
    };
    let all = voices(&clip.peaks(&post, spectra::PEAK_FLOOR_DB, spectra::PEAK_PROMINENCE_DB));
    let fresh = match (s.kind, pre.is_some() && diff.iter().sum::<f64>() > 0.0) {
        (Kind::New, true) => {
            voices(&clip.peaks(&diff, spectra::PEAK_FLOOR_DB, spectra::PEAK_PROMINENCE_DB))
        }
        (Kind::New, false) => all.clone(),
        _ => Vec::new(),
    };
    let seg = &clip.x[clip.at(a + SETTLE_SECS)..clip.at(span_end).max(clip.at(a + SETTLE_SECS))];
    let periodic = match seg.len() as f64 > PERIOD_LEAST_48K * clip.sr / 48_000.0 {
        true => period(seg, clip.sr),
        false => None,
    };
    let clarity = periodic.map_or(0.0, |p| p.1);
    let prominent = clip.peaks(&post, -50.0, spectra::PEAK_PROMINENCE_DB);
    let tonal_share = prominent
        .iter()
        .map(|&(hz, _)| {
            let k = (hz / clip.bin_hz) as usize;
            post[k.saturating_sub(3)..(k + 4).min(post.len())]
                .iter()
                .sum::<f64>()
        })
        .sum::<f64>()
        / post.iter().sum::<f64>();
    let sound = if tonal_share > TONAL_SHARE.0 || clarity > TONAL_PERIOD.0 {
        Sound::Tonal
    } else if tonal_share > TONAL_SHARE.1 || clarity > TONAL_PERIOD.1 {
        Sound::NoisyPitched
    } else {
        Sound::Noisy
    };

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
    let under = |db: f64| {
        frames
            .iter()
            .find(|(t, v)| *t > crest && *v < peak - db)
            .map(|(t, _)| *t)
    };
    let end = under(END_DB).unwrap_or(b);
    let short = under(HIT_DECAY_DB).is_some_and(|t| t - crest < HIT_SECS);
    let hit = match sound != Sound::Tonal || (short && s.kind == Kind::New) {
        true => hit(clip, (clip.at(a), clip.at(b.min(a + HIT_SECS)))),
        false => None,
    };

    let late = match b - a > HIT_SECS {
        true => clip
            .spectrum(a + LATE_LEAST_SECS.max(LATE_SHARE * (b - a)), b - 0.01)
            .map(|p| voices(&clip.peaks(&p, spectra::PEAK_FLOOR_DB, spectra::PEAK_PROMINENCE_DB))),
        false => None,
    };
    let late_from = a + LATE_LEAST_SECS.max(LATE_SHARE * (b - a));
    let late_seg = &clip.x[clip.at(late_from).min(clip.at(b - 0.01))..clip.at(b - 0.01)];
    let landed =
        match b - a > HIT_SECS && late_seg.len() as f64 > PERIOD_LEAST_48K * clip.sr / 48_000.0 {
            true => period(late_seg, clip.sr),
            false => None,
        }
        .or(periodic);
    let settle = |v: &Voice| -> (f64, Option<f64>) {
        let landing = late.as_ref().and_then(|late| {
            late.iter()
                .filter(|w| cents(w.hz, v.hz).abs() < BEND_CENTS)
                .min_by(|x, y| {
                    libm::fabs(libm::log2(x.hz / v.hz))
                        .total_cmp(&libm::fabs(libm::log2(y.hz / v.hz)))
                })
        });
        match landing {
            Some(w) => (w.hz, Some(cents(v.hz, w.hz).round_ties_even())),
            None => (v.hz, None),
        }
    };
    let fresh: Vec<(&Voice, (f64, Option<f64>))> = fresh.iter().map(|v| (v, settle(v))).collect();
    let others = all.iter().map(|v| (v, settle(v))).filter(|(_, (hz, _))| {
        fresh
            .iter()
            .all(|(_, (n, _))| cents(*hz, *n).abs() > SAME_VOICE_CENTS)
    });
    let read: Vec<(&Voice, Settled, bool)> = fresh
        .iter()
        .map(|&(v, settled)| (v, settled, true))
        .chain(others.map(|(v, settled)| (v, settled, s.kind != Kind::New)))
        .collect();
    let voiced: Vec<Voiced> = read
        .into_iter()
        .map(|(v, (hz, bend_cents), new)| {
            let mut loudest: Vec<&Partial> = v.partials.iter().collect();
            loudest.sort_by(|x, y| y.db.total_cmp(&x.db));
            loudest.truncate(3);
            let coherence = coherence(clip, (a + SETTLE_SECS, span_end), &loudest);
            let tone = match coherence.is_some_and(|c| c >= TONE_COHERENCE) {
                true => Tone::Tonal,
                false => Tone::Noisy,
            };
            let hz = match (tone, landed) {
                (Tone::Noisy, Some((per, _))) => {
                    let k = libm::log2(hz / per).round_ties_even();
                    let octave = per * libm::pow(2.0, k);
                    match k.abs() <= 2.0 && cents(hz, octave).abs() < PERIOD_CENTS {
                        true => per,
                        false => hz,
                    }
                }
                _ => hz,
            };
            Voiced {
                hz,
                tone,
                new,
                share: v.share,
                db: v.db,
                bend_cents,
                coherence,
                partials: v.partials.clone(),
            }
        })
        .collect();
    let pool: Vec<usize> = match voiced.iter().any(|v| v.new) {
        true => (0..voiced.len()).filter(|&i| voiced[i].new).collect(),
        false => (0..voiced.len()).collect(),
    };
    let main = pool.into_iter().fold(None::<usize>, |best, i| match best {
        Some(j) if voiced[i].share <= voiced[j].share => Some(j),
        _ => Some(i),
    });
    Some(Event {
        at_secs: a,
        end_secs: end.min(b),
        next_secs: b,
        kind: s.kind,
        cue: s.cue,
        sound,
        hit,
        attack_secs: crest - a,
        level_db: peak,
        period: periodic,
        voices: voiced,
        main,
    })
}
