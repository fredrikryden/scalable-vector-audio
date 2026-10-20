// Concern: subtracts two readings of one kind into a residual vector, each kind by its stated rule | Non-concern: taking or reducing them | IO: (Output, Output) -> Vec<f64>

use sva_engine::Output;

/// Every level is floored here before it subtracts: the profile's silence threshold.
pub const FLOOR_DB: f64 = -120.0;
/// What an event with no partner on the other side counts, in seconds.
pub const MISSED_ONSET_SECS: f64 = 1.0;
/// What a panel's note with no partner counts, in just-noticeable differences.
pub const MISSED_NOTE_JNDS: f64 = 10.0;
/// Each panel axis's just-noticeable difference, after PERCEPTUAL's provisional table.
pub const PITCH_JND_CENTS: (f64, f64) = (5.0, 10.0);
pub const SHORT_NOTE_SECS: f64 = 0.1;
pub const LOW_NOTE_HZ: f64 = 100.0;
/// Two voices further apart than this are no partners: a voicing differs, not a pitch.
pub const PARTNER_CENTS: f64 = 600.0;
pub const SHARPNESS_JND_PCT: f64 = 5.0;
pub const DECAY_JND_PCT: f64 = 15.0;
pub const DECAYING_PHON_PER_SEC: f64 = -5.0;
pub const ATTACK_JND_PCT: f64 = 25.0;
pub const BEATING_JND_PHON: f64 = 0.3;
/// What a note with no partner in its frame counts, in cents: an octave.
pub const MISSED_NOTE_CENTS: f64 = 1200.0;

/// One line per rule, as `--help` prints them.
pub const RULES: [(&str, &str); 9] = [
    (
        "envelope",
        "frame by frame, each frame's rms in dB, frame k of one side against frame k of the other",
    ),
    (
        "loudness",
        "the integrated loudness in LU, then each momentary block's, block k against block k",
    ),
    (
        "bands",
        "band by band, then frame by frame along each band's 2 kHz track, in dB",
    ),
    (
        "crest",
        "the broadband crest in dB, then each third-octave band's that both sides count",
    ),
    (
        "spectrum",
        "each third-octave band's level in dB, band by band",
    ),
    (
        "score",
        "its events a set by onset: nearest pairs first, each pair's distance in seconds less \
         both sides' `resolution_secs`, zero inside them, each event left unpaired counting 1 \
         s; then each pair's voices a set, nearest pairs first, in cents, each voice left \
         unpaired counting 1200 cents",
    ),
    (
        "pitch",
        "frame by frame, each frame's notes a set: nearest pairs first, in cents; each note \
         left unpaired counts 1200 cents",
    ),
    (
        "loudness_pattern",
        "frame by frame, band by band, 10 log2((a + T) / (b + T)) in phon, T the larger \
         `threshold_sone`, zero where both sides are at or under T, a frame or band one side \
         lacks read as silent",
    ),
    (
        "panel",
        "in just-noticeable differences, the whole's axes, then each note paired as `score` \
         pairs events: each voice's pitch paired nearest first within 600 cents, in cents over \
         5 (10 for a note under 100 ms or a voice under 100 Hz), \
         sharpness 100 ln(a / b) over 5 %, decay 100 ln(a / b) over 15 % where both fall \
         faster than 5 phon/s, attack 100 ln(a / b) over 25 %, beating a - b over 0.3 phon; \
         an axis either side lacks counts zero, each note left unpaired 10",
    ),
];

fn db(level: f64) -> f64 {
    match level > 0.0 {
        true => (20.0 * libm::log10(level)).max(FLOOR_DB),
        false => FLOOR_DB,
    }
}

fn floored(lufs: Option<f64>) -> f64 {
    lufs.filter(|v| v.is_finite())
        .map_or(FLOOR_DB, |v| v.max(FLOOR_DB))
}

/// Frame k against frame k; a frame one side lacks reads as the floor.
fn tracks(a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..a.len().max(b.len()))
        .map(|k| a.get(k).copied().unwrap_or(FLOOR_DB) - b.get(k).copied().unwrap_or(FLOOR_DB))
        .collect()
}

/// Pairs the nearest two left, ties to the earlier, until one side runs out: each pair, its
/// distance, and how many were left unpaired.
fn paired(a: &[f64], b: &[f64]) -> (Vec<(usize, usize, f64)>, usize) {
    let mut pairs: Vec<(f64, usize, usize)> = Vec::with_capacity(a.len() * b.len());
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            pairs.push(((x - y).abs(), i, j));
        }
    }
    pairs.sort_by(|p, q| p.0.total_cmp(&q.0).then((p.1, p.2).cmp(&(q.1, q.2))));
    let (mut used_a, mut used_b) = (vec![false; a.len()], vec![false; b.len()]);
    let mut out = Vec::new();
    for (distance, i, j) in pairs {
        if used_a[i] || used_b[j] {
            continue;
        }
        used_a[i] = true;
        used_b[j] = true;
        out.push((i, j, distance));
    }
    let left = used_a.iter().chain(&used_b).filter(|used| !**used).count();
    (out, left)
}

fn sets(a: &[f64], b: &[f64], zone: f64, missed: f64) -> Vec<f64> {
    let (pairs, left) = paired(a, b);
    let mut out: Vec<f64> = pairs.iter().map(|p| (p.2 - zone).max(0.0)).collect();
    out.extend(std::iter::repeat_n(missed, left));
    out
}

/// `100 ln(a / b)` over `jnd` percent, where both sides hold the axis.
fn ratio(a: Option<f64>, b: Option<f64>, jnd: f64) -> f64 {
    match (a, b) {
        (Some(a), Some(b)) if a > 0.0 && b > 0.0 => 100.0 * libm::log(a / b) / jnd,
        _ => 0.0,
    }
}

fn axes(a: &sva_engine::panel::Axes, b: &sva_engine::panel::Axes) -> [f64; 4] {
    let decaying = |d: Option<f64>| d.filter(|v| *v < DECAYING_PHON_PER_SEC).map(|v| -v);
    let beating = match (a.beating_phon, b.beating_phon) {
        (Some(a), Some(b)) => (a - b) / BEATING_JND_PHON,
        _ => 0.0,
    };
    [
        ratio(a.sharpness_acum, b.sharpness_acum, SHARPNESS_JND_PCT),
        ratio(
            decaying(a.decay_phon_per_sec),
            decaying(b.decay_phon_per_sec),
            DECAY_JND_PCT,
        ),
        ratio(a.attack_secs, b.attack_secs, ATTACK_JND_PCT),
        beating,
    ]
}

/// Each voice paired nearest first within half an octave, in cents over the pitch JND.
fn pitches(a: &sva_engine::panel::Note, b: &sva_engine::panel::Note) -> Vec<f64> {
    let short = a.length_secs.min(b.length_secs) < SHORT_NOTE_SECS;
    let at = |n: &sva_engine::panel::Note| -> Vec<f64> {
        n.pitches_hz.iter().map(|hz| cents(*hz)).collect()
    };
    let (x, y) = (at(a), at(b));
    let (pairs, _) = paired(&x, &y);
    pairs
        .into_iter()
        .filter(|p| p.2 <= PARTNER_CENTS)
        .map(|(i, j, _)| {
            let low = a.pitches_hz[i].min(b.pitches_hz[j]) < LOW_NOTE_HZ;
            let jnd = if short || low {
                PITCH_JND_CENTS.1
            } else {
                PITCH_JND_CENTS.0
            };
            (x[i] - y[j]) / jnd
        })
        .collect()
}

fn cents(hz: f64) -> f64 {
    1200.0 * libm::log2(hz)
}

/// `None` where the two are not one kind; the cost's typing already refused that.
pub fn residual(a: &Output, b: &Output) -> Option<Vec<f64>> {
    Some(match (a, b) {
        (Output::Envelope(a), Output::Envelope(b)) => {
            let level = |frames: &[sva_engine::EnvelopeFrame]| -> Vec<f64> {
                frames.iter().map(|f| db(f.rms)).collect()
            };
            tracks(&level(a), &level(b))
        }
        (Output::Loudness(a), Output::Loudness(b)) => {
            let blocks = |l: &sva_engine::Loudness| -> Vec<f64> {
                l.momentary.iter().map(|f| floored(Some(f.lufs))).collect()
            };
            let mut out = vec![floored(a.integrated_lufs) - floored(b.integrated_lufs)];
            out.extend(tracks(&blocks(a), &blocks(b)));
            out
        }
        (Output::Bands(a), Output::Bands(b)) => {
            let mut out = Vec::new();
            for k in 0..a.bands.len().max(b.bands.len()) {
                let track = |bands: &sva_engine::Bands| -> Vec<f64> {
                    bands
                        .bands
                        .get(k)
                        .map_or_else(Vec::new, |band| band.rms.iter().map(|v| db(*v)).collect())
                };
                out.extend(tracks(&track(a), &track(b)));
            }
            out
        }
        (Output::Crest(a), Output::Crest(b)) => {
            let mut out = vec![a.broadband_crest_db - b.broadband_crest_db];
            for (x, y) in a.bands.iter().zip(&b.bands) {
                if x.counted && y.counted {
                    out.push(x.crest_db - y.crest_db);
                }
            }
            out
        }
        (Output::Spectrum(a), Output::Spectrum(b)) => {
            let level = |s: &sva_engine::Spectrum| -> Vec<f64> {
                s.bands.iter().map(|band| band.db.max(FLOOR_DB)).collect()
            };
            tracks(&level(a), &level(b))
        }
        (Output::Score(a), Output::Score(b)) => {
            let times = |s: &sva_engine::Score| -> Vec<f64> {
                s.events.iter().map(|e| e.at_secs).collect()
            };
            let zone = a.resolution_secs + b.resolution_secs;
            let (pairs, left) = paired(&times(a), &times(b));
            let mut out: Vec<f64> = pairs.iter().map(|p| (p.2 - zone).max(0.0)).collect();
            out.extend(std::iter::repeat_n(MISSED_ONSET_SECS, left));
            for &(i, j, _) in &pairs {
                let voices = |e: &sva_engine::score::Event| -> Vec<f64> {
                    e.voices.iter().map(|v| cents(v.hz)).collect()
                };
                out.extend(sets(
                    &voices(&a.events[i]),
                    &voices(&b.events[j]),
                    0.0,
                    MISSED_NOTE_CENTS,
                ));
            }
            out
        }
        (Output::LoudnessPattern(a), Output::LoudnessPattern(b)) => {
            let t = a.threshold_sone.max(b.threshold_sone);
            let bands = a.centres_hz.len().max(b.centres_hz.len());
            let cell = |p: &sva_engine::loudness_pattern::LoudnessPattern, k: usize, z: usize| {
                p.frames
                    .get(k)
                    .and_then(|f| f.specific.get(z))
                    .copied()
                    .unwrap_or(0.0)
            };
            let mut out = Vec::new();
            for k in 0..a.frames.len().max(b.frames.len()) {
                for z in 0..bands {
                    let (x, y) = (cell(a, k, z), cell(b, k, z));
                    out.push(match x <= t && y <= t {
                        true => 0.0,
                        false => 10.0 * libm::log2((x + t) / (y + t)),
                    });
                }
            }
            out
        }
        (Output::Panel(a), Output::Panel(b)) => {
            let mut out = axes(&a.whole, &b.whole).to_vec();
            let times = |p: &sva_engine::panel::Panel| -> Vec<f64> {
                p.notes.iter().map(|n| n.at_secs).collect()
            };
            let (pairs, left) = paired(&times(a), &times(b));
            for (i, j, _) in pairs {
                let (x, y) = (&a.notes[i], &b.notes[j]);
                out.extend(pitches(x, y));
                out.extend(axes(&x.axes, &y.axes));
            }
            out.extend(std::iter::repeat_n(MISSED_NOTE_JNDS, left));
            out
        }
        (Output::Pitch(a), Output::Pitch(b)) => {
            let mut out = Vec::new();
            for k in 0..a.len().max(b.len()) {
                let notes = |frames: &[sva_engine::PitchFrame]| -> Vec<f64> {
                    frames.get(k).map_or_else(Vec::new, |frame| {
                        frame.notes.iter().map(|n| cents(n.hz)).collect()
                    })
                };
                out.extend(sets(&notes(a), &notes(b), 0.0, MISSED_NOTE_CENTS));
            }
            out
        }
        _ => return None,
    })
}
