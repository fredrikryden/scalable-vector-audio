// Concern: subtracts two readings of one kind into a residual vector, each kind by its stated rule | Non-concern: taking or reducing them | IO: (Output, Output) -> Vec<f64>

use sva_engine::Output;

/// Every level is floored here before it subtracts: the profile's silence threshold.
pub const FLOOR_DB: f64 = -120.0;
/// What an event with no partner on the other side counts, in seconds.
pub const MISSED_ONSET_SECS: f64 = 1.0;
/// What a note with no partner in its frame counts, in cents: an octave.
pub const MISSED_NOTE_CENTS: f64 = 1200.0;

/// One line per rule, as `--help` prints them.
pub const RULES: [(&str, &str); 10] = [
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
        "sharpness",
        "frame by frame, 100 ln(a / b) in percent, zero where either frame is silent or one \
         side lacks it",
    ),
    (
        "beating",
        "window by window, a - b in phon, zero where either window reads null or one side \
         lacks it",
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

fn cents(hz: f64) -> f64 {
    1200.0 * libm::log2(hz)
}

/// Frame k against frame k where both sides hold it, zero elsewhere.
fn held_pairs(a: &[Option<f64>], b: &[Option<f64>], apart: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    (0..a.len().max(b.len()))
        .map(
            |k| match (a.get(k).copied().flatten(), b.get(k).copied().flatten()) {
                (Some(x), Some(y)) => apart(x, y),
                _ => 0.0,
            },
        )
        .collect()
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
        (Output::Sharpness(a), Output::Sharpness(b)) => {
            let acum = |s: &sva_engine::sharpness::Sharpness| -> Vec<Option<f64>> {
                s.frames.iter().map(|f| f.acum).collect()
            };
            held_pairs(&acum(a), &acum(b), |x, y| 100.0 * libm::log(x / y))
        }
        (Output::Beating(a), Output::Beating(b)) => {
            let phon = |s: &sva_engine::beating::Beating| -> Vec<Option<f64>> {
                s.windows.iter().map(|w| w.phon).collect()
            };
            held_pairs(&phon(a), &phon(b), |x, y| x - y)
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
