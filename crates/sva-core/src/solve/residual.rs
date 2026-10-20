// Concern: subtracts two readings of one kind into a residual vector, each kind by its stated rule | Non-concern: taking or reducing them | IO: (Output, Output) -> Vec<f64>

use sva_engine::Output;

/// Every level is floored here before it subtracts: the profile's silence threshold.
pub const FLOOR_DB: f64 = -120.0;
/// What an onset with no partner on the other side counts, in seconds.
pub const MISSED_ONSET_SECS: f64 = 1.0;
/// What a note with no partner in its frame counts, in cents: an octave.
pub const MISSED_NOTE_CENTS: f64 = 1200.0;

/// One line per rule, as `--help` prints them.
pub const RULES: [(&str, &str); 7] = [
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
        "onsets",
        "a set: nearest pairs first, each pair's distance in seconds less both onsets' \
         `resolution_secs`, zero inside them; each onset left unpaired counts 1 s",
    ),
    (
        "pitch",
        "frame by frame, each frame's notes a set: nearest pairs first, in cents; each note \
         left unpaired counts 1200 cents",
    ),
];

fn db(level: f64) -> f64 {
    match level > 0.0 {
        true => (20.0 * level.log10()).max(FLOOR_DB),
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

/// Pairs the nearest two left, ties to the earlier, until one side runs out.
fn sets(a: &[f64], b: &[f64], zone: f64, missed: f64) -> Vec<f64> {
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
        out.push((distance - zone).max(0.0));
    }
    let left = used_a.iter().chain(&used_b).filter(|used| !**used).count();
    out.extend(std::iter::repeat_n(missed, left));
    out
}

fn cents(hz: f64) -> f64 {
    1200.0 * hz.log2()
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
        (Output::Onsets(a), Output::Onsets(b)) => {
            let times = |o: &sva_engine::Onsets| -> Vec<f64> {
                o.onsets.iter().map(|onset| onset.t_secs).collect()
            };
            let zone = a.resolution_secs + b.resolution_secs;
            sets(&times(a), &times(b), zone, MISSED_ONSET_SECS)
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
