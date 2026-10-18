// Concern: names each frame's spectral peaks as notes and flags likely harmonics | Non-concern: computing the spectrum (spectrum.rs), chord or key identity | IO: (planes, sample rate) -> frames

use crate::measure::spectrum::{Peak, magnitudes_of, peaks};
use crate::measure::stiff::{fit, stretch};

const NAMES: [&str; 12] = [
    "C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B",
];
pub const LOWEST_HZ: f64 = 25.0;
pub const HIGHEST_HZ: f64 = 5000.0;
const HARMONIC_TOLERANCE: f64 = 0.04;
const HIGHEST_HARMONIC: f64 = 8.0;
const COMPRESSION: f64 = 0.84;

#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub hz: f64,
    pub name: String,
    pub cents: f64,
    pub db: f64,
    pub harmonic_of: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchFrame {
    pub t_secs: f64,
    pub notes: Vec<Note>,
}

pub fn track(
    planes: &[&[f64]],
    sample_rate: f64,
    start_secs: f64,
    frame_secs: f64,
    max_notes: usize,
) -> Vec<PitchFrame> {
    let stride = ((frame_secs * sample_rate).round() as usize).max(1);
    let len = crate::measure::held(planes);
    (0..len)
        .step_by(stride)
        .enumerate()
        .map(|(n, from)| {
            let chunk: Vec<&[f64]> = planes
                .iter()
                .map(|p| &p[from..len.min(from + stride)])
                .collect();
            let (mags, bin_hz, ..) = magnitudes_of(&chunk, sample_rate, None);
            let found = peaks(&mags, bin_hz, max_notes * 4);
            let resolution_hz = sample_rate / chunk[0].len().max(1) as f64;
            PitchFrame {
                t_secs: start_secs + (n * stride) as f64 / sample_rate,
                notes: name_all(&found, max_notes, resolution_hz / 2.0),
            }
        })
        .collect()
}

/// The same naming over peaks a reading already holds exactly, loudest first.
pub fn name_peaks(found: &[Peak], max_notes: usize) -> Vec<Note> {
    name_all(found, max_notes, 0.0)
}

/// `spread_hz`: how far a peak may stand from where it was measured, 0 where exact.
fn name_all(found: &[Peak], max_notes: usize, spread_hz: f64) -> Vec<Note> {
    let listed: Vec<&Peak> = found
        .iter()
        .filter(|p| (LOWEST_HZ..=HIGHEST_HZ).contains(&p.hz))
        .take(max_notes)
        .collect();
    let roots = series_roots(&listed, spread_hz);

    listed
        .iter()
        .zip(roots)
        .map(|(p, harmonic_of)| {
            let midi = midi_of(p.hz);
            let nearest = midi.round();
            Note {
                hz: p.hz,
                name: note_name(nearest as i64),
                cents: (midi - nearest) * 100.0,
                db: p.db,
                harmonic_of,
            }
        })
        .collect()
}

pub fn midi_of(hz: f64) -> f64 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

pub fn note_name(midi: i64) -> String {
    format!("{}{}", class_name(midi), midi.div_euclid(12) - 1)
}

pub fn class_name(midi: i64) -> &'static str {
    NAMES[midi.rem_euclid(12) as usize]
}

/// Klapuri (ISMIR 2006): the greatest subharmonic sum (Hermes, JASA 83(1), 1988) roots a
/// series; a note 2 to 8 times a louder member joins it, or the root if their pair's sum there
/// reaches its amplitude. Exact peaks may stand as sharp as the series' stiff string fits.
fn series_roots(notes: &[&Peak], spread_hz: f64) -> Vec<Option<f64>> {
    let mut joined = vec![None; notes.len()];
    let mut left: Vec<usize> = (0..notes.len()).collect();
    while let Some((root, f0)) = most_salient(notes, &left, spread_hz) {
        left.retain(|&i| i != root);
        let mut series = vec![Member {
            at: root,
            n: 1,
            hz: f0,
        }];
        loop {
            let points: Vec<(u32, f64)> = series.iter().map(|m| (m.n, m.hz)).collect();
            let b = match spread_hz > 0.0 {
                true => 0.0,
                // No fit, or a negative slope, is no stiffness: the plain whole-multiple window.
                false => fit(&points).unwrap_or(0.0).max(0.0),
            };
            let joins = |m: &Member, i: usize| {
                over(notes[i].hz, m, b).filter(|&q| {
                    let (under, over) = (amp(notes[m.at].db), amp(notes[i].db));
                    under > over
                        || m.at == root && under + COMPRESSION.powi(q as i32 - 1) * over >= over
                })
            };
            let Some((at, m, q)) = left
                .iter()
                .enumerate()
                .find_map(|(at, &i)| series.iter().find_map(|m| joins(m, i).map(|q| (at, *m, q))))
            else {
                break;
            };
            let i = left.remove(at);
            joined[i] = Some(notes[m.at].hz);
            series.push(Member {
                at: i,
                n: q * m.n,
                hz: notes[i].hz,
            });
        }
    }
    joined
}

#[derive(Clone, Copy)]
struct Member {
    at: usize,
    n: u32,
    hz: f64,
}

/// q from 2 to 8, `hz` lying from q times `m` to where stiffness `b` puts partial q n over n.
fn over(hz: f64, m: &Member, b: f64) -> Option<u32> {
    let ratio = hz / m.hz;
    let q = ratio.round();
    let (n, k) = (f64::from(m.n), q * f64::from(m.n));
    let sharpest = q * stretch(k, b) / stretch(n, b);
    ((2.0..=HIGHEST_HARMONIC).contains(&q)
        && ratio > q - HARMONIC_TOLERANCE
        && ratio < sharpest.max(q) + HARMONIC_TOLERANCE)
        .then_some(q as u32)
}

fn amp(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn salience(notes: &[&Peak], among: &[usize], hz: f64) -> f64 {
    let mut strongest = [0.0f64; HIGHEST_HARMONIC as usize];
    for &i in among {
        if let Some(k) = multiple(notes[i].hz, hz) {
            let at = &mut strongest[k as usize - 1];
            *at = at.max(amp(notes[i].db));
        }
    }
    strongest
        .iter()
        .enumerate()
        .map(|(k, a)| COMPRESSION.powi(k as i32) * a)
        .sum::<f64>()
}

fn most_salient(notes: &[&Peak], left: &[usize], spread_hz: f64) -> Option<(usize, f64)> {
    left.iter()
        .map(|&c| {
            let f0 = fundamental(notes, left, c, spread_hz);
            (c, f0, salience(notes, left, f0))
        })
        .max_by(|(a, _, sa), (b, _, sb)| {
            sa.total_cmp(sb).then(notes[*b].hz.total_cmp(&notes[*a].hz))
        })
        .map(|(c, f0, _)| (c, f0))
}

/// Leakage pulls a measured peak further than stiffness: the fundamental in the root's spread
/// explaining every note its peak does, and more, with the greatest sum.
fn fundamental(notes: &[&Peak], left: &[usize], root: usize, spread_hz: f64) -> f64 {
    let at = notes[root].hz;
    if spread_hz <= 0.0 {
        return at;
    }
    let explained = |f: f64| -> Vec<bool> {
        left.iter()
            .map(|&i| multiple(notes[i].hz, f).is_some())
            .collect()
    };
    let own = explained(at);
    let keeps = |f: f64| explained(f).iter().zip(&own).all(|(e, o)| *e || !o);
    let near = |f: f64| (f - at).abs() <= spread_hz && (at / f - 1.0).abs() < HARMONIC_TOLERANCE;
    left.iter()
        .flat_map(|&i| (1..=HIGHEST_HARMONIC as u32).map(move |k| notes[i].hz / f64::from(k)))
        .filter(|f| near(*f) && keeps(*f))
        .map(|f| (f, salience(notes, left, f)))
        .fold((at, salience(notes, left, at)), |held, next| {
            let nearer = (held.0 - at).abs().total_cmp(&(next.0 - at).abs());
            match next.1.total_cmp(&held.1).then(nearer) {
                std::cmp::Ordering::Greater => next,
                _ => held,
            }
        })
        .0
}

fn multiple(hz: f64, root: f64) -> Option<u32> {
    let n = hz / root;
    let k = n.round();
    ((1.0..=HIGHEST_HARMONIC).contains(&k) && (n - k).abs() < HARMONIC_TOLERANCE)
        .then_some(k as u32)
}
