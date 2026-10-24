// Concern: states what naming a peak owes a tone and its partials | Non-concern: finding the peaks (measure_spectrum.rs) | IO: (peaks or samples) -> asserted notes

use sva_samples::measure::pitch::{PitchFrame, name_peaks, track};
use sva_samples::measure::spectrum::Peak;

use crate::helpers::partials::stiff_partial;

fn peak(hz: f64, db: f64) -> Peak {
    Peak { hz, db }
}

fn sine(hz: f64, rate: f64, secs: f64) -> Vec<f64> {
    let n = (rate * secs) as usize;
    (0..n)
        .map(|k| libm::sin(std::f64::consts::TAU * hz * k as f64 / rate))
        .collect()
}

#[test]
fn a_peak_is_named_by_the_equal_tempered_note_nearest_it() {
    let named = name_peaks(&[peak(440.0, -6.0), peak(261.6255653, -9.0)], 8);
    assert_eq!(named[0].name, "A4", "440 Hz is A4");
    assert_eq!(named[1].name, "C4", "261.63 Hz is middle C");
    for note in &named {
        assert!(note.cents.abs() < 0.01, "{note:?} sits on the note");
    }
}

fn string(f0: f64, b: f64, gain: f64) -> Vec<(f64, f64)> {
    (1..=8)
        .map(|k| (stiff_partial(f0, b, k), gain / f64::from(k).sqrt()))
        .collect()
}

fn exact_heads(partials: &[(f64, f64)]) -> Vec<String> {
    let mut found: Vec<Peak> = partials
        .iter()
        .map(|(hz, a)| peak(*hz, 20.0 * libm::log10(*a)))
        .collect();
    found.sort_by(|a, b| b.db.total_cmp(&a.db));
    let mut heads: Vec<String> = name_peaks(&found, 32)
        .into_iter()
        .filter(|n| n.harmonic_of.is_none())
        .map(|n| n.name)
        .collect();
    heads.sort();
    heads
}

fn measured(partials: &[(f64, f64)]) -> Vec<PitchFrame> {
    let rate = 44100.0;
    let tone: Vec<f64> = (0..(rate * 0.6) as usize)
        .map(|n| {
            let t = n as f64 / rate;
            let sum: f64 = partials
                .iter()
                .map(|(hz, a)| a * libm::sin(std::f64::consts::TAU * hz * t))
                .sum();
            sum * sva_formula::exp(-2.0 * t)
        })
        .collect();
    track(&[&tone[..]], rate, 0.0, 0.05, 16)
}

/// Each 50 ms frame's notes heading a series within 12 dB of its loudest.
fn measured_heads(partials: &[(f64, f64)]) -> Vec<Vec<String>> {
    measured(partials)
        .into_iter()
        .map(|frame| {
            let loudest = frame.notes.iter().map(|n| n.db).fold(f64::MIN, f64::max);
            let mut heads: Vec<String> = frame
                .notes
                .into_iter()
                .filter(|n| n.harmonic_of.is_none() && n.db >= loudest - 12.0)
                .map(|n| n.name)
                .collect();
            heads.sort();
            heads
        })
        .collect()
}

/// At B = 5e-4 the 7th partial stands 1.2% sharp of 7 f1, past the whole-multiple tolerance.
#[test]
fn a_stiff_string_reads_as_one_note() {
    let c4 = string(261.6256, 5e-4, 1.0);
    assert_eq!(exact_heads(&c4), ["C4"]);
    for heads in measured_heads(&c4) {
        assert_eq!(heads, ["C4"]);
    }
}

/// A neighbour's leakage pulls a measured fundamental a few hertz, its 3rd partial past the
/// tolerance over it: none of partials 3 to 8 may read as a note.
#[test]
fn a_fundamental_pulled_off_by_its_neighbour_still_heads_its_partials() {
    let chord = [string(261.6256, 1.5e-4, 0.7), string(207.6523, 1.5e-4, 1.0)].concat();
    for heads in measured_heads(&chord) {
        assert!(
            heads.contains(&"C4".to_string()) && heads.contains(&"Gs3".to_string()),
            "{heads:?}"
        );
        for partial in ["G5", "C6", "E6", "G6", "As6", "C7"] {
            assert!(!heads.contains(&partial.to_string()), "{heads:?}");
        }
    }
}

/// `peaks` counts notes: eighty louder sines over 5 kHz crowd no faint A3 out.
#[test]
fn peaks_past_the_named_band_crowd_no_note_out() {
    let rate = 44100.0;
    let mut tone: Vec<f64> = sine(220.0, rate, 0.2).iter().map(|x| 0.005 * x).collect();
    for k in 1..=80 {
        let high = sine(5200.0 + 97.0 * f64::from(k), rate, 0.2);
        tone.iter_mut().zip(high).for_each(|(x, h)| *x += h / 80.0);
    }
    for frame in track(&[&tone[..]], rate, 0.0, 0.05, 16) {
        let names: Vec<&str> = frame.notes.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["A3"], "at {} s", frame.t_secs);
    }
}
