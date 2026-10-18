// Concern: states that every reading off samples reads every channel in power, one convention for all | Non-concern: what any one reading measures (sva-samples) | IO: (planes) -> readings compared

use sva_core::{Output, Representation};
use sva_engine::{Buffer, PSYCHOACOUSTIC_V1, answer_buffer};

const RATE: u32 = 8_000;

const READINGS: [&str; 12] = [
    "spectrum",
    "pitch",
    "formants",
    "bands",
    "crest",
    "dissonance",
    "voice_leading",
    "tonal_distance",
    "inharmonicity",
    "periodicity",
    "modulation",
    "self_similarity",
];

/// Struck about twice a second and swelling at 3 Hz, so every reading has onsets and motion;
/// its strikes wobble and it grows, so no two windows tie for a maximum.
fn tone(gain: f64) -> Vec<f64> {
    (0..10 * RATE as usize)
        .map(|n| {
            let t = n as f64 / f64::from(RATE);
            let wobbled = t + 0.02 * (std::f64::consts::TAU * 0.37 * t).sin();
            let struck = (-12.0 * (wobbled % 0.5)).exp();
            let swell = (1.0 + 0.07 * t) * (1.0 + 0.3 * (std::f64::consts::TAU * 3.0 * t).sin());
            let partials: f64 = [(1.0, 0.5), (2.0, 0.25), (3.0, 0.125)]
                .iter()
                .map(|(k, a)| a * (std::f64::consts::TAU * 220.0 * k * t).sin())
                .sum();
            gain * struck * swell * partials
        })
        .collect()
}

fn read(name: &str, planes: Vec<Vec<f64>>) -> Output {
    let call = sva_core::call(name).expect("a call");
    let representation: Representation = sva_core::asked(&call)
        .expect("a reading")
        .representation(RATE, None)
        .expect("its arguments");
    let buffer = Buffer::of_planes(RATE, planes);
    answer_buffer(name, &buffer, representation, PSYCHOACOUSTIC_V1.name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .value
}

fn alike(name: &str, a: &Output, b: &Output) {
    let (a, b) = (format!("{a:?}"), format!("{b:?}"));
    let split = |s: &str| -> (Vec<String>, Vec<f64>) {
        let mut text = Vec::new();
        let mut numbers = Vec::new();
        let mut token = String::new();
        let numeric = |c: char| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | '+');
        for c in s.chars().chain(std::iter::once(' ')) {
            if numeric(c) {
                token.push(c);
                continue;
            }
            match token.parse::<f64>() {
                Ok(x) => numbers.push(x),
                Err(_) => text.push(token.clone()),
            }
            token.clear();
            text.push(c.to_string());
        }
        (text, numbers)
    };
    let ((text_a, xs), (text_b, ys)) = (split(&a), split(&b));
    assert_eq!(text_a.concat(), text_b.concat(), "{name}: one shape");
    assert_eq!(xs.len(), ys.len(), "{name}: as many numbers");
    for (x, y) in xs.iter().zip(&ys) {
        let tolerance = 1e-9 * x.abs().max(y.abs()).max(1.0);
        assert!((x - y).abs() <= tolerance, "{name}: {x} against {y}");
    }
}

/// A channel copied to two reads as itself, and a tone in the second channel beside silence
/// in the first reads as that tone at half its power, 3 dB down: each reading averages every
/// channel's power, never the first channel's alone.
#[test]
fn every_reading_off_samples_averages_every_channels_power() {
    let x = tone(1.0);
    let silence = vec![0.0; x.len()];
    let halved = tone(std::f64::consts::FRAC_1_SQRT_2);
    for name in READINGS {
        let mono = read(name, vec![x.clone()]);
        alike(name, &read(name, vec![x.clone(), x.clone()]), &mono);
        let right = read(name, vec![silence.clone(), x.clone()]);
        match name {
            "crest" => {
                let crest = |held: &Output| match held {
                    Output::Crest(crest) => crest.broadband_crest_db,
                    other => panic!("a crest, not {other:?}"),
                };
                let up = crest(&right) - crest(&mono);
                assert!((up - 10.0 * 2f64.log10()).abs() < 1e-9, "crest: {up}");
            }
            _ => alike(name, &right, &read(name, vec![halved.clone()])),
        }
    }
}
