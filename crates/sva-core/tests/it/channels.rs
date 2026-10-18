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

fn alike(name: &str, a: &dyn std::fmt::Debug, b: &dyn std::fmt::Debug) {
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

const EXACT: [&str; 6] = [
    "spectrum",
    "pitch",
    "dissonance",
    "voice_leading",
    "tonal_distance",
    "inharmonicity",
];

/// `x` holds steady lines, `y` a decaying partial beside a held one.
fn exact(name: &str, node: &str) -> Box<dyn std::fmt::Debug> {
    let held: sva_ast::Composition = [
        (
            "x",
            "sin(2*pi*220*t) + 0.5*sin(2*pi*277.18*t) + 0.25*sin(2*pi*330*t)\n",
        ),
        ("y", "exp(0 - 3*t)*sin(2*pi*220*t) + 0.5*sin(2*pi*330*t)\n"),
        ("twice", "join(@x, @x)\n"),
        ("opposed", "join(@x, 0 - @x)\n"),
        ("right", "join(0, @x)\n"),
        ("halved", "0.7071067811865476*@x\n"),
        ("y-twice", "join(@y, @y)\n"),
        ("y-right", "join(0, @y)\n"),
        ("y-halved", "0.7071067811865476*@y\n"),
    ]
    .into_iter()
    .collect();
    let asking = [sva_core::asked(&sva_core::call(name).expect("a call")).expect("a reading")];
    let target = format!("@{node}([0, 0.2s])");
    let job = sva_core::Job {
        asked: &asking,
        ..sva_core::Job::over(&held, &target)
    };
    let answer = sva_core::execute(job, &sva_engine::Tier::default())
        .and_then(|done| done.answered(&asking[0]))
        .unwrap_or_else(|e| panic!("{name} of {node}: {e:?}"));
    assert_eq!(answer.source, sva_engine::Source::Exact, "{name} of {node}");
    match answer.value {
        Output::Lines(lines) => Box::new(
            lines
                .iter()
                .map(|l| (l.hz, 20.0 * l.amp.abs().log10()))
                .collect::<Vec<_>>(),
        ),
        other => Box::new(other),
    }
}

/// The exact readings take the measured ones' convention: a channel copied to two reads as
/// itself, opposite phases cancel nothing, and a closed form in the second channel beside
/// silence reads at half its power.
#[test]
fn every_exact_reading_averages_every_channels_power() {
    for name in EXACT {
        let (x, twice, opposed, right, halved) = match name {
            "spectrum" | "pitch" => ("x", "twice", "opposed", "right", "halved"),
            _ => ("y", "y-twice", "opposed", "y-right", "y-halved"),
        };
        let mono = exact(name, x);
        alike(name, &*exact(name, twice), &*mono);
        if x == "x" {
            alike(name, &*exact(name, opposed), &*mono);
        }
        alike(name, &*exact(name, right), &*exact(name, halved));
    }
}
