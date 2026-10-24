// Concern: proves harmonic readings take exact partials off a law and measured ones off samples, agreeing, refusing by name | Non-concern: each model's arithmetic | IO: (a Render) -> Answer

use crate::fixtures::graph_of;
use sva_engine::harmony::voice_leading::Norm;
use sva_engine::{
    Ask, DEFAULT_MAX_PAIRS, Harmonic, Output, Render, RenderConfig, Representation, Source, Tier,
    answer, render,
};

const TRIAD: &str =
    "(0.3*cos(2*pi*C4*t) + 0.3*cos(2*pi*E4*t) + 0.3*cos(2*pi*G4*t))*exp(0 - t/0.5s)\n";

const PROGRESSION: &str = "crop(0.3*cos(2*pi*G3*t) + 0.3*cos(2*pi*B3*t) + 0.3*cos(2*pi*D4*t), \
                           0s, 0.5s) + crop(0.3*cos(2*pi*G3*t) + 0.3*cos(2*pi*C4*t) + \
                           0.3*cos(2*pi*E4*t), 0.5s, 1s)\n";

const DISSONANCE: Harmonic = Harmonic::Dissonance {
    max_pairs: DEFAULT_MAX_PAIRS,
};

fn harmony(reading: Harmonic, floor_db: f64) -> Representation {
    Representation::Harmony {
        reading,
        frame_secs: 0.05,
        floor_db,
    }
}

fn rendered(name: &str, files: &[(&str, &str)], asked: Representation) -> Render {
    let g = graph_of(name, files);
    let asks = vec![Ask {
        node: "node".to_string(),
        representation: asked,
    }];
    let config = RenderConfig::seconds(44_100, 1.0).asking(asks);
    render(&g, "node", config, &Tier::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn read(render: &Render, asked: Representation) -> sva_engine::Answer {
    let id = render.id("node").expect("the root");
    answer(render, id, asked).unwrap_or_else(|e| panic!("{e}"))
}

fn frames(answer: &sva_engine::Answer) -> Vec<sva_engine::harmony::dissonance::DissonanceFrame> {
    let Output::Dissonance(d) = &answer.value else {
        panic!("dissonance answers its curve, not {:?}", answer.value)
    };
    d.frames.clone()
}

fn curve(answer: &sva_engine::Answer) -> Vec<(f64, f64)> {
    frames(answer)
        .iter()
        .map(|f| (f.t_secs, f.roughness))
        .collect()
}

/// The decaying triad's exact roughness at each instant and the one measured off its samples
/// in a frame centred there step along one time axis, and agree within 1% where the frame
/// lies wholly inside the second, inside the bounds each measured frame reports.
#[test]
fn a_closed_form_triad_reads_exact_and_its_samples_measured_alike() {
    let asked = harmony(DISSONANCE, -40.0);
    let law = rendered("triad-exact", &[("node", TRIAD)], asked);
    let exact = read(&law, asked);
    assert_eq!((exact.source, exact.rate), (Source::Exact, None));
    let sampled = rendered(
        "triad-measured",
        &[("chord", TRIAD), ("node", "sample(@chord)\n")],
        asked,
    );
    let measured = read(&sampled, asked);
    assert_eq!(measured.source, Source::Measured);
    for (truth, frame) in curve(&exact).iter().zip(frames(&measured)).take(19).skip(1) {
        let (lo, hi) = frame.roughness_bounds;
        assert!(lo <= truth.1 && truth.1 <= hi, "{truth:?} in [{lo}, {hi}]");
    }
    let (exact, measured) = (curve(&exact), curve(&measured));
    assert_eq!(exact.len(), 20);
    assert_eq!(exact.len(), measured.len());
    for ((t, truth), (at, found)) in exact.iter().zip(&measured).take(19).skip(1) {
        assert!((t - at).abs() < 1e-12, "one time axis: {t} vs {at}");
        assert!(
            (found - truth).abs() <= 0.01 * truth,
            "at {t}s: {found} vs {truth}"
        );
    }
    assert!(exact[1].1 > exact[18].1, "the decay releases it");
}

#[test]
fn a_windowed_dominant_resolves_to_its_tonic_by_the_smallest_voice_leading_and_two_steps() {
    let leading = harmony(Harmonic::VoiceLeading { norm: Norm::L1 }, -40.0);
    let law = rendered("v-to-i", &[("node", PROGRESSION)], leading);
    let Output::VoiceLeading(v) = read(&law, leading).value else {
        panic!("voice_leading answers its transitions")
    };
    let chords: Vec<(f64, f64, Vec<i64>)> = v
        .chords
        .iter()
        .map(|c| (c.start_secs, c.end_secs, c.notes.clone()))
        .collect();
    assert_eq!(
        chords,
        vec![(0.0, 0.5, vec![55, 59, 62]), (0.5, 1.0, vec![55, 60, 64])]
    );
    assert_eq!(v.transitions[0].mapping, vec![(55, 55), (59, 60), (62, 64)]);
    assert_eq!((v.transitions[0].at_secs, v.total), (0.5, 3.0));

    let tonal = harmony(Harmonic::TonalDistance, -40.0);
    let law = rendered("v-to-i-tonnetz", &[("node", PROGRESSION)], tonal);
    let Output::TonalDistance(d) = read(&law, tonal).value else {
        panic!("tonal_distance answers its transitions")
    };
    let tension: Vec<Option<usize>> = d.chords.iter().map(|c| c.tension).collect();
    assert_eq!(tension, vec![Some(0), Some(2)]);
}

/// A 0.98 s render on a 0.05 s grid has its last instant at 0.95 s: the chord sounding there
/// ends with the render, never a hop on at 1 s, exact and measured alike.
#[test]
fn the_last_chord_ends_at_the_renders_end() {
    let leading = harmony(Harmonic::VoiceLeading { norm: Norm::L1 }, -40.0);
    let steady = "0.3*cos(2*pi*C4*t) + 0.3*cos(2*pi*E4*t) + 0.3*cos(2*pi*G4*t)\n";
    for (name, files) in [
        ("short-exact", vec![("node", steady)]),
        (
            "short-measured",
            vec![("chord", steady), ("node", "sample(@chord)\n")],
        ),
    ] {
        let g = graph_of(name, &files);
        let config = RenderConfig::seconds(44_100, 0.98).asking(vec![Ask {
            node: "node".to_string(),
            representation: leading,
        }]);
        let held = render(&g, "node", config, &Tier::default()).unwrap_or_else(|e| panic!("{e}"));
        let Output::VoiceLeading(v) = read(&held, leading).value else {
            panic!("voice_leading answers its chords")
        };
        let last = v.chords.last().expect("a chord sounds");
        assert_eq!(last.end_secs, 0.98, "{name}: {:?}", v.chords);
    }
}

fn inharmonic(answer: &sva_engine::Answer) -> &sva_engine::harmony::inharmonicity::Inharmonicity {
    let Output::Inharmonicity(r) = &answer.value else {
        panic!("inharmonicity answers its curve, not {:?}", answer.value)
    };
    r
}

/// The engine's `string` builtin lowers to Fletcher's law, so its exact partials fit back the
/// B it was asked for and the f0 it rings on, at every instant its members stand over the floor.
#[test]
fn the_string_builtins_exact_partials_fit_back_its_inharmonicity() {
    let asked = harmony(Harmonic::Inharmonicity, -40.0);
    let law = rendered(
        "string",
        &[("node", "string(110, inharmonicity=0.0004)\n")],
        asked,
    );
    let read = read(&law, asked);
    assert_eq!(read.source, Source::Exact);
    let fits: Vec<_> = inharmonic(&read)
        .frames
        .iter()
        .filter_map(|f| f.series.first().and_then(|s| s.fit))
        .collect();
    assert!(fits.len() >= 2, "{fits:?}");
    for fit in fits {
        assert!((fit.b - 4e-4).abs() < 1e-9 * 4e-4, "{fit:?}");
        assert!((fit.f0_hz - 110.0).abs() < 1e-9, "{fit:?}");
        assert_eq!(
            fit.b_bounds,
            Some((fit.b, fit.b)),
            "exact partials bound to a point"
        );
    }
}
