// Concern: proves every answer says which reading ran and under which profile | Non-concern: the arithmetic of any one reading (sva-samples) | IO: (a Render) -> Answer

use crate::fixtures::graph_of;
use sva_engine::{Ask, Output, Render, RenderConfig, Representation, Source, Tier, answer, render};

const CHORD: &str = "sin(2*pi*256*t) + sin(2*pi*512*t)\n";

fn rendered(name: &str, body: &str, asks: Vec<&str>) -> Render {
    rendered_of(name, &[("node", body)], asks)
}

fn rendered_of(name: &str, files: &[(&str, &str)], asks: Vec<&str>) -> Render {
    let g = graph_of(name, files);
    let asks = asks
        .into_iter()
        .map(|a| Ask {
            node: "node".to_string(),
            representation: Representation::from_name(a).expect("a known reading"),
        })
        .collect();
    let config = RenderConfig::seconds(8_192, 1.0).asking(asks);
    render(&g, "node", config, &Tier::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn every_representation_reports_source_and_profile() {
    let law = rendered("law-answers", CHORD, vec!["lines"]);
    let id = law.id("node").expect("the root");
    for name in ["lines", "atoms", "derivative", "envelope"] {
        let representation = Representation::from_name(name).expect("a known reading");
        let found = answer(&law, id, representation).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(found.source, Source::Exact, "{name}");
        assert_eq!(found.profile, "psychoacoustic-v1", "{name}");
        assert_eq!(found.rate, None, "{name}: a law answers without a rate");
    }

    let sampled = rendered(
        "sampled-answers",
        "sample(sin(2*pi*256*t))\n",
        vec!["samples"],
    );
    let id = sampled.id("node").expect("the root");
    for name in [
        "samples", "spectrum", "envelope", "bands", "crest", "loudness", "hits",
    ] {
        let representation = Representation::from_name(name).expect("a known reading");
        let found = answer(&sampled, id, representation).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(found.source, Source::Measured, "{name}");
        assert_eq!(found.profile, "psychoacoustic-v1", "{name}");
        assert_eq!(found.rate, Some(8_192), "{name}");
    }
}

/// The one place a reading can be wrong about itself: asking a closed form for samples that were
/// never held refuses instead of inventing them.
#[test]
fn a_reading_off_a_buffer_no_render_held_refuses() {
    let law = rendered("unheld", CHORD, vec!["lines"]);
    let id = law.id("node").expect("the root");
    let refused = answer(&law, id, Representation::Samples).expect_err("nothing was collapsed");
    assert_eq!(refused.code(), "engine.not_materialized");
}

/// One score stands for the whole node, so it has to be the component that aliases worst: the
/// first alone calls a joined pair clean whenever its quiet half is written first.
#[test]
fn an_alias_score_answers_the_component_that_aliases_worst() {
    const CLEAN: &str = "0.2*sin(2*pi*100*t)";
    const ALIASING: &str = "tanh(6*sin(2*pi*3000*t))";
    let asked = Representation::Alias { oversample: 4 };
    let scored = |name: &str, body: String| {
        let held = rendered(name, &format!("{body}\n"), vec!["alias"]);
        let id = held.id("node").expect("the root");
        let Output::Alias(score) = answer(&held, id, asked).expect("a score").value else {
            panic!("an alias reading answers a score");
        };
        *score
    };

    let alone = scored("alias-loud", ALIASING.to_string());
    let quiet = scored("alias-quiet", CLEAN.to_string());
    assert!(
        alone.nmr_peak_db > quiet.nmr_peak_db,
        "the fixture only says something if one component aliases more: {alone:?} {quiet:?}"
    );

    let quiet_first = scored("alias-quiet-first", format!("join({CLEAN}, {ALIASING})"));
    let loud_first = scored("alias-loud-first", format!("join({ALIASING}, {CLEAN})"));
    assert_eq!(
        quiet_first.asr_db, loud_first.asr_db,
        "the score does not turn on which component was written first"
    );
    assert_eq!(
        quiet_first.asr_db, alone.asr_db,
        "a joined pair scores as its worst half, not as the half written first"
    );
    assert_eq!(quiet_first.audible, alone.audible);
}

/// FORMAT 6.1: a finite `sum` is the type of its term, so the exact reading of a finite sum
/// of sinusoids is every line those terms place, never an empty list labelled exact.
#[test]
fn a_finite_sum_lists_every_line() {
    let law = rendered(
        "finite-sum",
        "sum(k, 2, 6, sin(2*pi*100*k*t))\n",
        vec!["lines"],
    );
    let id = law.id("node").expect("the root");
    let found = answer(&law, id, Representation::Lines).expect("an exact line list");
    assert_eq!(found.source, Source::Exact);
    let Output::Lines(lines) = found.value else {
        panic!("expected lines");
    };
    let mut hz: Vec<i64> = lines
        .iter()
        .filter(|l| l.hz > 0.0)
        .map(|l| l.hz.round() as i64)
        .collect();
    hz.sort_unstable();
    assert_eq!(hz, vec![200, 300, 400, 500, 600]);

    let found = answer(&law, id, Representation::Atoms).expect("an exact atom list");
    let Output::Atoms(atoms) = found.value else {
        panic!("expected atoms");
    };
    assert_eq!(atoms.len(), 10, "one atom per line, both signs");
}

/// A series answers the terms above the audibility floor with the tail beside
/// them. Three partials round a feedback comb are three waves per echo, and written
/// associativity is not a fact about how many lines a Neumann series has.
#[test]
fn a_long_comb_lists_its_kept_lines_and_names_the_dropped() {
    let law = rendered(
        "long-comb",
        "(sin(2*pi*200*t) + sin(2*pi*400*t) + sin(2*pi*600*t)) + 0.9*self(t - 0.01s)\n",
        vec!["lines"],
    );
    let id = law.id("node").expect("the root");
    let found = answer(&law, id, Representation::Lines).expect("an exact line list");
    assert_eq!(found.source, Source::Exact);
    let Output::Lines(lines) = &found.value else {
        panic!("expected lines");
    };

    let echoes = lines.len() / 6;
    assert!(
        echoes > 1 && lines.len() == echoes * 6,
        "each echo places three partials as six conjugate lines, got {}",
        lines.len()
    );
    let loudest = |set: &[sva_engine::Line]| set.iter().map(|l| l.amp.abs()).fold(0.0f64, f64::max);
    assert!(
        loudest(lines) > 0.0,
        "a truncated series still answers the lines it kept"
    );
    assert!(
        !found.dropped.is_empty(),
        "the terms the floor cut are named, not left to stand in the kept list"
    );
    assert!(
        loudest(&found.dropped) < loudest(lines),
        "what was dropped sits under what was kept"
    );
    assert!(
        found.tail_db.is_some_and(|db| db < 0.0),
        "the tail is stated in dB against the loudest line taken"
    );
}

/// The transform of a windowed term is its line convolved with the window's own: a shape with
/// a width, which no line holds. An empty list labelled exact says this node has no spectrum,
/// and a list of whatever sits unwindowed beside it says it has a smaller one.
#[test]
fn lines_under_a_window_are_listed_with_their_width_or_refused_by_name() {
    let windowed = rendered(
        "windowed-pair",
        "crop(sin(2*pi*440*t), 0s, 0.25s)\n",
        vec!["atoms"],
    );
    let id = windowed.id("node").expect("the root");
    for representation in [
        Representation::Lines,
        Representation::Spectrum {
            max_peaks: 8,
            frame_secs: None,
        },
        Representation::Pitch {
            max_notes: 4,
            frame_secs: 0.05,
        },
    ] {
        let name = representation.name();
        let refused = match answer(&windowed, id, representation) {
            Err(refused) => refused,
            Ok(held) => panic!(
                "`{name}` answered {:?} where it has no line to list",
                held.value
            ),
        };
        assert_eq!(refused.code(), "read.lines_need_unwindowed_lines", "{name}");
        let said = refused.to_string();
        assert!(said.contains("0.25"), "`{name}` names the window: {said}");
    }

    let mixed = rendered_of(
        "windowed-beside-bare",
        &[
            ("bare", "sin(2*pi*220*t)\n"),
            ("node", "@bare(t) + crop(sin(2*pi*440*t), 0s, 0.25s)\n"),
        ],
        vec!["atoms"],
    );
    let id = mixed.id("node").expect("the root");
    let refused = answer(&mixed, id, Representation::Lines)
        .expect_err("half of a line list reads as the whole of one");
    assert_eq!(refused.code(), "read.lines_need_unwindowed_lines");

    let bare = rendered("bare-pair", "sin(2*pi*440*t)\n", vec!["lines"]);
    let id = bare.id("node").expect("the root");
    let found = answer(&bare, id, Representation::Lines).expect("an exact line list");
    assert_eq!(found.source, Source::Exact);
    let Output::Lines(lines) = found.value else {
        panic!("expected lines");
    };
    assert_eq!(
        lines.len(),
        2,
        "the same term unwindowed is one conjugate pair"
    );
}

/// Each atom states the window it sounds in: a crop's own bounds, a placed note's moved with
/// its placement and its decay read from its onset, and no bound where nothing crops it.
#[test]
fn each_atom_states_the_window_it_sounds_in() {
    let held = rendered_of(
        "atom-windows",
        &[
            (
                "voice",
                "f0 = C4\ncrop(exp(-t/0.25s), 0s, 0.5s)*sin(2*pi*f0*t)\n",
            ),
            (
                "node",
                "0.1*sin(2*pi*50*t) + crop(sin(2*pi*440*t), 0.25s, 0.75s) + \
                 @voice(t - 1s, f0=330hz)\n",
            ),
        ],
        vec!["atoms"],
    );
    let id = held.id("node").expect("the root");
    let Output::Atoms(atoms) = answer(&held, id, Representation::Atoms)
        .expect("an atom list")
        .value
    else {
        panic!("expected atoms");
    };
    // A turn is the atom's radians over 2pi, so 330 Hz may read one ulp under.
    let at = |hz: f64| {
        let found: Vec<_> = atoms
            .iter()
            .filter(|a| (a.turns.abs() - hz).abs() <= hz * f64::EPSILON)
            .collect();
        assert_eq!(found.len(), 2, "{hz} Hz is one conjugate pair: {atoms:?}");
        found
    };
    for a in at(50.0) {
        assert_eq!((a.start, a.end), (None, None), "nothing crops it: {a:?}");
        assert_eq!((a.decay, a.degree), (0.0, 0), "a held line: {a:?}");
    }
    for a in at(440.0) {
        assert_eq!((a.start, a.end), (Some(0.25), Some(0.75)), "{a:?}");
    }
    for a in at(330.0) {
        assert_eq!((a.start, a.end), (Some(1.0), Some(1.5)), "{a:?}");
        assert_eq!((a.decay, a.decay_from), (4.0, 1.0), "{a:?}");
        assert!(a.sketch.contains("indicator"), "{a:?}");
    }
}
