// Concern: states what sources owe their nodes, rows and ordinals, a shaped noise's and a glide's too | Non-concern: fitting them (fit.rs) | IO: (sources) -> asserts

use sva_solve::sources::{
    Clip, SCORE, Sound, Source, Trajectory, birth, files_at, nodes, phase as phase_of, read, score,
    sounded,
};

const CLIP: Clip = Clip {
    rate: 16_000,
    len: 8_000,
};

fn knots(base: f64, knots: &[(f64, f64)]) -> Trajectory {
    Trajectory {
        base,
        knots: knots.to_vec(),
    }
}

/// The files a composition of `sources` holds at unit gain.
fn files(sources: &[Source]) -> Vec<(String, String)> {
    files_at(sources, &[], &CLIP, 0.0)
}

fn partial(hz: f64) -> Sound {
    Sound::Partial { hz, glide: 0.0 }
}

fn harmonic(hz: f64, stretch: f64, count: usize) -> Sound {
    Sound::Harmonic {
        hz,
        stretch,
        count,
        glide: 0.0,
    }
}

/// One of each kind.
fn unheld() -> Vec<Source> {
    vec![
        Source::new(
            0,
            0.3,
            partial(30.0),
            vec![knots(6.0, &[(0.2, -150.0), (0.9, 3.0)])],
        ),
        Source::new(
            1,
            -0.1,
            harmonic(4000.0, 0.0, 4),
            vec![
                knots(-50.0, &[(0.4, 80.0), (0.35, -10.0), (0.7, 0.0)]),
                knots(-40.0, &[(0.7, 0.0)]),
                knots(-45.0, &[(0.3, 1.0), (0.7, 0.0)]),
                knots(-60.0, &[(0.7, 0.0)]),
            ],
        ),
        Source::new(
            2,
            0.2,
            Sound::Noise {
                lo: 12_000.0,
                hi: 20.0,
            },
            vec![knots(-80.0, &[(0.1, 0.0)])],
        ),
    ]
}

#[test]
fn a_row_appended_leaves_every_earlier_row_and_writes_its_ordinal_as_its_key() {
    let sources = unheld();
    let before = score(&sources[..2]);
    let after = score(&sources);
    assert!(after.starts_with(&before), "{after}");
    assert_eq!(
        after.lines().count(),
        4,
        "the annotation and a row per source"
    );
    for (row, source) in after.lines().skip(1).zip(&sources) {
        assert!(row.contains(&format!("\"key\": {}", source.key)), "{row}");
        assert!(!row.contains("\"ph"), "no phase is written: {row}");
    }
    for (_, text) in nodes(&sources, &CLIP) {
        assert!(!text.contains("ph"), "{text}");
    }
}

/// The first birth is 0, each later one past the largest living, so a gap a removal left
/// stays one.
#[test]
fn a_birth_takes_one_past_the_largest_living_ordinal() {
    assert_eq!(birth(&[]), 0);
    let mut sources = unheld();
    assert_eq!(birth(&sources), 3);
    sources.remove(2);
    sources.remove(0);
    assert_eq!(birth(&sources), 2);
}

/// Keys out of order and with gaps, harmonics of one line and of many, read back as written.
#[test]
fn ordinals_and_harmonic_levels_read_back_from_the_written_rows() {
    let mut sources = unheld();
    sources.remove(1);
    sources.insert(
        0,
        Source::new(
            7,
            0.01,
            harmonic(110.0, 0.0, 1),
            vec![knots(-30.0, &[(0.2, 0.0)])],
        ),
    );
    sources.push(Source::new(
        4,
        0.125,
        harmonic(220.0, 4e-4, 4),
        vec![
            knots(-12.5, &[(0.3, -3.25), (0.45, 1.0 / 3.0)]),
            knots(0.1 + 0.2, &[(0.45, 1e-17)]),
            knots(-6.0, &[(0.2, 1.0), (0.45, 0.0)]),
            knots(-9.0, &[(0.45, 0.0)]),
        ],
    ));
    let text = score(&sources);
    let mut born = sources.clone();
    born.sort_by_key(|a| a.key);
    assert_eq!(read(&files(&sources)), Some(born), "{text}");
    let harmonic = text
        .lines()
        .find(|r| r.contains("\"key\": 4"))
        .expect("a row");
    assert!(
        harmonic.contains("\"l4\"") && !harmonic.contains("\"l5\""),
        "{harmonic}"
    );
    assert!(harmonic.contains("harmonic_h4_f0_l2-1-2-1_n"), "{harmonic}");
    let lone = vec![(
        SCORE.to_string(),
        "{\"at\": \"0s\", \"node\": \"@../sources/partial_f0_l1_n0\"}\n".to_string(),
    )];
    assert_eq!(read(&lone), None);
}

/// Each kind, knot count and level curve plays one node, its level curves knot tables of its
/// own numbers, named by its kind, its counts and their hash.
#[test]
fn each_kind_knot_count_and_level_plays_one_node() {
    let mut sources = unheld();
    sources.push(Source::new(
        3,
        0.0,
        partial(500.0),
        vec![knots(-10.0, &[(0.25, 0.0), (0.4, -6.0)])],
    ));
    let held = nodes(&sources, &CLIP);
    let names: Vec<&str> = held
        .iter()
        .map(|(path, _)| path.rsplit_once("_n").expect("a hash").0)
        .collect();
    assert_eq!(
        names,
        [
            "sources/harmonic_h4_f0_l3-1-2-1",
            "sources/noise_l1",
            "sources/partial_f0_l2",
            "sources/partial_f0_l2"
        ]
    );
    let mut twin = sources[3].clone();
    twin.key = 11;
    twin.on = 0.2;
    assert_eq!(twin.node(), sources[3].node(), "one level curve, one node");
    assert!(held[1].1.contains("period=0.5s"), "{}", held[1].1);
    assert!(
        held.iter()
            .any(|(_, text)| text.contains("pwl(t - on, 0s, 0, 0.25s, 0, 0.4s, -6)")),
        "{held:?}"
    );
    let mut longer = sources[3].clone();
    longer.levels[0].knots.push((0.45, -9.0));
    assert!(
        longer.node().starts_with("sources/partial_f0_l3_n"),
        "a knot more is the next count's node"
    );
}

#[test]
fn a_harmonic_writes_until_its_node_nests_past_the_parser() {
    let voice = |count: usize| {
        Source::new(
            0,
            0.1,
            harmonic(30.0, 1e-4, count),
            vec![knots(-20.0, &[(0.5, 0.0)]); count],
        )
    };
    assert!(voice(4).writes(&CLIP));
    assert!(!voice(400).writes(&CLIP));
}

/// A voice of 19 harmonics, each level through 51 knots, writes, reads back and sounds: its level
/// curves are knot tables, whose literals count toward no bound.
#[test]
fn a_long_envelope_voice_writes_and_sounds() {
    let curve = |k: usize| {
        let held: Vec<(f64, f64)> = (1..=51)
            .map(|i| {
                (
                    0.4 * i as f64 / 51.0,
                    -0.5 * i as f64 - (i % 3) as f64 - k as f64,
                )
            })
            .collect();
        knots(-20.0 - k as f64, &held)
    };
    let voice = Source::new(
        9,
        0.05,
        harmonic(110.0, 1e-4, 19),
        (1..=19).map(curve).collect(),
    );
    assert!(voice.writes(&CLIP), "{}", voice.node_text(&CLIP));
    let written = files(std::slice::from_ref(&voice));
    assert_eq!(read(&written), Some(vec![voice.clone()]));
    let heard = sounded(written, &CLIP).expect("the sources render");
    let power = heard.iter().map(|s| s * s).sum::<f64>() / heard.len() as f64;
    assert!(power > 0.0 && power.is_finite(), "{power}");
}

/// A shaped noise writes one node per band count, reads back as written, parses and sounds.
#[test]
fn a_shaped_noise_reads_back_and_sounds() {
    let shaped = Source::new(
        5,
        0.0,
        Sound::Shaped {
            bands: vec![(500.0, 1000.0, -6.0), (2000.0, 4000.0, 0.0)],
        },
        vec![knots(-20.0, &[(0.25, -3.0), (0.5, 0.0)])],
    );
    assert!(shaped.node().starts_with("sources/shaped_b2_l2_n"));
    assert!(shaped.writes(&CLIP));
    let written = files(std::slice::from_ref(&shaped));
    assert_eq!(read(&written), Some(vec![shaped.clone()]), "{written:?}");
    let heard = sounded(written, &CLIP).expect("the source renders");
    let power = heard.iter().map(|s| s * s).sum::<f64>() / heard.len() as f64;
    assert!(power > 0.0, "{power}");
}

/// A gliding partial and a gliding harmonic write one node each, read back as written, and the
/// partial sounds its closed form: its phase `hz (e^(glide t) - 1) / glide` from its onset.
#[test]
fn a_glide_reads_back_and_sounds_its_closed_form() {
    let glide = libm::log(2.0) / 0.2;
    let partial = Source::new(
        7,
        0.0,
        Sound::Partial { hz: 400.0, glide },
        vec![knots(-20.0, &[(0.5, 0.0)])],
    );
    let voice = Source::new(
        8,
        0.0,
        Sound::Harmonic {
            hz: 200.0,
            stretch: 1e-4,
            count: 3,
            glide: -glide,
        },
        vec![knots(-30.0, &[(0.5, 0.0)]); 3],
    );
    assert!(partial.node().starts_with("sources/partial_f0_g1_l1_n"));
    assert!(
        voice
            .node()
            .starts_with("sources/harmonic_h3_f0_g1_l1-1-1_n")
    );
    let both = vec![partial.clone(), voice];
    let written = files(&both);
    assert_eq!(read(&written), Some(both), "{written:?}");
    let heard = sounded(files(std::slice::from_ref(&partial)), &CLIP).expect("the source renders");
    let rate = f64::from(CLIP.rate);
    let worst = heard
        .iter()
        .enumerate()
        .map(|(n, s)| {
            let t = n as f64 / rate;
            let phase = std::f64::consts::TAU * 400.0 * libm::expm1(glide * t) / glide;
            let stated = 0.1 * libm::sin(phase + phase_of(7, 1));
            (s - stated).abs()
        })
        .fold(0.0, f64::max);
    assert!(worst < 1e-6, "{worst}");
}
