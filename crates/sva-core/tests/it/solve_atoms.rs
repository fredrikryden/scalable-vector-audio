// Concern: states what atoms owe their nodes, rows and ordinals, a shaped noise's too | Non-concern: fitting them (solve_fit.rs) | IO: (atoms) -> asserts

use sva_core::solve::atoms::{
    Atom, Clip, SCORE, Trajectory, birth, files, nodes, phase as phase_of, read, score, sounded,
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

/// One of each kind.
fn unheld() -> Vec<Atom> {
    vec![
        Atom::partial(
            0,
            0.3,
            knots(30.0, &[(0.1, 9000.0), (0.6, -20.0), (0.2, 50.0)]),
            knots(6.0, &[(0.2, -150.0), (0.9, 3.0)]),
        ),
        Atom::harmonic(
            1,
            -0.1,
            (knots(4000.0, &[(0.05, -3000.0)]), 0.0),
            vec![
                knots(-50.0, &[(0.4, 80.0), (0.35, -10.0), (0.7, 0.0)]),
                knots(-40.0, &[(0.7, 0.0)]),
                knots(-45.0, &[(0.3, 1.0), (0.7, 0.0)]),
                knots(-60.0, &[(0.7, 0.0)]),
            ],
        ),
        Atom::noise(2, 0.2, (12_000.0, 20.0), knots(-80.0, &[(0.1, 0.0)])),
    ]
}

#[test]
fn a_row_appended_leaves_every_earlier_row_and_writes_its_ordinal_as_its_key() {
    let atoms = unheld();
    let before = score(&atoms[..2]);
    let after = score(&atoms);
    assert!(after.starts_with(&before), "{after}");
    assert_eq!(
        after.lines().count(),
        4,
        "the annotation and a row per atom"
    );
    for (row, atom) in after.lines().skip(1).zip(&atoms) {
        assert!(row.contains(&format!("\"key\": {}", atom.key)), "{row}");
        assert!(!row.contains("\"ph"), "no phase is written: {row}");
    }
    for (_, text) in nodes(&atoms, &CLIP) {
        assert!(!text.contains("ph"), "{text}");
    }
}

/// The first birth is 0, each later one past the largest living, so a gap a removal left
/// stays one.
#[test]
fn a_birth_takes_one_past_the_largest_living_ordinal() {
    assert_eq!(birth(&[]), 0);
    let mut atoms = unheld();
    assert_eq!(birth(&atoms), 3);
    atoms.remove(2);
    atoms.remove(0);
    assert_eq!(birth(&atoms), 2);
}

/// Keys out of order and with gaps, harmonics of one line and of many, read back as written.
#[test]
fn ordinals_and_harmonic_levels_read_back_from_the_written_rows() {
    let mut atoms = unheld();
    atoms.remove(1);
    atoms.insert(
        0,
        Atom::harmonic(
            7,
            0.01,
            (Trajectory::steady(110.0), 0.0),
            vec![knots(-30.0, &[(0.2, 0.0)])],
        ),
    );
    atoms.push(Atom::harmonic(
        4,
        0.125,
        (knots(220.0, &[(0.25, -0.5)]), 4e-4),
        vec![
            knots(-12.5, &[(0.3, -3.25), (0.45, 1.0 / 3.0)]),
            knots(0.1 + 0.2, &[(0.45, 1e-17)]),
            knots(-6.0, &[(0.2, 1.0), (0.45, 0.0)]),
            knots(-9.0, &[(0.45, 0.0)]),
        ],
    ));
    let text = score(&atoms);
    let mut born = atoms.clone();
    born.sort_by_key(|a| a.key);
    assert_eq!(read(&files(&atoms, &CLIP)), Some(born), "{text}");
    let harmonic = text
        .lines()
        .find(|r| r.contains("\"key\": 4"))
        .expect("a row");
    assert!(
        harmonic.contains("\"l4\"") && !harmonic.contains("\"l5\""),
        "{harmonic}"
    );
    assert!(harmonic.contains("harmonic_h4_f1_l2-1-2-1_n"), "{harmonic}");
    let lone = vec![(
        SCORE.to_string(),
        "{\"at\": \"0s\", \"node\": \"@../atoms/partial_f0_l1_n0\"}\n".to_string(),
    )];
    assert_eq!(read(&lone), None);
}

/// Each kind, knot count and level curve plays one node, its level curves knot tables of its
/// own numbers, named by its kind, its counts and their hash.
#[test]
fn each_kind_knot_count_and_level_plays_one_node() {
    let mut atoms = unheld();
    atoms.push(Atom::partial(
        3,
        0.0,
        knots(500.0, &[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)]),
        knots(-10.0, &[(0.25, 0.0), (0.4, -6.0)]),
    ));
    let held = nodes(&atoms, &CLIP);
    let names: Vec<&str> = held
        .iter()
        .map(|(path, _)| path.rsplit_once("_n").expect("a hash").0)
        .collect();
    assert_eq!(
        names,
        [
            "atoms/harmonic_h4_f1_l3-1-2-1",
            "atoms/noise_l1",
            "atoms/partial_f3_l2",
            "atoms/partial_f3_l2"
        ]
    );
    let mut twin = atoms[3].clone();
    twin.key = 11;
    twin.on = 0.2;
    assert_eq!(twin.node(), atoms[3].node(), "one level curve, one node");
    assert!(held[1].1.contains("period=0.5s"), "{}", held[1].1);
    assert!(
        held.iter()
            .any(|(_, text)| text.contains("pwl(t - on, 0s, 0, 0.25s, 0, 0.4s, -6)")),
        "{held:?}"
    );
    let mut longer = atoms[3].clone();
    longer.level.knots.push((0.45, -9.0));
    assert!(
        longer.node().starts_with("atoms/partial_f3_l3_n"),
        "a knot more is the next count's node"
    );
}

#[test]
fn a_harmonic_writes_until_its_node_nests_past_the_parser() {
    let harmonic = |count: usize| {
        Atom::harmonic(
            0,
            0.1,
            (Trajectory::steady(30.0), 1e-4),
            vec![knots(-20.0, &[(0.5, 0.0)]); count],
        )
    };
    assert!(harmonic(4).writes(&CLIP));
    assert!(!harmonic(400).writes(&CLIP));
}

/// A voice of 19 harmonics, each level through 51 knots, its Hz through one, writes, reads back
/// and sounds: its level curves are knot tables, whose literals count toward no bound.
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
    let voice = Atom::harmonic(
        9,
        0.05,
        (knots(110.0, &[(0.2, 2.0)]), 1e-4),
        (1..=19).map(curve).collect(),
    );
    assert!(voice.writes(&CLIP), "{}", voice.node_text(&CLIP));
    let written = files(std::slice::from_ref(&voice), &CLIP);
    assert_eq!(read(&written), Some(vec![voice.clone()]));
    let heard = sounded(written, &CLIP).expect("the atoms render");
    let power = heard.iter().map(|s| s * s).sum::<f64>() / heard.len() as f64;
    assert!(power > 0.0 && power.is_finite(), "{power}");
}

/// A shaped noise writes one node per band count, reads back as written, parses and sounds.
#[test]
fn a_shaped_noise_reads_back_and_sounds() {
    let shaped = Atom::shaped(
        5,
        0.0,
        vec![(500.0, 1000.0, -6.0), (2000.0, 4000.0, 0.0)],
        knots(-20.0, &[(0.25, -3.0), (0.5, 0.0)]),
    );
    assert!(shaped.node().starts_with("atoms/shaped_b2_l2_n"));
    assert!(shaped.writes(&CLIP));
    let written = files(std::slice::from_ref(&shaped), &CLIP);
    assert_eq!(read(&written), Some(vec![shaped.clone()]), "{written:?}");
    let heard = sounded(files(&[shaped], &CLIP), &CLIP).expect("the atom renders");
    let power = heard.iter().map(|s| s * s).sum::<f64>() / heard.len() as f64;
    assert!(power > 0.0, "{power}");
}

/// A gliding partial and a gliding harmonic write one node each, read back as written, and the
/// partial sounds its closed form: its phase `hz (e^(glide t) - 1) / glide` from its onset.
#[test]
fn a_glide_reads_back_and_sounds_its_closed_form() {
    let glide = libm::log(2.0) / 0.2;
    let partial = Atom::partial(
        7,
        0.0,
        Trajectory::steady(400.0),
        knots(-20.0, &[(0.5, 0.0)]),
    )
    .glided(glide);
    let harmonic = Atom::harmonic(
        8,
        0.0,
        (Trajectory::steady(200.0), 1e-4),
        vec![knots(-30.0, &[(0.5, 0.0)]); 3],
    )
    .glided(-glide);
    assert!(partial.node().starts_with("atoms/partial_f0_g1_l1_n"));
    assert!(
        harmonic
            .node()
            .starts_with("atoms/harmonic_h3_f0_g1_l1-1-1_n")
    );
    let both = vec![partial.clone(), harmonic];
    let written = files(&both, &CLIP);
    assert_eq!(read(&written), Some(both), "{written:?}");
    let heard =
        sounded(files(std::slice::from_ref(&partial), &CLIP), &CLIP).expect("the atom renders");
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
