// Concern: states what atoms owe their nodes, rows and ordinals | Non-concern: fitting them (solve_fit.rs) | IO: (atoms) -> asserts

use sva_core::solve::atoms::{Atom, Clip, Trajectory, birth, nodes, read, score};

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
    assert_eq!(read(&text), Some(born), "{text}");
    let harmonic = text
        .lines()
        .find(|r| r.contains("\"key\": 4"))
        .expect("a row");
    assert!(
        harmonic.contains("\"l4\"") && !harmonic.contains("\"l5\""),
        "{harmonic}"
    );
    assert!(harmonic.contains("harmonic_h4_f1_l2-1-2-1"), "{harmonic}");
    assert_eq!(
        read("{\"at\": \"0s\", \"node\": \"@../atoms/partial_f0_l1\"}\n"),
        None
    );
}

#[test]
fn each_kind_and_knot_count_plays_one_node() {
    let mut atoms = unheld();
    atoms.push(Atom::partial(
        3,
        0.0,
        knots(500.0, &[(0.1, 1.0), (0.2, 2.0), (0.3, 3.0)]),
        knots(-10.0, &[(0.25, 0.0), (0.4, -6.0)]),
    ));
    let held = nodes(&atoms, &CLIP);
    let names: Vec<&str> = held.iter().map(|(path, _)| path.as_str()).collect();
    assert_eq!(
        names,
        [
            "atoms/harmonic_h4_f1_l3-1-2-1",
            "atoms/noise_l1",
            "atoms/partial_f3_l2"
        ]
    );
    assert!(held[1].1.contains("period=0.5s"), "{}", held[1].1);
    let mut longer = atoms[3].clone();
    longer.level.knots.push((0.45, -9.0));
    assert_eq!(
        longer.node(),
        "atoms/partial_f3_l3",
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
