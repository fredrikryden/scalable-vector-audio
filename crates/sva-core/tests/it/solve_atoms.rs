// Concern: states what atoms owe their numbers, nodes, rows, keys and the feasible set after projection | Non-concern: what the ear reads of one (solve_spectrum.rs) | IO: (atoms) -> asserts

use std::f64::consts::TAU;

use sva_core::solve::atoms::{Atom, Clip, Feasible, Quantity, Sound, Trajectory, nodes, score};

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

/// One of each kind, every number off its range or out of order.
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
            knots(4000.0, &[(0.05, -3000.0)]),
            vec![0.0, 200.0, -300.0],
            knots(-50.0, &[(0.4, 80.0), (0.35, -10.0), (0.7, 0.0)]),
        ),
        Atom::noise(2, 0.2, (12_000.0, 20.0), knots(-80.0, &[(0.1, 0.0)])),
    ]
}

fn weights(atom: &Atom) -> Vec<f64> {
    (0..atom.numbers().len())
        .map(|i| 1.0 + 0.37 * (i % 5) as f64)
        .collect()
}

/// Every constraint the feasible set is, as a half-space `a.x <= b` over an atom's numbers in
/// `numbers`' order, written from its definition alone.
fn halfspaces(atom: &Atom, feasible: &Feasible) -> Vec<(Vec<(usize, f64)>, f64)> {
    let numbers = atom.numbers();
    let mut out = Vec::new();
    let mut between = |terms: Vec<(usize, f64)>, (lo, hi): (f64, f64)| {
        out.push((terms.clone(), hi));
        out.push((terms.iter().map(|(i, a)| (*i, -a)).collect(), -lo));
    };
    let mut last_time = 0;
    let mut base = 0;
    for (i, (q, _)) in numbers.iter().enumerate() {
        match q {
            Quantity::Time => {
                between(vec![(i, 1.0)], feasible.secs);
                if i > 0 {
                    between(vec![(last_time, 1.0), (i, -1.0)], (f64::NEG_INFINITY, 0.0));
                }
                last_time = i;
            }
            Quantity::Hz | Quantity::Db => {
                base = i;
                last_time = 0;
            }
            _ => {}
        }
        let hz = match &atom.sound {
            Sound::Harmonic { offsets, .. } => {
                (feasible.hz.0, feasible.hz.1 / offsets.len() as f64)
            }
            _ => feasible.hz,
        };
        let width = feasible.db.1 - feasible.db.0;
        match q {
            Quantity::Hz => between(vec![(i, 1.0)], hz),
            Quantity::HzDeviation => between(vec![(base, 1.0), (i, 1.0)], hz),
            Quantity::Db => between(vec![(i, 1.0)], feasible.db),
            Quantity::DbDeviation => between(vec![(base, 1.0), (i, 1.0)], feasible.db),
            Quantity::Offset => between(vec![(i, 1.0)], (-width, width)),
            Quantity::Edge => between(vec![(i, 1.0)], feasible.hz),
            Quantity::Time => {}
        }
    }
    if let Sound::Noise { .. } = atom.sound {
        out.push((vec![(1, 1.0), (2, -1.0)], -feasible.spacing));
    }
    out
}

/// Dykstra's alternating projections in the weighted metric: the nearest point of the
/// intersection, to as many sweeps as the test runs.
fn dykstra(start: &[f64], w: &[f64], sets: &[(Vec<(usize, f64)>, f64)]) -> Vec<f64> {
    let mut x = start.to_vec();
    let mut increments = vec![vec![0.0; x.len()]; sets.len()];
    for _ in 0..20_000 {
        for ((a, b), p) in sets.iter().zip(&mut increments) {
            let y: Vec<f64> = x.iter().zip(p.iter()).map(|(x, p)| x + p).collect();
            let over = a.iter().map(|(i, a)| a * y[*i]).sum::<f64>() - b;
            let norm: f64 = a.iter().map(|(i, a)| a * a / w[*i]).sum();
            let mut next = y.clone();
            if over > 0.0 {
                for (i, a) in a {
                    next[*i] -= over / norm * a / w[*i];
                }
            }
            for k in 0..x.len() {
                p[k] = y[k] - next[k];
            }
            x = next;
        }
    }
    x
}

#[test]
fn a_projected_atom_holds_and_is_the_nearest_atom_that_holds() {
    let feasible = Feasible::of(&CLIP, -90.0);
    for atom in unheld() {
        assert!(!feasible.holds(&atom), "{atom:?} starts off its ranges");
        let w = weights(&atom);
        let projected = feasible.project(&atom, &w).expect("a feasible set");
        assert!(feasible.holds(&projected), "{projected:?}");
        let start: Vec<f64> = atom.numbers().iter().map(|(_, v)| *v).collect();
        let nearest = dykstra(&start, &w, &halfspaces(&atom, &feasible));
        for ((_, got), want) in projected.numbers().iter().zip(&nearest) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1.0),
                "{got} against {want}"
            );
        }
        assert_eq!(feasible.project(&projected, &w), Some(projected.clone()));
    }
}

/// Edges anywhere, the floor's and the top's own included, land on a band at least a line
/// spacing wide inside the span, whatever the clip's length.
#[test]
fn a_noise_band_projects_onto_one_holding_a_line() {
    let edges = [
        -9.0, -5.0, 0.0, 49.0, 49.0113, 61.9, 1000.0, 7999.0, 8000.0, 9000.0,
    ];
    for len in [1_234, 8_000, 16_001] {
        let clip = Clip { rate: 16_000, len };
        let feasible = Feasible::of(&clip, -90.0);
        for lo in edges {
            for hi in edges {
                let atom = Atom::noise(0, 0.0, (lo, hi), knots(-20.0, &[(0.05, 0.0)]));
                let projected = feasible.project(&atom, &weights(&atom)).expect("a band");
                assert!(
                    feasible.holds(&projected),
                    "{lo}, {hi} at {len}: {projected:?}"
                );
            }
        }
    }
}

/// Atoms of every kind with every number drawn far off its range land inside it.
#[test]
fn any_atom_projects_into_the_feasible_set() {
    let feasible = Feasible::of(&CLIP, -90.0);
    let mut step = 0;
    let mut draw = |lo: f64, hi: f64| {
        step += 1;
        lo + (hi - lo) * sva_formula::draw(7, step)
    };
    for _ in 0..200 {
        for atom in unheld() {
            let values: Vec<f64> = atom
                .numbers()
                .iter()
                .map(|(q, _)| match q {
                    Quantity::Time => draw(-1.0, 2.0),
                    Quantity::Db | Quantity::DbDeviation | Quantity::Offset => draw(-200.0, 200.0),
                    _ => draw(-20_000.0, 30_000.0),
                })
                .collect();
            let atom = atom.with(&values);
            let w: Vec<f64> = values.iter().map(|_| draw(0.01, 100.0)).collect();
            let projected = feasible.project(&atom, &w).expect("a feasible set");
            assert!(feasible.holds(&projected), "{atom:?} to {projected:?}");
        }
    }
}

#[test]
fn nothing_projects_where_nothing_is_feasible() {
    let deaf = Feasible::of(&CLIP, 3.0);
    let atom = &unheld()[0];
    assert_eq!(
        deaf.project(atom, &weights(atom)),
        None,
        "a floor over full scale"
    );
    let crowded = Atom::harmonic(
        0,
        0.1,
        Trajectory::steady(100.0),
        vec![0.0; 400],
        knots(-20.0, &[(0.2, 0.0)]),
    );
    let feasible = Feasible::of(&CLIP, -90.0);
    assert_eq!(
        feasible.project(&crowded, &weights(&crowded)),
        None,
        "400 harmonics under 8 kHz"
    );
}

#[test]
fn an_atom_reads_its_numbers_back() {
    for atom in unheld() {
        let values: Vec<f64> = atom.numbers().iter().map(|(_, v)| *v).collect();
        assert_eq!(atom.with(&values), atom);
    }
}

#[test]
fn a_row_appended_leaves_every_earlier_row_its_phases_and_key() {
    let atoms = unheld();
    let before = score(&atoms[..2]);
    let after = score(&atoms);
    assert!(after.starts_with(&before), "{after}");
    assert_eq!(
        after.lines().count(),
        4,
        "the annotation and a row per atom"
    );
    for (ordinal, atom) in atoms.iter().enumerate() {
        let drawn = |k: usize| TAU * sva_formula::draw_nearest(ordinal as u64, k as f64).unwrap();
        match &atom.sound {
            Sound::Partial { phase, .. } => assert_eq!(*phase, drawn(1)),
            Sound::Harmonic { phases, .. } => {
                assert_eq!(*phases, (1..=phases.len()).map(drawn).collect::<Vec<_>>());
            }
            Sound::Noise { key, .. } => assert_eq!(*key, ordinal as u64),
        }
    }
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
            "atoms/harmonic_h3_f1_l3",
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
