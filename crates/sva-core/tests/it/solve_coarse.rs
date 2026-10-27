// Concern: states the coarse model reads a tone and a band near their samples, its tangents as its differences, silence as nothing | Non-concern: the sample model | IO: (atoms) -> asserts

use sva_core::solve::atoms::{Atom, Clip, Trajectory};
use sva_core::solve::objective::{Objective, reach};

const SPL: f64 = 100.0;
const SCALE: f64 = 2.828_427_124_746_190_3;
const CLIP: Clip = Clip {
    rate: 16_000,
    len: 16_000,
};

fn tone() -> Atom {
    Atom::partial(
        0,
        0.1,
        Trajectory::steady(1100.0),
        Trajectory {
            base: -30.0,
            knots: vec![(0.9, -6.0)],
        },
    )
}

fn band() -> Atom {
    Atom::noise(
        1,
        0.1,
        (2000.0, 4000.0),
        Trajectory {
            base: -30.0,
            knots: vec![(0.9, 0.0)],
        },
    )
}

/// The clip `atoms` write, so the sample model reads them as the engine renders them.
fn objectives(atoms: &[Atom]) -> (Objective, Objective) {
    let samples = sva_core::solve::calibrate::rendered(atoms, &CLIP);
    let sampled = Objective::new(&samples, CLIP, SPL, SCALE, 1.0);
    let coarse = Objective::new(&samples, CLIP, SPL, SCALE, 1.0).coarse();
    (sampled, coarse)
}

/// The largest gap in dB at a middle frame over the channels centred in `(lo, hi)` Hz.
fn gap_db(atoms: &[Atom], (lo, hi): (f64, f64)) -> f64 {
    let (sampled, coarse) = objectives(atoms);
    let m = (0.5 / coarse.heard().hop_secs) as i64;
    let at = |o: &Objective| o.excitations(atoms, true, (m, m)).remove(0);
    let (s, c) = (at(&sampled), at(&coarse));
    coarse
        .ear()
        .centres()
        .iter()
        .zip(s.iter().zip(&c))
        .filter(|(hz, _)| lo < **hz && **hz < hi)
        .map(|(_, (s, c))| (10.0 * libm::log10(c / s)).abs())
        .fold(0.0, f64::max)
}

/// The coarse model drops each window's leakage and a tone's own lobe, so about a tone and
/// within a band the two readings agree to a fraction of a dB.
#[test]
fn a_tone_and_a_band_read_near_their_samples() {
    for (name, atom, span) in [
        ("tone", tone(), (800.0, 1500.0)),
        ("band", band(), (2000.0, 4000.0)),
    ] {
        let gap = gap_db(std::slice::from_ref(&atom), span);
        println!("{name}: {gap} dB from its samples' excitation");
        assert!(gap < 0.5, "{name}: {gap} dB");
    }
}

#[test]
fn tangents_are_the_excitations_differences() {
    let atoms = [tone(), band()];
    let (_, coarse) = objectives(&atoms);
    let reaches: Vec<(i64, i64)> = atoms
        .iter()
        .map(|a| reach(coarse.ear(), &CLIP, a))
        .collect();
    let read = coarse.coarse_read(&atoms, &reaches).expect("coarse");
    let m = (0.12 / coarse.heard().hop_secs) as i64;
    let seeds: Vec<(usize, usize)> = (0..atoms.len())
        .flat_map(|a| (0..atoms[a].values().len()).map(move |j| (a, j)))
        .collect();
    let (excitation, along) = read.tangents(m, &seeds);
    for ((a, j), d) in seeds.iter().zip(along) {
        let d = d.expect("both atoms reach the frame");
        let h = 1e-6 * atoms[*a].values()[*j].abs().max(1.0);
        let moved = |by: f64| {
            let mut x = atoms[*a].values();
            x[*j] += by;
            let mut state = atoms.to_vec();
            state[*a] = atoms[*a].lifted(&x);
            let read = coarse.coarse_read(&state, &reaches).expect("coarse");
            read.excitation(m)
        };
        let (up, down) = (moved(h), moved(-h));
        let top = excitation.iter().fold(0.0f64, |p, e| p.max(*e));
        for c in 0..excitation.len() {
            let diff = (up[c] - down[c]) / (2.0 * h);
            let scale = diff.abs().max(1e-9 * top / h);
            assert!(
                (d[c] - diff).abs() <= 1e-4 * scale,
                "atom {a} number {j} channel {c}: {} against {diff}",
                d[c]
            );
        }
    }
}

#[test]
fn silence_excites_nothing() {
    let (_, coarse) = objectives(&[]);
    let frames = coarse.ear().frames(CLIP.len);
    assert!(
        coarse
            .excitations(&[], true, frames)
            .iter()
            .all(|e| e.iter().all(|v| *v == 0.0))
    );
}
