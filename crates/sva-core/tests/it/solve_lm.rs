// Concern: states the descent recovers a partial and a noise started precision units off, at a coarse scale | Non-concern: the tangents' values (solve_spectrum.rs) | IO: (atoms, clip) -> asserts

use sva_core::solve::atoms::{Atom, Clip, Feasible, Quantity, Trajectory};
use sva_core::solve::bits::{
    Look, Precision, Priced, curvature_delta, frequency_delta, level_delta, precision,
};
use sva_core::solve::calibrate::{n0, rendered};
use sva_core::solve::fisher::Fit;
use sva_core::solve::lm::{Ended, Limits, curvatures, descend};
use sva_core::solve::objective::Objective;
use sva_core::solve::spectrum::Model;

const SPL: f64 = 100.0;
/// The coarsest scale both recover at: at 4 and 8 the partial ends 8 frequency units off.
const SCALE: f64 = 2.0;
const CLIP: Clip = Clip {
    rate: 4_000,
    len: 1_600,
};
/// Units off at the start, and the most off the descent may end.
const OFF: f64 = 2.0;
const WITHIN: f64 = 3.0;

fn partial() -> Atom {
    Atom::partial(
        0,
        0.0503,
        Trajectory::steady(500.0),
        Trajectory {
            base: -30.0,
            knots: vec![(0.3001, 0.0)],
        },
    )
}

fn noise() -> Atom {
    Atom::noise(
        1,
        0.0503,
        (1000.3, 2000.3),
        Trajectory {
            base: -35.0,
            knots: vec![(0.3001, 0.0)],
        },
    )
}

fn looks(objective: &Objective, atom: &Atom) -> Vec<Look> {
    let ear = objective.ear();
    let model = Model::new(std::slice::from_ref(atom), CLIP, true);
    let thresholds = ear.thresholds();
    let (first, last) = ear.frames(CLIP.len);
    (first..=last)
        .map(|m| {
            let e = ear.excite(m, &ear.compose(m, &mut |f| model.bins(f).power));
            let peak = e
                .iter()
                .zip(&thresholds)
                .fold(0.0f64, |p, (e, t)| p.max(e / t));
            Look {
                sl_db: 10.0 * libm::log10(peak),
                hz: atom.hz().map_or(0.0, |h| h.base),
            }
        })
        .collect()
}

/// Each number's precision at `truth`, by looks or by the Fisher diagonal.
fn priced(objective: &Objective, truth: &Atom) -> Priced {
    let feasible = Feasible::of(&CLIP, -120.0);
    let ones = Priced {
        feasible,
        deltas: vec![1.0; truth.values().len()],
    };
    let probe = Fit::new(objective, vec![truth.clone()], vec![ones], true);
    let a = curvatures(&probe, &probe.numbers());
    let hop = objective.heard().hop_secs;
    let l = looks(objective, truth);
    let deltas = truth
        .numbers()
        .iter()
        .zip(&a)
        .map(|((q, _), a)| match precision(truth, *q) {
            Precision::Level => level_delta(hop, &l),
            Precision::Frequency => frequency_delta(hop, &l),
            Precision::Curvature => curvature_delta(*a),
        })
        .collect();
    Priced { feasible, deltas }
}

/// The descent from `truth` moved `OFF` units along `moved`: how it ended, and each number's
/// distance from the truth in units.
fn recover(truth: Atom, moved: &[usize]) -> (Ended, Vec<f64>) {
    let threads = std::thread::available_parallelism().expect("a thread count");
    let n0 = n0(CLIP.rate, SCALE, SPL, threads);
    let objective = Objective::new(
        &rendered(std::slice::from_ref(&truth), &CLIP),
        CLIP,
        SPL,
        SCALE,
        n0,
    );
    let priced = priced(&objective, &truth);
    let deltas = priced.deltas.clone();
    let want = truth.values();
    let start: Vec<f64> = want
        .iter()
        .zip(&deltas)
        .enumerate()
        .map(|(i, (v, d))| match moved.contains(&i) {
            true => v + OFF * d,
            false => *v,
        })
        .collect();
    let fit = Fit::new(&objective, vec![truth.lifted(&start)], vec![priced], true);
    let never = || false;
    let out = descend(
        &fit,
        &start,
        &deltas,
        &Limits {
            steps: None,
            stopped: &never,
        },
    )
    .expect("feasible");
    let off: Vec<f64> = out
        .x
        .iter()
        .zip(&want)
        .zip(&deltas)
        .map(|((x, w), d)| (x - w) / d)
        .collect();
    (out.ended, off)
}

#[test]
fn a_partial_off_in_onset_frequency_and_level_is_recovered() {
    let truth = partial();
    let at = |q: Quantity| truth.numbers().iter().position(|(n, _)| *n == q);
    let moved: Vec<usize> = [Quantity::Time, Quantity::Hz, Quantity::Db]
        .into_iter()
        .filter_map(at)
        .collect();
    let (ended, off) = recover(truth, &moved);
    assert_eq!(ended, Ended::Converged);
    assert!(off.iter().all(|o| o.abs() <= WITHIN), "off {off:?}");
}

#[test]
fn a_noise_off_in_level_is_recovered() {
    let truth = noise();
    let at = truth
        .numbers()
        .iter()
        .position(|(n, _)| *n == Quantity::Db)
        .expect("a level");
    let (ended, off) = recover(truth, &[at]);
    assert_eq!(ended, Ended::Converged);
    assert!(off.iter().all(|o| o.abs() <= WITHIN), "off {off:?}");
}
