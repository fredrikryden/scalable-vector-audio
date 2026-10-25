// Concern: states an atom's closed form reads as its render, bin for bin within its bound, noise in expectation, and its floor | Non-concern: rows (solve_atoms.rs) | IO: (atoms, render) -> asserts

use std::f64::consts::TAU;

use sva_ast::Composition;
use sva_core::solve::atoms::{Atom, Clip, NEPER_PER_DB, Trajectory, nodes, score};
use sva_core::solve::spectrum::{Model, floor_db};
use sva_core::{Job, Output, Representation, asked, call, execute};
use sva_engine::Tier;
use sva_engine::hearing::{ear_gain_db, threshold_excitation_db};
use sva_engine::internal::{Ear, Frame};

const SPL: f64 = 100.0;

/// The clip `atoms` write, rendered by the engine from their own nodes and score.
fn rendered(atoms: &[Atom], clip: &Clip) -> Vec<f64> {
    let mut held = Composition::new();
    for (path, text) in nodes(atoms, clip) {
        held.insert(path, text);
    }
    held.insert("score/clip.jsonl", score(atoms));
    held.insert(
        "master",
        "; Models: the clip | Neglects: nothing | IO: (t) -> amplitude | Tags: master\n\
         @score/clip(t)\n",
    );
    let samples = [asked(&call("samples").expect("a call")).expect("a reading")];
    let target = format!("@master([0, {}s])", clip.secs());
    let job = Job {
        asked: &samples,
        rate: Some(clip.rate),
        ..Job::over(&held, &target)
    };
    let rendered = execute(job, &Tier::default()).unwrap_or_else(|e| panic!("{e}"));
    let answer = rendered
        .answer(&rendered.target, Representation::Samples)
        .unwrap_or_else(|e| panic!("{e}"));
    let Output::Samples(buffer) = answer.value else {
        panic!("samples answer samples");
    };
    assert_eq!(buffer.plane(0).len(), clip.len, "the render is the clip");
    buffer.plane(0).to_vec()
}

fn knots(base: f64, knots: &[(f64, f64)]) -> Trajectory {
    Trajectory {
        base,
        knots: knots.to_vec(),
    }
}

/// Every frame and range of `ear` over the clip, each beside `check`.
fn each_frame(ear: &Ear, clip: &Clip, check: &mut dyn FnMut(&Frame)) {
    let (first, last) = ear.frames(clip.len);
    for m in first..=last {
        ear.excitation(m, &mut |frame| {
            check(frame);
            vec![0.0; frame.hz().len()]
        });
    }
}

/// The largest distance between the model's bins and the render's over twice the model's
/// bound (the render's own rounding is the same formula's), and the largest bound over its
/// frame's whole power; the render's excitation within twice the model's bracket.
fn against_render(atoms: &[Atom], clip: &Clip, scale: f64) -> (f64, f64) {
    let render = rendered(atoms, clip);
    let ear = Ear::of(f64::from(clip.rate), SPL, scale);
    let model = Model::new(atoms, *clip, true);
    let (mut over_bound, mut over_whole) = (0.0f64, 0.0f64);
    each_frame(&ear, clip, &mut |frame| {
        let heard = frame.power(&[&render], 0);
        let bins = model.bins(frame);
        let whole: f64 = heard.iter().sum();
        for ((h, p), b) in heard.iter().zip(&bins.power).zip(&bins.bound) {
            let gap = (h - p).abs();
            assert!(gap <= 2.0 * b, "{h} against {p}, bound {b}");
            if *b > 0.0 {
                over_bound = over_bound.max(gap / (2.0 * b));
            }
            if whole > 0.0 {
                over_whole = over_whole.max(b / whole);
            }
        }
    });
    let (first, last) = ear.frames(clip.len);
    for m in (first..=last).step_by(11) {
        let heard = ear.excitation(m, &mut |frame| frame.power(&[&render], 0));
        let read = model.excitation(&ear, m);
        for (h, ((e, least), most)) in heard
            .iter()
            .zip(read.excitation.iter().zip(&read.least).zip(&read.most))
        {
            assert!(least <= e && e <= most, "{least} <= {e} <= {most}");
            let (low, high) = (least - (e - least), most + (most - e));
            assert!(
                low <= *h && h <= &high,
                "the render's {h} in [{low}, {high}]"
            );
        }
    }
    (over_bound, over_whole)
}

const RATE: u32 = 16_000;

fn steady() -> Atom {
    Atom::partial(
        0,
        0.05,
        Trajectory::steady(1000.0),
        knots(-12.0, &[(0.25, -30.0)]),
    )
}

/// A glide from the onset, then a held note and a legato step down.
fn glide() -> Atom {
    Atom::partial(
        1,
        0.021,
        knots(440.0, &[(0.08, 220.0), (0.15, 220.0), (0.151, 54.0)]),
        knots(-20.0, &[(0.03, 8.0), (0.27, -4.0)]),
    )
}

fn harmonic() -> Atom {
    Atom::harmonic(
        2,
        0.0,
        knots(220.0, &[(0.2, -11.0)]),
        vec![0.0, -6.0, 3.0, -40.0],
        knots(-18.0, &[(0.1, -3.0), (0.3, -20.0)]),
    )
}

#[test]
fn each_kind_reads_as_its_render_bin_for_bin_within_its_bound() {
    let clip = Clip {
        rate: RATE,
        len: 4_800,
    };
    let lone = Atom::harmonic(
        4,
        0.03,
        knots(330.0, &[(0.12, 30.0)]),
        vec![-12.0],
        knots(-10.0, &[(0.2, 0.0)]),
    );
    for atoms in [vec![steady()], vec![glide()], vec![harmonic()], vec![lone]] {
        let (over_bound, over_whole) = against_render(&atoms, &clip, 1.0);
        assert!(over_bound <= 1.0, "{over_bound}");
        assert!(
            over_whole < 1e-9,
            "a bound far under the power it bounds: {over_whole}"
        );
    }
}

/// The three summed in samples, as the finest level reads them, through a wider ear too.
#[test]
fn atoms_summed_read_as_their_render_through_any_ear() {
    let clip = Clip {
        rate: RATE,
        len: 4_800,
    };
    for scale in [1.0, 2.0] {
        let (over_bound, over_whole) =
            against_render(&[steady(), glide(), harmonic()], &clip, scale);
        assert!(over_bound <= 1.0, "{over_bound}");
        assert!(over_whole < 1e-9, "{over_whole}");
    }
}

fn band() -> Atom {
    Atom::noise(
        3,
        0.017,
        (500.0, 1000.0),
        knots(-20.0, &[(0.05, -6.0), (0.09, 0.0)]),
    )
}

/// `E|X_j|^2 = sum over lines of (a^2/4)(|Y(b_j - w)|^2 + |Y(b_j + w)|^2)`, `Y` the weighted
/// envelope's transform: each line's phase uniform and independent, every line summed.
#[test]
fn a_noise_reads_its_exact_phase_ensemble_average() {
    let clip = Clip {
        rate: 4_000,
        len: 400,
    };
    let atom = band();
    let model = Model::new(std::slice::from_ref(&atom), clip, true);
    let ear = Ear::of(f64::from(clip.rate), SPL, 1.0);
    let period = clip.secs();
    let half = libm::pow(period, -0.5) / 2.0;
    let turning: Vec<f64> = (50..=100)
        .map(|l| TAU / period * l as f64 / f64::from(clip.rate))
        .collect();
    let envelope = |s: i64| {
        let t = s as f64 / f64::from(clip.rate);
        let level = match t {
            t if !(0.017..0.09).contains(&t) => return 0.0,
            t if t < 0.05 => -20.0 + (-6.0) * (t - 0.017) / (0.05 - 0.017),
            t => -26.0 + 6.0 * (t - 0.05) / (0.09 - 0.05),
        };
        sva_formula::exp(NEPER_PER_DB * level) / 500f64.sqrt()
    };
    let mut worst = 0.0f64;
    let (first, last) = ear.frames(clip.len);
    for m in (first..=last).step_by(9) {
        ear.excitation(m, &mut |frame| {
            let bins = model.bins(frame);
            let y: Vec<(i64, f64)> = frame
                .weights()
                .iter()
                .enumerate()
                .map(|(k, w)| (k as i64, w * envelope(frame.first() + k as i64)))
                .collect();
            let transform = |nu: f64| {
                y.iter().fold((0.0, 0.0), |(re, im), (k, v)| {
                    let a = nu * *k as f64;
                    (re + v * libm::cos(a), im - v * libm::sin(a))
                })
            };
            let n = frame.n() as f64;
            let peak = bins.power.iter().fold(0.0f64, |a, b| a.max(*b));
            for (j, (p, b)) in bins.power.iter().zip(&bins.bound).enumerate() {
                let beta = TAU * j as f64 / n;
                let exact: f64 = turning
                    .iter()
                    .map(|w| {
                        let (a, b) = (transform(beta - w), transform(beta + w));
                        half * half * (a.0 * a.0 + a.1 * a.1 + b.0 * b.0 + b.1 * b.1)
                    })
                    .sum();
                let gap = (p - exact).abs();
                assert!(gap <= b + 1e-9 * peak, "bin {j}: {p} against {exact}");
                if peak > 0.0 {
                    worst = worst.max(gap / peak);
                }
            }
            bins.power
        });
    }
    assert!(worst < 1e-9, "{worst}");
}

/// The mean of keyed renders, each key its own realisation, against the expectation: each
/// range's power summed over every frame within four of its standard errors across keys.
#[test]
fn a_noise_reads_in_expectation_as_its_keyed_renders_mean() {
    let clip = Clip {
        rate: 8_000,
        len: 1_600,
    };
    let level = knots(-20.0, &[(0.12, -6.0), (0.18, 0.0)]);
    let atom = |key| Atom::noise(key, 0.03, (500.0, 1500.0), level.clone());
    let keys = 48;
    let renders: Vec<Vec<f64>> = (1..=keys).map(|k| rendered(&[atom(k)], &clip)).collect();
    let expected = [atom(0)];
    let model = Model::new(&expected, clip, true);
    let ear = Ear::of(f64::from(clip.rate), SPL, 1.0);
    let mut totals: Vec<(usize, Vec<f64>, f64)> = Vec::new();
    each_frame(&ear, &clip, &mut |frame| {
        let at = match totals.iter().position(|(n, ..)| *n == frame.n()) {
            Some(at) => at,
            None => {
                totals.push((frame.n(), vec![0.0; renders.len()], 0.0));
                totals.len() - 1
            }
        };
        for (total, x) in totals[at].1.iter_mut().zip(&renders) {
            *total += frame.power(&[x], 0).iter().sum::<f64>();
        }
        totals[at].2 += model.bins(frame).power.iter().sum::<f64>();
    });
    let (mut heard, mut predicted) = (0.0, 0.0);
    for (n, per_key, expect) in &totals {
        let mean = per_key.iter().sum::<f64>() / keys as f64;
        let spread =
            per_key.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / (keys - 1) as f64;
        let error = (spread / keys as f64).sqrt();
        assert!(
            (mean - expect).abs() <= 4.0 * error,
            "{n}-point frames: {mean} against {expect}, standard error {error}"
        );
        heard += mean;
        predicted += expect;
    }
    let ratio = heard / predicted;
    assert!((ratio - 1.0).abs() < 0.02, "{ratio}");
}

/// At its floor, to the rounding of the floor's own sum, the ear first hears a steady tone,
/// within the window's spread of the level whose eardrum excitation is the threshold's.
#[test]
fn a_tone_is_first_heard_at_its_floor() {
    let clip = Clip {
        rate: RATE,
        len: 1_600,
    };
    let hz = 1000.0;
    let atom = Atom::partial(
        0,
        0.02,
        Trajectory::steady(hz),
        knots(-40.0, &[(0.08, 0.0)]),
    );
    let ear = Ear::of(f64::from(clip.rate), SPL, 1.0);
    let floor = floor_db(&atom, &clip, &ear);
    assert!(floor.bound < 1e-6, "{floor:?}");
    let floor = floor.db;
    let model = Model::new(std::slice::from_ref(&atom), clip, true);
    let thresholds = ear.thresholds();
    let loudest = |c: f64| {
        let gain = libm::pow(10.0, c / 10.0);
        let (first, last) = ear.frames(clip.len);
        (first..=last).fold(0.0f64, |most, m| {
            let e = ear.excitation(m, &mut |frame| {
                model.bins(frame).power.iter().map(|p| p * gain).collect()
            });
            e.iter()
                .zip(&thresholds)
                .fold(most, |most, (e, t)| most.max(e / t))
        })
    };
    let shift = floor - atom.level.base;
    let rounding = 64.0 * f64::EPSILON;
    assert!(loudest(shift) >= 1.0 - rounding, "heard at its floor");
    assert!(
        loudest(shift - rounding * floor.abs()) < 1.0,
        "unheard under it"
    );
    let steady = threshold_excitation_db(hz) - ear_gain_db(hz) - SPL;
    assert!((floor - steady).abs() < 2.0, "{floor} against {steady}");
}
