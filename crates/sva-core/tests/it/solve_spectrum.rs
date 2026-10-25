// Concern: states an atom's model reads as its render within its stated bounds, noise in expectation, its tangents, its floor | Non-concern: rows (solve_atoms.rs) | IO: (atoms, render) -> asserts

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

/// What a comparison measured: the largest gap over what its bounds allow, the largest gap
/// past twice the rounding bound over the sampling bound, and the largest sampling bound over
/// its frame's whole power.
#[derive(Debug, Default)]
struct Gaps {
    over_bounds: f64,
    over_sampling: f64,
    sampling_share: f64,
}

/// The model's bins against the render's, within twice the model's rounding bound (the
/// render's own rounding is the same formula's) plus its sampling bound; the render's
/// excitation within the model's render bracket, widened by the rounding again.
fn against_render(atoms: &[Atom], clip: &Clip, scale: f64) -> Gaps {
    let render = rendered(atoms, clip);
    let ear = Ear::of(f64::from(clip.rate), SPL, scale);
    let model = Model::new(atoms, *clip, true);
    let mut gaps = Gaps::default();
    each_frame(&ear, clip, &mut |frame| {
        let heard = frame.power(&[&render], 0);
        let bins = model.bins(frame);
        let whole: f64 = heard.iter().sum();
        for (j, h) in heard.iter().enumerate() {
            let (p, b, s) = (bins.power[j], bins.bound[j], bins.sampling[j]);
            let gap = (h - p).abs();
            assert!(
                gap <= 2.0 * b + s,
                "{h} against {p}, bound {b}, sampling {s}"
            );
            if b + s > 0.0 {
                gaps.over_bounds = gaps.over_bounds.max(gap / (2.0 * b + s));
            }
            if s > 0.0 {
                gaps.over_sampling = gaps.over_sampling.max((gap - 2.0 * b).max(0.0) / s);
            }
            if whole > 0.0 {
                gaps.sampling_share = gaps.sampling_share.max(s / whole);
            }
        }
    });
    let (first, last) = ear.frames(clip.len);
    for m in (first..=last).step_by(11) {
        let heard = ear.excitation(m, &mut |frame| frame.power(&[&render], 0));
        let read = model.excitation(&ear, m);
        for (c, h) in heard.iter().enumerate() {
            let (e, least, most) = (read.excitation[c], read.least[c], read.most[c]);
            assert!(least <= e && e <= most, "{least} <= {e} <= {most}");
            let low = read.render_least[c] - (e - least);
            let high = read.render_most[c] + (most - e);
            assert!(
                low <= *h && *h <= high,
                "the render's {h} in [{low}, {high}]"
            );
        }
    }
    gaps
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
        vec![0.0, -6.0, 3.0],
        knots(-18.0, &[(0.1, -3.0), (0.3, -20.0)]),
    )
}

/// Edges on sample instants, so each edge's gate is a half off the crop's, and edges between
/// them.
#[test]
fn each_kind_reads_as_its_render_within_its_stated_bounds() {
    let clip = Clip {
        rate: RATE,
        len: 4_800,
    };
    let lone = Atom::harmonic(
        4,
        0.03,
        knots(330.0, &[(0.12, 30.0)]),
        Vec::new(),
        knots(-10.0, &[(0.2, 0.0)]),
    );
    let between = Atom::partial(
        5,
        0.050_031_7,
        Trajectory::steady(1000.0),
        knots(-12.0, &[(0.250_004_1, -30.0)]),
    );
    for (name, atoms) in [
        ("partial", vec![steady()]),
        ("partial between samples", vec![between]),
        ("glide", vec![glide()]),
        ("harmonic", vec![harmonic()]),
        ("one harmonic", vec![lone]),
    ] {
        let gaps = against_render(&atoms, &clip, 1.0);
        println!("{name}: {gaps:?}");
        assert!(gaps.over_bounds <= 1.0, "{gaps:?}");
        assert!(gaps.over_sampling <= 1.0, "{gaps:?}");
        assert!(
            gaps.over_sampling > 0.9,
            "a sampling bound some bin reaches: {gaps:?}"
        );
        assert!(gaps.sampling_share < 0.25, "{gaps:?}");
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
        let gaps = against_render(&[steady(), glide(), harmonic()], &clip, scale);
        println!("summed at scale {scale}: {gaps:?}");
        assert!(gaps.over_bounds <= 1.0, "{gaps:?}");
        assert!(
            gaps.over_sampling > 0.9,
            "a sampling bound some bin reaches: {gaps:?}"
        );
        assert!(gaps.sampling_share < 0.25, "{gaps:?}");
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

/// The render's own expectation, `E|X_j|^2 = sum over lines of (a^2/4)(|Y(b_j - w)|^2 +
/// |Y(b_j + w)|^2)`, `Y` the transform of the crop's weighted envelope: each line's phase
/// uniform and independent, every line summed. The model's band is continuous and its edges
/// gated, so it is within its rounding and sampling bounds.
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
    let (mut over, mut share) = (0.0f64, 0.0f64);
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
            for (j, p) in bins.power.iter().enumerate() {
                let (b, s) = (bins.bound[j], bins.sampling[j]);
                let beta = TAU * j as f64 / n;
                let exact: f64 = turning
                    .iter()
                    .map(|w| {
                        let (a, b) = (transform(beta - w), transform(beta + w));
                        half * half * (a.0 * a.0 + a.1 * a.1 + b.0 * b.0 + b.1 * b.1)
                    })
                    .sum();
                let gap = (p - exact).abs();
                assert!(
                    gap <= b + s + 1e-9 * peak,
                    "bin {j}: {p} against {exact}, bound {b}, sampling {s}"
                );
                over = over.max(gap / (b + s + 1e-9 * peak));
                if peak > 0.0 {
                    share = share.max(s / peak);
                }
            }
            bins.power
        });
    }
    println!(
        "noise against its exact expectation: gap over bounds {over}, sampling over peak {share}"
    );
    assert!(over > 0.5, "a sampling bound some bin comes near: {over}");
    assert!(share < 2.0, "{share}");
}

/// The mean of keyed renders, each key its own realisation, against the expectation: each
/// range's power summed over every frame within four of its standard errors across keys, plus
/// the sampling bound summed alike.
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
    let mut totals: Vec<(usize, Vec<f64>, f64, f64)> = Vec::new();
    each_frame(&ear, &clip, &mut |frame| {
        let at = match totals.iter().position(|(n, ..)| *n == frame.n()) {
            Some(at) => at,
            None => {
                totals.push((frame.n(), vec![0.0; renders.len()], 0.0, 0.0));
                totals.len() - 1
            }
        };
        for (total, x) in totals[at].1.iter_mut().zip(&renders) {
            *total += frame.power(&[x], 0).iter().sum::<f64>();
        }
        let bins = model.bins(frame);
        totals[at].2 += bins.power.iter().sum::<f64>();
        totals[at].3 += bins.sampling.iter().sum::<f64>();
    });
    let (mut heard, mut predicted) = (0.0, 0.0);
    for (n, per_key, expect, sampling) in &totals {
        let mean = per_key.iter().sum::<f64>() / keys as f64;
        let spread =
            per_key.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / (keys - 1) as f64;
        let error = (spread / keys as f64).sqrt();
        assert!(
            (mean - expect).abs() <= 4.0 * error + sampling,
            "{n}-point frames: {mean} against {expect}, standard error {error}, sampling {sampling}"
        );
        println!(
            "{n}-point frames: mean/expected {}, gap {} standard errors, sampling/expected {}",
            mean / expect,
            (mean - expect).abs() / error,
            sampling / expect
        );
        heard += mean;
        predicted += expect;
    }
    println!(
        "keyed renders' mean over expected, every frame: {}",
        heard / predicted
    );
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

/// A time a quarter-odd way between samples, off the instants knots kink at and the cells'
/// edges gates kink at.
fn off(samples: f64) -> f64 {
    (samples + 0.27) / f64::from(RATE)
}

/// One of each kind with every time off the grid, a glide and a harmonic summed coherently.
fn smooth() -> Vec<Vec<Atom>> {
    let partial = Atom::partial(
        0,
        off(800.0),
        knots(1000.0, &[(off(2400.0), 12.5)]),
        knots(-12.0, &[(off(1920.0), -3.0), (off(4000.0), -30.0)]),
    );
    let glide = Atom::partial(
        1,
        off(336.0),
        knots(
            440.0,
            &[
                (off(1280.0), 220.0),
                (off(2400.0), 220.0),
                (off(2416.0), 54.0),
            ],
        ),
        knots(-20.0, &[(off(480.0), 8.0), (off(4320.0), -4.0)]),
    );
    let harmonic = Atom::harmonic(
        2,
        off(18.0),
        knots(220.0, &[(off(3200.0), -11.0)]),
        vec![0.5, -6.0, 3.0],
        knots(-18.0, &[(off(1600.0), -3.0), (off(4640.0), -20.0)]),
    );
    let noise = Atom::noise(
        3,
        off(272.0),
        (500.3, 999.7),
        knots(-20.0, &[(off(800.0), -6.0), (off(1440.0), 0.0)]),
    );
    vec![
        vec![partial],
        vec![glide.clone()],
        vec![harmonic.clone()],
        vec![noise],
        vec![glide, harmonic],
    ]
}

/// Each number's tangent per bin against the model's own central difference, a check of the
/// forward mode only; each step a ten-millionth of its number, under any kink's distance.
#[test]
fn tangents_agree_with_the_models_central_difference() {
    let clip = Clip {
        rate: RATE,
        len: 4_800,
    };
    let ear = Ear::of(f64::from(clip.rate), SPL, 1.0);
    for atoms in smooth() {
        let model = Model::new(&atoms, clip, true);
        let mut worst = 0.0f64;
        let (first, last) = ear.frames(clip.len);
        for m in (first..=last).step_by(23) {
            ear.excitation(m, &mut |frame| {
                for (i, atom) in atoms.iter().enumerate() {
                    let values = atom.values();
                    for (j, tangent) in model.tangents(frame, i).iter().enumerate() {
                        let step = 1e-7 * values[j].abs().max(1.0);
                        let at = |sign: f64| {
                            let mut moved = values.clone();
                            moved[j] += sign * step;
                            let mut held = atoms.clone();
                            held[i] = atom.lifted(&moved);
                            Model::new(&held, clip, true).bins(frame).power
                        };
                        let (up, down) = (at(1.0), at(-1.0));
                        let peak = up.iter().fold(0.0f64, |a, p| a.max(p.abs()));
                        let scale = tangent.iter().fold(0.0f64, |a, t| a.max(t.abs()));
                        for (b, t) in tangent.iter().enumerate() {
                            let central = (up[b] - down[b]) / (2.0 * step);
                            let gap = (t - central).abs();
                            let rounding = 1e-12 * peak / step;
                            assert!(
                                gap <= 1e-4 * scale + rounding,
                                "atom {i} number {j} bin {b}: {t} against {central}"
                            );
                            if scale > rounding {
                                worst = worst.max(gap / scale);
                            }
                        }
                    }
                }
                vec![0.0; frame.hz().len()]
            });
        }
        println!(
            "{} atoms: tangent gap over its largest {worst}",
            atoms.len()
        );
    }
}
