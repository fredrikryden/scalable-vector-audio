// Concern: states what the objective owes: no bits for the clip itself, a tone's derived bits, a noise's variance over keyed renders, Rice | Non-concern: N0's fit | IO: (atoms, clip) -> asserts

use std::f64::consts::{LN_2, PI};

use sva_core::solve::atoms::{Atom, Clip, Trajectory};
use sva_core::solve::bits::{self, Priced};
use sva_core::solve::calibrate::rendered;
use sva_core::solve::objective::{Expected, Kernel, Objective, power_spectrum, rice, total};
use sva_core::solve::spectrum::gamma;
use sva_engine::fft;
use sva_engine::internal::modulation::Element;

const SPL: f64 = 100.0;
/// Every assertion here holds at any N0.
const N0: f64 = 1.0;
const CLIP: Clip = Clip {
    rate: 4_000,
    len: 1_000,
};

/// Edges at (2n - 1)/(2 rate), where the gate's shares are the crop's.
fn tone() -> Atom {
    Atom::partial(
        0,
        401.0 / 8000.0,
        Trajectory::steady(1000.0),
        Trajectory {
            base: -40.0,
            knots: vec![(1601.0 / 8000.0, -6.0)],
        },
    )
}

fn gaussian(a: &[f64], b: &[f64], heard: &sva_engine::internal::Internal) -> f64 {
    let mut i = 0;
    let mut nats = 0.0;
    for channel in &heard.channels {
        for band in &channel.bands {
            for _ in &band.elements {
                let d = a[i] - b[i];
                nats += 0.5 * d * d / (N0 * band.rate_hz);
                i += 1;
            }
        }
    }
    nats / LN_2
}

fn flat(read: &sva_engine::internal::Internal) -> Vec<f64> {
    read.channels
        .iter()
        .flat_map(|c| c.bands.iter().flat_map(|b| b.elements.iter().copied()))
        .collect()
}

#[test]
fn the_clip_against_itself_costs_no_residual_bits() {
    let silence = Objective::new(&vec![0.0; CLIP.len], CLIP, SPL, 1.0, N0);
    assert_eq!(silence.bits(&[], true), 0.0);
    let atoms = [tone()];
    let itself = Objective::new(&rendered(&atoms, &CLIP), CLIP, SPL, 1.0, N0);
    let residual = itself.bits(&atoms, true);
    println!("a tone against its own render: {residual} bits");
    assert!((0.0..1e-12).contains(&residual), "{residual}");
    let priced: Vec<Priced> = atoms
        .iter()
        .map(|a| Priced {
            feasible: sva_core::solve::atoms::Feasible::of(&CLIP, -120.0),
            deltas: vec![0.01; a.numbers().len()],
        })
        .collect();
    assert_eq!(
        total(residual, &atoms, &priced),
        residual + bits::bits(&atoms, &priced)
    );
}

/// Silence's elements are 0, so the tone's own elements' Gaussian code.
#[test]
fn a_known_tone_against_silence_costs_the_derived_bits() {
    let atoms = [tone()];
    let silence = Objective::new(&vec![0.0; CLIP.len], CLIP, SPL, 1.0, N0);
    let bits = silence.bits(&atoms, true);
    let heard = silence.ear().read(&[&rendered(&atoms, &CLIP)], 0.0);
    let derived = gaussian(&flat(&heard), &vec![0.0; flat(&heard).len()], &heard);
    println!("a tone against silence: {bits} bits, derived {derived}");
    assert!(derived > 1.0, "{derived}");
    assert!(
        (bits - derived).abs() <= 1e-6 * derived,
        "{bits} against {derived}"
    );
}

/// Lowpass and real bands are what the lines option was measured on.
#[test]
fn a_noise_atom_sits_within_its_stated_variance_over_keyed_renders() {
    const RENDERS: u64 = 48;
    let clip = Clip {
        rate: 8_000,
        len: 4_000,
    };
    let noise = |key| {
        Atom::noise(
            key,
            0.0,
            (1000.0, 1250.0),
            Trajectory {
                base: -40.0,
                knots: vec![(clip.secs(), 0.0)],
            },
        )
    };
    let model = Objective::new(&vec![0.0; clip.len], clip, SPL, 1.0, N0);
    let expected = model.expect(&[noise(0)], true);
    let reads: Vec<Vec<f64>> = (1..=RENDERS)
        .map(|key| flat(&model.ear().read(&[&rendered(&[noise(key)], &clip)], 0.0)))
        .collect();
    let n = RENDERS as f64;
    let mut sums = [(0.0, 0.0); 3];
    let mut i = 0;
    for (channel, bands) in model.heard().channels.iter().zip(&expected) {
        for (band, elements) in channel.bands.iter().zip(bands) {
            let kind = match band.band.element {
                Element::Lowpass => 0,
                Element::Real => 1,
                Element::Magnitude => 2,
            };
            for (j, predicted) in elements.iter().enumerate() {
                let secs = j as f64 / band.rate_hz;
                if (1000.0..1250.0).contains(&channel.hz) && (0.15..0.35).contains(&secs) {
                    let mean = reads.iter().map(|r| r[i]).sum::<f64>() / n;
                    let spread = reads.iter().map(|r| (r[i] - mean).powi(2)).sum::<f64>();
                    sums[kind].0 += spread / (n - 1.0);
                    sums[kind].1 += predicted.variance;
                }
                i += 1;
            }
        }
    }
    let ratios = sums.map(|(renders, predicted)| renders / predicted);
    println!("renders' variance over predicted: lowpass, real, magnitude {ratios:?}");
    for ratio in &ratios[..2] {
        assert!((ratio - 1.0).abs() < 0.15, "{ratios:?}");
    }
    assert!(
        (ratios[2] - 1.89).abs() < 0.01,
        "objective's MODEL states 1.89: {ratios:?}"
    );
}

/// The kernel over every lag but 0 of an autocorrelation summed directly lag by lag,
/// against Parseval's sum over the transform: within the stated bound plus the direct sums'
/// own rounding (every term positive), at lengths from one line to a thousand, flat and
/// falling over 40 orders of magnitude.
#[test]
fn the_pair_sum_by_transform_reads_as_the_direct_sum() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut over = 0.0f64;
    for (len, fall) in [
        (1usize, 0.0),
        (2, 0.0),
        (3, 0.0),
        (125, 0.0),
        (1001, 0.0),
        (1001, 0.09),
    ] {
        let x: Vec<f64> = (0..len)
            .map(|k| next() * sva_formula::exp(-fall * k as f64))
            .collect();
        let kernel: Vec<f64> = (0..2 * len - 1).map(|_| next()).collect();
        let twiddles = fft::twiddles((2 * len).next_power_of_two());
        let (sum, bound) = Kernel::of(&kernel, &twiddles).apply(&power_spectrum(&x, &twiddles));
        let mut direct = 0.0;
        for n in 1..len {
            let acf: f64 = (0..len - n).map(|k| x[k] * x[k + n]).sum();
            direct += acf * (kernel[len - 1 + n] + kernel[len - 1 - n]);
        }
        let allowed = bound + gamma(2.0 * len as f64 + 4.0) * direct;
        assert!(
            (sum - direct).abs() <= allowed,
            "{len} lines: {sum} against {direct}, allowed {allowed}"
        );
        if direct > 0.0 {
            over = over.max((sum - direct).abs() / allowed);
        }
    }
    println!("pair sum by transform against the direct sum: gap over bounds {over}");
}

/// How far an element's bits move with `v` within its bound, its moments within theirs twice:
/// a magnitude's mean by `g/4 sqrt(pi/v)` per unit (heat equation; Anderson's inequality) and
/// `g sqrt(pi dv)/2` in all (`|y|` 1-Lipschitz in a Gaussian's scale), its variance by `dv +
/// (2 mean + d) d`.
fn bits_reach(model: &Expected, clip: f64, floor: f64, element: Element, gain: f64) -> f64 {
    let Expected {
        mean,
        variance,
        mean_bound,
        variance_bound,
        noise,
    } = *model;
    let dv = noise.bound;
    let (mean_moved, variance_moved) = match element {
        Element::Magnitude => {
            let low = (noise.v - dv).max(0.0);
            let per = (std::f64::consts::PI / low).sqrt() / 4.0 * dv;
            let whole = (std::f64::consts::PI * dv).sqrt() / 2.0;
            let moved = gain * per.min(whole);
            let rice = mean / gain;
            let spread = dv + (2.0 * rice + moved / gain) * moved / gain;
            (
                moved + 2.0 * mean_bound,
                gain * gain * spread + 2.0 * variance_bound,
            )
        }
        _ => (0.0, dv),
    };
    let d = (mean - clip).abs() + mean_moved;
    let low = floor + (variance - variance_moved).max(0.0);
    (d * mean_moved / low
        + d * d * variance_moved / (2.0 * low * low)
        + variance_moved / (2.0 * low))
        / LN_2
}

/// `v_i` and bits on the noise fixture above and on two noises sharing lines (ordered pairs
/// of distinct atoms at one line), pinned as per-line transforms and direct pair sums give
/// them: each `v_i` and their sum within the stated bounds, the bits within what those bounds
/// move them plus the sum's rounding.
#[test]
fn v_and_bits_read_as_the_per_line_transforms_within_their_stated_bounds() {
    let noise = |key, (lo, hi), len| {
        Atom::noise(
            key,
            0.0,
            (lo, hi),
            Trajectory {
                base: -40.0,
                knots: vec![(len as f64 / 8_000.0, 0.0)],
            },
        )
    };
    type Pinned = [((usize, usize, usize), f64); 6];
    let fixtures: [(Vec<Atom>, usize, f64, f64, Pinned); 2] = [
        (
            vec![noise(0, (1000.0, 1250.0), 4_000)],
            4_000,
            32_346.617_761_460_85,
            17_645.813_549_648_432,
            [
                ((61, 0, 2), 3.699_150_861_300_768_4e-1),
                ((61, 4, 16), 1.529_207_463_488_195_4),
                ((61, 7, 60), 7.904_617_577_991_47e-1),
                ((61, 8, 49), 7.089_059_933_163_373e-1),
                ((61, 6, 0), 1.129_587_922_153_046_7),
                ((100, 3, 10), 3.683_243_590_896_457e-3),
            ],
        ),
        (
            vec![
                noise(0, (1000.0, 1250.0), 1_600),
                noise(1, (1100.0, 1300.0), 1_600),
            ],
            1_600,
            16_000.780_766_403_59,
            6_114.823_930_599_546,
            [
                ((61, 0, 0), 2.540_002_601_080_393e-1),
                ((61, 4, 6), 1.502_184_238_076_817),
                ((61, 7, 20), 7.909_654_226_937_03e-1),
                ((61, 8, 19), 7.022_147_859_924_868e-1),
                ((61, 6, 0), 1.164_822_850_022_962_7),
                ((100, 3, 4), 1.514_883_304_576_697_6e-2),
            ],
        ),
    ];
    for (atoms, len, pinned_sum, pinned_bits, pinned) in fixtures {
        let clip = Clip { rate: 8_000, len };
        let model = Objective::new(&vec![0.0; clip.len], clip, SPL, 1.0, N0);
        let expected = model.expect(&atoms, true);
        for ((c, k, j), old) in pinned {
            let held = expected[c][k][j].noise;
            assert!(
                (held.v - old).abs() <= held.bound,
                "v at {c} {k} {j}: {held:?} against {old}"
            );
        }
        let (mut sum, mut bound, mut count) = (0.0, 0.0, 0.0);
        let mut reach = 0.0;
        for (channel, bands) in model.heard().channels.iter().zip(&expected) {
            for (band, elements) in channel.bands.iter().zip(bands) {
                let floor = N0 * band.rate_hz;
                for (clip, element) in band.elements.iter().zip(elements) {
                    sum += element.noise.v;
                    bound += element.noise.bound;
                    count += 1.0;
                    reach += bits_reach(element, *clip, floor, band.band.element, band.band.gain());
                }
            }
        }
        let allowed = bound + gamma(count) * sum + gamma(1.0) * pinned_sum;
        println!(
            "{} noises: sum of v {sum} against {pinned_sum}, allowed {allowed}",
            atoms.len()
        );
        assert!((sum - pinned_sum).abs() <= allowed);
        let bits = model.residual_bits(&expected);
        let allowed = reach + gamma(8.0 * count) * (bits + pinned_bits);
        println!("bits {bits} against {pinned_bits}, allowed {allowed}");
        assert!((bits - pinned_bits).abs() <= allowed);
    }
}

/// Against mpmath's `besseli` at 50 digits, `v = 2`, `a = sqrt(2x)`.
#[test]
fn rice_moments_hold_on_both_sides_of_the_switch() {
    let v = 2.0;
    let zero = rice(0.0, v);
    assert!(
        (zero.mean - (PI * v).sqrt() / 2.0).abs() < 1e-15,
        "{zero:?}"
    );
    assert!(
        (zero.variance - (1.0 - PI / 4.0) * v).abs() < 1e-15,
        "{zero:?}"
    );
    let mpmath = [
        (0.01, 1.259_572_887_823_704_7, 0.433_476_140_259_453_1),
        (0.5, 1.548_572_460_551_145_4, 0.601_923_334_422_571_3),
        (2.0, 2.272_383_428_068_742_5, 0.836_273_555_838_550_1),
        (10.0, 4.585_464_850_286_769, 0.973_512_106_784_538_9),
        (30.0, 7.810_792_478_547_886, 0.991_520_857_059_774_2),
        (37.0, 8.660_649_611_399_652, 0.993_148_308_563_053_1),
        (40.0, 9.000_351_714_267_215, 0.993_669_019_487_196_2),
        (45.0, 9.539_686_539_666_86, 0.994_380_724_898_934_7),
        (60.0, 11.000_191_008_212_586, 0.995_797_782_838_960_3),
        (200.0, 20.025_015_684_057_22, 0.998_746_853_262_429_2),
    ];
    for (x, m, s) in mpmath {
        let held = rice((x * v).sqrt(), v);
        let (mean_gap, variance_gap) = ((held.mean - m).abs(), (held.variance - s).abs());
        assert!(
            mean_gap <= held.mean_bound + f64::EPSILON * m,
            "x {x}: {held:?} against {m}"
        );
        assert!(
            variance_gap <= held.variance_bound + f64::EPSILON * s,
            "x {x}: {held:?} against {s}"
        );
        assert!(
            held.mean_bound < 1e-12 * m && held.variance_bound < 1e-10 * s,
            "{held:?}"
        );
    }
}
