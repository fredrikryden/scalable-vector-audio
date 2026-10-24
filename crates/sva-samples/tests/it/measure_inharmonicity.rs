// Concern: what inharmonicity owes a stiff string, exact and measured | Non-concern: closed forms | IO: (partials or samples) -> asserts

use crate::helpers::partials::stiff_partial;
use sva_samples::measure::harmony::inharmonicity::{InharmonicFrame, measure};
use sva_samples::measure::harmony::{Framing, Moment, Partial, measured};

const RATE: f64 = 44_100.0;

const EXACT: Framing = Framing {
    hop_secs: 0.05,
    window: None,
};

fn exact(hz: f64, amplitude: f64) -> Partial {
    Partial {
        hz,
        amplitude,
        hz_bounds: (hz, hz),
        amplitude_bounds: (amplitude, amplitude),
    }
}

fn frame_of(partials: Vec<Partial>) -> InharmonicFrame {
    let moment = Moment {
        t_secs: 0.0,
        partials,
    };
    measure(&[moment], EXACT, -40.0).frames.remove(0)
}

fn stiff(b: f64) -> Vec<(f64, f64)> {
    (1..=8)
        .map(|k| (stiff_partial(110.0, b, k), 0.4 / f64::from(k)))
        .collect()
}

fn render(partials: &[(f64, f64)], secs: f64) -> Vec<f64> {
    (0..(RATE * secs) as usize)
        .map(|n| {
            let t = n as f64 / RATE;
            partials
                .iter()
                .map(|(f, a)| a * libm::cos(std::f64::consts::TAU * f * t))
                .sum()
        })
        .collect()
}

/// Fletcher's string law read back: n f0 sqrt(1 + B n^2) fits its own B and f0 exactly, and
/// each member stands sharp of n f1 by 1200 log2(n sqrt(1 + B n^2) / (n sqrt(1 + B))).
#[test]
fn a_stiff_strings_partials_fit_their_own_b_and_stand_sharp_by_the_law() {
    let b = 2e-4;
    let read = frame_of(stiff(b).iter().map(|(f, a)| exact(*f, *a)).collect());
    let [series] = &read.series[..] else {
        panic!("one series: {read:?}")
    };
    assert_eq!(series.members.len(), 8);
    for m in &series.members {
        let n = f64::from(m.n);
        let law = 1200.0 * libm::log2(((1.0 + b * n * n) / (1.0 + b)).sqrt());
        assert!(
            (m.cents - law).abs() < 1e-9,
            "n {}: {} vs {law}",
            m.n,
            m.cents
        );
    }
    let fit = series.fit.expect("eight distinct n");
    assert!((fit.b - b).abs() < 1e-9 * b, "{fit:?}");
    assert!((fit.f0_hz - 110.0).abs() < 1e-9, "{fit:?}");
    assert!(read.cents.expect("members").value > 3.0, "{read:?}");
}

#[test]
fn a_measured_stiff_string_agrees_with_its_exact_partials_inside_its_bounds() {
    let b = 2e-4;
    let truth = frame_of(stiff(b).iter().map(|(f, a)| exact(*f, *a)).collect());
    let cents = truth.cents.expect("members").value;
    let (moments, framing) = measured::moments(&[&render(&stiff(b), 0.5)[..]], RATE, 0.0, 0.05);
    let read = measure(&moments, framing, -40.0);
    assert_eq!(read.frames.len(), 10);
    for frame in &read.frames[1..9] {
        let [series] = &frame.series[..] else {
            panic!("one series: {frame:?}")
        };
        let found = frame.cents.expect("members");
        assert!((found.value - cents).abs() < 0.15, "{found:?} vs {cents}");
        assert!(
            found.bounds.0 <= cents && cents <= found.bounds.1,
            "{found:?}"
        );
        let fit = series.fit.expect("eight distinct n");
        let (lo, hi) = fit.b_bounds.expect("a positive f0^2 over the box");
        assert!((fit.b - b).abs() < 0.015 * b, "{fit:?}");
        assert!(lo <= b && b <= hi, "{b} in [{lo}, {hi}]");
        assert_eq!(frame.unexplained.expect("partials").value, 0.0);
    }
}
