// Concern: proves a series is a value, truncated once against a ceiling | Non-concern: summing its lines into a buffer (sva-samples) | IO: (a Series) -> the lines it yields

use crate::fixtures::{cosine, part, saw_series, term};
use sva_formula::{Body, Bound, IndexId, Series, Var, lines, noise, normalize_closed_form};

fn enumerate(s: &Series, ceiling: f64) -> Option<sva_formula::Lines> {
    lines(s, ceiling, -20.0, 2f64.powi(-24))
}

fn series_of(f: &Body) -> Series {
    let mut sum = normalize_closed_form(&term(Var::T, f.clone())).expect("a series value");
    sum.lanes.remove(0).series.remove(0)
}

#[test]
fn a_series_truncates_against_the_ceiling_and_reports_its_tail() {
    let answer = enumerate(&series_of(&saw_series(220.0)), 22050.0).expect("a saw's lines");
    let highest = answer
        .taken
        .iter()
        .map(|l| l.hz.abs())
        .fold(0.0f64, f64::max);
    assert!(highest < 22050.0, "nothing at or above the ceiling is kept");
    assert!((highest - 220.0 * 100.0).abs() < 1e-6, "{highest}");
    assert!(
        !answer.dropped.is_empty(),
        "the first line out of band is listed"
    );
    assert!(
        answer.tail_db < 0.0,
        "the tail is quieter than what was kept"
    );
}

/// `sum(cos(2*pi*440*t) / k^2)` piles every term onto one line, and no ratio bounds `1/k^2`:
/// the walk stops only where everything it drops sums under the floor, and the tail it
/// states is that sum, never one term.
#[test]
fn a_tail_no_ratio_bounds_is_dropped_only_where_its_whole_sum_is_under_the_floor() {
    let k = IndexId(0);
    let series = Series {
        index: k,
        lo: 1,
        hi: Bound::Infinite,
        term: part(Body::Div(
            part(cosine(440.0)),
            part(Body::Pow(part(Body::Index(k)), 2)),
        )),
    };
    let answer = enumerate(&series, 20_000.0).expect("a bounded tail");
    let last = answer.taken.len() / 2;
    let kept: f64 = answer.taken.iter().map(|l| l.amp.abs()).sum();
    let dropped = (std::f64::consts::PI.powi(2) / 6.0 - kept) / 2.0;
    let loudest = answer
        .taken
        .iter()
        .map(|l| l.amp.abs())
        .fold(0.0f64, f64::max);
    assert!(
        dropped < loudest * 0.1,
        "the {last} terms kept leave {} dB of tail, over the -20 dB floor",
        20.0 * libm::log10(dropped / loudest)
    );
    assert!(
        answer.tail_db >= 20.0 * libm::log10(dropped / loudest),
        "the stated tail {} dB covers the {} dB dropped",
        answer.tail_db,
        20.0 * libm::log10(dropped / loudest)
    );
}

#[test]
fn a_tail_no_bound_sums_names_no_lines() {
    let k = IndexId(0);
    let series = Series {
        index: k,
        lo: 1,
        hi: Bound::Infinite,
        term: part(Body::Div(part(cosine(440.0)), part(Body::Index(k)))),
    };
    assert_eq!(enumerate(&series, 20_000.0), None);
}

fn band_power(period: f64, color: f64) -> f64 {
    let answer = enumerate(&noise(7, period, color), 2_000.0).expect("noise lines");
    answer
        .taken
        .iter()
        .filter(|l| (500.1..1_500.1).contains(&l.hz.abs()))
        .map(|l| l.amp.abs().powi(2))
        .sum()
}

/// The builtins table's contract: a level a hertz the period never moves, flat noise half a
/// unit of power a hertz, and a color tilting that density alike at every period.
#[test]
fn noise_holds_its_level_a_hertz_whatever_its_period() {
    for period in [0.25, 1.0, 4.0] {
        let flat = band_power(period, 1.0);
        assert!(
            (flat - 500.0).abs() < 1e-6,
            "period {period} holds {flat} over 1000 Hz, not 500"
        );
    }
    for color in [libm::pow(10f64, -3.0 / 20.0), 2.0] {
        let at_one = band_power(1.0, color);
        for period in [0.25, 4.0] {
            let db = 10.0 * libm::log10(band_power(period, color) / at_one);
            assert!(
                db.abs() < 0.05,
                "color {color} at period {period} moves the band {db} dB from period 1"
            );
        }
    }
}
