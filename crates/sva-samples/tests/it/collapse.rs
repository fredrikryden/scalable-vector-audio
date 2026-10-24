// Concern: states what a collapse owes lines, series, crops and noise: values, tails, refusals | Non-concern: the spectral sum under it (sva-formula) | IO: (a ClosedForm) -> Buffer and Label

use crate::helpers::{part, render, whole_second};
use std::f64::consts::{PI, TAU};

use sva_formula::{Body, Bound, C64, ClosedForm, Edge, IndexId, Origin, Part, Series, Unary, Var};
use sva_samples::{CollapseError, Detail, PSYCHOACOUSTIC_V1, Rule, Source};

const RATE: u32 = 8_192;

/// The terms a geometric series keeps, from a first term bound and a ratio: up to the first
/// whose whole tail stays within half the profile's least significant bit.
fn kept(first: f64, g: f64) -> i32 {
    (0..)
        .find(|n| first * g.powi(*n) / (1.0 - g) <= PSYCHOACOUSTIC_V1.half_lsb())
        .expect("a tail under the precision")
        .max(1)
}

fn form(var: Var, body: Body) -> ClosedForm {
    ClosedForm {
        var,
        body,
        origin: Origin::new(0),
    }
}

fn constant(x: f64) -> Body {
    Body::Const(C64::real(x))
}

fn cosine(hz: f64, amp: f64) -> Body {
    Body::Mul(vec![
        part(constant(amp)),
        part(Body::Apply(
            Unary::Cos,
            part(Body::Mul(vec![part(constant(TAU * hz)), part(Body::Line)])),
        )),
    ])
}

fn sum(parts: Vec<Body>) -> Body {
    Body::Add(parts.into_iter().map(part).collect())
}

/// A sinc tail thins as `1/d^2`, so past a ceiling `d` hertz off one line it integrates to
/// `1/(2*pi^2*d)`, and against the window's own energy that is one more factor of `width`.
fn sinc_tail_db(hz: f64, width: f64) -> f64 {
    let ceiling = f64::from(RATE) / 2.0;
    let spread = 1.0 / (ceiling - hz) + 1.0 / (ceiling + hz);
    10.0 * libm::log10(spread / (2.0 * PI * PI * width))
}

#[test]
fn a_line_spectrum_collapse_is_exact_and_lists_dropped() {
    let law = form(
        Var::T,
        sum(vec![
            cosine(256.0, 1.0),
            cosine(1024.0, 0.5),
            cosine(5000.0, 0.25),
        ]),
    );
    let (buffer, label) = render(&law, RATE, whole_second(), &PSYCHOACOUSTIC_V1).expect("lines");

    assert_eq!(label.source, Source::Exact);
    assert_eq!(buffer.len(), RATE as usize);
    let Detail::Lines {
        summed,
        dropped,
        dropped_more,
        ..
    } = &label.detail
    else {
        panic!("expected a line label, got {:?}", label.detail);
    };
    assert_eq!(
        *summed, 2,
        "two kept frequencies, each a conjugate pair folded"
    );
    assert_eq!(*dropped_more, 0);
    assert_eq!(dropped.len(), 1, "5000 Hz is above 8192/2");
    assert!((dropped[0].hz - 5000.0).abs() < 1e-6);
    assert!((dropped[0].db - 20.0 * libm::log10(0.25f64)).abs() < 1e-9);

    for i in [0usize, 1, 37, 4096, 8191] {
        let t = i as f64 / f64::from(RATE);
        let want = libm::cos(TAU * 256.0 * t) + 0.5 * libm::cos(TAU * 1024.0 * t);
        assert!(
            (buffer.at(0, i) - want).abs() < 1e-10,
            "sample {i}: {} vs {want}",
            buffer.at(0, i)
        );
    }
}

#[test]
fn a_delta_in_ct_refuses() {
    let law = form(
        Var::T,
        Body::Delta {
            at: part(Body::Add(vec![part(Body::Line), part(constant(-0.25))])),
            order: 0,
        },
    );
    let refused = render(&law, RATE, whole_second(), &PSYCHOACOUSTIC_V1).expect_err("a delta");
    assert_eq!(refused.code(), "collapse.singular_in_ct");
}

/// Lines above the ceiling are not rendered, so a form holding no other is silence, each line
/// still listed as dropped: a high note on a shared harmonic table plays nothing, never fails.
#[test]
fn every_line_above_the_ceiling_renders_silence_and_lists_each() {
    let law = form(Var::T, sum(vec![cosine(6000.0, 1.0), cosine(7000.0, 0.5)]));
    let (buffer, label) =
        render(&law, RATE, whole_second(), &PSYCHOACOUSTIC_V1).expect("silence, not a refusal");
    assert_eq!(buffer.len(), RATE as usize);
    assert!((0..buffer.len()).all(|i| buffer.at(0, i) == 0.0));
    let Detail::Lines {
        summed, dropped, ..
    } = &label.detail
    else {
        panic!("expected a line label, got {:?}", label.detail);
    };
    assert_eq!(*summed, 0);
    let mut hz: Vec<f64> = dropped.iter().map(|d| d.hz).collect();
    hz.sort_by(f64::total_cmp);
    assert_eq!(hz, vec![6000.0, 7000.0]);
}

/// A series reaching an instant is truncated once, against the profile's own precision: the
/// terms before its tail rounds away, and no more.
#[test]
fn a_series_under_a_nonlinearity_takes_the_terms_the_precision_leaves() {
    let k = IndexId(1);
    let falling = Body::Apply(
        Unary::Exp,
        part(Body::Mul(vec![
            part(Body::Index(k)),
            part(constant(libm::log(0.5f64))),
        ])),
    );
    let series = Body::Series(Box::new(Series {
        index: k,
        lo: 0,
        hi: Bound::Infinite,
        term: part(Body::Mul(vec![part(falling), part(cosine(220.0, 1.0))])),
    }));
    let law = Body::Mul(vec![
        part(series),
        part(Body::Apply(Unary::Tanh, part(Body::Line))),
    ]);
    let (buffer, label) = render(&form(Var::T, law), RATE, (0.0, 0.02), &PSYCHOACOUSTIC_V1)
        .expect("a series has a value once it is truncated");
    assert_eq!(label.source, Source::Measured);
    let taken: f64 = (0..kept(1.0, 0.5)).map(|n| 0.5f64.powi(n)).sum();
    for i in 0..buffer.len() {
        let t = i as f64 / f64::from(RATE);
        let want = taken * libm::cos(TAU * 220.0 * t) * libm::tanh(t);
        assert!(
            (buffer.at(0, i) - want).abs() < 1e-9,
            "sample {i}: {} against {want}",
            buffer.at(0, i)
        );
    }
}

/// Nothing bounds how many terms of a series an instant needs unless the term carries a
/// coefficient that falls, and a count nothing bounds is a refusal, not a guess.
#[test]
fn a_series_no_coefficient_bounds_refuses_at_a_point() {
    let k = IndexId(1);
    let law = Body::Mul(vec![
        part(Body::Series(Box::new(Series {
            index: k,
            lo: 1,
            hi: Bound::Infinite,
            term: part(Body::Fold(
                sva_formula::Fold::Mod,
                vec![part(Body::Line), part(Body::Index(k))],
            )),
        }))),
        part(Body::Apply(Unary::Tanh, part(Body::Line))),
    ]);
    let refused = render(&form(Var::T, law), RATE, (0.0, 0.01), &PSYCHOACOUSTIC_V1)
        .expect_err("no coefficient bounds the count");
    assert_eq!(refused.code(), "collapse.not_evaluable");
}

/// `sum(cos(2*pi*440*t) / k)` piles every term onto one line and its tail sums to no bound,
/// so no count of terms is its value: it refuses rather than stop where one term is quiet.
#[test]
fn a_line_series_whose_tail_no_bound_sums_refuses() {
    let k = IndexId(1);
    let law = Body::Series(Box::new(Series {
        index: k,
        lo: 1,
        hi: Bound::Infinite,
        term: part(Body::Div(part(cosine(440.0, 1.0)), part(Body::Index(k)))),
    }));
    let refused = render(&form(Var::T, law), RATE, (0.0, 0.01), &PSYCHOACOUSTIC_V1)
        .expect_err("no bound sums the tail");
    assert_eq!(refused.code(), "collapse.not_evaluable");
}

#[test]
fn a_series_reading_a_node_no_bound_holds_refuses() {
    let k = IndexId(1);
    let turning = Body::Apply(
        Unary::Cos,
        part(Body::Mul(vec![
            part(Body::Node(sva_formula::NodeId(0))),
            part(Body::Line),
        ])),
    );
    let series = Body::Series(Box::new(Series {
        index: k,
        lo: 0,
        hi: Bound::Infinite,
        term: part(Body::Mul(vec![
            part(Body::Apply(
                Unary::Exp,
                part(Body::Mul(vec![
                    part(Body::Index(k)),
                    part(constant(libm::log(0.5f64))),
                ])),
            )),
            part(turning),
        ])),
    }));
    let band = sva_samples::Audible::of(&PSYCHOACOUSTIC_V1, RATE);
    let refused = sva_samples::truncate_written(&series, band).expect_err("no bound");
    assert_eq!(refused.code(), "collapse.not_evaluable");
}

/// `sum(sin(k*(2*pi*220*t + 10*sin(2*pi*5*t)))/k)`, a saw whose pitch moves between 170 and
/// 270 Hz, sums at each instant every harmonic under the ceiling there and no other; the
/// loudest it may drop is the first the fastest pitch carries past the ceiling.
#[test]
fn a_saw_whose_pitch_moves_sums_every_harmonic_under_the_ceiling_at_each_instant() {
    let k = IndexId(1);
    let wobble = Body::Mul(vec![
        part(constant(10.0)),
        part(Body::Apply(
            Unary::Sin,
            part(Body::Mul(vec![part(constant(TAU * 5.0)), part(Body::Line)])),
        )),
    ]);
    let phase = Body::Add(vec![
        part(Body::Mul(vec![
            part(constant(TAU * 220.0)),
            part(Body::Line),
        ])),
        part(wobble),
    ]);
    let carrier = Body::Apply(
        Unary::Sin,
        part(Body::Mul(vec![part(Body::Index(k)), part(phase)])),
    );
    let law = Body::Series(Box::new(Series {
        index: k,
        lo: 1,
        hi: Bound::Infinite,
        term: part(Body::Div(part(carrier), part(Body::Index(k)))),
    }));
    let (buffer, label) = render(&form(Var::T, law), RATE, (0.0, 0.25), &PSYCHOACOUSTIC_V1)
        .expect("a moving pitch sums instant by instant");
    let ceiling = PSYCHOACOUSTIC_V1.ceiling(RATE);
    for i in 0..buffer.len() {
        let t = i as f64 / f64::from(RATE);
        let (angle, hz) = (
            TAU * 220.0 * t + 10.0 * libm::sin(TAU * 5.0 * t),
            220.0 + 50.0 * libm::cos(TAU * 5.0 * t),
        );
        let want: f64 = (1..)
            .take_while(|n| f64::from(*n) * hz < ceiling - 1e-6)
            .map(|n| libm::sin(f64::from(n) * angle) / f64::from(n))
            .sum();
        assert!(
            (buffer.at(0, i) - want).abs() < 1e-9,
            "sample {i}: {} against {want}",
            buffer.at(0, i)
        );
    }
    let Detail::Point { tail_db, .. } = label.detail else {
        panic!("a point row, not {:?}", label.detail);
    };
    let first = (1..)
        .find(|n| f64::from(*n) * 270.0 >= ceiling)
        .expect("a harmonic past it");
    let loudest = 20.0 * libm::log10(1.0 / f64::from(first));
    let tail = tail_db.expect("a bound on what it drops");
    assert!((tail - loudest).abs() < 1e-9, "{tail} dB against {loudest}");
}

/// Series inside series expand as their counts multiply. Where that product is past what
/// one instant expands, the refusal says how deep the nesting goes and what it would take.
#[test]
fn nested_series_beyond_the_bound_refuse_naming_the_count() {
    let deep = (0..5).fold(cosine(220.0, 1.0), |inner, level| {
        let k = IndexId(level + 1);
        Body::Series(Box::new(Series {
            index: k,
            lo: 0,
            hi: Bound::Infinite,
            term: part(Body::Mul(vec![
                part(Body::Apply(
                    Unary::Exp,
                    part(Body::Mul(vec![
                        part(Body::Index(k)),
                        part(constant(libm::log(0.82f64))),
                    ])),
                )),
                part(inner),
            ])),
        }))
    });
    let refused = render(&form(Var::T, deep), RATE, (0.0, 0.01), &PSYCHOACOUSTIC_V1)
        .expect_err("five nested series are past the bound");
    assert_eq!(refused.code(), "collapse.series_nesting");
    let CollapseError::NestedSeries {
        depth,
        terms,
        bound,
    } = refused
    else {
        panic!("the refusal prices the nesting, not {refused:?}");
    };
    assert_eq!(depth, 5, "five series deep");
    let terms = terms.expect("a count a usize holds");
    let whole = 1.0 / (1.0 - 0.82f64);
    let levels: usize = (0..5)
        .map(|level| kept(whole.powi(level), 0.82) as usize)
        .product();
    assert_eq!(
        terms, levels,
        "each level's terms under the sum it holds, multiplied five deep"
    );
    assert!(terms > bound, "{terms} is past the bound {bound}");
    assert!(
        refused.to_string().contains("5 deep"),
        "the message names the depth: {refused}"
    );
}

/// A series' price counts series terms. A term holding no series is one value at an
/// instant however many addends its spelling multiplies out to, so forty terms each a product
/// of sums (a formant weight over a moving pitch, say) take forty terms, not those addends.
#[test]
fn a_finite_series_of_products_of_sums_prices_its_own_terms() {
    let k = IndexId(1);
    // `(k + tanh(t) + 1)` nine times over: 3^9 addends were once read as 19683 terms.
    let factor = || {
        part(Body::Add(vec![
            part(Body::Index(k)),
            part(Body::Apply(Unary::Tanh, part(Body::Line))),
            part(constant(1.0)),
        ]))
    };
    let mut factors: Vec<Part> = (0..9).map(|_| factor()).collect();
    factors.push(part(cosine(220.0, 1e-12)));
    let series = Body::Series(Box::new(Series {
        index: k,
        lo: 1,
        hi: Bound::Finite(40),
        term: part(Body::Mul(factors)),
    }));
    render(&form(Var::T, series), RATE, (0.0, 0.01), &PSYCHOACOUSTIC_V1)
        .expect("forty terms are forty terms at one instant");
}

/// A nesting whose count overflows a `usize` says so, and never prints the saturated word.
#[test]
fn a_nesting_past_any_count_refuses_without_a_saturated_count() {
    let deep = (0..4).fold(cosine(220.0, 1.0), |inner, level| {
        let k = IndexId(level + 1);
        Body::Series(Box::new(Series {
            index: k,
            lo: 1,
            hi: Bound::Finite(1_000_000),
            term: part(Body::Mul(vec![
                part(Body::Apply(Unary::Tanh, part(Body::Index(k)))),
                part(inner),
            ])),
        }))
    });
    let refused = render(&form(Var::T, deep), RATE, (0.0, 0.01), &PSYCHOACOUSTIC_V1)
        .expect_err("a million terms four deep are past the bound");
    let CollapseError::NestedSeries { depth, terms, .. } = &refused else {
        panic!("the refusal prices the nesting, not {refused:?}");
    };
    assert_eq!(*depth, 4);
    assert_eq!(*terms, None, "10^24 terms is past what a usize counts");
    let said = refused.to_string();
    assert!(
        !said.contains(&usize::MAX.to_string()),
        "no saturated count leaks: {said}"
    );
    assert!(said.contains("4 deep"), "the depth is still named: {said}");
}

/// FORMAT 3.1: the indicator is half-open, so two windows that abut fire on disjoint
/// instants and their sum never reaches twice one segment's own height at the seam.
#[test]
fn abutting_crops_do_not_double_at_the_seam() {
    let segment = |from: f64, to: f64, hz: f64| Body::Crop {
        of: part(cosine(hz, 1.0)),
        l: Edge::at(from),
        r: Edge::at(to),
        rise: 0.0,
        fall: 0.0,
    };
    let law = form(
        Var::T,
        sum(vec![segment(0.0, 0.5, 440.0), segment(0.5, 1.0, 660.0)]),
    );
    let (buffer, _) =
        render(&law, RATE, whole_second(), &PSYCHOACOUSTIC_V1).expect("two abutting crops");

    let seam = (0.5 * f64::from(RATE)) as usize;
    let (below, above) = (
        libm::cos(TAU * 440.0 * 0.5f64),
        libm::cos(TAU * 660.0 * 0.5f64),
    );
    assert!(
        (buffer.at(0, seam) - above).abs() < 1e-9,
        "the seam instant belongs to the later window alone, got {} against {above} and {below}",
        buffer.at(0, seam)
    );
    for i in 0..buffer.len() {
        assert!(
            buffer.at(0, i).abs() <= 1.0 + 1e-9,
            "sample {i} is {}, past either segment's own height",
            buffer.at(0, i)
        );
    }
}

/// The cropped-pair row: the tail is the energy the window's own dual carries past the
/// ceiling, so a longer window, whose dual is narrower, loses less of it.
#[test]
fn tail_db_falls_as_the_window_lengthens() {
    let cropped = |width: f64| {
        let law = form(
            Var::T,
            Body::Crop {
                of: part(cosine(440.0, 1.0)),
                l: Edge::at(0.0),
                r: Edge::at(width),
                rise: 0.0,
                fall: 0.0,
            },
        );
        let (_, label) =
            render(&law, RATE, (0.0, 2.0), &PSYCHOACOUSTIC_V1).expect("a cropped pair");
        let Detail::Cropped { tail_db, .. } = label.detail else {
            panic!("expected a cropped label, got {:?}", label.detail);
        };
        tail_db.expect("a sinc window states its tail")
    };
    let mut held = f64::INFINITY;
    for width in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6] {
        let found = cropped(width);
        assert!(
            (found - sinc_tail_db(440.0, width)).abs() < 0.05,
            "a {width}s window states {found} dB where the sinc tail is {} dB",
            sinc_tail_db(440.0, width)
        );
        assert!(
            found < held - 2.9,
            "a {width}s window states {found} dB against {held} dB for the shorter one"
        );
        held = found;
    }
}

#[test]
fn noise_alone_sums_its_lines() {
    let law = form(
        Var::T,
        Body::Series(Box::new(sva_formula::noise(1, 0.4, 1.0))),
    );
    let (buffer, label) =
        render(&law, RATE, whole_second(), &PSYCHOACOUSTIC_V1).expect("a noise series");
    let Detail::Lines { summed, .. } = &label.detail else {
        panic!("expected a line label, got {:?}", label.detail);
    };
    assert_eq!(label.rule(), Rule::LineSpectrumSummed);
    assert!(*summed > 1_000, "{summed} lines under the ceiling");

    let placed = sva_formula::lines(
        &sva_formula::noise(1, 0.4, 1.0),
        4_096.0,
        -20.0,
        PSYCHOACOUSTIC_V1.half_lsb(),
    )
    .expect("noise lines");
    for i in [0usize, 137, 4_001] {
        let t = i as f64 / f64::from(RATE);
        let direct: f64 = placed
            .taken
            .iter()
            .map(|l| (l.amp * C64::new(0.0, TAU * l.hz * t).exp()).re)
            .sum();
        assert!(
            (buffer.at(0, i) - direct).abs() < 1e-6,
            "sample {i} places {} where the lines sum to {direct}",
            buffer.at(0, i)
        );
    }
}

#[test]
fn a_shoulder_over_a_noise_series_keeps_a_pair() {
    let law = form(
        Var::T,
        Body::Crop {
            of: part(Body::Series(Box::new(sva_formula::noise(3, 0.25, 1.0)))),
            l: Edge::at(0.2),
            r: Edge::at(0.4),
            rise: 0.01,
            fall: 0.01,
        },
    );
    let (buffer, label) =
        render(&law, RATE, whole_second(), &PSYCHOACOUSTIC_V1).expect("a shouldered noise series");
    assert_eq!(
        label.rule(),
        Rule::CroppedPair,
        "a windowed line spectrum is a cropped pair"
    );
    let at = |secs: f64| (secs * f64::from(RATE)) as usize;
    for i in [0usize, at(0.19), at(0.41), buffer.len() - 1] {
        assert_eq!(buffer.at(0, i), 0.0, "sample {i} sits outside the window");
    }
    assert!(
        (at(0.2)..at(0.4)).any(|i| buffer.at(0, i) != 0.0),
        "the window itself is not silent"
    );
}
