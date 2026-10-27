// Concern: states what B(x) owes: complete codes, its normaliser, silence, atoms' and knots' codes, precisions by looks | Non-concern: the looks | IO: (atoms) -> asserts

use std::f64::consts::LN_2;

use sva_core::solve::atoms::{Atom, Clip, Feasible, Quantity, Sound, Trajectory};
use sva_core::solve::bits::{
    Look, NORMALISER, Priced, atom, bits, count, frequency_delta, integer, level_delta, log_star,
    number,
};
use sva_engine::hearing::{frequency_jnd_hz, level_jnd_db};

const CLIP: Clip = Clip {
    rate: 16_000,
    len: 16_000,
};

const FLOOR_DB: f64 = -90.0;

/// The iterated log2s while positive, written from the definition alone.
fn iterated(n: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut term = libm::log2(n);
    while term > 0.0 {
        out.push(term);
        term = libm::log2(term);
    }
    out
}

fn star(n: f64) -> f64 {
    iterated(n).iter().sum()
}

/// `2^-log*(n)` and its derivative in `n`.
fn weight(n: f64) -> (f64, f64) {
    let logs = iterated(n);
    let f = libm::exp2(-logs.iter().sum::<f64>());
    let mut product = 1.0;
    let mut slope = 1.0;
    for (j, l) in logs.iter().enumerate().take(logs.len().saturating_sub(1)) {
        product *= l;
        slope += 1.0 / (libm::pow(LN_2, (j + 1) as f64) * product);
    }
    (f, -f * slope / n)
}

/// `sum_{n=1}^{last} 2^-log*(n)`, Neumaier-compensated.
fn head(last: u64) -> f64 {
    let (mut sum, mut lost) = (0.0f64, 0.0f64);
    for n in 1..=last {
        let term = weight(n as f64).0;
        let next = sum + term;
        lost += match sum.abs() >= term.abs() {
            true => (sum - next) + term,
            false => (term - next) + sum,
        };
        sum = next;
    }
    sum + lost
}

/// `sum_{n > last} 2^-log*(n)` for `16 < log2 last < 65536`: the integral, each region of k
/// positive terms `(ln 2)^k`, by Euler and Maclaurin.
fn tail(last: u64) -> f64 {
    let n = last as f64;
    let logs = iterated(n);
    assert_eq!(logs.len(), 5, "the fifth region");
    let (f, slope) = weight(n);
    libm::pow(LN_2, 5.0) * (1.0 - logs[4]) + libm::pow(LN_2, 6.0) / (1.0 - LN_2)
        - f / 2.0
        - slope / 12.0
}

fn knots(base: f64, knots: &[(f64, f64)]) -> Trajectory {
    Trajectory {
        base,
        knots: knots.to_vec(),
    }
}

fn feasible() -> Feasible {
    Feasible::of(&CLIP, (FLOOR_DB, 0.0))
}

/// Each number's precision by its quantity.
fn priced(atom: &Atom) -> Priced {
    let deltas = atom
        .numbers()
        .iter()
        .map(|(q, _)| match q {
            Quantity::Stretch => 1e-5,
            Quantity::Glide => 0.05,
            Quantity::Time => 0.003,
            Quantity::Hz | Quantity::HzDeviation | Quantity::Edge => 2.9,
            Quantity::Db | Quantity::DbDeviation => 0.4,
        })
        .collect();
    Priced {
        feasible: feasible(),
        deltas,
    }
}

fn atoms() -> Vec<Atom> {
    vec![
        Atom::partial(
            0,
            0.1,
            knots(440.0, &[(0.3, 20.0)]),
            knots(-20.0, &[(0.2, 3.0), (0.6, -30.0)]),
        ),
        Atom::harmonic(
            1,
            0.2,
            (knots(110.0, &[]), 0.0),
            vec![
                knots(-30.0, &[(0.9, -40.0)]),
                knots(-27.0, &[(0.9, -40.0)]),
                knots(-32.0, &[(0.4, -6.0), (0.9, -40.0)]),
                knots(-29.0, &[(0.9, -40.0)]),
            ],
        ),
        Atom::noise(2, 0.0, (1000.0, 4000.0), knots(-40.0, &[(0.5, 0.0)])),
    ]
}

fn total(atoms: &[Atom]) -> f64 {
    bits(atoms, &atoms.iter().map(priced).collect::<Vec<_>>())
}

/// The cells of number `i` of `atom` at its precision, from §2.3's ranges.
fn written(atom: &Atom, i: usize) -> f64 {
    let f = feasible();
    let (q, _) = atom.numbers()[i];
    let hz = (f.hz.1 / atom.harmonics() as f64) - f.hz.0;
    let db = f.db.1 - f.db.0;
    let range = match q {
        Quantity::Time => CLIP.secs(),
        Quantity::Hz => hz,
        Quantity::HzDeviation => 2.0 * hz,
        Quantity::Db => db,
        Quantity::DbDeviation => 2.0 * db,
        Quantity::Edge => f.hz.1 - f.hz.0,
        Quantity::Glide => 2.0 * libm::log(f.hz.1 / f.hz.0) / CLIP.secs(),
        Quantity::Stretch => {
            let h = atom.harmonics() as f64;
            let top = f.hz.1 / (h * 110.0);
            (top * top - 1.0) / (h * h)
        }
    };
    libm::log2(libm::ceil(range / priced(atom).deltas[i]).max(1.0))
}

#[test]
fn the_normaliser_is_the_sum_of_two_to_minus_log_star() {
    let last = 1_000_000;
    let sum = head(last) + tail(last);
    assert!(
        (sum - NORMALISER).abs() <= 4.0 * f64::EPSILON * NORMALISER,
        "sum {sum:.17} vs {NORMALISER:.17}"
    );
}

#[test]
fn log_star_reads_rissanen_on_towers_of_two_and_between() {
    for (n, expected) in [(1, 0.0), (2, 1.0), (4, 3.0), (16, 7.0), (65_536, 23.0)] {
        assert_eq!(log_star(n), expected, "log*({n})");
    }
    for n in [3u64, 5, 17, 1000] {
        assert_eq!(log_star(n), star(n as f64), "log*({n})");
    }
    assert_eq!(integer(5), libm::log2(NORMALISER) + star(5.0));
    assert_eq!(
        count(4),
        integer(5),
        "a count from 0 is its successor's integer"
    );
}

#[test]
fn every_code_meets_kraft_and_the_complete_ones_with_equality() {
    let mut partial = 0.0;
    for n in 1..=100_000u64 {
        partial += libm::exp2(-integer(n));
        assert!(partial <= 1.0, "the first {n} integers sum to {partial}");
    }
    let complete = (head(1_000_000) + tail(1_000_000)) / NORMALISER;
    assert!((complete - 1.0).abs() < 1e-14, "integers: {complete}");
}

#[test]
fn silence_costs_exactly_the_normalisers_bits() {
    assert_eq!(bits(&[], &[]), libm::log2(NORMALISER));
}

#[test]
fn adding_an_atom_costs_its_stated_code() {
    let all = atoms();
    let c = libm::log2(NORMALISER);
    for n in 0..all.len() {
        let born = &all[n];
        let numbers = (0..born.numbers().len())
            .map(|i| written(born, i))
            .sum::<f64>();
        let level: f64 = born
            .levels()
            .iter()
            .map(|l| c + star(l.knots.len() as f64))
            .sum();
        let stated = match born.hz() {
            None => level,
            Some(hz) => {
                let shared = match born.harmonics() {
                    1 => 0.0,
                    h => c + star(h as f64),
                };
                level + c + star((hz.knots.len() + 1) as f64) + shared
            }
        } + libm::log2(6.0)
            + numbers;
        let grown = total(&all[..=n]) - total(&all[..n]);
        let counted = star((n + 2) as f64) - star((n + 1) as f64);
        assert!(
            (grown - counted - stated).abs() < 1e-11,
            "atom {n}: {grown} vs {counted} + {stated}"
        );
    }
}

#[test]
fn adding_a_knot_costs_its_count_its_time_and_its_deviation() {
    let c = atoms()[0].clone();
    let at = |a: &Atom| atom(a, &priced(a));
    let mut levelled = c.clone();
    levelled.level.knots.insert(1, (0.4, -10.0));
    let mut glided = c.clone();
    if let Sound::Partial { hz, .. } = &mut glided.sound {
        hz.knots.push((0.5, 40.0));
    }
    for (grown, time, counts) in [
        (&levelled, 7, star(3.0) - star(2.0)),
        (&glided, 4, star(3.0) - star(2.0)),
    ] {
        let stated = counts + written(grown, time) + written(grown, time + 1);
        let grew = at(grown) - at(&c);
        assert!((grew - stated).abs() < 1e-11, "{grew} vs {stated}");
    }
}

#[test]
fn a_number_whose_precision_spans_its_range_costs_nothing() {
    for range in [1.0, 96.0, 17_000.0] {
        assert_eq!(number(range, range), 0.0);
        assert_eq!(number(range, 2.0 * range), 0.0);
        assert_eq!(number(range, f64::INFINITY), 0.0);
        assert_eq!(number(range, range * (1.0 - f64::EPSILON)), 1.0);
    }
    let a = atoms()[0].clone();
    let coarse = Priced {
        feasible: feasible(),
        deltas: vec![f64::INFINITY; a.numbers().len()],
    };
    let c = libm::log2(NORMALISER);
    let counts = libm::log2(6.0) + c + star(2.0) + c + star(2.0);
    assert!((atom(&a, &coarse) - counts).abs() < 1e-12);
}

#[test]
fn a_deviations_code_does_not_change_with_its_value() {
    let a = atoms()[0].clone();
    let p = priced(&a);
    let base = atom(&a, &p);
    for (d, f) in [(-89.0, 1.0), (89.0, -400.0), (0.0, 7000.0)] {
        let mut moved = a.clone();
        moved.level.knots[0].1 = d;
        moved.level.base = -60.0;
        if let Sound::Partial { hz, .. } = &mut moved.sound {
            hz.knots[0].1 = f;
            hz.base = 2000.0;
        }
        assert_eq!(atom(&moved, &p), base, "deviations {d} dB, {f} Hz");
    }
}

#[test]
fn a_jnd_of_looks_over_a_tone_as_long_as_the_papers_is_their_jnd() {
    let hop = 0.001;
    let looks = vec![
        Look {
            sl_db: 40.0,
            hz: 1000.0
        };
        500
    ];
    let level = level_delta(hop, &looks);
    let frequency = frequency_delta(hop, &looks);
    assert!(
        (level / level_jnd_db(40.0) - 1.0).abs() < 1e-13,
        "{level} dB"
    );
    assert!(
        (frequency / frequency_jnd_hz(1000.0, 40.0) - 1.0).abs() < 1e-13,
        "{frequency} Hz"
    );
    let quarter = level_delta(hop, &looks[..125]);
    assert!(
        (quarter / level - 2.0).abs() < 1e-13,
        "a quarter the looks, twice the step"
    );
}

#[test]
fn an_inaudible_frame_changes_no_precision() {
    let heard = [
        Look {
            sl_db: 12.0,
            hz: 300.0,
        },
        Look {
            sl_db: 55.0,
            hz: 310.0,
        },
    ];
    let mut with = heard.to_vec();
    for (i, sl_db) in [(0, 0.0), (2, -15.0), (3, -0.0)] {
        with.insert(i, Look { sl_db, hz: 305.0 });
    }
    assert_eq!(level_delta(0.001, &with), level_delta(0.001, &heard));
    assert_eq!(
        frequency_delta(0.001, &with),
        frequency_delta(0.001, &heard)
    );
    let silent = [Look {
        sl_db: 0.0,
        hz: 300.0,
    }];
    assert_eq!(level_delta(0.001, &silent), f64::INFINITY);
    assert_eq!(number(96.0, frequency_delta(0.001, &silent)), 0.0);
}
