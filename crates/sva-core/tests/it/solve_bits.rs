// Concern: states what B(x) owes: complete codes, its normaliser, silence, atoms' and knots' codes, offsets, precisions | Non-concern: the looks and curvatures | IO: (atoms) -> asserts

use std::f64::consts::LN_2;

use sva_core::solve::atoms::{Atom, Clip, Feasible, Quantity, Sound, Trajectory};
use sva_core::solve::bits::{
    Look, NORMALISER, Precision, Priced, atom, bits, count, curvature_delta, frequency_delta,
    integer, level_delta, log_star, number, offsets, precision,
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
    Feasible::of(&CLIP, FLOOR_DB)
}

/// Each number's precision by its quantity, the offsets one.
fn priced(atom: &Atom) -> Priced {
    let deltas = atom
        .numbers()
        .iter()
        .map(|(q, _)| match q {
            Quantity::Offset => 0.7,
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
            knots(110.0, &[]),
            vec![3.0, -2.0, 1.0],
            knots(-30.0, &[(0.9, -40.0)]),
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
        Quantity::DbDeviation | Quantity::Offset => 2.0 * db,
        Quantity::Edge => f.hz.1 - f.hz.0,
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
    for h in 2..8usize {
        let deltas = vec![0.7; h - 1];
        let states = (h as f64).sqrt() * libm::pow(192.0 / 0.7, (h - 1) as f64);
        let sum = states * libm::exp2(-offsets(192.0, &deltas));
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "H = {h}: the basis' cells sum to {sum}"
        );
    }
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
            .filter(|i| born.numbers()[*i].0 != Quantity::Offset)
            .map(|i| written(born, i))
            .sum::<f64>();
        let level = c + star(born.level.knots.len() as f64);
        let stated = match born.hz() {
            None => level,
            Some(hz) => {
                let shared = match born.harmonics() {
                    1 => 0.0,
                    h => {
                        let states =
                            (h as f64).sqrt() * libm::pow(-2.0 * FLOOR_DB / 0.7, (h - 1) as f64);
                        c + star(h as f64) + libm::log2(states.max(1.0))
                    }
                };
                level + c + star((hz.knots.len() + 1) as f64) + shared
            }
        } + libm::log2(3.0)
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
    if let Sound::Partial { hz } = &mut glided.sound {
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

/// `det(M^T M)` for the written offsets mapped into all `h` with `dependent` at minus the sum.
fn gram(h: usize, dependent: usize) -> f64 {
    let column = |k: usize| -> Vec<f64> {
        let free: Vec<usize> = (0..h).filter(|i| *i != dependent).collect();
        (0..h)
            .map(|row| match row == dependent {
                true => -1.0,
                false => f64::from(u8::from(row == free[k])),
            })
            .collect()
    };
    let n = h - 1;
    let cols: Vec<Vec<f64>> = (0..n).map(column).collect();
    let mut g: Vec<Vec<f64>> = (0..n)
        .map(|a| {
            (0..n)
                .map(|b| (0..h).map(|r| cols[a][r] * cols[b][r]).sum())
                .collect()
        })
        .collect();
    let mut det = 1.0;
    for p in 0..n {
        det *= g[p][p];
        let pivot = g[p].clone();
        for row in &mut g[p + 1..] {
            let factor = row[p] / pivot[p];
            for (v, above) in row.iter_mut().zip(&pivot).skip(p) {
                *v -= factor * above;
            }
        }
    }
    det
}

#[test]
fn the_offsets_cost_the_same_whichever_coordinate_is_dependent() {
    for h in 2..9usize {
        let deltas = vec![0.7; h - 1];
        let cells = (h - 1) as f64 * libm::log2(192.0 / 0.7);
        let jacobian = offsets(192.0, &deltas) - cells;
        for dependent in 0..h {
            let volume = 0.5 * libm::log2(gram(h, dependent));
            assert!(
                (jacobian - volume).abs() < 1e-12,
                "H = {h}, o_{dependent} dependent: {jacobian} vs {volume}"
            );
        }
    }
    assert_eq!(offsets(192.0, &[]), 0.0, "one harmonic writes no offset");
}

#[test]
fn offsets_of_one_state_cost_nothing() {
    for h in 2..9usize {
        let wide = 192.0 * (h as f64).sqrt();
        assert_eq!(offsets(192.0, &vec![wide; h - 1]), 0.0, "H = {h}");
        assert_eq!(offsets(192.0, &vec![f64::INFINITY; h - 1]), 0.0, "H = {h}");
    }
}

#[test]
fn offsets_of_many_cells_cost_the_ceiled_cells_and_the_basis_volume_within_the_ceil() {
    for h in 2..9usize {
        for delta in [0.7, 0.13, 0.031] {
            let ratio = 192.0 / delta;
            let ceiled =
                (h - 1) as f64 * libm::log2(libm::ceil(ratio)) + 0.5 * libm::log2(h as f64);
            let ceil = (h - 1) as f64 * libm::log2(libm::ceil(ratio) / ratio);
            let gap = ceiled - offsets(192.0, &vec![delta; h - 1]);
            assert!(
                (-1e-12..=ceil + 1e-12).contains(&gap),
                "H = {h}, delta {delta}: {gap} vs the ceil's {ceil}"
            );
        }
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
    let counts = libm::log2(3.0) + c + star(2.0) + c + star(2.0);
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
        if let Sound::Partial { hz } = &mut moved.sound {
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

#[test]
fn a_curvatures_precision_minimises_its_cells_and_quantisation() {
    for a in [0.01, 1.0, 250.0] {
        let delta = curvature_delta(a);
        let cost = |d: f64| libm::log(1.0 / d) + a * d * d / 24.0;
        let (mut lo, mut hi) = (delta / 10.0, delta * 10.0);
        for _ in 0..200 {
            let (l, r) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            match cost(l) < cost(r) {
                true => hi = r,
                false => lo = l,
            }
        }
        assert!(((lo + hi) / 2.0 / delta - 1.0).abs() < 1e-6, "A = {a}");
    }
    assert_eq!(curvature_delta(0.0), f64::INFINITY);
}

#[test]
fn each_number_reads_its_precision_from_its_cited_jnd_or_its_curvature() {
    let all = atoms();
    let read = |a: &Atom| -> Vec<Precision> {
        a.numbers().iter().map(|(q, _)| precision(a, *q)).collect()
    };
    use Precision::{Curvature as C, Frequency as F, Level as L};
    assert_eq!(read(&all[0]), [C, F, C, F, L, C, L, C, L]);
    assert_eq!(read(&all[1]), [C, C, L, L, L, L, C, L]);
    assert_eq!(read(&all[2]), [C, C, C, L, C, L]);
}
