// Concern: B(x) of a source: its counts and numbers as a price, each number's range, precisions by looks | Non-concern: the codes (optimize/), the looks (fit/) | IO: (Source, Priced) -> Price

use sva_engine::hearing::{JESTEADT_TONE_SECS, WIER_TONE_SECS, frequency_jnd_hz, level_jnd_db};

use crate::optimize::code::{Count, Number, Price};
use crate::sources::{Feasible, Quantity, Sound, Source};

pub const MODEL: &str = "B(x) in bits: the source count n by Rissanen's (1983, doi \
     10.1214/aos/1176346150) universal code for integers from 1, log2 c + log*(n + 1), log*(n) \
     the sum of the positive terms of log2 n, log2 log2 n, ..., and c = sum_{n >= 1} \
     2^-log*(n) = 2.865108415839907, so the code meets Kraft's inequality with equality: each \
     region where log* has k positive terms integrates to (ln 2)^k, as d/dx of the k-th \
     iterated log2 is (ln 2)^-k / (x times the k - 1 before it), so the sum is the direct sum \
     to N plus (ln 2)^5 (1 - log2^5(N)) + (ln 2)^6/(1 - ln 2) - f(N)/2 - f'(N)/12 (Euler and \
     Maclaurin) for 16 < log2 N < 65536; then per source log2 6 for its kind (partial, \
     gliding partial, harmonic, gliding harmonic, noise band or shaped noise), each level's knot \
     count n >= 1 as log2 c + log*(n), a partial's or harmonic's frequency knot count, 0 as written, \
     as log2 c + log*(0 + 1), a harmonic's H >= 1 as log2 c + log*(H), a shaped noise's \
     band count n >= 1 as log2 c + log*(n); a number by a uniform \
     code over its range in cells of its precision delta, log2 max(1, ceil(range/delta)); a \
     level or level deviation at delta = (sum_m (h/T_J) DL_J(SL_m)^-2)^-1/2 and a \
     partial's frequency at the same form in Wier's DLF(F_m, SL_m) \
     over T_W, multiple looks (Viemeister and Wakefield 1991, doi 10.1121/1.401953) over the \
     frames m the number moves, a frame with SL_m <= 0 no look; every other number at the \
     precision its caller states (fit/); a stretch from 0 to where the highest harmonic \
     meets the top of the feasible set; a glide over minus to plus the rate that moves the \
     feasible set's lowest Hz to its highest over the clip; keys and phases are the ordinal's, \
     0 bits; a deviation's range is minus to plus its quantity's width, a periodic level's depth \
     a level deviation's; which shared variable, if any, a source references the ledger's \
     (optimize/)";

/// One frame a number moves: the source's own excitation peak over threshold in dB, and its
/// trajectory's frequency there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Look {
    pub(crate) sl_db: f64,
    pub(crate) hz: f64,
}

fn looked(hop_secs: f64, tone_secs: f64, looks: &[Look], jnd: impl Fn(&Look) -> f64) -> f64 {
    let information = looks
        .iter()
        .filter(|look| look.sl_db > 0.0)
        .fold(0.0, |sum, look| {
            let step = jnd(look);
            sum + hop_secs / tone_secs / (step * step)
        });
    1.0 / libm::sqrt(information)
}

pub(crate) fn level_delta(hop_secs: f64, looks: &[Look]) -> f64 {
    looked(hop_secs, JESTEADT_TONE_SECS, looks, |look| {
        level_jnd_db(look.sl_db)
    })
}

pub(crate) fn frequency_delta(hop_secs: f64, looks: &[Look]) -> f64 {
    looked(hop_secs, WIER_TONE_SECS, looks, |look| {
        frequency_jnd_hz(look.hz, look.sl_db)
    })
}

fn range(source: &Source, feasible: &Feasible, quantity: Quantity) -> f64 {
    let width = |(lo, hi): (f64, f64)| hi - lo;
    match quantity {
        Quantity::Time => width(feasible.secs),
        Quantity::Hz => width(feasible.hz_of(source)),
        Quantity::Stretch => {
            let h = source.harmonics() as f64;
            let f0 = source.hz().unwrap_or(0.0);
            let top = feasible.hz.1 / (h * f0);
            ((top * top - 1.0) / (h * h)).max(0.0)
        }
        Quantity::Glide => 2.0 * libm::log(feasible.hz.1 / feasible.hz.0) / width(feasible.secs),
        Quantity::Db => width(feasible.db),
        Quantity::DbDeviation | Quantity::Depth => 2.0 * feasible.offset(),
        Quantity::Edge => width(feasible.hz),
    }
}

/// A source's feasible set and each number's precision in `numbers()`' order, refreshed by the
/// caller between calls.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Priced {
    pub(crate) feasible: Feasible,
    pub(crate) deltas: Vec<f64>,
}

/// `source`'s price: one of its six kinds, its counts, and each number at its range and
/// precision; a tonal one's frequency knot count the 0 its node's name writes (`f0`).
pub(crate) fn price(source: &Source, priced: &Priced) -> Price {
    let numbers = source.numbers();
    assert_eq!(
        priced.deltas.len(),
        numbers.len(),
        "one precision per number"
    );
    let shape = match &source.sound {
        Sound::Partial { .. } => vec![Count::From0(0)],
        Sound::Harmonic { count, .. } => vec![Count::From1(*count as u64), Count::From0(0)],
        Sound::Noise { .. } => Vec::new(),
        Sound::Shaped { bands } => vec![Count::From1(bands.len() as u64)],
    };
    let levels = source
        .levels
        .iter()
        .map(|l| Count::From1(l.knots.len() as u64));
    Price {
        alternatives: 6,
        counts: shape.into_iter().chain(levels).collect(),
        numbers: numbers
            .iter()
            .zip(&priced.deltas)
            .map(|((q, _), delta)| Number {
                range: range(source, &priced.feasible, *q),
                delta: *delta,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use sva_engine::hearing::{frequency_jnd_hz, level_jnd_db};

    use super::{Look, Priced, frequency_delta, level_delta, price};
    use crate::optimize::code::{NORMALISER, count, number};
    use crate::sources::{Clip, Feasible, Quantity, Sound, Source, Trajectory};

    const CLIP: Clip = Clip {
        rate: 16_000,
        len: 16_000,
    };

    const FLOOR_DB: f64 = -90.0;

    fn star(n: f64) -> f64 {
        let mut sum = 0.0;
        let mut term = libm::log2(n);
        while term > 0.0 {
            sum += term;
            term = libm::log2(term);
        }
        sum
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
    fn priced(source: &Source) -> Priced {
        let deltas = source
            .numbers()
            .iter()
            .map(|(q, _)| match q {
                Quantity::Stretch => 1e-5,
                Quantity::Glide => 0.05,
                Quantity::Time => 0.003,
                Quantity::Hz | Quantity::Edge => 2.9,
                Quantity::Db | Quantity::DbDeviation | Quantity::Depth => 0.4,
            })
            .collect();
        Priced {
            feasible: feasible(),
            deltas,
        }
    }

    fn sources() -> Vec<Source> {
        vec![
            Source::new(
                0,
                0.1,
                Sound::Partial {
                    hz: 440.0,
                    glide: 0.0,
                },
                vec![knots(-20.0, &[(0.2, 3.0), (0.6, -30.0)])],
            ),
            Source::new(
                1,
                0.2,
                Sound::Harmonic {
                    hz: 110.0,
                    stretch: 0.0,
                    count: 4,
                    glide: 0.0,
                },
                vec![
                    knots(-30.0, &[(0.9, -40.0)]),
                    knots(-27.0, &[(0.9, -40.0)]),
                    knots(-32.0, &[(0.4, -6.0), (0.9, -40.0)]),
                    knots(-29.0, &[(0.9, -40.0)]),
                ],
            ),
            Source::new(
                2,
                0.0,
                Sound::Noise {
                    lo: 1000.0,
                    hi: 4000.0,
                },
                vec![knots(-40.0, &[(0.5, 0.0)])],
            ),
        ]
    }

    fn total(sources: &[Source]) -> f64 {
        sources.iter().fold(count(sources.len()), |sum, a| {
            sum + price(a, &priced(a)).bits()
        })
    }

    /// The cells of number `i` of `source` at its precision, from its stated range.
    fn written(source: &Source, i: usize) -> f64 {
        let f = feasible();
        let (q, _) = source.numbers()[i];
        let hz = (f.hz.1 / source.harmonics() as f64) - f.hz.0;
        let db = f.db.1 - f.db.0;
        let range = match q {
            Quantity::Time => CLIP.secs(),
            Quantity::Hz => hz,
            Quantity::Db => db,
            Quantity::DbDeviation | Quantity::Depth => 2.0 * db,
            Quantity::Edge => f.hz.1 - f.hz.0,
            Quantity::Glide => 2.0 * libm::log(f.hz.1 / f.hz.0) / CLIP.secs(),
            Quantity::Stretch => {
                let h = source.harmonics() as f64;
                let top = f.hz.1 / (h * 110.0);
                (top * top - 1.0) / (h * h)
            }
        };
        libm::log2(libm::ceil(range / priced(source).deltas[i]).max(1.0))
    }

    #[test]
    fn adding_a_source_costs_its_stated_code() {
        let all = sources();
        let c = libm::log2(NORMALISER);
        for n in 0..all.len() {
            let born = &all[n];
            let numbers = (0..born.numbers().len())
                .map(|i| written(born, i))
                .sum::<f64>();
            let level: f64 = born
                .levels
                .iter()
                .map(|l| c + star(l.knots.len() as f64))
                .sum();
            let stated = match born.hz() {
                None => level,
                Some(_) => {
                    let shared = match born.harmonics() {
                        1 => 0.0,
                        h => c + star(h as f64),
                    };
                    level + c + shared
                }
            } + libm::log2(6.0)
                + numbers;
            let grown = total(&all[..=n]) - total(&all[..n]);
            let counted = star((n + 2) as f64) - star((n + 1) as f64);
            assert!(
                (grown - counted - stated).abs() < 1e-11,
                "source {n}: {grown} vs {counted} + {stated}"
            );
        }
    }

    #[test]
    fn adding_a_knot_costs_its_count_its_time_and_its_deviation() {
        let c = sources()[0].clone();
        let at = |a: &Source| price(a, &priced(a)).bits();
        let mut levelled = c.clone();
        levelled.levels[0].knots.insert(1, (0.4, -10.0));
        let stated = star(3.0) - star(2.0) + written(&levelled, 5) + written(&levelled, 6);
        let grew = at(&levelled) - at(&c);
        assert!((grew - stated).abs() < 1e-11, "{grew} vs {stated}");
    }

    #[test]
    fn a_number_whose_precision_spans_its_range_costs_nothing() {
        let a = sources()[0].clone();
        let coarse = Priced {
            feasible: feasible(),
            deltas: vec![f64::INFINITY; a.numbers().len()],
        };
        let c = libm::log2(NORMALISER);
        let counts = libm::log2(6.0) + c + star(1.0) + c + star(2.0);
        assert!((price(&a, &coarse).bits() - counts).abs() < 1e-12);
    }

    #[test]
    fn a_deviations_code_does_not_change_with_its_value() {
        let a = sources()[0].clone();
        let p = priced(&a);
        let base = price(&a, &p).bits();
        for (d, hz) in [(-89.0, 2000.0), (89.0, 30.0), (0.0, 7000.0)] {
            let mut moved = a.clone();
            moved.levels[0].knots[0].1 = d;
            moved.levels[0].base = -60.0;
            moved.sound = Sound::Partial { hz, glide: 0.0 };
            assert_eq!(
                price(&moved, &p).bits(),
                base,
                "deviation {d} dB at {hz} Hz"
            );
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
}
