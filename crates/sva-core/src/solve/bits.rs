// Concern: B(x) of an atom: its counts and numbers as a price, each number's range, precisions by looks | Non-concern: the codes (optimize/), the looks (fit.rs) | IO: (atom, Priced) -> Price

use sva_engine::hearing::{JESTEADT_TONE_SECS, WIER_TONE_SECS, frequency_jnd_hz, level_jnd_db};

use super::atoms::{Atom, Feasible, Quantity, Sound};
use super::optimize::code::{Count, Number, Price};

pub const MODEL: &str = "B(x) in bits: the atom count n by Rissanen's (1983, doi \
     10.1214/aos/1176346150) universal code for integers from 1, log2 c + log*(n + 1), log*(n) \
     the sum of the positive terms of log2 n, log2 log2 n, ..., and c = sum_{n >= 1} \
     2^-log*(n) = 2.865108415839907, so the code meets Kraft's inequality with equality: each \
     region where log* has k positive terms integrates to (ln 2)^k, as d/dx of the k-th \
     iterated log2 is (ln 2)^-k / (x times the k - 1 before it), so the sum is the direct sum \
     to N plus (ln 2)^5 (1 - log2^5(N)) + (ln 2)^6/(1 - ln 2) - f(N)/2 - f'(N)/12 (Euler and \
     Maclaurin) for 16 < log2 N < 65536; then per atom log2 6 for its kind (partial, \
     gliding partial, harmonic, gliding harmonic, noise band or shaped noise), each level's knot \
     count n >= 1 as log2 c + log*(n), a partial's or harmonic's frequency knot count n >= 0 \
     as log2 c + log*(n + 1), a harmonic's H >= 1 as log2 c + log*(H), a shaped noise's \
     band count n >= 1 as log2 c + log*(n); a number by a uniform \
     code over its range in cells of its precision delta, log2 max(1, ceil(range/delta)); a \
     level or level deviation at delta = (sum_m (h/T_J) DL_J(SL_m)^-2)^-1/2 and a \
     partial's frequency or frequency deviation at the same form in Wier's DLF(F_m, SL_m) \
     over T_W, multiple looks (Viemeister and Wakefield 1991, doi 10.1121/1.401953) over the \
     frames m the number moves, a frame with SL_m <= 0 no look; every other number at the \
     precision its caller states (fit.rs); a stretch from 0 to where the highest harmonic \
     meets the top of the feasible set; a glide over minus to plus the rate that moves the \
     feasible set's lowest Hz to its highest over the clip; keys and phases are the ordinal's, \
     0 bits; a deviation's range is minus to plus its quantity's width";

/// One frame a number moves: the atom's own excitation peak over threshold in dB, and its
/// trajectory's frequency there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    pub sl_db: f64,
    pub hz: f64,
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

pub fn level_delta(hop_secs: f64, looks: &[Look]) -> f64 {
    looked(hop_secs, JESTEADT_TONE_SECS, looks, |look| {
        level_jnd_db(look.sl_db)
    })
}

pub fn frequency_delta(hop_secs: f64, looks: &[Look]) -> f64 {
    looked(hop_secs, WIER_TONE_SECS, looks, |look| {
        frequency_jnd_hz(look.hz, look.sl_db)
    })
}

pub fn range(atom: &Atom, feasible: &Feasible, quantity: Quantity) -> f64 {
    let width = |(lo, hi): (f64, f64)| hi - lo;
    match quantity {
        Quantity::Time => width(feasible.secs),
        Quantity::Hz => width(feasible.hz_of(atom)),
        Quantity::HzDeviation => 2.0 * width(feasible.hz_of(atom)),
        Quantity::Stretch => {
            let h = atom.harmonics() as f64;
            let f0 = atom.hz().map_or(0.0, |t| t.base);
            let top = feasible.hz.1 / (h * f0);
            ((top * top - 1.0) / (h * h)).max(0.0)
        }
        Quantity::Glide => 2.0 * libm::log(feasible.hz.1 / feasible.hz.0) / width(feasible.secs),
        Quantity::Db => width(feasible.db),
        Quantity::DbDeviation => 2.0 * feasible.offset(),
        Quantity::Edge => width(feasible.hz),
    }
}

/// An atom's feasible set and each number's precision in `values()`' order, refreshed by the
/// caller between calls.
#[derive(Clone, Debug, PartialEq)]
pub struct Priced {
    pub feasible: Feasible,
    pub deltas: Vec<f64>,
}

/// `atom`'s price: one of its six kinds, its counts, and each number at its range and precision.
pub fn price(atom: &Atom, priced: &Priced) -> Price {
    let numbers = atom.numbers();
    assert_eq!(
        priced.deltas.len(),
        numbers.len(),
        "one precision per number"
    );
    let shape = match &atom.sound {
        Sound::Partial { hz, .. } => vec![Count::From0(hz.knots.len())],
        Sound::Harmonic { hz, .. } => vec![
            Count::From1(atom.harmonics() as u64),
            Count::From0(hz.knots.len()),
        ],
        Sound::Noise { .. } => Vec::new(),
        Sound::Shaped { bands } => vec![Count::From1(bands.len() as u64)],
    };
    let levels = atom
        .levels()
        .iter()
        .map(|l| Count::From1(l.knots.len() as u64))
        .collect();
    Price {
        alternatives: 6,
        counts: vec![shape, levels],
        numbers: numbers
            .iter()
            .zip(&priced.deltas)
            .map(|((q, _), delta)| Number {
                range: range(atom, &priced.feasible, *q),
                delta: *delta,
            })
            .collect(),
    }
}
