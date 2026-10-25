// Concern: B(x): the counts' universal codes, each number's cells at its precision, precisions by looks or curvature | Non-concern: the looks and curvatures | IO: (atoms, Priced) -> bits

use sva_engine::hearing::{JESTEADT_TONE_SECS, WIER_TONE_SECS, frequency_jnd_hz, level_jnd_db};

use super::atoms::{Atom, Feasible, Quantity, Sound};

pub const MODEL: &str = "B(x) in bits: the atom count n by Rissanen's (1983, doi \
     10.1214/aos/1176346150) universal code for integers from 1, log2 c + log*(n + 1), log*(n) \
     the sum of the positive terms of log2 n, log2 log2 n, ..., and c = sum_{n >= 1} \
     2^-log*(n) = 2.865108415839907, so the code meets Kraft's inequality with equality: each \
     region where log* has k positive terms integrates to (ln 2)^k, as d/dx of the k-th \
     iterated log2 is (ln 2)^-k / (x times the k - 1 before it), so the sum is the direct sum \
     to N plus (ln 2)^5 (1 - log2^5(N)) + (ln 2)^6/(1 - ln 2) - f(N)/2 - f'(N)/12 (Euler and \
     Maclaurin) for 16 < log2 N < 65536; then per atom log2 3 for its kind, its level knot \
     count n >= 1 as log2 c + log*(n), a partial's or harmonic's frequency knot count n >= 0 \
     as log2 c + log*(n + 1), a harmonic's H >= 1 as log2 c + log*(H); a number by a uniform \
     code over its range in cells of its precision delta, log2 max(1, ceil(range/delta)); a \
     level, level deviation or offset at delta = (sum_m (h/T_J) DL_J(SL_m)^-2)^-1/2 and a \
     partial's frequency or frequency deviation at the same form in Wier's DLF(F_m, SL_m) \
     over T_W, multiple looks (Viemeister and Wakefield 1991, doi 10.1121/1.401953) over the \
     frames m the number moves, a frame with SL_m <= 0 no look; every other number at Wallace \
     and Freeman's (1987, doi 10.1111/j.2517-6161.1987.tb01695.x) delta = sqrt(12/A_ii), the \
     minimiser of log(range/delta) + A_ii delta^2/24, 1/12 a unit uniform quantiser's second \
     moment; a harmonic's H - 1 offsets, written with the last at minus their sum, share one \
     delta and cost log2 max(1, sqrt(H) (2W/delta)^(H - 1)), the volume of the sum-zero \
     offsets in an orthonormal basis over the cell volume delta^(H - 1), floored at one state: \
     the written box (2W)^(H - 1) maps into that basis with volume sqrt(det(M^T M)) = sqrt(H) \
     times its own whichever coordinate M makes dependent; keys and phases are the ordinal's, \
     0 bits; a deviation's range is minus to plus its quantity's width";

/// `sum_{n >= 1} 2^-log*(n)`, as MODEL derives it; the bits suite recomputes it.
pub const NORMALISER: f64 = 2.865_108_415_839_907;

pub fn log_star(n: u64) -> f64 {
    let mut sum = 0.0;
    let mut term = libm::log2(n as f64);
    while term > 0.0 {
        sum += term;
        term = libm::log2(term);
    }
    sum
}

/// Rissanen's code length for `n >= 1`.
pub fn integer(n: u64) -> f64 {
    assert!(n >= 1, "the universal code starts at 1");
    libm::log2(NORMALISER) + log_star(n)
}

/// A count from 0.
pub fn count(n: usize) -> f64 {
    integer(n as u64 + 1)
}

pub fn cells(range: f64, delta: f64) -> f64 {
    libm::ceil(range / delta).max(1.0)
}

pub fn number(range: f64, delta: f64) -> f64 {
    libm::log2(cells(range, delta))
}

/// A harmonic's written offsets, `deltas` theirs, one shared; `range` each's.
pub fn offsets(range: f64, deltas: &[f64]) -> f64 {
    let Some(delta) = deltas.first() else {
        return 0.0;
    };
    assert!(
        deltas.iter().all(|d| d == delta),
        "a harmonic's offsets share one precision"
    );
    let harmonics = (deltas.len() + 1) as f64;
    // log2 max(1, sqrt(H) (range/delta)^(H - 1)), taken in logs so no power overflows.
    ((harmonics - 1.0) * libm::log2(range / delta) + 0.5 * libm::log2(harmonics)).max(0.0)
}

/// What a number's precision is read from (§2.3): the level JND, the frequency JND, or the
/// curvature where no JND is cited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Level,
    Frequency,
    Curvature,
}

pub fn precision(atom: &Atom, quantity: Quantity) -> Precision {
    match (quantity, &atom.sound) {
        (Quantity::Db | Quantity::DbDeviation | Quantity::Offset, _) => Precision::Level,
        (Quantity::Hz | Quantity::HzDeviation, Sound::Partial { .. }) => Precision::Frequency,
        _ => Precision::Curvature,
    }
}

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

/// `curvature` is `A_ii`, infinite at 0.
pub fn curvature_delta(curvature: f64) -> f64 {
    assert!(curvature >= 0.0, "a Fisher diagonal is never negative");
    libm::sqrt(12.0 / curvature)
}

pub fn range(atom: &Atom, feasible: &Feasible, quantity: Quantity) -> f64 {
    let width = |(lo, hi): (f64, f64)| hi - lo;
    match quantity {
        Quantity::Time => width(feasible.secs),
        Quantity::Hz => width(feasible.hz_of(atom)),
        Quantity::HzDeviation => 2.0 * width(feasible.hz_of(atom)),
        Quantity::Db => width(feasible.db),
        Quantity::DbDeviation | Quantity::Offset => 2.0 * feasible.offset(),
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

pub fn atom(atom: &Atom, priced: &Priced) -> f64 {
    let numbers = atom.numbers();
    assert_eq!(
        priced.deltas.len(),
        numbers.len(),
        "one precision per number"
    );
    let (shared, each): (Vec<_>, Vec<_>) = numbers
        .iter()
        .zip(&priced.deltas)
        .partition(|((q, _), _)| *q == Quantity::Offset);
    let shared: Vec<f64> = shared.into_iter().map(|(_, d)| *d).collect();
    let counts = match &atom.sound {
        Sound::Partial { hz } => count(hz.knots.len()),
        Sound::Harmonic { hz, .. } => integer(atom.harmonics() as u64) + count(hz.knots.len()),
        Sound::Noise { .. } => 0.0,
    } + integer(atom.level.knots.len() as u64);
    let written = each.into_iter().fold(0.0, |sum, ((q, _), d)| {
        sum + number(range(atom, &priced.feasible, *q), *d)
    });
    libm::log2(3.0)
        + counts
        + offsets(range(atom, &priced.feasible, Quantity::Offset), &shared)
        + written
}

pub fn bits(atoms: &[Atom], priced: &[Priced]) -> f64 {
    assert_eq!(atoms.len(), priced.len(), "one pricing per atom");
    atoms
        .iter()
        .zip(priced)
        .fold(count(atoms.len()), |sum, (a, p)| sum + atom(a, p))
}
