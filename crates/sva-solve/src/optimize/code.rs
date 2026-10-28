// Concern: bits: Rissanen's code for counts, a number's cells at its precision, a price | Non-concern: what a number means, a move's DL change (the ledger) | IO: (Price) -> bits

pub(crate) const NORMALISER: f64 = 2.865_108_415_839_907;

fn log_star(n: u64) -> f64 {
    let mut sum = 0.0;
    let mut term = libm::log2(n as f64);
    while term > 0.0 {
        sum += term;
        term = libm::log2(term);
    }
    sum
}

fn integer(n: u64) -> f64 {
    assert!(n >= 1, "the universal code starts at 1");
    libm::log2(NORMALISER) + log_star(n)
}

pub(crate) fn count(n: usize) -> f64 {
    integer(n as u64 + 1)
}

fn cells(range: f64, delta: f64) -> f64 {
    libm::ceil(range / delta).max(1.0)
}

pub(crate) fn number(range: f64, delta: f64) -> f64 {
    libm::log2(cells(range, delta))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Count {
    From1(u64),
    From0(usize),
}

impl Count {
    pub(crate) fn bits(self) -> f64 {
        match self {
            Count::From1(n) => integer(n),
            Count::From0(n) => count(n),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Number {
    pub(crate) range: f64,
    pub(crate) delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Price {
    pub(crate) alternatives: usize,
    pub(crate) counts: Vec<Count>,
    pub(crate) numbers: Vec<Number>,
}

impl Price {
    pub(crate) fn bits(&self) -> f64 {
        let counts: f64 = self.counts.iter().map(|c| c.bits()).sum();
        let written = self
            .numbers
            .iter()
            .fold(0.0, |sum, n| sum + number(n.range, n.delta));
        libm::log2(self.alternatives as f64) + counts + written
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::LN_2;

    use super::{NORMALISER, count, integer, log_star, number};

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
    fn no_thing_costs_exactly_the_normalisers_bits() {
        assert_eq!(count(0), libm::log2(NORMALISER));
    }

    #[test]
    fn a_number_whose_precision_spans_its_range_costs_nothing() {
        for range in [1.0, 96.0, 17_000.0] {
            assert_eq!(number(range, range), 0.0);
            assert_eq!(number(range, 2.0 * range), 0.0);
            assert_eq!(number(range, f64::INFINITY), 0.0);
            assert_eq!(number(range, range * (1.0 - f64::EPSILON)), 1.0);
        }
    }
}
