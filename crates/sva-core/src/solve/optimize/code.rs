// Concern: bits: Rissanen's code for counts, a number's cells at its precision, a price, the things' bits | Non-concern: what a number means, a move's DL change (the ledger) | IO: (Price) -> bits

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

pub fn integer(n: u64) -> f64 {
    assert!(n >= 1, "the universal code starts at 1");
    libm::log2(NORMALISER) + log_star(n)
}

pub fn count(n: usize) -> f64 {
    integer(n as u64 + 1)
}

pub fn cells(range: f64, delta: f64) -> f64 {
    libm::ceil(range / delta).max(1.0)
}

pub fn number(range: f64, delta: f64) -> f64 {
    libm::log2(cells(range, delta))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Count {
    From1(u64),
    From0(usize),
}

impl Count {
    pub fn bits(self) -> f64 {
        match self {
            Count::From1(n) => integer(n),
            Count::From0(n) => count(n),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Number {
    pub range: f64,
    pub delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Price {
    pub alternatives: usize,
    pub counts: Vec<Count>,
    pub numbers: Vec<Number>,
}

impl Price {
    pub fn bits(&self) -> f64 {
        let counts: f64 = self.counts.iter().map(|c| c.bits()).sum();
        let written = self
            .numbers
            .iter()
            .fold(0.0, |sum, n| sum + number(n.range, n.delta));
        libm::log2(self.alternatives as f64) + counts + written
    }
}

pub fn bits(prices: &[Price]) -> f64 {
    prices
        .iter()
        .fold(count(prices.len()), |sum, p| sum + p.bits())
}
