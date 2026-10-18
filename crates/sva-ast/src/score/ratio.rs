// Concern: an exact rational, the one form a bar position or duration is decided in | Non-concern: what a bar lasts in seconds (song/tempo.rs) | IO: (f64 or i64) -> Ratio -> f64

use std::cmp::Ordering;

/// `num / den` in lowest terms, `den > 0`. Both stay within 2^62 so a comparison's cross
/// products fit an `i128`; arithmetic past that answers `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ratio {
    num: i128,
    den: i128,
}

const LIMIT: i128 = 1 << 62;

fn gcd(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

impl Ratio {
    pub const ZERO: Ratio = Ratio { num: 0, den: 1 };

    fn new(num: i128, den: i128) -> Option<Ratio> {
        if den == 0 {
            return None;
        }
        let sign = if den < 0 { -1 } else { 1 };
        let g = gcd(num, den).max(1);
        let (num, den) = (sign * num / g, sign * den / g);
        (num.abs() <= LIMIT && den <= LIMIT).then_some(Ratio { num, den })
    }

    pub fn int(n: i64) -> Ratio {
        Ratio {
            num: i128::from(n),
            den: 1,
        }
    }

    /// The double's own value, exactly: every finite double is a dyadic rational.
    pub fn of_f64(x: f64) -> Option<Ratio> {
        if !x.is_finite() {
            return None;
        }
        if x == 0.0 {
            return Some(Ratio::ZERO);
        }
        let bits = x.to_bits();
        let sign: i128 = if bits >> 63 == 0 { 1 } else { -1 };
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = i128::from(bits & ((1u64 << 52) - 1));
        let (mantissa, exponent) = match exponent {
            0 => (fraction, -1074),
            e => (fraction | (1 << 52), e - 1075),
        };
        let shift = mantissa.trailing_zeros() as i32;
        let (mantissa, exponent) = (mantissa >> shift, exponent + shift);
        match exponent >= 0 {
            true if exponent < 62 => Ratio::new(sign * (mantissa << exponent), 1),
            true => None,
            false if exponent > -63 => Ratio::new(sign * mantissa, 1 << -exponent),
            false => None,
        }
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn plus(self, o: Ratio) -> Option<Ratio> {
        Ratio::new(
            self.num
                .checked_mul(o.den)?
                .checked_add(o.num.checked_mul(self.den)?)?,
            self.den.checked_mul(o.den)?,
        )
    }

    pub fn minus(self, o: Ratio) -> Option<Ratio> {
        self.plus(o.negated())
    }

    pub fn times(self, o: Ratio) -> Option<Ratio> {
        Ratio::new(self.num.checked_mul(o.num)?, self.den.checked_mul(o.den)?)
    }

    pub fn over(self, o: Ratio) -> Option<Ratio> {
        Ratio::new(self.num.checked_mul(o.den)?, self.den.checked_mul(o.num)?)
    }

    pub fn negated(self) -> Ratio {
        Ratio {
            num: -self.num,
            den: self.den,
        }
    }

    pub fn is_negative(self) -> bool {
        self.num < 0
    }

    pub fn is_positive(self) -> bool {
        self.num > 0
    }

    /// The whole number this is, where it is one.
    pub fn whole(self) -> Option<i64> {
        (self.den == 1)
            .then(|| i64::try_from(self.num).ok())
            .flatten()
    }
}

impl Ord for Ratio {
    fn cmp(&self, o: &Ratio) -> Ordering {
        (self.num * o.den).cmp(&(o.num * self.den))
    }
}

impl PartialOrd for Ratio {
    fn partial_cmp(&self, o: &Ratio) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_double_is_read_exactly_and_sums_stay_exact() {
        let third = Ratio::int(1).over(Ratio::int(3)).unwrap();
        let tenth = Ratio::of_f64(0.1).unwrap();
        assert_eq!(tenth.to_f64(), 0.1);
        assert_eq!(Ratio::of_f64(0.375).unwrap(), Ratio::new(3, 8).unwrap());
        let sum = third.plus(third).unwrap().plus(third).unwrap();
        assert_eq!(sum, Ratio::int(1));
        assert!(third < Ratio::of_f64(0.34).unwrap());
        assert_eq!(Ratio::of_f64(-2.5).unwrap().to_f64(), -2.5);
        assert_eq!(Ratio::int(6).over(Ratio::int(2)).unwrap().whole(), Some(3));
        assert_eq!(Ratio::of_f64(1e30), None);
    }
}
