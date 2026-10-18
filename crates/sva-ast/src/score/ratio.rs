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

    /// A decimal as written, `digits[.digits][e±digits]`, exactly: `"0.01"` is 1/100, where
    /// the double nearest it is not.
    pub fn of_decimal(text: &str) -> Option<Ratio> {
        let (mantissa, exponent) = match text.find(['e', 'E']) {
            Some(at) => (&text[..at], text[at + 1..].parse::<i32>().ok()?),
            None => (text, 0),
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let digits = format!("{whole}{fraction}");
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let digits = digits.trim_start_matches('0');
        let num: i128 = match digits.is_empty() {
            true => 0,
            false => digits.parse().ok()?,
        };
        let shift = exponent.checked_sub(i32::try_from(fraction.len()).ok()?)?;
        let ten = |n: i32| 10i128.checked_pow(n.unsigned_abs());
        match shift >= 0 {
            true => Ratio::new(num.checked_mul(ten(shift)?)?, 1),
            false => Ratio::new(num, ten(shift)?),
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

    #[test]
    fn a_written_decimal_is_read_as_the_decimal_it_spells() {
        let hundredth = Ratio::of_decimal("0.01").unwrap();
        assert_eq!(hundredth, Ratio::new(1, 100).unwrap());
        assert_ne!(Ratio::of_f64(0.01).unwrap(), hundredth);
        assert_eq!(
            Ratio::of_decimal("2.5e-3").unwrap(),
            Ratio::new(1, 400).unwrap()
        );
        assert_eq!(Ratio::of_decimal("1.2E3").unwrap(), Ratio::int(1200));
        assert_eq!(Ratio::of_decimal(".5").unwrap(), Ratio::new(1, 2).unwrap());
        assert_eq!(Ratio::of_decimal("007").unwrap(), Ratio::int(7));
        assert_eq!(Ratio::of_decimal("0.000").unwrap(), Ratio::ZERO);
        assert_eq!(Ratio::of_decimal("1e40"), None);
        assert_eq!(Ratio::of_decimal(""), None);
        let mut at = Ratio::int(48);
        for _ in 0..1000 {
            at = at.plus(hundredth).unwrap();
        }
        assert_eq!(at, Ratio::int(58));
    }
}
