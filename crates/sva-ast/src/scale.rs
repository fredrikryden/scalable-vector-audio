// Concern: reads a scale file's rows of cents and names each degree's ratio | Non-concern: which scale a composition plays in (tuning.rs) | IO: (rows) -> Scale; (degree) -> ratio

use crate::diag::{ByteSpan, Diag, DiagCode};

/// Rows in cents above an unwritten tonic, the last the period, as a Scala `.scl` lists them.
#[derive(Clone, Debug, PartialEq)]
pub struct Scale {
    pub cents: Vec<f64>,
}

pub fn is_scale(rows: &[(usize, &str)]) -> bool {
    rows.len() > 1
        && rows
            .iter()
            .all(|(_, row)| row.trim().parse::<f64>().is_ok())
}

pub fn parse_scale(rows: &[(usize, &str)]) -> Result<Scale, Diag> {
    let mut cents: Vec<f64> = Vec::with_capacity(rows.len());
    for (at, row) in rows {
        let value: f64 = row
            .trim()
            .parse()
            .expect("is_scale read every row as a number");
        let below = cents.last().copied().unwrap_or(0.0);
        if !value.is_finite() || value <= below {
            return Err(Diag::new(
                DiagCode::BadScale,
                ByteSpan::new(*at, at + row.len()),
                format!(
                    "a scale row is a degree above the tonic in cents, each above the one \
                     before and the last the period, so `{}` cannot follow `{below}`",
                    row.trim()
                ),
            ));
        }
        cents.push(value);
    }
    Ok(Scale { cents })
}

impl Scale {
    /// Degree 1 is the tonic; every other integer counts on through the rows and periods.
    /// `None` for a degree so far out its ratio is no finite, nonzero number.
    pub fn ratio(&self, degree: i64) -> Option<f64> {
        let n = self.cents.len() as i64;
        let period = *self.cents.last().expect("a scale holds a row");
        let index = degree.checked_sub(1)?;
        let (periods, step) = (index.div_euclid(n), index.rem_euclid(n));
        let within = match step {
            0 => 0.0,
            k => self.cents[k as usize - 1],
        };
        let ratio = 2f64.powf((periods as f64 * period + within) / 1200.0);
        (ratio.is_finite() && ratio > 0.0).then_some(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn major() -> Scale {
        Scale {
            cents: vec![200.0, 400.0, 500.0, 700.0, 900.0, 1100.0, 1200.0],
        }
    }

    #[test]
    fn degrees_count_from_one_and_continue_through_periods_both_ways() {
        let st = |n: f64| 2f64.powf(n / 12.0);
        let close = |a: Option<f64>, b: f64| a.is_some_and(|a| (a - b).abs() < 1e-12);
        assert!(close(major().ratio(1), 1.0));
        assert!(close(major().ratio(5), st(7.0)));
        assert!(close(major().ratio(8), 2.0));
        assert!(close(major().ratio(10), st(16.0)));
        assert!(close(major().ratio(0), st(-1.0)), "the leading tone below");
        assert!(close(major().ratio(-6), 0.5));
        assert_eq!(major().ratio(i64::MIN), None);
        assert_eq!(major().ratio(i64::MAX), None);
    }

    #[test]
    fn a_row_not_above_the_one_before_refuses_located() {
        let rows = [(0, "200"), (4, "100")];
        let refused = parse_scale(&rows).unwrap_err();
        assert_eq!(refused.code, DiagCode::BadScale);
        assert_eq!(refused.span, ByteSpan::new(4, 7));
    }
}
