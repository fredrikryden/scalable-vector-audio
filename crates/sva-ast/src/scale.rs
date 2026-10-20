// Concern: a scale's rows of cents, the scales the language names, each degree's ratio | Non-concern: which scale a composition plays in (reserved.rs) | IO: (rows | name) -> Scale; (degree) -> ratio

use crate::diag::{ByteSpan, Diag, DiagCode};

/// Rows in cents above an unwritten tonic, the last the period, as a Scala `.scl` lists them.
#[derive(Clone, Debug, PartialEq)]
pub struct Scale {
    pub cents: Vec<f64>,
}

const TWELVE: &str = "twelve-tone equal temperament";

/// maqamworld.com's quarter-tone spelling read as 24-tone equal; a performer's neutral second
/// or third varies by region.
const MAQAM: &str = "24-tone equal temperament, as maqamworld.com notates the maqam";

const MAJOR: &[f64] = &[200.0, 400.0, 500.0, 700.0, 900.0, 1100.0, 1200.0];
const MINOR: &[f64] = &[200.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1200.0];

/// `(name, cents with the period last, their tuning)`: the scales `variables/scale` may name
/// bare; any other is a scale file named by a ref.
pub const BUILTIN_SCALES: [(&str, &[f64], &str); 19] = [
    ("major", MAJOR, TWELVE),
    ("minor", MINOR, TWELVE),
    ("ionian", MAJOR, TWELVE),
    (
        "dorian",
        &[200.0, 300.0, 500.0, 700.0, 900.0, 1000.0, 1200.0],
        TWELVE,
    ),
    (
        "phrygian",
        &[100.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1200.0],
        TWELVE,
    ),
    (
        "lydian",
        &[200.0, 400.0, 600.0, 700.0, 900.0, 1100.0, 1200.0],
        TWELVE,
    ),
    (
        "mixolydian",
        &[200.0, 400.0, 500.0, 700.0, 900.0, 1000.0, 1200.0],
        TWELVE,
    ),
    ("aeolian", MINOR, TWELVE),
    (
        "locrian",
        &[100.0, 300.0, 500.0, 600.0, 800.0, 1000.0, 1200.0],
        TWELVE,
    ),
    (
        "harmonic-minor",
        &[200.0, 300.0, 500.0, 700.0, 800.0, 1100.0, 1200.0],
        TWELVE,
    ),
    (
        "melodic-minor",
        &[200.0, 300.0, 500.0, 700.0, 900.0, 1100.0, 1200.0],
        TWELVE,
    ),
    (
        "major-pentatonic",
        &[200.0, 400.0, 700.0, 900.0, 1200.0],
        TWELVE,
    ),
    (
        "minor-pentatonic",
        &[300.0, 500.0, 700.0, 1000.0, 1200.0],
        TWELVE,
    ),
    (
        "blues",
        &[300.0, 500.0, 600.0, 700.0, 1000.0, 1200.0],
        TWELVE,
    ),
    (
        "whole-tone",
        &[200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0],
        TWELVE,
    ),
    (
        "chromatic",
        &[
            100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0,
        ],
        TWELVE,
    ),
    (
        "rast",
        &[200.0, 350.0, 500.0, 700.0, 900.0, 1050.0, 1200.0],
        MAQAM,
    ),
    (
        "bayati",
        &[150.0, 300.0, 500.0, 700.0, 800.0, 1000.0, 1200.0],
        MAQAM,
    ),
    (
        "hijaz",
        &[100.0, 400.0, 500.0, 700.0, 800.0, 1000.0, 1200.0],
        MAQAM,
    ),
];

pub fn builtin_scale(name: &str) -> Option<Scale> {
    BUILTIN_SCALES
        .iter()
        .find(|(known, ..)| *known == name)
        .map(|(_, cents, _)| Scale {
            cents: cents.to_vec(),
        })
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
        let ratio = libm::exp2((periods as f64 * period + within) / 1200.0);
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
        let st = |n: f64| libm::exp2(n / 12.0);
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
    fn every_builtin_scale_is_rows_a_scale_file_could_hold() {
        for (name, cents, _) in BUILTIN_SCALES {
            let text: Vec<String> = cents.iter().map(f64::to_string).collect();
            let rows: Vec<(usize, &str)> = text.iter().map(|row| (0, row.as_str())).collect();
            assert!(is_scale(&rows), "{name}");
            assert_eq!(parse_scale(&rows).ok(), builtin_scale(name), "{name}");
            assert_eq!(cents.last(), Some(&1200.0), "{name} repeats at the octave");
            let named = BUILTIN_SCALES.iter().filter(|(n, ..)| *n == name).count();
            assert_eq!(named, 1, "{name} is named once");
        }
        assert_eq!(builtin_scale("major"), Some(major()));
        assert_eq!(builtin_scale("ionian"), builtin_scale("major"));
        assert_eq!(builtin_scale("aeolian"), builtin_scale("minor"));
        assert_eq!(builtin_scale("Major"), None);
    }

    #[test]
    fn a_row_not_above_the_one_before_refuses_located() {
        let rows = [(0, "200"), (4, "100")];
        let refused = parse_scale(&rows).unwrap_err();
        assert_eq!(refused.code, DiagCode::BadScale);
        assert_eq!(refused.span, ByteSpan::new(4, 7));
    }
}
