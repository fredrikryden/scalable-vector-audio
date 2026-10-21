// Concern: the whole samples after which a formula repeats exactly, where it does | Non-concern: folding a demand by it (segments.rs), evaluating it | IO: (form, its rows, grid) -> a period or none

use sva_formula::{Body, ClosedForm, Fold};
use sva_samples::{Grid, Rows};

use crate::time::{Lattice, Q};

/// Undamped commensurate lines repeat after `1/gcd`, a written `(a*t + b) % P` after `P/|a|`;
/// in samples, the numerator of `P*rate/step`.
pub(crate) fn period(form: Option<&ClosedForm>, rows: &Rows, grid: Grid) -> Option<i64> {
    match rows.lines() {
        Some(hz) => lines_period(&hz, grid),
        None => samples(wrapped(&form?.body)?, grid),
    }
}

/// The samples after which lines at `hz` repeat, where they do.
pub(crate) fn lines_period(hz: &[f64], grid: Grid) -> Option<i64> {
    samples(lines(hz)?, grid)
}

fn samples(seconds: Q, grid: Grid) -> Option<i64> {
    let samples = seconds
        .mul(Q::int(i64::from(grid.rate)))?
        .div(grid.step())?;
    i64::try_from(samples.num()).ok().filter(|n| *n > 0)
}

/// The whole number of cycles a line at `hz` turns over `n` samples, signed as `hz`.
pub(crate) fn bin(hz: f64, n: i64, grid: Grid) -> Option<i64> {
    let cycles = written(hz.abs())?
        .mul(Q::int(n))?
        .mul(grid.step())?
        .div(Q::int(i64::from(grid.rate)))?;
    let whole = i64::try_from(cycles.num()).ok()?;
    cycles.is_integer().then_some(whole * hz.signum() as i64)
}

fn lines(hz: &[f64]) -> Option<Q> {
    let mut held: Option<Q> = None;
    let mut last: Option<(f64, Q)> = None;
    for f in hz.iter().filter(|f| **f != 0.0).map(|f| f.abs()) {
        let q = match last {
            Some((was, q)) if was == f => q,
            _ => written(f)?,
        };
        last = Some((f, q));
        held = Some(match held {
            None => q,
            Some(g) if divides(g, q) => g,
            Some(g) => gcd(g, q)?,
        });
    }
    Q::ONE.div(held?)
}

/// The decimal a product with 2π rounded away from: the fewest figures, at most 15, whose
/// double lies within 4 ulps.
fn written(hz: f64) -> Option<Q> {
    let ulp = f64::from_bits(hz.to_bits() + 1) - hz;
    let exact = exactly(hz).or_else(|| shortest(hz));
    (1..=15usize).find_map(
        |digits| match exact.and_then(|e| rounded(e, digits as u32)) {
            Some((near, (kept, shift))) => ((near - hz).abs() <= 4.0 * ulp)
                .then(|| Q::digits(i128::from(kept), shift))
                .flatten(),
            None => {
                let text = format!("{:.*e}", digits - 1, hz);
                let near: f64 = text.parse().ok()?;
                ((near - hz).abs() <= 4.0 * ulp)
                    .then(|| Q::decimal(near))
                    .flatten()
            }
        },
    )
}

/// A double's exact value as `n * 10^p`, where `n` fits.
fn exactly(x: f64) -> Option<(u128, i32)> {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if x <= 0.0 || biased == 0 || biased == 0x7ff {
        return None;
    }
    let whole = (bits & ((1 << 52) - 1)) | (1 << 52);
    let zeros = whole.trailing_zeros();
    let (m, e) = (u128::from(whole >> zeros), biased - 1075 + zeros as i32);
    match e >= 0 {
        true => (e <= 60).then(|| (m << e, 0)),
        false => (e >= -30).then(|| (m * 5u128.pow(e.unsigned_abs()), e)),
    }
}

/// The shortest decimal that parses to `x`, as `n * 10^p`.
fn shortest(x: f64) -> Option<(u128, i32)> {
    let text = format!("{x:e}");
    let (mantissa, exp) = text.split_once('e')?;
    let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let part: u64 = match frac.is_empty() {
        true => 0,
        false => frac.parse().ok()?,
    };
    let whole: u64 = whole.parse().ok()?;
    let n = whole
        .checked_mul(10u64.checked_pow(frac.len() as u32)?)?
        .checked_add(part)?;
    Some((u128::from(n), exp.parse::<i32>().ok()? - frac.len() as i32))
}

/// `x` to `digits` figures as the formatter rounds it, and the double that parses to. Its
/// shortest decimal rounds alike but at an inexact midpoint, `None`: one between would be
/// shorter or nearer. 15 figures or fewer are their double's shortest, as `Q::decimal` reads.
fn rounded((n, p): (u128, i32), digits: u32) -> Option<(f64, (u64, i32))> {
    let figures = n.checked_ilog10()? + 1;
    let (mut kept, mut shift) = match figures > digits {
        true => {
            let cut = 10u128.pow(figures - digits);
            let (q, r) = (n / cut, n % cut);
            let kept = match r.cmp(&(cut / 2)) {
                std::cmp::Ordering::Less => q,
                std::cmp::Ordering::Greater => q + 1,
                std::cmp::Ordering::Equal if is_double((n, p)) => q + (q & 1),
                std::cmp::Ordering::Equal => return None,
            };
            (kept, p + (figures - digits) as i32)
        }
        false => (n, p),
    };
    while kept % 10 == 0 {
        kept /= 10;
        shift += 1;
    }
    let kept = u64::try_from(kept).ok().filter(|k| *k < 1 << 53)?;
    let power = POWERS.get(shift.unsigned_abs() as usize)?;
    let near = match shift >= 0 {
        true => kept as f64 * power,
        false => kept as f64 / power,
    };
    Some((near, (kept, shift)))
}

/// Whether `n * 10^p` is exactly a double.
fn is_double((n, p): (u128, i32)) -> bool {
    let fits = |m: u128| m < 1 << 53;
    match p >= 0 {
        true => 10u128
            .checked_pow(p.unsigned_abs())
            .and_then(|ten| n.checked_mul(ten))
            .is_some_and(fits),
        false => 5u128
            .checked_pow(p.unsigned_abs())
            .is_some_and(|five| n % five == 0 && fits(n / five)),
    }
}

const POWERS: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `b` a whole multiple of `a`, so `gcd(a, b)` is `a`, in 64 bits where nothing overflows.
fn divides(a: Q, b: Q) -> bool {
    let small = [a.num(), a.den(), b.num(), b.den()]
        .iter()
        .all(|x| x.unsigned_abs() < 1 << 31);
    small && {
        let narrow = |x: i128| x as i64;
        let (top, bottom) = (
            narrow(b.num()) * narrow(a.den()),
            narrow(a.num()) * narrow(b.den()),
        );
        bottom != 0 && top % bottom == 0
    }
}

fn gcd(a: Q, b: Q) -> Option<Q> {
    let whole = |x: i128, y: i128| {
        let (mut x, mut y) = (x.abs(), y.abs());
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    let den = a.den().checked_mul(b.den())? / whole(a.den(), b.den());
    let (x, y) = (a.num() * (den / a.den()), b.num() * (den / b.den()));
    Q::new(whole(x, y), den)
}

fn wrapped(body: &Body) -> Option<Q> {
    let mut periods = Vec::new();
    if !mods(body, &mut periods) || periods.is_empty() {
        return None;
    }
    periods.into_iter().try_fold(None, |held: Option<Q>, p| {
        Some(Some(match held {
            None => p,
            Some(q) => q.mul(p)?.div(gcd(q, p)?)?,
        }))
    })?
}

fn mods(body: &Body, out: &mut Vec<Q>) -> bool {
    match body {
        Body::Line => false,
        Body::Fold(Fold::Mod, parts) => match parts.as_slice() {
            [x, p] => match (affine(&x.body), constant(&p.body)) {
                (Some(a), Some(p)) if !a.is_zero() && p > Q::ZERO => match p.div(abs(a)) {
                    Some(period) => {
                        out.push(period);
                        true
                    }
                    None => false,
                },
                _ => false,
            },
            _ => false,
        },
        other => sva_formula::closed_form::children(other)
            .iter()
            .all(|p| mods(&p.body, out)),
    }
}

fn abs(q: Q) -> Q {
    match q < Q::ZERO {
        true => q.neg(),
        false => q,
    }
}

fn affine(body: &Body) -> Option<Q> {
    let coeffs = sva_formula::affine::polynomial(body)?;
    let exact = |c: &sva_formula::affine::Coeff| {
        let c = c.exact()?;
        (c.im == 0.0).then(|| Q::decimal(c.re)).flatten()
    };
    match coeffs.as_slice() {
        [_, a] => exact(a),
        [_] => Some(Q::ZERO),
        _ => None,
    }
}

fn constant(body: &Body) -> Option<Q> {
    match body {
        Body::Const(c) if c.im == 0.0 => Q::decimal(c.re),
        _ => None,
    }
}
