// Concern: where one partial of a rendered string rings, and how loud | Non-concern: rendering it (fd.rs) | IO: (samples, rate, Hz) -> Hz, dB

/// Partial `k` of a stiff string: `k f0 sqrt(1 + b k^2)`.
pub fn stiff_partial(f0: f64, b: f64, k: u32) -> f64 {
    f64::from(k) * f0 * (1.0 + b * f64::from(k * k)).sqrt()
}
