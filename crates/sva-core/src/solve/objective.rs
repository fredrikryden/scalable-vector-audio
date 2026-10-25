// Concern: OBJECTIVE: a state's expected elements, variances N0 rho + v, Rice moments, residual bits against the clip, the total with B(x) | Non-concern: tangents, search | IO: (clip, atoms, N0) -> bits

use std::f64::consts::{LN_2, LN_10, PI, SQRT_2};

use sva_engine::fft;
use sva_engine::internal::modulation::Element;
use sva_engine::internal::{Ear, Internal};

use super::atoms::{Atom, Clip, Sound};
use super::bits::{self, Priced};
use super::spectrum::{Model, fft_bound, gamma, line_span};

pub const MODEL: &str = "residual bits = sum_i [d_i^2 log2(e)/2 + log2(sigma_i/sigma_0,i)], \
     the Gaussian code length of the clip's elements given the model's over the noise-free \
     code, channels summed, place only (no temporal cue), d_i = (e_model,i - e_clip,i)/sigma_i; \
     sigma_0,i^2 = N0 rho_kc, equal noise density in every band (calibrate.rs), and sigma_i^2 \
     = N0 rho_kc + v_i; the model's elements are the reading of each frame's expected \
     excitation, partials and harmonics as written and noise in expectation over its lines' \
     phases (spectrum.rs); v_i is the variance the render's own noise lines k/T give element i \
     at the frame it reads, each line's excitation e_k through that frame's bins, filters set \
     by the frame's whole expected spectrum, and V linearised, s = 10 log10(e)/(E_T + E), a \
     model form whose error is measured, not bounded: over 48 keyed renders of a 1 to 1.25 \
     kHz band at 8 kHz the renders' variance is 0.94, 1.02 and 1.89 times v in the lowpass, \
     real and magnitude bands, the log's curvature the linear form drops: v = \
     s^2 sum over ordered pairs of distinct lines of e_k e_l K((l - k)/T) (Isserlis 1918, doi \
     10.1093/biomet/12.1-2.134: a pair's beat at their difference, of random phase, so pairs \
     do not covary), a line's own term (n = 0) never counted, so v >= 0, summed as s^2 [K(0) \
     q + sum_(n != 0) K(n/T) r(n)], q the ordered pairs of distinct atoms on one line, r the \
     autocorrelation of x_k, line k's excitation summed over atoms, the second sum by \
     Parseval, (1/M) sum_f Re K^(f) |X(f)|^2 over M = 2^ceil(log2 2L) >= 2L - 1 points (no \
     lag wraps; K^ the transform of K placed circularly, 0 at n = 0, so n = 0 is excluded \
     exactly), X by one M/2-point transform of x's even samples plus i its odd, then \
     clamped at 0, where the exact value lies; within its rounding of the direct sum: |X^ - \
     X|_2 <= phi' sqrt(M) |x|_2, phi' = phi_(M/2) + 2 sqrt(2) (gamma_5 + mu)(1 + phi_(M/2)) \
     (Higham 2002, Accuracy and Stability of Numerical Algorithms, 2nd ed., Thm 24.2: phi_n \
     = log2(n) eta/(1 - log2(n) eta), eta = mu + gamma_4 (sqrt(2) + mu), twiddles within mu \
     = 12U), so |P^ - P|_2 <= M |x|_2^2 [phi' (2 + phi') + gamma_2 (1 + phi')^2], K^ within \
     phi_M sqrt(M) |K|_2, the sum within (sqrt(M) |K|_2 dP + dK^ |P^|_2)/M plus gamma_(M/2 \
     + 4) of its terms' moduli; each x_k within e_x = sum_b W_b e_b (1 + gamma_B) + \
     gamma_(B + A) max x (e_b its bins' bound, spectrum.rs, W the filters' nonnegative \
     weights over B bins, A atoms), so r and q within 2 e_x sum x + L e_x^2 at every lag, \
     q's sum within gamma_(2A + L) more, moving v by s^2 sum_(n != 0) K and s^2 K(0); v is \
     reported beside that bound plus gamma_10 v for s^2, K = |H|^2 from E to \
     the band's output: the squared window's transform of the channel's range, squared, times \
     the reading's own digital filters at the hop, a real band through (M(nu) + conj M(-nu))/2; \
     a real element's mean is its output and its variance v; a magnitude element's output y \
     is the expected one plus a complex Gaussian of variance v, v/2 per quadrature, so its \
     element is gain |y| with Rice's (1945, BSTJ 24(1) 46-156, doi \
     10.1002/j.1538-7305.1945.tb00453.x, eqs 3.10-11 to 3.10-13) mean and variance, written \
     free of cancellation as mean = a + (v/2a)(1 - D) and variance = v D - (v/2a)^2 (1 - D)^2, \
     a = |E y|, x = a^2/v, z = x/2, D = (1 + x)(1 - S_0) + x (1 - S_1), S_nu = sqrt(2 pi z) \
     e^-z I_nu(z) by Hankel's expansion (DLMF 10.40.1), its remainder after l terms within 2 \
     chi(l) |a_l(nu)| z^-l exp(|nu^2 - 1/4| pi/(2z)) (DLMF 10.40.10-12 at z e^(pi i), through \
     10.34.2, chi(l) = sqrt(pi) Gamma(l/2 + 1)/Gamma(l/2 + 1/2)), taken where that bound on D \
     is under 2^-53 of D, else the power series of e^-z I_nu(z), all its terms positive, \
     summed until its geometric tail bound is under 2^-53 of the sum, mean = sqrt(v/2) \
     sqrt(pi/2) e^-z ((1 + x) I_0 + x I_1) and variance = (v/2)(2 + 2x) - mean^2, each \
     moment beside its proven bound (gamma_n = nU/(1 - nU) per rounding, the remainders \
     twice, x's rounding through |y|'s 1-Lipschitz modulus); the total \
     is the residual bits plus B(x) (bits.rs) at the caller's precisions";

const U: f64 = f64::EPSILON / 2.0;

pub struct Objective {
    ear: Ear,
    clip: Clip,
    heard: Internal,
    n0: f64,
}

/// The noise atoms' line spans in `atoms`' order, the grid from `lowest` over `span` lines,
/// `kernels[c][k]` the band's `K(n/T)`, the transform's twiddles and each channel's `E_T`.
struct Lines {
    spans: Vec<(i64, i64)>,
    lowest: i64,
    span: usize,
    kernels: Vec<Vec<Kernel>>,
    twiddles: Vec<(f64, f64)>,
    thresholds: Vec<f64>,
}

/// A band's `K(n/T)` for `|n| < L`: `K(0)`, the sum and 2-norm of the rest, and the real part
/// of the `M`-point transform of the rest placed circularly, `K(0)` there set to 0, at
/// `f <= M/2`, beside a bound on its 2-norm distance over all `M` from the exact transform.
pub struct Kernel {
    zero: f64,
    sum: f64,
    norm: f64,
    half: Vec<f64>,
    bound: f64,
}

impl Kernel {
    /// `kernel[L - 1 + n]` is `K(n/T)`; `twiddles` are of `M >= 2L - 1` points.
    pub fn of(kernel: &[f64], twiddles: &[(f64, f64)]) -> Kernel {
        let m = 2 * twiddles.len();
        let span = kernel.len().div_ceil(2);
        assert!(m >= kernel.len(), "no lag wraps");
        let (mut re, mut im) = (vec![0.0; m], vec![0.0; m]);
        for (i, k) in kernel.iter().enumerate() {
            let n = i as i64 - (span as i64 - 1);
            if n != 0 {
                re[n.rem_euclid(m as i64) as usize] = *k;
            }
        }
        let rest = || kernel.iter().enumerate().filter(|(i, _)| *i != span - 1);
        let sum: f64 = rest().map(|(_, k)| k).sum();
        let norm = libm::sqrt(rest().map(|(_, k)| k * k).sum::<f64>());
        fft::fft_with(&mut re, &mut im, twiddles);
        re.truncate(m / 2 + 1);
        let reach = 1.0 + gamma(kernel.len() as f64 + 2.0);
        Kernel {
            zero: kernel[span - 1],
            sum: sum * reach,
            norm: norm * reach,
            half: re,
            bound: fft_bound(m) * libm::sqrt(m as f64) * norm * reach * (1.0 + 2.0 * U),
        }
    }

    /// `sum_(n != 0) K(n/T) r(n)` for `r` the autocorrelation whose transform `spectrum` holds
    /// (Parseval), beside the bound `MODEL` states on its distance from the exact sum: the
    /// spectrum's and this transform's, `(sqrt(M) |K|_2 spectrum.bound + bound |spectrum|_2)/M`,
    /// and the products' and the sum's rounding.
    pub fn apply(&self, spectrum: &Spectrum) -> (f64, f64) {
        let power = &spectrum.power;
        let last = power.len() - 1;
        let inner = 1..last;
        let (mut sum, mut size) = (0.0, 0.0);
        for (k, p) in self.half[inner.clone()].iter().zip(&power[inner]) {
            sum += k * p;
            size += k.abs() * p;
        }
        let ends = [0, last].map(|f| self.half[f] * power[f]);
        let m = 2.0 * last as f64;
        let sum = 2.0 * sum + ends[0] + ends[1];
        let size = 2.0 * size + ends[0].abs() + ends[1].abs();
        let moved = libm::sqrt(m) * self.norm * spectrum.bound + self.bound * spectrum.norm;
        let bound = (moved + gamma(m / 2.0 + 4.0) * size) / m * (1.0 + 4.0 * U);
        (sum / m, bound)
    }
}

/// `|X(f)|^2` at `f <= M/2`, the bound `MODEL` states on the 2-norm over all `M` of its
/// distance from the exact one, and a bound on its own 2-norm over all `M`.
pub struct Spectrum {
    pub power: Vec<f64>,
    pub bound: f64,
    pub norm: f64,
}

/// `v_i` within `bound` of its exact value (`MODEL`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Variance {
    pub v: f64,
    pub bound: f64,
}

/// `variance` is `v_i`, a magnitude's by Rice, each moment within its bound of the exact one
/// at `noise.v`; `noise` is `v_i` itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expected {
    pub mean: f64,
    pub variance: f64,
    pub mean_bound: f64,
    pub variance_bound: f64,
    pub noise: Variance,
}

pub type Elements = Vec<Vec<Vec<Expected>>>;

impl Objective {
    pub fn new(samples: &[f64], clip: Clip, spl_db: f64, scale: f64, n0: f64) -> Objective {
        let ear = Ear::of(f64::from(clip.rate), spl_db, scale);
        let heard = ear.read(&[samples], 0.0);
        Objective {
            ear,
            clip,
            heard,
            n0,
        }
    }

    pub fn ear(&self) -> &Ear {
        &self.ear
    }

    pub fn heard(&self) -> &Internal {
        &self.heard
    }

    /// Partials and harmonics summed in samples where `coherent`.
    pub fn expect(&self, atoms: &[Atom], coherent: bool) -> Elements {
        let model = Model::new(atoms, self.clip, coherent);
        let lines = self.lines(atoms);
        let (first, _) = self.ear.frames(self.clip.len);
        let mut variances: Vec<Option<Vec<Vec<Variance>>>> = Vec::new();
        let outputs = self.ear.elements(self.clip.len, &mut |m| {
            let power = self.ear.compose(m, &mut |frame| model.bins(frame).power);
            let excitation = self.ear.excite(m, &power);
            variances.push(self.variances(&model, &lines, m, &power, &excitation));
            excitation
        });
        outputs
            .into_iter()
            .enumerate()
            .map(|(c, bands)| {
                bands
                    .into_iter()
                    .enumerate()
                    .map(|(k, ys)| {
                        let band = self.heard.channels[c].bands[k].band;
                        ys.into_iter()
                            .enumerate()
                            .map(|(j, y)| {
                                let at = (self.ear.sampled(c, k, j) - first) as usize;
                                let noise = variances[at]
                                    .as_ref()
                                    .map_or(Variance::default(), |v| v[c][k]);
                                match band.element {
                                    Element::Magnitude => {
                                        let moments = rice(y.abs(), noise.v);
                                        let g = band.gain();
                                        Expected {
                                            mean: g * moments.mean,
                                            variance: g * g * moments.variance,
                                            mean_bound: g * moments.mean_bound
                                                + U * (g * moments.mean).abs(),
                                            variance_bound: g * g * moments.variance_bound
                                                + gamma(2.0) * (g * g * moments.variance).abs(),
                                            noise,
                                        }
                                    }
                                    _ => Expected {
                                        mean: band.element(y),
                                        variance: noise.v,
                                        mean_bound: 0.0,
                                        variance_bound: 0.0,
                                        noise,
                                    },
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn lines(&self, atoms: &[Atom]) -> Lines {
        let spans: Vec<(i64, i64)> = atoms
            .iter()
            .filter_map(|a| match a.sound {
                Sound::Noise { lo, hi } => Some(line_span(lo, hi, &self.clip).unwrap_or((1, 0))),
                _ => None,
            })
            .collect();
        let held = || spans.iter().filter(|(from, to)| from <= to);
        let lowest = held().map(|s| s.0).min().unwrap_or(1);
        let span = held()
            .map(|s| s.1)
            .max()
            .map_or(0, |top| (top - lowest + 1) as usize);
        let period = self.clip.secs();
        let twiddles = fft::twiddles((2 * span).next_power_of_two());
        let kernels = match span {
            0 => Vec::new(),
            _ => self
                .heard
                .channels
                .iter()
                .enumerate()
                .map(|(c, channel)| {
                    (0..channel.bands.len())
                        .map(|k| {
                            let kernel: Vec<f64> = (0..2 * span - 1)
                                .map(|i| {
                                    let n = i as f64 - (span - 1) as f64;
                                    self.ear.kernel(c, k, n / period)
                                })
                                .collect();
                            Kernel::of(&kernel, &twiddles)
                        })
                        .collect()
                })
                .collect(),
        };
        Lines {
            spans,
            lowest,
            span,
            kernels,
            twiddles,
            thresholds: self.ear.thresholds(),
        }
    }

    /// Frame `m`'s `v` per channel and band, as `MODEL` sums it over the lines, beside its
    /// bound; none without.
    fn variances(
        &self,
        model: &Model,
        lines: &Lines,
        m: i64,
        power: &[f64],
        excitation: &[f64],
    ) -> Option<Vec<Vec<Variance>>> {
        let Lines {
            spans,
            lowest,
            span,
            kernels,
            twiddles,
            thresholds,
        } = lines;
        let (lowest, span) = (*lowest, *span);
        if span == 0 {
            return None;
        }
        let owners: Vec<i64> = spans.iter().flat_map(|(from, to)| *from..=*to).collect();
        let atoms = spans.len();
        let mut parts: Vec<Vec<f64>> = vec![Vec::new(); owners.len() + atoms];
        self.ear.compose(m, &mut |frame| {
            let bins = frame.hz().len();
            let mut at = 0;
            for (a, (held, (from, to))) in model.lines(frame).iter().zip(spans).enumerate() {
                let count = (to - from + 1).max(0) as usize;
                for i in 0..count {
                    match held.power.is_empty() {
                        true => parts[at + i].extend(std::iter::repeat_n(0.0, bins)),
                        false => parts[at + i].extend_from_slice(&held.power[i]),
                    }
                }
                parts[owners.len() + a].extend(std::iter::repeat_n(held.bound, bins));
                at += count;
            }
            vec![(); bins]
        });
        let each = self.ear.excite_each(m, power, &parts);
        let (each, reach) = each.split_at(owners.len());
        let rows = self.ear.hz().len() as f64;
        let v = (0..excitation.len())
            .map(|c| {
                let mut lines = vec![0.0; span];
                let mut pairs = vec![0.0; span];
                for (k, e) in owners.iter().zip(each) {
                    let i = (k - lowest) as usize;
                    pairs[i] += 2.0 * lines[i] * e[c];
                    lines[i] += e[c];
                }
                let same: f64 = pairs.iter().sum();
                let total: f64 = lines.iter().sum();
                let top = lines.iter().fold(0.0, |a: f64, b| a.max(*b));
                let eps = reach.iter().map(|r| r[c]).sum::<f64>() * (1.0 + gamma(rows))
                    + gamma(rows + atoms as f64) * top;
                let moved = (2.0 * eps * total + span as f64 * eps * eps) * (1.0 + 4.0 * U);
                let paired = match atoms > 1 {
                    true => moved + gamma(2.0 * atoms as f64 + span as f64) * same,
                    false => 0.0,
                };
                let spectrum = power_spectrum(&lines, twiddles);
                let slope = 10.0 / LN_10 / (thresholds[c] + excitation[c]);
                kernels[c]
                    .iter()
                    .map(|kernel| {
                        let (rest, rest_bound) = kernel.apply(&spectrum);
                        let out = kernel.zero * same + rest;
                        let v = slope * slope * out.max(0.0);
                        let bound = slope
                            * slope
                            * (kernel.zero * paired
                                + kernel.sum * moved
                                + rest_bound
                                + U * out.abs())
                            + gamma(10.0) * v;
                        Variance {
                            v,
                            bound: bound * (1.0 + 8.0 * U),
                        }
                    })
                    .collect()
            })
            .collect();
        Some(v)
    }

    /// `sum_i [d_i^2 log2(e)/2 + log2(sigma_i/sigma_0,i)]` of `expected` against the clip.
    pub fn residual_bits(&self, expected: &Elements) -> f64 {
        let mut nats = 0.0;
        for (channel, bands) in self.heard.channels.iter().zip(expected) {
            for (band, elements) in channel.bands.iter().zip(bands) {
                let floor = self.n0 * band.rate_hz;
                for (clip, model) in band.elements.iter().zip(elements) {
                    let variance = floor + model.variance;
                    let d = model.mean - clip;
                    nats += 0.5 * d * d / variance + 0.5 * libm::log1p(model.variance / floor);
                }
            }
        }
        nats / LN_2
    }

    pub fn bits(&self, atoms: &[Atom], coherent: bool) -> f64 {
        self.residual_bits(&self.expect(atoms, coherent))
    }
}

/// `X` the transform of `x` zero-padded to `M = 2 twiddles.len()` points, through one
/// transform of `M/2` points of `x`'s even samples plus `i` its odd (`MODEL`).
pub fn power_spectrum(x: &[f64], twiddles: &[(f64, f64)]) -> Spectrum {
    let m = 2 * twiddles.len();
    let half = m / 2;
    assert!(m >= 2 * x.len().max(1) - 1, "no lag wraps");
    let (mut re, mut im) = (vec![0.0; half], vec![0.0; half]);
    for (i, v) in x.iter().enumerate() {
        match i % 2 {
            0 => re[i / 2] = *v,
            _ => im[i / 2] = *v,
        }
    }
    let inner: Vec<(f64, f64)> = twiddles.iter().step_by(2).take(half / 2).copied().collect();
    fft::fft_with(&mut re, &mut im, &inner);
    let power: Vec<f64> = (0..=half)
        .map(|j| {
            let (a, b) = (j % half, (half - j) % half);
            let even = ((re[a] + re[b]) / 2.0, (im[a] - im[b]) / 2.0);
            let odd = ((im[a] + im[b]) / 2.0, (re[b] - re[a]) / 2.0);
            let (c, s) = match j < half {
                true => twiddles[j],
                false => (-1.0, 0.0),
            };
            let x = (
                even.0 + c * odd.0 - s * odd.1,
                even.1 + c * odd.1 + s * odd.0,
            );
            x.0 * x.0 + x.1 * x.1
        })
        .collect();
    let phi = fft_bound(half);
    let phi = phi + 2.0 * SQRT_2 * (gamma(5.0) + 12.0 * U) * (1.0 + phi);
    let energy = x.iter().map(|v| v * v).sum::<f64>() * (1.0 + gamma(x.len() as f64));
    let bound = m as f64 * energy * (phi * (2.0 + phi) + gamma(2.0) * (1.0 + phi) * (1.0 + phi));
    let squares: f64 = power
        .iter()
        .enumerate()
        .map(|(f, p)| match f == 0 || f == half {
            true => p * p,
            false => 2.0 * p * p,
        })
        .sum();
    Spectrum {
        power,
        bound: bound * (1.0 + 4.0 * U),
        norm: libm::sqrt(squares) * (1.0 + gamma(half as f64 + 3.0)),
    }
}

pub fn total(residual_bits: f64, atoms: &[Atom], priced: &[Priced]) -> f64 {
    residual_bits + bits::bits(atoms, priced)
}

/// `a_k(nu)` of DLMF 10.17.1 for `nu` 0 and 1, `k` from 0.
fn hankel(nu: f64, k: usize) -> f64 {
    let mu = 4.0 * nu * nu;
    (1..=k).fold(1.0, |a, j| {
        let odd = (2 * j - 1) as f64;
        a * (mu - odd * odd) / (8.0 * j as f64)
    })
}

/// `chi(l) = sqrt(pi) Gamma(l/2 + 1)/Gamma(l/2 + 1/2)`, by `chi(l + 2) = chi(l) (l + 2)/(l + 1)`
/// from `chi(0) = 1` and `chi(1) = pi/2`.
fn chi(l: usize) -> f64 {
    let mut out = match l % 2 {
        0 => 1.0,
        _ => PI / 2.0,
    };
    let mut at = l % 2;
    while at < l {
        out *= (at + 2) as f64 / (at + 1) as f64;
        at += 2;
    }
    out
}

/// `D` by Hankel's expansion where its remainder's bound is under 2^-53 of it, beside a bound
/// on its distance from the exact `D` at `x`: the remainder twice over, each term's `2l + 3`
/// roundings (one ulp each for `pow`) and each partial sum's `l`, on the terms' moduli.
fn asymptotic(x: f64) -> Option<(f64, f64)> {
    let z = x / 2.0;
    let (mut d0, mut d1, mut size0, mut size1) = (0.0, 0.0, 0.0, 0.0);
    let mut last = f64::INFINITY;
    for l in 1.. {
        let term = |nu: f64| hankel(nu, l) / libm::pow(z, l as f64);
        let bound = |nu: f64| {
            2.0 * chi(l)
                * term(nu).abs()
                * sva_formula::exp((nu * nu - 0.25).abs() * PI / (2.0 * z))
        };
        let d = (1.0 + x) * d0 + x * d1;
        let reach = (1.0 + x) * bound(0.0) + x * bound(1.0);
        if reach <= U * d.abs() {
            let each = gamma((l * l + 2 * l + 3) as f64);
            let (e0, e1) = (each * size0, each * size1);
            let rounding = (1.0 + x) * (e0 + gamma(2.0) * d0.abs())
                + x * (e1 + gamma(1.0) * d1.abs())
                + U * d.abs();
            return Some((d, rounding + 2.0 * reach));
        }
        if reach >= last {
            return None;
        }
        last = reach;
        let sign = match l % 2 {
            0 => -1.0,
            _ => 1.0,
        };
        let (t0, t1) = (term(0.0), term(1.0));
        d0 += sign * t0;
        d1 += sign * t1;
        size0 += t0.abs();
        size1 += t1.abs();
    }
    None
}

/// `e^-z I_nu(z)` by its power series, every term positive, until the tail after a term,
/// within it times `r/(1 - r)` once the terms' ratio `r` falls under 1, is under 2^-53 of the
/// sum; beside its relative bound: each term's roundings, the sum's, the tail and `exp`'s.
fn scaled_bessel(nu: u32, z: f64) -> (f64, f64) {
    let quarter = z * z / 4.0;
    let mut term = libm::pow(z / 2.0, f64::from(nu)) / f64::from((1..=nu).product::<u32>());
    let mut sum = term;
    let mut terms = 0.0;
    for m in 0u32.. {
        let ratio = quarter / (f64::from(m + 1) * f64::from(m + 1 + nu));
        term *= ratio;
        sum += term;
        terms += 1.0;
        if ratio < 1.0 && term * ratio / (1.0 - ratio) <= U * sum {
            break;
        }
    }
    (sum * sva_formula::exp(-z), gamma(4.0 * terms + 10.0))
}

/// Each bound is on the distance from the exact moment at `(a, v)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    pub mean: f64,
    pub variance: f64,
    pub mean_bound: f64,
    pub variance_bound: f64,
}

/// The moments of `|y|`, `y` of mean modulus `a` plus a complex Gaussian of variance `v`, as
/// `MODEL` computes them. `x` rounded is `a` moved by at most `a gamma_2`, and `|y|` is
/// 1-Lipschitz in `a`, so the mean moves by at most that and the variance by twice it times
/// `sqrt(v)`, its covariance with a slope of modulus at most 1.
pub fn rice(a: f64, v: f64) -> Moments {
    if v == 0.0 {
        return Moments {
            mean: a,
            variance: 0.0,
            mean_bound: 0.0,
            variance_bound: 0.0,
        };
    }
    let x = a * a / v;
    let moved = a * gamma(2.0);
    if x > 0.0
        && let Some((d, d_bound)) = asymptotic(x)
    {
        let scale = v / (2.0 * a);
        let shift = scale * (1.0 - d);
        let shift_bound =
            scale * (d_bound + gamma(4.0) * (1.0 - d).abs()) + moved * shift.abs() / a;
        let mean = a + shift;
        let variance = v * d - shift * shift;
        return Moments {
            mean,
            variance,
            mean_bound: 2.0 * moved + shift_bound + U * mean.abs(),
            variance_bound: v * (d_bound + gamma(1.0) * d.abs())
                + (2.0 * shift.abs() + shift_bound) * shift_bound
                + gamma(2.0) * shift * shift
                + U * variance.abs()
                + 2.0 * libm::sqrt(v) * moved,
        };
    }
    let z = x / 2.0;
    let sigma = libm::sqrt(v / 2.0);
    let ((i0, e0), (i1, e1)) = (scaled_bessel(0, z), scaled_bessel(1, z));
    let g = libm::sqrt(PI / 2.0) * ((1.0 + x) * i0 + x * i1);
    let mean = sigma * g;
    let relative = e0.max(e1) + gamma(9.0);
    let whole = (v / 2.0) * (2.0 + 2.0 * x);
    let variance = whole - mean * mean;
    Moments {
        mean,
        variance,
        mean_bound: relative * mean + moved,
        variance_bound: gamma(3.0) * whole
            + (2.0 * relative + gamma(1.0)) * mean * mean
            + U * variance.abs()
            + 2.0 * libm::sqrt(v) * moved,
    }
}
