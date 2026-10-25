// Concern: OBJECTIVE: a state's expected elements, variances N0 rho + v, Rice moments, residual bits against the clip, the total with B(x) | Non-concern: tangents, search | IO: (clip, atoms, N0) -> bits

use std::f64::consts::{LN_2, LN_10, PI};

use sva_engine::internal::modulation::Element;
use sva_engine::internal::{Ear, Internal};

use super::atoms::{Atom, Clip, Sound};
use super::bits::{self, Priced};
use super::spectrum::{Model, line_span};

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
     do not covary), a line's own term (n = 0) never counted, so v >= 0, K = |H|^2 from E to \
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
/// `kernels[c][k][span - 1 + n]` the band's `K(n/T)`, and each channel's `E_T`.
struct Lines {
    spans: Vec<(i64, i64)>,
    lowest: i64,
    span: usize,
    kernels: Vec<Vec<Vec<f64>>>,
    thresholds: Vec<f64>,
}

/// `variance` is `v_i`, a magnitude's by Rice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Expected {
    pub mean: f64,
    pub variance: f64,
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
        let mut variances: Vec<Option<Vec<Vec<f64>>>> = Vec::new();
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
                                let v = variances[at].as_ref().map_or(0.0, |v| v[c][k]);
                                match band.element {
                                    Element::Magnitude => {
                                        let Moments { mean, variance, .. } = rice(y.abs(), v);
                                        let g = band.gain();
                                        Expected {
                                            mean: g * mean,
                                            variance: g * g * variance,
                                        }
                                    }
                                    _ => Expected {
                                        mean: band.element(y),
                                        variance: v,
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
                            (0..2 * span - 1)
                                .map(|i| {
                                    let n = i as f64 - (span - 1) as f64;
                                    self.ear.kernel(c, k, n / period)
                                })
                                .collect()
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
            thresholds: self.ear.thresholds(),
        }
    }

    /// Frame `m`'s `v` per channel and band, as `MODEL` sums it over the lines; none without.
    fn variances(
        &self,
        model: &Model,
        lines: &Lines,
        m: i64,
        power: &[f64],
        excitation: &[f64],
    ) -> Option<Vec<Vec<f64>>> {
        let Lines {
            spans,
            lowest,
            span,
            kernels,
            thresholds,
        } = lines;
        let (lowest, span) = (*lowest, *span);
        let owners: Vec<i64> = spans.iter().flat_map(|(from, to)| *from..=*to).collect();
        if span == 0 {
            return None;
        }
        let per_bin = self.ear.compose(m, &mut |frame| {
            let bins = frame.hz().len();
            let mut each = vec![Vec::with_capacity(owners.len()); bins];
            for (lines, (from, to)) in model.lines(frame).iter().zip(spans.iter()) {
                let count = (to - from + 1).max(0) as usize;
                for i in 0..count {
                    for (b, bin) in each.iter_mut().enumerate() {
                        bin.push(match lines.power.is_empty() {
                            true => 0.0,
                            false => lines.power[i][b],
                        });
                    }
                }
            }
            each
        });
        let parts: Vec<Vec<f64>> = (0..owners.len())
            .map(|i| per_bin.iter().map(|b| b[i]).collect())
            .collect();
        let each = self.ear.excite_each(m, power, &parts);
        let v = (0..excitation.len())
            .map(|c| {
                let mut lines = vec![0.0; span];
                let mut pairs = vec![0.0; span];
                for (k, e) in owners.iter().zip(&each) {
                    let i = (k - lowest) as usize;
                    pairs[i] += 2.0 * lines[i] * e[c];
                    lines[i] += e[c];
                }
                let same: f64 = pairs.iter().sum();
                let acf: Vec<f64> = (0..span)
                    .map(|n| (0..span - n).map(|k| lines[k] * lines[k + n]).sum())
                    .collect();
                let slope = 10.0 / LN_10 / (thresholds[c] + excitation[c]);
                kernels[c]
                    .iter()
                    .map(|kernel| {
                        let mut sum = kernel[span - 1] * same;
                        for (n, a) in acf.iter().enumerate().skip(1) {
                            sum += a * (kernel[span - 1 + n] + kernel[span - 1 - n]);
                        }
                        slope * slope * sum
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

fn gamma(n: f64) -> f64 {
    n * U / (1.0 - n * U)
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
