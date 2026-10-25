// Concern: Jepsen et al.'s (2008) modulation stage at one hop: the 150 Hz lowpass, the bank, each band's element rate | Non-concern: the series it reads (mod.rs) | IO: (V, hop) -> elements

use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};

use sva_formula::C64;

use super::rate::Rate;

pub(super) const PRE_LOWPASS_HZ: f64 = 150.0;
pub(super) const LOWPASS_HZ: f64 = 2.5;
pub(super) const FIRST_CENTRE_HZ: f64 = 5.0;
pub(super) const CONSTANT_BANDWIDTH_HZ: f64 = 5.0;
pub(super) const CONSTANT_UNTIL_HZ: f64 = 10.0;
pub(super) const Q: f64 = 2.0;
pub(super) const TOP_SHARE: f64 = 0.25;
pub(super) const TOP_HZ: f64 = 1000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Lowpass,
    Real,
    Magnitude,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    pub hz: f64,
    pub bandwidth_hz: f64,
    pub element: Element,
}

impl Band {
    fn at(hz: f64) -> Band {
        match hz <= CONSTANT_UNTIL_HZ {
            true => Band {
                hz,
                bandwidth_hz: CONSTANT_BANDWIDTH_HZ,
                element: Element::Real,
            },
            false => Band {
                hz,
                bandwidth_hz: hz / Q,
                element: Element::Magnitude,
            },
        }
    }

    pub(super) fn upper_edge_hz(&self) -> f64 {
        match self.element {
            Element::Lowpass => LOWPASS_HZ,
            _ => self.hz + self.bandwidth_hz / 2.0,
        }
    }

    pub(super) fn gain(&self) -> f64 {
        match self.element {
            Element::Magnitude => FRAC_1_SQRT_2,
            _ => 1.0,
        }
    }

    pub(super) fn power(&self, nu: f64) -> f64 {
        match self.element {
            Element::Lowpass => {
                let x = nu / LOWPASS_HZ;
                1.0 / (1.0 + x * x * x * x)
            }
            _ => {
                let x = (nu - self.hz) / (self.bandwidth_hz / 2.0);
                1.0 / (1.0 + x * x)
            }
        }
    }

    /// The element its output `y` maps to: the real part, or `gain |y|` above 10 Hz.
    pub fn element(&self, y: C64) -> f64 {
        match self.element {
            Element::Magnitude => self.gain() * y.abs(),
            _ => y.re,
        }
    }

    /// `element`'s slope along `dy` at `y`: `gain Re(conj(y) dy)/|y|` (0 at `y = 0`), or `Re dy`.
    pub fn element_slope(&self, y: C64, dy: C64) -> f64 {
        match self.element {
            Element::Magnitude => match y.abs() > 0.0 {
                true => self.gain() * (y.re * dy.re + y.im * dy.im) / y.abs(),
                false => 0.0,
            },
            _ => dy.re,
        }
    }
}

/// Each resonator's lower -3 dB edge is the last one's upper edge.
pub(super) fn bank(centre_hz: f64, hop_secs: f64) -> Vec<Band> {
    let nyquist = 0.5 / hop_secs;
    let top = (TOP_SHARE * centre_hz).min(TOP_HZ);
    let lowpass = Band {
        hz: 0.0,
        bandwidth_hz: 0.0,
        element: Element::Lowpass,
    };
    let mut out: Vec<Band> = [lowpass]
        .into_iter()
        .filter(|b| b.upper_edge_hz() < nyquist)
        .collect();
    let mut band = Band::at(FIRST_CENTRE_HZ);
    while band.hz <= top && band.upper_edge_hz() < nyquist {
        out.push(band);
        let edge = band.upper_edge_hz();
        band = match edge + CONSTANT_BANDWIDTH_HZ / 2.0 <= CONSTANT_UNTIL_HZ {
            true => Band::at(edge + CONSTANT_BANDWIDTH_HZ / 2.0),
            false => Band::at(edge / (1.0 - 0.5 / Q)),
        };
    }
    out
}

fn one_pole(hz: f64, hop_secs: f64) -> f64 {
    sva_formula::exp(-TAU * hz * hop_secs)
}

/// Per band, its output `y` at each frame `round(j / per_frame)` of `v[from..]`, before
/// `Band::element` maps it.
pub(super) fn outputs(
    v: &[f64],
    from: usize,
    hop_secs: f64,
    bands: &[(Band, Rate)],
) -> Vec<Vec<C64>> {
    let a = one_pole(PRE_LOWPASS_HZ, hop_secs);
    let mut y = 0.0;
    let pre: Vec<f64> = v
        .iter()
        .map(|x| {
            y = a * y + (1.0 - a) * x;
            y
        })
        .collect();
    let frames = v.len() - from;
    bands
        .iter()
        .map(|(band, rate)| {
            let per_frame = rate.per_frame;
            let out = filtered(&pre, band, hop_secs);
            let count = libm::floor((frames - 1) as f64 * per_frame) as usize + 1;
            (0..count)
                .map(|j| out[from + (j as f64 / per_frame).round_ties_even() as usize])
                .collect()
        })
        .collect()
}

fn filtered(x: &[f64], band: &Band, hop_secs: f64) -> Vec<C64> {
    match band.element {
        Element::Lowpass => {
            let w = TAU * LOWPASS_HZ * hop_secs * FRAC_1_SQRT_2;
            let r = sva_formula::exp(-w);
            let (c1, c2) = (2.0 * r * libm::cos(w), -r * r);
            let g = 1.0 - c1 - c2;
            let (mut y1, mut y2) = (0.0, 0.0);
            x.iter()
                .map(|v| {
                    let y = c1 * y1 + c2 * y2 + g * v;
                    y2 = y1;
                    y1 = y;
                    C64::new(y, 0.0)
                })
                .collect()
        }
        _ => {
            let r = sva_formula::exp(-PI * band.bandwidth_hz * hop_secs);
            let (s, c) = libm::sincos(TAU * band.hz * hop_secs);
            let pole = C64::new(r * c, r * s);
            let mut y = C64::new(0.0, 0.0);
            x.iter()
                .map(|v| {
                    y = pole * y + C64::new((1.0 - r) * v, 0.0);
                    y
                })
                .collect()
        }
    }
}
