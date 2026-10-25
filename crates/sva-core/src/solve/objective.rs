// Concern: OBJECTIVE: a state's expected elements, residual bits against the clip at variance N0 rho, the total with B(x) | Non-concern: tangents, search | IO: (clip, atoms, N0) -> bits

use std::f64::consts::LN_2;

use sva_engine::internal::{Ear, Internal};
use sva_formula::C64;

use super::atoms::{Atom, Clip};
use super::bits::{self, Priced};
use super::spectrum::Model;

pub const MODEL: &str = "residual bits = sum_i d_i^2 log2(e)/2, the Gaussian code length of \
     the clip's elements given the model's over the noise-free code, channels summed, place \
     only (no temporal cue), d_i = (e_model,i - e_clip,i)/sigma_i, sigma_i^2 = N0 rho_kc, equal \
     noise density in every band (calibrate.rs); the model's elements are the reading of each \
     frame's expected excitation, partials and harmonics as written and noise in expectation \
     over its lines' phases (spectrum.rs), so a noise is compared by its expected excitation \
     alone; the total is the residual bits plus B(x) (bits.rs) at the caller's precisions";

pub struct Objective {
    ear: Ear,
    clip: Clip,
    heard: Internal,
    n0: f64,
}

/// Per channel, band and element, the model's expected element.
pub type Elements = Vec<Vec<Vec<f64>>>;

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

    pub fn clip(&self) -> Clip {
        self.clip
    }

    pub fn n0(&self) -> f64 {
        self.n0
    }

    /// Partials and harmonics summed in samples where `coherent`.
    pub fn expect(&self, atoms: &[Atom], coherent: bool) -> Elements {
        let model = Model::new(atoms, self.clip, coherent);
        let outputs = self.ear.elements(self.clip.len, &mut |m| {
            let power = self.ear.compose(m, &mut |frame| model.bins(frame).power);
            self.ear.excite(m, &power)
        });
        self.elements(&outputs)
    }

    /// Each band's elements of `outputs`, its outputs before `Band::element`.
    pub fn elements(&self, outputs: &[Vec<Vec<C64>>]) -> Elements {
        outputs
            .iter()
            .zip(&self.heard.channels)
            .map(|(bands, channel)| {
                bands
                    .iter()
                    .zip(&channel.bands)
                    .map(|(ys, band)| ys.iter().map(|y| band.band.element(*y)).collect())
                    .collect()
            })
            .collect()
    }

    /// `sum_i d_i^2 log2(e)/2` of `expected` against the clip.
    pub fn residual_bits(&self, expected: &Elements) -> f64 {
        let mut nats = 0.0;
        for (channel, bands) in self.heard.channels.iter().zip(expected) {
            for (band, elements) in channel.bands.iter().zip(bands) {
                let variance = self.n0 * band.rate_hz;
                for (clip, model) in band.elements.iter().zip(elements) {
                    let d = model - clip;
                    nats += 0.5 * d * d / variance;
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
