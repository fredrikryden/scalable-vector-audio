// Concern: what a sound puts in the clip's grid: its columns' templates, a kept source's own render, its links on it, the residual | Non-concern: writing it (written.rs) | IO: (Sound, gains) -> power

use std::borrow::Cow;
use std::sync::Arc;

use super::problem::Problem;
use super::shared::linked;
use super::written::Written;
use crate::analysis::floor_db;
use crate::optimize;
use crate::sources::{Link, Sound, Source};

/// A column's template over the feasible bins, one for every frame or one per frame.
#[derive(Clone)]
pub(super) struct Template(pub(super) Vec<Vec<f64>>);

impl Template {
    pub(super) fn at(&self, f: usize) -> &[f64] {
        match self.0.len() {
            1 => &self.0[0],
            _ => &self.0[f],
        }
    }
}

pub(super) struct Columns {
    pub(super) starts: Vec<f64>,
    pub(super) paths: Vec<Vec<f64>>,
    pub(super) heard: Vec<f64>,
    pub(super) floors: Vec<f64>,
}

pub(super) fn centre((lo, hi): (f64, f64)) -> f64 {
    libm::sqrt(lo * hi)
}

pub(super) fn db(power: f64) -> f64 {
    10.0 * libm::log10(power)
}

pub(super) fn power(db: f64) -> f64 {
    libm::pow(10.0, db / 10.0)
}

/// Each column's Hz at the clip's start: a noise band's its geometric centre, a shaped noise's
/// its first band's.
pub(super) fn hzs(sound: &Sound) -> Vec<f64> {
    match sound {
        Sound::Partial { hz, .. } => vec![*hz],
        Sound::Harmonic {
            hz, stretch, count, ..
        } => (1..=*count)
            .map(|k| {
                let k = k as f64;
                hz * k * libm::sqrt(1.0 + stretch * k * k)
            })
            .collect(),
        Sound::Noise { lo, hi } => vec![centre((*lo, *hi))],
        Sound::Shaped { bands } => vec![centre((bands[0].0, bands[0].1))],
    }
}

/// This partial or voice gliding at `rate`, its Hz at the start moved so its path holds at
/// `middle`.
pub(super) fn glided(sound: &Sound, rate: f64, middle: f64) -> Sound {
    let moved = |hz: f64| hz * sva_formula::exp((sound.glide() - rate) * middle);
    match sound {
        Sound::Partial { hz, .. } => Sound::Partial {
            hz: moved(*hz),
            glide: rate,
        },
        Sound::Harmonic {
            hz, stretch, count, ..
        } => Sound::Harmonic {
            hz: moved(*hz),
            stretch: *stretch,
            count: *count,
            glide: rate,
        },
        Sound::Noise { .. } | Sound::Shaped { .. } => sound.clone(),
    }
}

impl Problem {
    pub(super) fn templates(&self, sound: &Sound) -> Vec<Template> {
        match sound {
            Sound::Noise { lo, hi } => vec![Template(vec![
                self.grid.noise_bins((*lo, *hi))[self.range.clone()].to_vec(),
            ])],
            Sound::Shaped { bands } => {
                let mut sum = vec![0.0; self.range.len()];
                for (lo, hi, gain) in bands {
                    let (bins, gain) = (self.grid.noise_bins((*lo, *hi)), power(*gain));
                    for (s, b) in sum.iter_mut().zip(&bins[self.range.clone()]) {
                        *s += gain * b;
                    }
                }
                vec![Template(vec![sum])]
            }
            Sound::Partial { .. } | Sound::Harmonic { .. } => {
                let rate = sound.glide();
                let at = |hz: f64| self.range.clone().map(|b| self.grid.sine(hz, b)).collect();
                let bent = match rate != 0.0 {
                    true => self.bent(rate),
                    false => Vec::new(),
                };
                hzs(sound)
                    .iter()
                    .map(|hz| {
                        Template(match rate != 0.0 {
                            true => self
                                .times
                                .iter()
                                .map(|t| self.swept(hz * sva_formula::exp(rate * t), &bent))
                                .collect(),
                            false => vec![at(*hz)],
                        })
                    })
                    .collect()
            }
        }
    }

    /// Radians per Hz at the window's centre a path gliding at `rate` turns by each sample,
    /// `2 pi (e^(rate s) - 1) / rate`, `s` its time from the centre.
    fn bent(&self, rate: f64) -> Vec<f64> {
        let long = self.grid.long;
        (0..long)
            .map(|k| {
                let from = (k as f64 - (long / 2) as f64) / self.grid.rate;
                std::f64::consts::TAU * libm::expm1(rate * from) / rate
            })
            .collect()
    }

    /// The power a unit sine at `hz` at a window's centre, turning `bent` per Hz there, puts
    /// in each feasible bin of that window: its periodogram, positive frequencies alone.
    fn swept(&self, hz: f64, bent: &[f64]) -> Vec<f64> {
        let n = self.grid.n;
        let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
        for (k, (w, turns)) in self.window.iter().zip(bent).enumerate() {
            let (sin, cos) = libm::sincos(hz * turns);
            (re[k], im[k]) = (0.5 * w * cos, 0.5 * w * sin);
        }
        sva_engine::fft::fft_with(&mut re, &mut im, &self.twiddle);
        self.range
            .clone()
            .map(|b| re[b] * re[b] + im[b] * im[b])
            .collect()
    }

    pub(super) fn columns(&self, sound: &Sound) -> Columns {
        let rate = sound.glide();
        let hzs = hzs(sound);
        let heard: Vec<f64> = hzs
            .iter()
            .map(|hz| hz * sva_formula::exp(rate * self.middle()))
            .collect();
        let floors = match sound {
            Sound::Shaped { bands } => vec![
                bands
                    .iter()
                    .map(|(lo, hi, g)| floor_db(centre((*lo, *hi)), self.spl_db) - g)
                    .fold(f64::INFINITY, f64::min),
            ],
            _ => heard.iter().map(|hz| floor_db(*hz, self.spl_db)).collect(),
        };
        Columns {
            starts: hzs.clone(),
            paths: hzs
                .iter()
                .map(|hz| {
                    self.times
                        .iter()
                        .map(|t| hz * sva_formula::exp(rate * t))
                        .collect()
                })
                .collect(),
            heard,
            floors,
        }
    }

    pub(super) fn moments(&self, f: usize, t: &[f64], r: &[f64]) -> (f64, f64) {
        self.weight[f]
            .iter()
            .zip(t)
            .zip(r)
            .fold((0.0, 0.0), |(e, c), ((w, t), r)| {
                let wt = w * t;
                (e + wt * t, c + wt * r)
            })
    }

    /// `templates` summed per frame, column `c` at `gain(c, f)` in frame `f`, a nought gain
    /// left out.
    pub(super) fn summed(
        &self,
        templates: &[Template],
        gain: &dyn Fn(usize, usize) -> f64,
    ) -> Vec<Vec<f64>> {
        (0..self.frames())
            .map(|f| {
                let mut u = vec![0.0; self.range.len()];
                for (c, t) in templates.iter().enumerate() {
                    let g = gain(c, f);
                    if g != 0.0 {
                        u.iter_mut().zip(t.at(f)).for_each(|(u, t)| *u += g * t);
                    }
                }
                u
            })
            .collect()
    }

    pub(super) fn residual(&self, kept: &[Held], shared: &[optimize::Shared]) -> Vec<Vec<f64>> {
        let mut out = self.power.clone();
        for h in kept {
            for (r, u) in out.iter_mut().zip(h.render(self, shared).iter()) {
                r.iter_mut().zip(u).for_each(|(r, u)| *r -= u);
            }
        }
        out
    }

    pub(super) fn model(&self, kept: &[Held], shared: &[optimize::Shared]) -> Vec<Vec<f64>> {
        self.residual(kept, shared)
            .iter()
            .zip(&self.power)
            .map(|(r, y)| y.iter().zip(r).map(|(y, r)| y - r).collect())
            .collect()
    }

    pub(super) fn full(&self, model: &[Vec<f64>]) -> Vec<Vec<f64>> {
        model
            .iter()
            .map(|m| {
                let mut out = vec![0.0; self.grid.bins()];
                out[self.range.clone()].copy_from_slice(m);
                out
            })
            .collect()
    }
}

/// The residual's weighted square's change, `sum w (u^2 - 2 r u)`, `u` its change `delta`.
pub(super) fn change(p: &Problem, residual: &[Vec<f64>], delta: &[Vec<f64>]) -> f64 {
    (0..p.frames())
        .map(|f| {
            delta[f]
                .iter()
                .zip(&residual[f])
                .zip(&p.weight[f])
                .map(|((u, r), w)| w * (u * u - 2.0 * r * u))
                .sum::<f64>()
        })
        .sum()
}

/// A kept source: each column's template, the source as written, and its own render, its
/// expected power per frame before its links, computed once.
#[derive(Clone)]
pub(super) struct Held {
    pub(super) templates: Vec<Template>,
    pub(super) written: Written,
    own: Arc<Vec<Vec<f64>>>,
}

impl Held {
    pub(super) fn new(p: &Problem, templates: Vec<Template>, written: Written) -> Held {
        let own = p.summed(&templates, &|c, f| written.gains[c][f]);
        Held {
            templates,
            written,
            own: Arc::new(own),
        }
    }

    /// Itself rewritten as `written`, which holds its gains and references other links.
    pub(super) fn relinked(&self, written: Written) -> Held {
        Held {
            templates: self.templates.clone(),
            written,
            own: self.own.clone(),
        }
    }

    pub(super) fn sound(&self) -> &Sound {
        &self.written.source.sound
    }

    pub(super) fn links(&self) -> &[Link] {
        &self.written.source.links
    }

    pub(super) fn key(&self) -> u64 {
        self.written.source.key
    }

    pub(super) fn render(&self, p: &Problem, shared: &[optimize::Shared]) -> Cow<'_, [Vec<f64>]> {
        match self.links().is_empty() {
            true => Cow::Borrowed(&self.own[..]),
            false => {
                let mut out = self.own.to_vec();
                linked(p, self.links(), shared, &mut out);
                Cow::Owned(out)
            }
        }
    }
}

pub(super) fn sources(kept: &[Held]) -> Vec<Source> {
    kept.iter().map(|h| h.written.source.clone()).collect()
}
