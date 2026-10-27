// Concern: OBJECTIVE: a state's expected elements, residual bits against the clip at variance N0 rho, the total with B(x) | Non-concern: tangents, search | IO: (clip, atoms, N0) -> bits

use std::f64::consts::LN_2;

use sva_engine::internal::{Ear, Internal};
use sva_engine::{default_threads, each};
use sva_formula::C64;

use super::atoms::{Atom, Clip, Sound};
use super::bits::{self, Priced};
use super::coarse::{Coarse, Read};
use super::spectrum::{Model, Summed};

pub const MODEL: &str = "residual bits = sum_i d_i^2 log2(e)/2, the Gaussian code length of the \
     clip's elements given the model's over the noise-free code, channels summed, place only (no \
     temporal cue), d_i = (e_model,i - e_clip,i)/sigma_i, sigma_i^2 = N0 rho_kc, equal noise \
     density in every band (calibrate.rs); the model's elements are the reading of each frame's \
     expected excitation, partials and harmonics as written and noise in expectation over its \
     lines' phases (spectrum.rs), so a noise is compared by its expected excitation alone; the \
     total is the residual bits plus B(x) (bits.rs) at the caller's precisions; an evaluation \
     sums its partials in samples over the clip once (spectrum.rs), and reads a noise whose \
     numbers the last evaluation read as it read them, the same sum; atoms held while others move \
     are read once: their partials so summed, their noise's expected power per frame, and their \
     excitation alone per frame, which a frame no moving atom reaches takes as it is (an atom \
     reaches a frame when any window of the frame weighs a sample whose cell meets [on, end)); at \
     a coarse objective every frame's excitation is the coarse model's (coarse.rs), the held \
     atoms read beside the moving, and the clip's elements stay its samples' reading";

pub struct Objective {
    ear: Ear,
    clip: Clip,
    heard: Internal,
    n0: f64,
    coarse: Option<Coarse>,
}

/// Per channel, band and element, the model's expected element.
pub type Elements = Vec<Vec<Vec<f64>>>;

impl Objective {
    pub fn new(samples: &[f64], clip: Clip, spl_db: f64, scale: f64, n0: f64) -> Objective {
        let ear = Ear::of(f64::from(clip.rate), spl_db, scale).remembering();
        let heard = ear.read(&[samples], 0.0);
        Objective {
            ear,
            clip,
            heard,
            n0,
            coarse: None,
        }
    }

    /// The same objective reading its atoms by the coarse model (coarse.rs).
    pub fn coarse(mut self) -> Objective {
        self.coarse = Some(Coarse::of(&self.ear, &self.clip));
        self
    }

    pub fn is_coarse(&self) -> bool {
        self.coarse.is_some()
    }

    /// The coarse model's excitation of `atoms` at each frame of `frames`, where coarse.
    fn read_coarse(&self, atoms: &[Atom], frames: (i64, i64)) -> Option<Vec<Vec<f64>>> {
        let coarse = self.coarse.as_ref()?;
        let reach: Vec<(i64, i64)> = atoms
            .iter()
            .map(|a| reach(&self.ear, &self.clip, a))
            .collect();
        Some(Read::of(coarse, atoms, &reach).excitations(frames))
    }

    /// The coarse model's state of `atoms`, each atom's reach beside it, where coarse.
    pub fn coarse_read<'a>(
        &'a self,
        atoms: &'a [Atom],
        reach: &'a [(i64, i64)],
    ) -> Option<Read<'a>> {
        Some(Read::of(self.coarse.as_ref()?, atoms, reach))
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

    /// The elements one evaluation compares.
    pub fn count(&self) -> u64 {
        self.heard
            .channels
            .iter()
            .flat_map(|c| &c.bands)
            .map(|b| b.elements.len() as u64)
            .sum()
    }

    /// Partials and harmonics summed in samples where `coherent`.
    pub fn expect(&self, atoms: &[Atom], coherent: bool) -> Elements {
        self.expect_beside(atoms, coherent, None)
    }

    /// `expect` beside `held`.
    pub fn expect_beside(&self, atoms: &[Atom], coherent: bool, held: Option<&Held>) -> Elements {
        let frames = self.ear.frames(self.clip.len);
        if self.is_coarse() {
            let all: Vec<Atom> = atoms
                .iter()
                .chain(held.iter().flat_map(|h| &h.atoms))
                .cloned()
                .collect();
            let each = self.read_coarse(&all, frames).expect("a coarse objective");
            return self.elements_of(&each);
        }
        let (first, _) = frames;
        let reach: Vec<(i64, i64)> = atoms
            .iter()
            .map(|a| reach(&self.ear, &self.clip, a))
            .collect();
        let summed = coherent.then(|| summed_beside(atoms, self.clip, held));
        let each = framed_beside(
            atoms,
            self.clip,
            coherent,
            self.ear.frames(self.clip.len),
            (summed.as_ref().or(held.map(|h| &h.summed)), coherent),
            &|model, m| match held {
                Some(h) if !reach.iter().any(|(f, l)| *f <= m && m <= *l) => {
                    h.excitation[(m - first) as usize].clone()
                }
                _ => {
                    let mut power = self.ear.compose(m, &mut |frame| model.bins(frame).power);
                    if let Some(h) = held {
                        h.add_noise(m - first, &mut power);
                    }
                    excited(&self.ear, m, &power)
                }
            },
        );
        self.elements_of(&each)
    }

    /// The elements of `each` frame's excitation from the first.
    fn elements_of(&self, each: &[Vec<f64>]) -> Elements {
        let (first, _) = self.ear.frames(self.clip.len);
        let outputs = self
            .ear
            .elements(self.clip.len, &mut |m| each[(m - first) as usize].clone());
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

    /// Each frame's excitation of `atoms` from frame `from` to `to`, as `expect` reads it.
    pub fn excitations(
        &self,
        atoms: &[Atom],
        coherent: bool,
        (from, to): (i64, i64),
    ) -> Vec<Vec<f64>> {
        if let Some(each) = self.read_coarse(atoms, (from, to)) {
            return each;
        }
        let summed = coherent.then(|| Summed::of(atoms, self.clip));
        framed_beside(
            atoms,
            self.clip,
            coherent,
            (from, to),
            (summed.as_ref(), coherent),
            &|model, m| {
                let power = self.ear.compose(m, &mut |frame| model.bins(frame).power);
                excited(&self.ear, m, &power)
            },
        )
    }

    /// The noise `atom`'s composed expected power at each frame it reaches.
    pub fn noise_power(&self, atom: &Atom) -> NoisePower {
        let (first, _) = self.ear.frames(self.clip.len);
        let (from, to) = reach(&self.ear, &self.clip, atom);
        let power = framed(
            std::slice::from_ref(atom),
            self.clip,
            true,
            (from, to),
            &|model, m| self.ear.compose(m, &mut |frame| model.bins(frame).power),
        );
        NoisePower {
            at: from - first,
            power,
        }
    }

    /// `expect_beside`, `atoms`' noises' power given in their order.
    pub fn expect_read(
        &self,
        atoms: &[Atom],
        noise: &[&NoisePower],
        held: Option<&Held>,
    ) -> Elements {
        if self.is_coarse() {
            return self.expect_beside(atoms, true, held);
        }
        let tonal: Vec<Atom> = atoms
            .iter()
            .filter(|a| !matches!(a.sound, Sound::Noise { .. }))
            .cloned()
            .collect();
        let reach: Vec<(i64, i64)> = atoms
            .iter()
            .map(|a| reach(&self.ear, &self.clip, a))
            .collect();
        let frames = self.ear.frames(self.clip.len);
        let each = self.read_beside(&tonal, noise, held, (frames, &reach));
        let outputs = self.ear.elements(self.clip.len, &mut |m| {
            each[(m - frames.0) as usize].clone()
        });
        self.elements(&outputs)
    }

    /// Each frame's excitation of `tonal` summed in samples, `noise` and the held added; a frame
    /// no `reach` holds takes the held's.
    fn read_beside(
        &self,
        tonal: &[Atom],
        noise: &[&NoisePower],
        held: Option<&Held>,
        ((from, to), reach): ((i64, i64), &[(i64, i64)]),
    ) -> Vec<Vec<f64>> {
        let (first, _) = self.ear.frames(self.clip.len);
        let summed = summed_beside(tonal, self.clip, held);
        let none: [Atom; 0] = [];
        framed_beside(
            &none,
            self.clip,
            true,
            (from, to),
            (Some(&summed), true),
            &|model, m| {
                if let Some(h) = held.filter(|_| !reach.iter().any(|(f, l)| *f <= m && m <= *l)) {
                    return h.excitation[(m - first) as usize].clone();
                }
                let mut power = self.ear.compose(m, &mut |frame| model.bins(frame).power);
                for n in noise {
                    n.add(m - first, &mut power);
                }
                if let Some(h) = held {
                    h.add_noise(m - first, &mut power);
                }
                excited(&self.ear, m, &power)
            },
        )
    }

    /// `excitations` of `tonal` beside `noise`.
    pub fn excitations_beside(
        &self,
        tonal: &[Atom],
        noise: &[&NoisePower],
        frames: (i64, i64),
    ) -> Vec<Vec<f64>> {
        self.read_beside(tonal, noise, None, (frames, &[]))
    }

    /// The residual bits of the elements read off `excitation(m)` at each frame `m`.
    pub fn bits_of(&self, excitation: &mut dyn FnMut(i64) -> Vec<f64>) -> f64 {
        let outputs = self.ear.elements(self.clip.len, excitation);
        self.residual_bits(&self.elements(&outputs))
    }

    pub fn bits(&self, atoms: &[Atom], coherent: bool) -> f64 {
        self.residual_bits(&self.expect(atoms, coherent))
    }
}

/// A noise's composed expected power per frame it reaches, from frame index `at`.
pub struct NoisePower {
    at: i64,
    power: Vec<Vec<f64>>,
}

impl NoisePower {
    /// Adds the `at`-th frame's power.
    pub fn add(&self, at: i64, power: &mut [f64]) {
        let Some(own) = usize::try_from(at - self.at)
            .ok()
            .and_then(|i| self.power.get(i))
        else {
            return;
        };
        for (p, n) in power.iter_mut().zip(own) {
            *p += n;
        }
    }
}

/// Atoms held while others move, read once as `MODEL` states; at a coarse objective only
/// `atoms`, read beside the moving.
pub struct Held {
    pub atoms: Vec<Atom>,
    pub summed: Summed,
    noise: Vec<Option<Vec<f64>>>,
    pub excitation: Vec<Vec<f64>>,
}

impl Held {
    pub fn of(objective: &Objective, atoms: &[Atom]) -> Held {
        let (ear, clip) = (objective.ear(), objective.clip());
        if objective.is_coarse() {
            return Held {
                atoms: atoms.to_vec(),
                summed: Summed::of(&[], clip),
                noise: Vec::new(),
                excitation: Vec::new(),
            };
        }
        let summed = Summed::of(atoms, clip);
        let noises: Vec<Atom> = atoms
            .iter()
            .filter(|a| matches!(a.sound, Sound::Noise { .. }))
            .cloned()
            .collect();
        let frames = ear.frames(clip.len);
        let noise: Vec<Option<Vec<f64>>> = framed(&noises, clip, true, frames, &|model, m| {
            let power = ear.compose(m, &mut |frame| model.bins(frame).power);
            power.iter().any(|p| *p != 0.0).then_some(power)
        });
        let none: [Atom; 0] = [];
        let excitation = framed_beside(
            &none,
            clip,
            true,
            frames,
            (Some(&summed), false),
            &|tonal, m| {
                let mut power = ear.compose(m, &mut |frame| tonal.bins(frame).power);
                if let Some(n) = &noise[(m - frames.0) as usize] {
                    for (p, n) in power.iter_mut().zip(n) {
                        *p += n;
                    }
                }
                excited(ear, m, &power)
            },
        );
        Held {
            atoms: Vec::new(),
            summed,
            noise,
            excitation,
        }
    }

    /// Adds the `at`-th frame's held noise power.
    pub fn add_noise(&self, at: i64, power: &mut [f64]) {
        if let Some(noise) = &self.noise[at as usize] {
            for (p, n) in power.iter_mut().zip(noise) {
                *p += n;
            }
        }
    }
}

/// `atoms`' partials and harmonics summed in samples beside the held's.
pub fn summed_beside(atoms: &[Atom], clip: Clip, held: Option<&Held>) -> Summed {
    let own = Summed::of(atoms, clip);
    match held {
        Some(h) => own.plus(&h.summed),
        None => own,
    }
}

/// The frames `atom` reaches as `MODEL` states, `first > last` where none.
pub fn reach(ear: &Ear, clip: &Clip, atom: &Atom) -> (i64, i64) {
    let rate = f64::from(clip.rate);
    let (first, last) = ear.frames(clip.len);
    let lo = libm::floor(atom.on * rate - 0.5) as i64;
    let hi = libm::ceil(atom.end() * rate + 0.5) as i64;
    let met = |m: i64| {
        let (f, l) = ear.reach(m);
        f <= hi && l >= lo
    };
    let from = (first..=last).find(|m| met(*m));
    let to = (first..=last).rev().find(|m| met(*m));
    match (from, to) {
        (Some(f), Some(t)) => (f, t),
        _ => (1, 0),
    }
}

/// E per channel at frame `m`, 0 unsummed where every bin is.
pub fn excited(ear: &Ear, m: i64, power: &[f64]) -> Vec<f64> {
    match power.iter().all(|p| *p == 0.0) {
        true => vec![0.0; ear.thresholds().len()],
        false => ear.excite(m, power),
    }
}

/// `per` at each frame in order, every `threads`-th frame on one thread with its own `Model`,
/// so the frames an atom reaches spread over every thread.
pub fn framed<R: Send>(
    atoms: &[Atom],
    clip: Clip,
    coherent: bool,
    frames: (i64, i64),
    per: &(dyn Fn(&Model, i64) -> R + Sync),
) -> Vec<R> {
    framed_beside(atoms, clip, coherent, frames, (None, false), per)
}

/// `framed` with `summed` beside, holding `atoms`' partials and harmonics too where `summing`.
pub fn framed_beside<R: Send>(
    atoms: &[Atom],
    clip: Clip,
    coherent: bool,
    (first, last): (i64, i64),
    (summed, summing): (Option<&Summed>, bool),
    per: &(dyn Fn(&Model, i64) -> R + Sync),
) -> Vec<R> {
    let threads = default_threads();
    let frames = (last - first + 1).max(0) as usize;
    let runs = threads.get().min(frames).max(1);
    let mut each_run = each(threads, (0..runs).collect(), |r| {
        let model = Model::new(atoms, clip, coherent);
        let model = match (summed, summing) {
            (Some(s), true) => model.summing(s),
            (Some(s), false) => model.beside(s),
            (None, _) => model,
        };
        (r..frames)
            .step_by(runs)
            .map(|k| per(&model, first + k as i64))
            .collect::<Vec<R>>()
            .into_iter()
    });
    (0..frames)
        .map(|k| each_run[k % runs].next().expect("one answer per frame"))
        .collect()
}

pub fn total(residual_bits: f64, atoms: &[Atom], priced: &[Priced]) -> f64 {
    residual_bits + bits::bits(atoms, priced)
}
