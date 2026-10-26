// Concern: the objective's gradient and Fisher matrix at a state by forward tangents through the ear and the stage; seeds shared by colouring | Non-concern: the search (lm.rs) | IO: (atoms) -> Linear

use std::f64::consts::LN_2;
use std::rc::Rc;

use sva_engine::internal::{Ear, Frame};
use sva_formula::C64;

use super::atoms::{Atom, Clip};
use super::bits::Priced;
use super::dual::Dual;
use super::lm::{Linear, Problem};
use super::objective::{Held, Objective, excited, framed_beside, reach};

pub const MODEL: &str = "the residual in nats f = sum_i d_i^2/2, d_i = (m_i - e_i)/sigma_i, \
     sigma_i^2 = N0 rho (objective.rs), differentiated in forward mode, one seed per number: \
     each bin's tangent (spectrum.rs), then the ear's excitation with the lower skirts' p_l \
     moving with X, V's slope 10 log10(e)/(E_T + E), the modulation stage, linear from rest, \
     applied to the tangent of V, a magnitude element through d|y| = Re(conj(y) dy)/|y| (0 at \
     y = 0); the gradient g_j = sum_i d_i r_ij and the Fisher matrix (Gauss-Newton's, the \
     variance held) A = sum_i r_i r_i^T, r_ij = dm_i/dx_j/sigma_i; an atom reaches a frame when \
     any window of the frame weighs a sample whose cell meets [on, end), atoms reaching no \
     common frame share a colour (Curtis, Powell and Reid 1974, IMA J. Appl. Math. 13(1) \
     117-119, doi 10.1093/imamat/13.1.117; intervals coloured greedily by first frame, which is \
     optimal), and the seeds are the passes, one per colour and number slot, each frame's \
     tangents those of the one atom of that colour reaching it, split by atom before the \
     stage, whose memory reaches every later frame; held atoms sound and seed nothing, read \
     once (objective.rs), so a frame no moving atom reaches takes their excitation, a pinned \
     number seeds nothing, its gradient entry and Fisher row and column 0; the tangents carry no bound: they propose \
     steps and curvatures, and every step is taken on the true cost";

/// Each atom's reach `(first, last)` in frames (empty where `first > last`), its colour, and
/// the seed passes: per colour its atoms' most numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Colouring {
    pub reach: Vec<(i64, i64)>,
    pub colours: Vec<usize>,
    pub passes: usize,
}

pub fn colouring(ear: &Ear, clip: &Clip, atoms: &[Atom]) -> Colouring {
    let reach: Vec<(i64, i64)> = atoms.iter().map(|a| reach(ear, clip, a)).collect();
    let mut order: Vec<usize> = (0..atoms.len()).collect();
    order.sort_by_key(|a| (reach[*a].0, *a));
    let mut ends: Vec<i64> = Vec::new();
    let mut colours = vec![0; atoms.len()];
    for a in order {
        let (from, to) = reach[a];
        let free = ends.iter().position(|end| *end < from);
        let c = match free {
            Some(c) => c,
            None => {
                ends.push(i64::MIN);
                ends.len() - 1
            }
        };
        if from <= to {
            ends[c] = to;
        }
        colours[a] = c;
    }
    let passes = (0..ends.len())
        .map(|c| {
            (0..atoms.len())
                .filter(|a| colours[*a] == c)
                .map(|a| atoms[a].values().len())
                .max()
                .unwrap_or(0)
        })
        .sum();
    Colouring {
        reach,
        colours,
        passes,
    }
}

/// The cost `f` in nats at a state, its `Linear`, and the colouring's passes.
pub struct Linearised {
    pub cost: f64,
    pub linear: Linear,
    pub passes: usize,
}

/// Composes, per bin, what `per` answers each range's frame with by number.
fn composed(
    ear: &Ear,
    m: i64,
    numbers: usize,
    per: &mut dyn FnMut(&Frame) -> Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::new(); numbers];
    ear.compose(m, &mut |frame| {
        let each = per(frame);
        let bins = frame.hz().len();
        for (j, o) in out.iter_mut().enumerate() {
            match each.get(j).filter(|e| !e.is_empty()) {
                Some(e) => o.extend_from_slice(e),
                None => o.extend(std::iter::repeat_n(0.0, bins)),
            }
        }
        vec![(); bins]
    });
    out
}

/// `atoms`' cost and its linearisation in the numbers of the first `moving`, the rest held,
/// partials and harmonics summed in samples where `coherent`.
pub fn linearise(
    objective: &Objective,
    atoms: &[Atom],
    moving: usize,
    coherent: bool,
) -> Linearised {
    let numbers = atoms[..moving].iter().map(|a| a.values().len()).sum();
    linearise_beside(
        objective,
        atoms,
        moving,
        coherent,
        (None, &vec![false; numbers]),
    )
}

/// `linearise` beside the atoms `held` read (objective.rs), partials summed in samples: a frame
/// no moving atom reaches takes their excitation as it is and seeds nothing; a number `still`
/// marks is seeded nowhere, its column 0.
pub fn linearise_beside(
    objective: &Objective,
    atoms: &[Atom],
    moving: usize,
    coherent: bool,
    (held, still): (Option<&Held>, &[bool]),
) -> Linearised {
    let (ear, clip) = (objective.ear(), objective.clip());
    let colouring = colouring(ear, &clip, &atoms[..moving]);
    let numbers: Vec<usize> = atoms[..moving].iter().map(|a| a.values().len()).collect();
    let offsets: Vec<usize> = numbers
        .iter()
        .scan(0, |at, n| {
            let here = *at;
            *at += n;
            Some(here)
        })
        .collect();
    let columns: usize = numbers.iter().sum();
    assert_eq!(still.len(), columns, "one mark per number");
    let (first, last) = ear.frames(clip.len);
    let frames = (last - first + 1) as usize;
    let channels = ear.thresholds().len();
    let summed = held.map(|h| &h.summed);
    let per = framed_beside(atoms, clip, coherent, (first, last), summed, &|model, m| {
        let reached = colouring.reach.iter().any(|(f, t)| *f <= m && m <= *t);
        if let Some(h) = held.filter(|_| !reached) {
            return (ear.levels(&h.excitation[(m - first) as usize]), Vec::new());
        }
        let mut power = ear.compose(m, &mut |frame| model.bins(frame).power);
        if let Some(h) = held {
            h.add_noise(m - first, &mut power);
        }
        let levels = ear.levels(&excited(ear, m, &power));
        let mut moved = Vec::new();
        for (a, (f, t)) in colouring.reach.iter().enumerate() {
            if m < *f || m > *t {
                continue;
            }
            let own = &still[offsets[a]..offsets[a] + numbers[a]];
            let d_power = composed(ear, m, numbers[a], &mut |frame| {
                model.tangents_of(frame, a, own)
            });
            for (j, d) in d_power.iter().enumerate() {
                if own[j] {
                    continue;
                }
                let tangent: Vec<Dual> = power
                    .iter()
                    .zip(d)
                    .map(|(v, d)| Dual::new(*v, *d))
                    .collect();
                let excited = ear.excite(m, &tangent);
                let slopes: Vec<f64> = ear.levels(&excited).iter().map(|v| v.tangent).collect();
                moved.push((offsets[a] + j, slopes));
            }
        }
        (levels, moved)
    });
    let mut levels = vec![vec![0.0; frames]; channels];
    let mut d_levels = vec![vec![vec![0.0; frames]; channels]; columns];
    for (at, (held, moved)) in per.into_iter().enumerate() {
        for (c, v) in held.into_iter().enumerate() {
            levels[c][at] = v;
        }
        for (column, slopes) in moved {
            for (c, v) in slopes.into_iter().enumerate() {
                d_levels[column][c][at] = v;
            }
        }
    }
    let outputs = ear.filtered(&levels);
    let expected = objective.elements(&outputs);
    let cost = objective.residual_bits(&expected) * LN_2;
    let d_outputs: Vec<(usize, Vec<Vec<Vec<C64>>>)> = d_levels
        .iter()
        .enumerate()
        .filter(|(p, _)| !still[*p])
        .map(|(p, d)| (p, ear.filtered(d)))
        .collect();
    let mut gradient = vec![0.0; columns];
    let mut fisher = vec![vec![0.0; columns]; columns];
    let mut rows: Vec<(usize, f64)> = Vec::with_capacity(columns);
    for (c, channel) in objective.heard().channels.iter().enumerate() {
        for (k, band) in channel.bands.iter().enumerate() {
            let sd = (objective.n0() * band.rate_hz).sqrt();
            for (e, clipped) in band.elements.iter().enumerate() {
                let y = outputs[c][k][e];
                let d = (expected[c][k][e] - clipped) / sd;
                rows.clear();
                for (p, column) in &d_outputs {
                    let p = *p;
                    let r = band.band.element_slope(y, column[c][k][e]) / sd;
                    if r != 0.0 {
                        gradient[p] += d * r;
                        rows.push((p, r));
                    }
                }
                for (p, r) in &rows {
                    for (q, s) in &rows {
                        fisher[*p][*q] += r * s;
                    }
                }
            }
        }
    }
    Linearised {
        cost,
        linear: Linear { gradient, fisher },
        passes: colouring.passes,
    }
}

/// A fixed structure's numbers as one vector in `values()`' order atom by atom, beside atoms
/// it holds as they are: its cost in nats, its linearisation, its projection atom by atom
/// (atoms.rs), each pinned number kept where it started.
pub struct Fit<'a> {
    objective: &'a Objective,
    atoms: Vec<Atom>,
    held: Vec<Atom>,
    read: Option<Rc<Held>>,
    priced: Vec<Priced>,
    coherent: bool,
    pinned: Vec<Option<f64>>,
}

impl<'a> Fit<'a> {
    pub fn new(
        objective: &'a Objective,
        atoms: Vec<Atom>,
        priced: Vec<Priced>,
        coherent: bool,
    ) -> Fit<'a> {
        assert_eq!(atoms.len(), priced.len(), "one pricing per atom");
        let numbers = atoms.iter().map(|a| a.values().len()).sum();
        Fit {
            objective,
            atoms,
            held: Vec::new(),
            read: None,
            priced,
            coherent,
            pinned: vec![None; numbers],
        }
    }

    /// The same fit with `held` sounding beside its atoms, none of their numbers moving: read
    /// once where partials are summed in samples (objective.rs), else sounded at each call.
    pub fn holding(mut self, held: Vec<Atom>) -> Fit<'a> {
        self.read = self
            .coherent
            .then(|| Rc::new(Held::of(self.objective, &held)));
        self.held = held;
        self
    }

    /// `holding` the atoms `read` holds, as `Held::of` read them (partials summed in samples).
    pub fn holding_read(mut self, read: Rc<Held>) -> Fit<'a> {
        assert!(self.coherent, "held partials are summed in samples");
        self.read = Some(read);
        self.held = Vec::new();
        self
    }

    /// The same fit with each number `pinned` marks, in `numbers()`' order, kept where it is.
    pub fn pinning(mut self, pinned: &[bool]) -> Fit<'a> {
        self.pinned = self
            .numbers()
            .into_iter()
            .zip(pinned)
            .map(|(v, pin)| pin.then_some(v))
            .collect();
        self
    }

    pub fn numbers(&self) -> Vec<f64> {
        self.atoms.iter().flat_map(|a| a.values()).collect()
    }

    pub fn deltas(&self) -> Vec<f64> {
        self.priced
            .iter()
            .flat_map(|p| p.deltas.iter().copied())
            .collect()
    }

    pub fn atoms_at(&self, x: &[f64]) -> Vec<Atom> {
        let mut at = 0;
        self.atoms
            .iter()
            .map(|a| {
                let n = a.values().len();
                let held = a.lifted(&x[at..at + n]);
                at += n;
                held
            })
            .collect()
    }

    /// The moving atoms at `x`, then the held unless they were read once.
    fn sounding(&self, x: &[f64]) -> Vec<Atom> {
        let mut all = self.atoms_at(x);
        if self.read.is_none() {
            all.extend(self.held.iter().cloned());
        }
        all
    }

    pub fn linearise(&self, x: &[f64]) -> Linearised {
        let still: Vec<bool> = self.pinned.iter().map(Option::is_some).collect();
        let mut out = linearise_beside(
            self.objective,
            &self.sounding(x),
            self.atoms.len(),
            self.coherent,
            (self.read.as_deref(), &still),
        );
        for (i, pin) in self.pinned.iter().enumerate() {
            if pin.is_some() {
                out.linear.gradient[i] = 0.0;
                for row in out.linear.fisher.iter_mut() {
                    row[i] = 0.0;
                }
                out.linear.fisher[i].fill(0.0);
            }
        }
        out
    }
}

impl Problem for Fit<'_> {
    fn cost(&self, x: &[f64]) -> f64 {
        let expected =
            self.objective
                .expect_beside(&self.sounding(x), self.coherent, self.read.as_deref());
        self.objective.residual_bits(&expected) * LN_2
    }

    fn linear(&self, x: &[f64]) -> Linear {
        self.linearise(x).linear
    }

    fn project(&self, x: &[f64], weights: &[f64]) -> Option<Vec<f64>> {
        let kept: Vec<f64> = x
            .iter()
            .zip(&self.pinned)
            .map(|(v, pin)| pin.unwrap_or(*v))
            .collect();
        let mut at = 0;
        let mut out = Vec::with_capacity(x.len());
        for (atom, priced) in self.atoms_at(&kept).iter().zip(&self.priced) {
            let n = atom.values().len();
            let held = priced.feasible.project(atom, &weights[at..at + n])?;
            out.extend(held.values());
            at += n;
        }
        Some(out)
    }
}
