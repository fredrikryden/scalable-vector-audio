// Concern: the objective's gradient at a state by the stage's adjoint and its Fisher matrix by forward tangents; seeds shared by colouring | Non-concern: the search (lm.rs) | IO: (atoms) -> Linear

use std::cell::RefCell;
use std::f64::consts::LN_2;
use std::rc::Rc;

use sva_engine::internal::{Ear, Frame};
use sva_formula::C64;

use super::atoms::{Atom, Clip, Sound};
use super::bits::Priced;
use super::lm::{self, Groups, Linear, Problem};
use super::objective::{Held, NoisePower, Objective, excited, framed_beside, reach, summed_beside};
use super::spectrum::{Series, series};
use sva_engine::{default_threads, each};

pub const MODEL: &str = "the residual in nats f = sum_i d_i^2/2, d_i = (m_i - e_i)/sigma_i, \
     sigma_i^2 = N0 rho (objective.rs), differentiated in forward mode, one seed per number: \
     a partial's or harmonic's sample tangents read once over the samples it sounds on (a \
     harmonic's offsets in closed form, spectrum.rs) and sliced per window, each bin's tangent \
     2 Re(conj(X) dX) against the state's transform, a noise's through its expected power \
     (spectrum.rs), then the ear's excitation with the lower skirts' p_l moving with X, V's \
     slope 10 log10(e)/(E_T + E), at the frames the atom reaches; the gradient g_j = sum_i d_i \
     r_ij, r_ij = dm_i/dx_j/sigma_i, exact, by the modulation stage's adjoint (each filter's \
     recursion run back in time on d_i/sigma_i times the element's gradient, Re y or |y| \
     through d|y| = Re(conj(y) dy)/|y|, 0 at y = 0) against each seed's V tangent; the Fisher \
     matrix (Gauss-Newton's, the variance held) A = sum_i r_i r_i^T with each seed's r_i read \
     over the elements of the frames its atom reaches, the stage run from rest at its first, \
     so two atoms couple only through the frames they share (the stage's memory past an \
     atom's last frame, read by the gradient, proposes no curvature), or to the clip's end \
     where the curvatures are read whole (precisions, bits.rs); an atom reaches a frame when \
     any window of the frame weighs a sample whose cell meets [on, end), atoms reaching no \
     common frame share a colour (Curtis, Powell and Reid 1974, IMA J. Appl. Math. 13(1) \
     117-119, doi 10.1093/imamat/13.1.117; intervals coloured greedily by first frame, which is \
     optimal), and the seeds are the passes, one per colour and number slot; held atoms sound \
     and seed nothing, read once (objective.rs), so a frame no moving atom reaches takes their \
     excitation, a pinned or frozen number seeds nothing, its gradient entry and Fisher row \
     and column 0; the tangents carry no bound: they propose steps and curvatures, and every \
     step is taken on the true cost";

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
        false,
    )
}

/// One number's V tangent per channel over the frames its atom reaches, from frame index `at`
/// (from the first), beside its Fisher window's last frame index.
struct Column {
    at: usize,
    last: usize,
    slopes: Vec<Vec<f64>>,
}

/// `linearise` beside the atoms `held` read (objective.rs), partials summed in samples: a frame
/// no moving atom reaches takes their excitation as it is and seeds nothing; a number `still`
/// marks is seeded nowhere, its column 0; the Fisher matrix read to the clip's end where
/// `tails`, else over the frames each pair of atoms both reach.
pub fn linearise_beside(
    objective: &Objective,
    atoms: &[Atom],
    moving: usize,
    coherent: bool,
    (held, still): (Option<&Held>, &[bool]),
    tails: bool,
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
    let sampled = !objective.is_coarse();
    let summed = (coherent && sampled).then(|| summed_beside(atoms, clip, held));
    let beside = (summed.as_ref(), coherent);
    let tonal = match sampled {
        true => moving,
        false => 0,
    };
    let series: Vec<Vec<Option<Series>>> = each(default_threads(), (0..tonal).collect(), |a| {
        let own = &still[offsets[a]..offsets[a] + numbers[a]];
        series(&atoms[a], &clip, own)
    });
    let reaches: Vec<(i64, i64)> = atoms.iter().map(|a| reach(ear, &clip, a)).collect();
    let per = match objective.coarse_read(atoms, &reaches) {
        Some(read) => each(default_threads(), (first..=last).collect(), |m| {
            let mut seeds = Vec::new();
            let mut pairs = Vec::new();
            for (a, (f, t)) in colouring.reach.iter().enumerate() {
                if m < *f || m > *t {
                    continue;
                }
                for j in 0..numbers[a] {
                    if !still[offsets[a] + j] {
                        seeds.push(offsets[a] + j);
                        pairs.push((a, j));
                    }
                }
            }
            if pairs.is_empty() {
                return (ear.levels(&read.excitation(m)), Vec::new());
            }
            let (excitation, along) = read.tangents(m, &pairs);
            let moved = seeds
                .into_iter()
                .zip(along)
                .filter_map(|(p, d)| d.map(|d| (p, ear.levels_along(&excitation, &d))))
                .collect();
            (ear.levels(&excitation), moved)
        }),
        None => framed_beside(atoms, clip, coherent, (first, last), beside, &|model, m| {
            model.forget();
            let reached = colouring.reach.iter().any(|(f, t)| *f <= m && m <= *t);
            if let Some(h) = held.filter(|_| !reached) {
                return (ear.levels(&h.excitation[(m - first) as usize]), Vec::new());
            }
            let mut power = ear.compose(m, &mut |frame| model.bins(frame).power);
            if let Some(h) = held {
                h.add_noise(m - first, &mut power);
            }
            let excitation = excited(ear, m, &power);
            let levels = ear.levels(&excitation);
            let mut seeds = Vec::new();
            let mut along = Vec::new();
            for (a, (f, t)) in colouring.reach.iter().enumerate() {
                if m < *f || m > *t {
                    continue;
                }
                let own = &still[offsets[a]..offsets[a] + numbers[a]];
                if own.iter().all(|s| *s) {
                    continue;
                }
                let d_power = composed(ear, m, numbers[a], &mut |frame| {
                    model.tangents_read(frame, a, &series[a])
                });
                for (j, d) in d_power.into_iter().enumerate() {
                    if !own[j] {
                        seeds.push(offsets[a] + j);
                        along.push(d);
                    }
                }
            }
            let moved: Vec<(usize, Vec<f64>)> = match along.is_empty() {
                true => Vec::new(),
                false => seeds
                    .into_iter()
                    .zip(ear.excite_along(m, &power, &along))
                    .map(|(p, d)| (p, ear.levels_along(&excitation, &d)))
                    .collect(),
            };
            (levels, moved)
        }),
    };
    let mut levels = vec![vec![0.0; frames]; channels];
    let mut d_levels: Vec<Option<Column>> = (0..columns)
        .map(|p| {
            let a = offsets.partition_point(|o| *o <= p) - 1;
            let (f, t) = colouring.reach[a];
            (!still[p] && f <= t).then(|| {
                let (at, end) = ((f - first) as usize, (t - first) as usize);
                Column {
                    at,
                    last: match tails {
                        true => frames - 1,
                        false => end,
                    },
                    slopes: vec![vec![0.0; end - at + 1]; channels],
                }
            })
        })
        .collect();
    for (at, (held, moved)) in per.into_iter().enumerate() {
        for (c, v) in held.into_iter().enumerate() {
            levels[c][at] = v;
        }
        for (p, slopes) in moved {
            let column = d_levels[p].as_mut().expect("a seeded number has a column");
            for (c, v) in slopes.into_iter().enumerate() {
                column.slopes[c][at - column.at] = v;
            }
        }
    }
    let outputs = ear.filtered(&levels);
    let expected = objective.elements(&outputs);
    let cost = objective.residual_bits(&expected) * LN_2;
    let heard = &objective.heard().channels;
    let pulls: Vec<Vec<Vec<C64>>> = heard
        .iter()
        .enumerate()
        .map(|(c, channel)| {
            channel
                .bands
                .iter()
                .enumerate()
                .map(|(k, band)| {
                    let variance = objective.n0() * band.rate_hz;
                    band.elements
                        .iter()
                        .enumerate()
                        .map(|(e, clipped)| {
                            let z = band.band.pull(outputs[c][k][e]);
                            let w = (expected[c][k][e] - clipped) / variance;
                            C64::new(z.re * w, z.im * w)
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let pulled = ear.pulled(&pulls, frames);
    let gradient: Vec<f64> = d_levels
        .iter()
        .map(|column| {
            column.as_ref().map_or(0.0, |column| {
                column
                    .slopes
                    .iter()
                    .zip(&pulled)
                    .map(|(d, w)| {
                        d.iter()
                            .zip(&w[column.at..])
                            .map(|(d, w)| d * w)
                            .sum::<f64>()
                    })
                    .sum()
            })
        })
        .collect();
    let fisher = fisher(objective, &outputs, &d_levels, frames);
    Linearised {
        cost,
        linear: Linear { gradient, fisher },
        passes: colouring.passes,
    }
}

/// Per channel and band, a column's first element in its window beside its residual's slopes
/// `r_i = dm_i/dx/sigma_i` from there to the window's end.
type Rows = Vec<Vec<(usize, Vec<f64>)>>;

/// The Fisher matrix `A = sum_i r_i r_i^T` over each pair of columns' shared window, columns in
/// windows ordered by their first frame, each pair read where both are live.
fn fisher(
    objective: &Objective,
    outputs: &[Vec<Vec<C64>>],
    columns: &[Option<Column>],
    frames: usize,
) -> Vec<Vec<f64>> {
    let ear = objective.ear();
    let heard = &objective.heard().channels;
    let n = columns.len();
    let mut out = vec![vec![0.0; n]; n];
    let rows_of = |p: usize| -> Rows {
        let column = columns[p].as_ref().expect("a seeded column");
        let len = column.last - column.at + 1;
        column
            .slopes
            .iter()
            .enumerate()
            .map(|(c, d)| {
                let mut v = d.clone();
                v.resize(len, 0.0);
                ear.filtered_within(c, &v, (column.at, frames))
                    .into_iter()
                    .zip(&heard[c].bands)
                    .enumerate()
                    .map(|(k, ((j0, ys), band))| {
                        let sd = (objective.n0() * band.rate_hz).sqrt();
                        let r = ys
                            .iter()
                            .enumerate()
                            .map(|(i, dy)| band.band.element_slope(outputs[c][k][j0 + i], *dy) / sd)
                            .collect();
                        (j0, r)
                    })
                    .collect()
            })
            .collect()
    };
    let dot = |a: &Rows, b: &Rows| -> f64 {
        let mut sum = 0.0;
        for (a, b) in a.iter().zip(b) {
            for ((ja, ra), (jb, rb)) in a.iter().zip(b) {
                let from = (*ja).max(*jb);
                let to = (ja + ra.len()).min(jb + rb.len());
                if from < to {
                    sum += lm::dot(&ra[from - ja..to - ja], &rb[from - jb..to - jb]);
                }
            }
        }
        sum
    };
    let mut order: Vec<usize> = (0..n).filter(|p| columns[*p].is_some()).collect();
    order.sort_by_key(|p| (columns[*p].as_ref().map_or(0, |c| c.at), *p));
    let window = |p: usize| {
        let c = columns[p].as_ref().expect("a seeded column");
        (c.at, c.last)
    };
    let mut live: Vec<(usize, Rows)> = Vec::new();
    let mut k = 0;
    while k < order.len() {
        let at = window(order[k]).0;
        let batch: Vec<usize> = order[k..]
            .iter()
            .take_while(|p| window(**p).0 == at)
            .copied()
            .collect();
        k += batch.len();
        live.retain(|(q, _)| window(*q).1 >= at);
        let rows: Vec<(usize, Rows)> = each(default_threads(), batch, |p| (p, rows_of(p)));
        let entries: Vec<Vec<(usize, usize, f64)>> =
            each(default_threads(), (0..rows.len()).collect(), |b| {
                let (p, r) = &rows[b];
                live.iter()
                    .chain(&rows[..=b])
                    .map(|(q, s)| (*p, *q, dot(r, s)))
                    .collect()
            });
        for (p, q, v) in entries.into_iter().flatten() {
            out[p][q] = v;
            out[q][p] = v;
        }
        live.extend(rows);
    }
    out
}

/// A noise's power as one evaluation read it, by its key and its numbers' bits.
type ReadNoise = (u64, Vec<u64>, Rc<NoisePower>);

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
    /// The noises the last evaluation read, by key and numbers' bits.
    noises: RefCell<Vec<ReadNoise>>,
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
            noises: RefCell::default(),
        }
    }

    /// The same fit with `held` sounding beside its atoms, none of their numbers moving: read
    /// once where partials are summed in samples (objective.rs), else (or at a coarse
    /// objective) sounded at each call.
    pub fn holding(mut self, held: Vec<Atom>) -> Fit<'a> {
        self.read = (self.coherent && !self.objective.is_coarse())
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

    /// The linearisation at `x`, no number `frozen` or pinned seeded, the Fisher matrix read
    /// to the clip's end where `tails`.
    pub fn linearise(&self, x: &[f64], frozen: &[bool], tails: bool) -> Linearised {
        let still: Vec<bool> = self
            .pinned
            .iter()
            .zip(frozen)
            .map(|(p, f)| p.is_some() || *f)
            .collect();
        linearise_beside(
            self.objective,
            &self.sounding(x),
            self.atoms.len(),
            self.coherent,
            (self.read.as_deref(), &still),
            tails,
        )
    }
}

impl Problem for Fit<'_> {
    /// The residual at `x`, partials summed in samples, each noise's power read once while
    /// its numbers stay (a frozen atom's), else as `expect` reads it.
    fn cost(&self, x: &[f64]) -> f64 {
        let atoms = self.sounding(x);
        if self.objective.is_coarse() {
            return self
                .objective
                .residual_bits(&self.objective.expect(&atoms, true))
                * LN_2;
        }
        let expected = match self.coherent {
            true => {
                let last = self.noises.take();
                let read: Vec<ReadNoise> = atoms
                    .iter()
                    .filter(|a| matches!(a.sound, Sound::Noise { .. }))
                    .map(|a| {
                        let bits: Vec<u64> = a.values().iter().map(|v| v.to_bits()).collect();
                        let kept = last.iter().find(|(k, b, _)| *k == a.key && *b == bits);
                        let power = match kept {
                            Some((_, _, p)) => p.clone(),
                            None => Rc::new(self.objective.noise_power(a)),
                        };
                        (a.key, bits, power)
                    })
                    .collect();
                let noise: Vec<&NoisePower> = read.iter().map(|(_, _, p)| p.as_ref()).collect();
                let expected = self
                    .objective
                    .expect_read(&atoms, &noise, self.read.as_deref());
                self.noises.replace(read);
                expected
            }
            false => self.objective.expect_beside(&atoms, false, None),
        };
        self.objective.residual_bits(&expected) * LN_2
    }

    fn linear(&self, x: &[f64], frozen: &[bool]) -> Linear {
        self.linearise(x, frozen, false).linear
    }

    fn groups(&self, x: &[f64]) -> Groups {
        let atoms = self.atoms_at(x);
        let (ear, clip) = (self.objective.ear(), self.objective.clip());
        let reach: Vec<(i64, i64)> = atoms.iter().map(|a| reach(ear, &clip, a)).collect();
        let meets = |a: (i64, i64), b: (i64, i64)| a.0 <= b.1 && b.0 <= a.1;
        Groups {
            sizes: atoms.iter().map(|a| a.values().len()).collect(),
            neighbours: (0..atoms.len())
                .map(|a| {
                    (0..atoms.len())
                        .filter(|b| *b != a && meets(reach[a], reach[*b]))
                        .collect()
                })
                .collect(),
        }
    }

    fn curvatures(&self, x: &[f64]) -> Vec<f64> {
        let linear = self.linearise(x, &vec![false; x.len()], true).linear;
        (0..x.len()).map(|i| linear.fisher[i][i]).collect()
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
