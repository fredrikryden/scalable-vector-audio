// Concern: the clip on its grid as the optimizer's problem: power, weights, each pass's reading and candidates, MODEL | Non-concern: scoring a move (adds.rs, adjusts.rs) | IO: (Grid, Clip) -> Problem

use std::f64::consts::LOG2_E;
use std::num::NonZeroUsize;
use std::ops::Range;

use sva_engine::fft;
use sva_engine::hearing::erb_hz;

use super::adds::{add, adding};
use super::adjusts::adjusts;
use super::peaks::{Lobes, collide, lines};
use super::proposals::{averaged, candidates};
use super::render::{Held, Proposal};
use super::shared;
use crate::analysis::{floor_db, hann, span_hz};
use crate::grid::Grid;
use crate::optimize::code::Price;
use crate::optimize::{self, Ledger, Scored};
use crate::sources::{Clip, SharedKind};

pub const MODEL: &str = "one sound fitted by passes of one move each: coordinate descent on the sources' levels and frequencies beside matching pursuit for new sources (Mallat and Zhang 1993, doi 10.1109/78.258082); \
     DL = B(x) (bits.rs) plus the residual's bits, the one objective; \
     every pass over one problem: the clip's grid, its weights read once off the clip and never off a residual; \
     every source spans the clip; a partial's or voice's Hz f(t) = f(0) e^(glide t), glide 0 a fixed tone (sources.rs); \
     the clip read at its rate over the largest whole factor keeping Cam 38.9 under the Nyquist; \
     scaled to unit RMS, one master gain in dB bringing it back, every floor read at the SPL plus that gain; \
     the residual on the coarse grid's 48 ms frames by its analysis bins (grid.rs), octave bands being too coarse to tell a sine from noise inside one; \
     a cell's power P the averaged periodogram of K looks, K the frame over half the window times the window over the padded transform (Welch 1967); \
     a cell's heard power P' = P + F, the floor F a noise the ear's own; \
     the tonal lobes the voices' sieve reads (below) against a local noise floor: a bin's median over one Cam (at least a main lobe's width either side of its own lobe), outside the tonal lobes, on each side of the bin's own main lobe, over the median's share of the mean of K looks, (1 - 1/(9K))^3 (Wilson and Hilferty 1931), the higher side; \
     a tonal lobe the main lobe (2 window bins either side) of a local maximum resolved from every higher bin within a main lobe's width and over 2 ln M times that floor, M the feasible bins: an exponential's tail, the periodogram's at one look and a bound at more, so noise alone passes in one bin with probability 1/M^2 and holds a false lobe in a spectrum with under 1/M; \
     its lobes found first against the lower side's median over all bins, then barred from the floor read against the higher; \
     F a noise at threshold: its power in one ERB a sine's at its threshold (analysis.rs), so per bin that sine's peak-bin power times f_s sum w^2 / (ERB (sum w)^2), the sine's own peak-bin power overstating a threshold noise's by 2 to 17 dB from 100 Hz to 10 kHz; \
     the residual weighted by the clip's envelope, as the reported distance is: a cell's error its relative error r / P', r = P less the model's expected power, its dB error times ln 10 / 10 to first order with the log's power clipped at the floor so a cell the ear does not hear stays finite, weighted by its share P' / sum P' of the clip's heard power over every feasible cell; \
     that share-weighted mean square scaled to bits so a clip of uniform heard power codes each cell's error as a Gaussian at a noise periodogram's variance P'^2 / K over K looks: one unit of it N K / 2 nats, N the feasible cells; \
     so residual bits 1/2 sum w r^2 log2(e), w = K / (P' mean P'), the mean over the clip's feasible cells, read once off the clip; bins outside the feasible set unweighted; \
     templates in expected power: a sine through the window's closed-form kernel, a noise band's expected periodogram (grid.rs), a shaped noise's the sum of its bands' at their gains; \
     a gliding column's template in frame m the periodogram, by one transform, of a unit sine on its path in the window centred on the frame, its sweep across that window held, the frame's other hops left out: that template is the model's prediction, as a fixed sine's lobe is, not an estimate of another one, so its gap from the frame's pooled hops is misfit the residual holds and the DL prices; \
     a gliding column's floor at its Hz at the frames' mean time, its frequency looks at its path's Hz in each frame; \
     each column's gain per frame its power; \
     each pass scores every move against the residual and makes the one that lowers the DL most, or none, and the fit stops at the first pass with none, so the DL falls at every pass, \
     or after the first pass whose drop is no more than the stopping threshold times the larger of the DL before and after it and 1, the relative tolerance optimisers' ftol reads (default 1e-3), a resumed fit's default 0; \
     a fit resumed from a composition holds its sources as written, each kind and level curve read off its numbers, a glide at the rate floor's precision (its slope's standard error the add's, unwritten), its passes numbered on from those that made it; \
     a move an add or an adjust; \
     the model's power the sum of its sources' powers, which holds only where no two sources' tonal lines, a voice's every harmonic a line, lie within the main lobe's half-width (2 window bins) of each other in any frame: inside it two lines beat and their powers do not add; \
     so no add or frequency adjust leaves two sources' lines within it: a candidate whose lines fall within it of a kept source's line is no add, that kept source's own adjusts (its level, frequency or rethin) the moves there, and two voices overlap where any of their lines do; \
     a move whose source's node the parser or the engine refuses (sources.rs) skipped for the next best, and named in its pass's line; \
     an add a new source off the residual averaged over the frames, at its best level: \
     a free partial at each local maximum over its floor, its Hz by QIFFT (Smith and Serra 1987); \
     a partial or voice candidate, refined, glides where its columns' peak tracks glide: a column's track in each frame the highest local maximum over its floor, by QIFFT, in the basin of the averaged residual about its Hz, from the local maximum it climbs to down to the local minimum either side; \
     a glide one slope of ln Hz on time by least squares over its columns' tracks, each its own intercept, its precision the coarser of that slope's standard error and the rate moving the first column's Hz by its written precision over the frames, 0 within that of 0; \
     a glide scored only where it moves some line's Hz over the frames by the main lobe's half-width or more, a narrower sweep staying inside the place two lines there would share; \
     a gliding candidate starts at its tracks' intercepts, is refined as a fixed one is, and is kept where its DL is the lower; \
     a voice at each fundamental the sieve (analysis.rs) finds among those peaks that are tonal lobes of the averaged residual at K times its frames looks, taken from the pool with its members till none is found; \
     a voice's harmonic k at k f0 sqrt(1 + B k^2) (Fletcher 1964), f0 and B by least squares of (f_k / k)^2 = f0^2 + f0^2 B k^2 over its member peaks, B at least 0; \
     members reassigned within half an analysis bin of the stretched Hz and refitted till the set repeats; \
     every harmonic to the highest member its own column and its own level; \
     one noise band per octave band no kept noise band holds; \
     one shaped noise: a gain per octave band times one envelope over the frames (rank 1), both at least 0, by alternating least squares against the residual from a flat envelope till an alternation no longer lowers the weighted misfit, which never rises, its bands' cross terms left out of the alternation and in its score; a band whose gain falls to 0 carved out, the rest's gains over the highest's written in dB; \
     an add scored in closed form: each column's gain per frame its weighted correlation with the residual over its weighted energy, at least 0, written, the DL drop those gains give less its bits; \
     each partial's and voice's frequencies refined against the residual as it is scored, its glide held: a partial's Hz by golden section (Kiefer 1953) over half an analysis bin of its peak to the finer of the precision it is written at and its standard deviation under the weights, sigma = (sum w (du/dx)^2)^-1/2, u its expected power, du/dx by central difference at the cube root of epsilon; a voice's harmonics each so as a partial, then f0 and B by that least squares weighted by the inverse variance of (f_k / k)^2, (2 f_k sigma_k / k^2)^-2, kept where the voice's score falls; \
     an adjust changes one kept source in its own written numbers, three for each: \
     level, one gain over its whole level curve, 1 + sum w r u / sum w u^2 in closed form (u its expected power), its levels' bases moved by it in dB; or to 0, which removes it, where that gives the lower DL or the gain is not positive; \
     frequency, for a partial or voice: its tracks read off the residual with the source added back, in each frame the local maximum over its floor nearest its path's Hz in Cam within a main lobe of it, give a glide as an add's do; \
     at its own glide, at none and at that one where it moves a line so, its Hz at the start moved so its path holds at the frames' mean time, a partial's Hz by golden section over half a bin to the finer of its written precision and its standard deviation, a voice refined as an add's against that residual, its levels kept, the lowest DL kept; \
     rethin, its level curve thinned afresh from each column's written gain per frame plus the residual's weighted correlation with the column over its weighted energy, at least 0; \
     the DL of every move the written state's, in closed form from the residual, its drop exact; the residual derived afresh from the kept sources at each pass, a move whose repriced DL does not fall, its drop under rounding, no move; \
     ties broken by the move's order: adds in the candidates' order, then adjusts in the sources', each source's level, frequency, then rethin; \
     a column's level 10 log10 of its gain, held at its floor; a shaped noise's floor the least of its bands' floors less their gains; \
     each level through the frame centres from 0 to the clip's end, thinned by Ramer-Douglas-Peucker at the level JND at its peak sensation level (Jesteadt, Wier and Green 1977); \
     the DL the written sources': levels at the level JND by looks, frequencies at the frequency JND by looks (bits.rs), every frame a look at its level over its floor; \
     a shaped noise's band gain at the level JND by looks of that band's level over its own floor, its envelope by its looks over the envelope's floor; \
     a voice's f0 at the JND at f0, its B where its highest harmonic moves by that harmonic's JND; a glide at its precision; \
     a time at the frame, a noise band's edge one of the grid's band edges; \
     a key one past the last born; \
     a shared variable numbers several sources reference, each source its own numbers beside them (optimize/); one kind, a periodic level: its rate and phase shared, each referencing source's depth in dB its own, its level moved by depth sin(2 pi rate t + phase), written as a shared node each referencing source's node reads; \
     a frame's power of a source so moved its level's power times that factor's mean under the frame's kernel, the squared window over each hop the frame pools at the analysis hop; \
     an add also scored attached to each active periodic level and to the spare, the sources attached already set beside it, their levels held: per frame each source's level in dB the weighted correlation of the residual, those sources added back, with its template at its level (a new source's at its columns' mean gains) over its weighted energy; \
     the rate started at the peak of those levels' summed modulation spectrum, each less its least-squares line, at a whole count of cycles over the frames' span to their Nyquist, then by golden section a cycle over the span either side to its precision; \
     at a rate each source's line, sine and cosine by least squares, the phase the direction their sine and cosine pairs share most, each depth its pair along it over the kernel's attenuation at that rate; the new source's gains over the factor written as an add's; the whole edit priced by the ledger; \
     a periodic level's rate from 0 to the frames' Nyquist and its phase over a turn, the phase at the finest of its sources' depth precisions over their depths, the rate at that phase over the clip; a depth at its source's level precision; \
     each pass reported as it ends: its move and what an adjust changed, the source's level before and after, the DL's drop and total, the residual's grid, the moves it skipped";

/// The clip on its grid, as `MODEL` states: power and weights over the feasible bins, frame by
/// frame.
pub(super) struct Problem {
    pub(super) grid: Grid,
    pub(super) clip: Clip,
    pub(super) spl_db: f64,
    pub(super) top_db: f64,
    pub(super) range: Range<usize>,
    /// K, the looks one cell holds.
    pub(super) looks: f64,
    pub(super) lobes: Lobes,
    pub(super) power: Vec<Vec<f64>>,
    pub(super) weight: Vec<Vec<f64>>,
    pub(super) times: Vec<f64>,
    pub(super) window: Vec<f64>,
    pub(super) twiddle: Vec<(f64, f64)>,
    /// The share of a frame's power each time from its centre holds, as `MODEL` states.
    pub(super) kernel: Vec<(f64, f64)>,
    pub(super) threads: NonZeroUsize,
}

/// A fit's state: its kept sources and the optimizer's bookkeeping.
pub(super) type State = optimize::State<Problem>;

impl Problem {
    /// `grid` read as `MODEL` states, its power the target, computed on at most `threads`.
    pub(super) fn of(
        grid: Grid,
        clip: Clip,
        spl_db: f64,
        gain_db: f64,
        threads: NonZeroUsize,
    ) -> Problem {
        let range = feasible(&grid);
        let looks = grid.hop as f64 / (grid.long as f64 / 2.0) * grid.long as f64 / grid.n as f64;
        let floors: Vec<f64> = range
            .clone()
            .map(|b| {
                let hz = grid.hz(b);
                let amplitude = libm::pow(10.0, floor_db(hz, spl_db) / 20.0);
                amplitude * amplitude * grid.rate * grid.sum_w2 / (4.0 * erb_hz(hz))
            })
            .collect();
        let power: Vec<Vec<f64>> = grid
            .power
            .iter()
            .map(|f| f[range.clone()].to_vec())
            .collect();
        let lobes = Lobes::of(&grid, &range);
        // P' = P + F per cell, and its mean over every feasible cell of the clip.
        let heard: Vec<Vec<f64>> = power
            .iter()
            .map(|f| f.iter().zip(&floors).map(|(p, fl)| p + fl).collect())
            .collect();
        let cells = heard.iter().map(Vec::len).sum::<usize>().max(1) as f64;
        let mean = heard.iter().flatten().sum::<f64>() / cells;
        let weight: Vec<Vec<f64>> = heard
            .iter()
            .map(|f| f.iter().map(|h| looks / (h * mean)).collect())
            .collect();
        let times = (0..grid.power.len()).map(|f| grid.centre_secs(f)).collect();
        let window = hann(grid.long);
        let twiddle = fft::twiddles(grid.n);
        let kernel = grid.kernel();
        Problem {
            grid,
            clip,
            spl_db,
            top_db: -gain_db,
            range,
            looks,
            lobes,
            power,
            weight,
            times,
            window,
            twiddle,
            kernel,
            threads,
        }
    }

    pub(super) fn frames(&self) -> usize {
        self.power.len()
    }

    pub(super) fn middle(&self) -> f64 {
        self.times.iter().sum::<f64>() / self.times.len().max(1) as f64
    }

    /// Whether `proposal`'s lines lie apart from every kept source's but source `but`'s.
    pub(super) fn apart(&self, reading: &Reading, proposal: &Proposal, but: Option<usize>) -> bool {
        let mine = lines(self, proposal);
        reading
            .lines
            .iter()
            .enumerate()
            .all(|(i, theirs)| Some(i) == but || !collide(self, &mine, theirs))
    }
}

/// The bins the feasible set's Hz hold, as `MODEL` states.
pub(super) fn feasible(grid: &Grid) -> Range<usize> {
    let (lo, hi) = span_hz(grid.rate);
    (libm::ceil(grid.bin_of(lo)) as usize).max(1)
        ..(libm::floor(grid.bin_of(hi)) as usize + 1).min(grid.bins().saturating_sub(1))
}

/// What a pass reads off the kept sources: the residual, its mean over the frames, each kept
/// source's lines, and the shared variables an add may attach to: per kind every active one
/// and a spare.
pub(super) struct Reading {
    pub(super) residual: Vec<Vec<f64>>,
    pub(super) mean: Vec<f64>,
    pub(super) lines: Vec<Vec<Vec<f64>>>,
    pub(super) shared: Vec<optimize::Shared>,
    pub(super) targets: Vec<usize>,
}

/// A new source, its proposal beside its glide's precision, or one its fit made beside what it
/// set.
pub(super) enum Draft {
    Proposed(Proposal, f64),
    Made(Box<Held>),
}

/// A candidate: an add of a proposal, or a kept source's adjusts.
pub(super) enum Candidate {
    Add(Proposal),
    Adjust(usize),
}

impl optimize::Problem for Problem {
    type Thing = Held;
    type Draft = Draft;
    type Label = super::Tweak;
    type Candidate = Candidate;
    type Reading = Reading;

    fn missed(&self, kept: &[Held], shared: &[optimize::Shared]) -> f64 {
        0.5 * LOG2_E
            * self
                .residual(kept, shared)
                .iter()
                .zip(&self.weight)
                .map(|(r, w)| r.iter().zip(w).map(|(r, w)| w * r * r).sum::<f64>())
                .sum::<f64>()
    }

    fn price(&self, thing: &Held) -> Price {
        thing.written.price.clone()
    }

    fn attached(&self, thing: &Held) -> Vec<usize> {
        thing.links().iter().map(|l| l.to).collect()
    }

    fn spares(&self) -> Vec<optimize::Shared> {
        SharedKind::ALL
            .iter()
            .map(|kind| optimize::Shared {
                kind: *kind as usize,
                numbers: vec![0.0; kind.numbers().len()],
            })
            .collect()
    }

    fn price_shared(&self, s: usize, kept: &[Held]) -> Price {
        let written = kept.iter().map(|h| &h.written);
        shared::price(self, written, shared::kind_of(kept, s), s)
    }

    fn read(&self, kept: &[Held], shared: &[optimize::Shared]) -> Reading {
        let residual = self.residual(kept, shared);
        let used: Vec<usize> = kept.iter().flat_map(|h| self.attached(h)).collect();
        let mut targets = Vec::new();
        for kind in 0..SharedKind::ALL.len() {
            let of = |s: &usize| shared[*s].kind == kind;
            targets.extend((0..shared.len()).filter(|s| of(s) && used.contains(s)));
            targets.extend((0..shared.len()).find(|s| of(s) && !used.contains(s)));
        }
        Reading {
            mean: averaged(self, &residual),
            lines: kept.iter().map(|h| lines(self, &h.proposal())).collect(),
            residual,
            shared: shared.to_vec(),
            targets,
        }
    }

    /// The adds the residual proposes whose lines lie apart from the kept sources', then each
    /// kept source's adjusts.
    fn propose(&self, kept: &[Held], reading: &Reading) -> Vec<Candidate> {
        candidates(self, kept, &reading.residual)
            .into_iter()
            .filter(|proposal| self.apart(reading, proposal, None))
            .map(Candidate::Add)
            .chain((0..kept.len()).map(Candidate::Adjust))
            .collect()
    }

    fn fit(
        &self,
        kept: &[Held],
        ledger: &Ledger,
        reading: &Reading,
        key: u64,
        candidate: Candidate,
    ) -> Vec<Scored<Problem>> {
        match candidate {
            Candidate::Add(proposal) => add(self, kept, ledger, reading, key, proposal),
            Candidate::Adjust(a) => adjusts(self, kept, ledger, reading, a),
        }
    }

    fn make(&self, reading: &Reading, key: u64, draft: Draft) -> Held {
        match draft {
            Draft::Proposed(proposal, step) => {
                let (_, written) = adding(self, &reading.residual, &proposal, step, key);
                Held::new(self, self.templates(&proposal), written)
            }
            Draft::Made(held) => *held,
        }
    }

    fn admits(&self, thing: &Held) -> bool {
        thing.written.source.writes(&self.clip)
    }
}
