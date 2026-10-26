// Concern: N0, the one calibrated constant: Jesteadt's grid through the reading, its closed-form fit and residuals, cached | Non-concern: the cost it prices (objective.rs) | IO: (rate, s, spl) -> N0

use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};

use sva_ast::Composition;
use sva_engine::hearing::{JESTEADT_TONE_SECS, level_jnd_db};
use sva_engine::internal::{Ear, Internal};
use sva_engine::{PSYCHOACOUSTIC_V1, Tier, each};

use super::atoms::{Atom, Clip, FULL_SCALE_DBFS, Trajectory, files};
use super::spectrum::floor_db;
use crate::{Job, Output, Representation, asked, call, execute};

pub const MODEL: &str = "N0, the internal noise's density in every modulation band (Dau, \
     Kollmeier and Kohlrausch 1997a, doi 10.1121/1.420344; Jepsen, Ewert and Dau 2008, doi \
     10.1121/1.2924135: equal-variance noise on every sample of every band), so element i of \
     band k in channel c has variance N0 rho_kc at its information rate rho_kc; fitted to \
     Jesteadt, Wier and Green's (1977, JASA 61(1) 169-177, doi 10.1121/1.381278) grid, 200, \
     400, 600, 800, 1000, 2000, 4000 and 8000 Hz at 5, 10, 20, 40 and 80 dB SL (the grid Wier, \
     Jesteadt and Green's 1975 abstract, JASA 58(S1) S35, gives the same experiment), each a \
     500 ms tone at the clip's rate under its own rectangular gate, a clip of its length, as \
     the engine renders its own node, a level step its power times 10^(D/10), a tone the \
     render does not hold (at or over the rate's audible ceiling) dropped; SL over \
     the reading's own threshold, the atom's floor; channels summed, d'^2 = sum_i (e_i(L + \
     D) - e_i(L))^2/(N0 rho_i) over every element, place only; DL(1) the exact d' = 1 root \
     at N0 = 1, the smallest double at which d' reaches 1 above the first doubling of \
     Jesteadt's own DL that does; DL scales with sqrt(N0) to first order, so the log fit is \
     closed form, log sqrt(N0) = mean_j (log DL_J,j - log DL_j(1)), DL_J Jesteadt's fit at d' \
     = 1; the residuals are the exact roots at that N0 over DL_J; N0 does not depend on spl, \
     which the floor cancels, and is kept per rate, scale and spl so its bits do";

/// Jesteadt, Wier and Green's (1977) frequencies and sensation levels.
pub const FREQUENCIES_HZ: [f64; 8] = [200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0, 4000.0, 8000.0];
pub const SENSATION_LEVELS_DB: [f64; 5] = [5.0, 10.0, 20.0, 40.0, 80.0];

/// One condition of the grid: its tone's floor and its DL at N0 = 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Condition {
    pub hz: f64,
    pub sl_db: f64,
    pub floor_db: f64,
    pub unit_dl_db: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub rate: u32,
    pub scale: f64,
    pub spl_db: f64,
    pub n0: f64,
    pub conditions: Vec<Condition>,
}

/// Each condition's exact DL at the fitted N0 over Jesteadt's, the log residuals' root mean
/// square and the largest as a ratio.
#[derive(Clone, Debug, PartialEq)]
pub struct Residuals {
    pub ratios: Vec<f64>,
    pub rms: f64,
    pub most: f64,
}

impl Calibration {
    /// The exact d' = 1 roots at N0, each condition's tone read again, on up to `threads`.
    pub fn residuals(&self, threads: NonZeroUsize) -> Residuals {
        let ear = Ear::of(f64::from(self.rate), self.spl_db, self.scale);
        let clip = clip(self.rate);
        let root_n0 = libm::sqrt(self.n0);
        let ratios = each(threads, self.conditions.clone(), |c| {
            let mut d = level_step(&ear, &clip, &tone(c.hz, c.floor_db + c.sl_db));
            root(&mut d, root_n0, c.unit_dl_db * root_n0) / level_jnd_db(c.sl_db)
        });
        let logs: Vec<f64> = ratios.iter().map(|r| libm::log(*r)).collect();
        let rms = libm::sqrt(logs.iter().map(|l| l * l).sum::<f64>() / logs.len() as f64);
        let most = sva_formula::exp(logs.iter().fold(0.0f64, |m, l| m.max(l.abs())));
        Residuals { ratios, rms, most }
    }
}

/// `d'` at N0 = 1 of a level step in dB on `tone`'s render, its power times `10^(D/10)`.
fn level_step<'a>(ear: &'a Ear, clip: &'a Clip, tone: &Atom) -> impl FnMut(f64) -> f64 + 'a {
    let powers = powers(ear, clip, &rendered(std::slice::from_ref(tone), clip));
    let base = reading(ear, clip, &powers, 1.0);
    move |db: f64| {
        let gain = libm::pow(10.0, db / 10.0);
        libm::sqrt(distance(&reading(ear, clip, &powers, gain), &base))
    }
}

/// The grid's tone: one partial from 0 under its own gate, its level knot ending it.
pub fn tone(hz: f64, db: f64) -> Atom {
    Atom::partial(
        0,
        0.0,
        Trajectory::steady(hz),
        Trajectory {
            base: db,
            knots: vec![(JESTEADT_TONE_SECS, 0.0)],
        },
    )
}

pub fn clip(rate: u32) -> Clip {
    Clip {
        rate,
        len: libm::ceil(JESTEADT_TONE_SECS * f64::from(rate)) as usize,
    }
}

/// The clip `atoms` write, rendered by the engine from their own nodes and score.
pub fn rendered(atoms: &[Atom], clip: &Clip) -> Vec<f64> {
    let mut held = Composition::new();
    for (path, text) in files(atoms, clip) {
        held.insert(path, text);
    }
    let samples = [asked(&call("samples").expect("a call")).expect("a reading")];
    let target = format!("@master([0, {}s])", clip.secs());
    let job = Job {
        asked: &samples,
        rate: Some(clip.rate),
        ..Job::over(&held, &target)
    };
    let rendered = execute(job, &Tier::default()).expect("atoms' own nodes render");
    let answer = rendered
        .answer(&rendered.target, Representation::Samples)
        .expect("a render answers its samples");
    let Output::Samples(buffer) = answer.value else {
        unreachable!("samples answer samples")
    };
    buffer.plane(0).to_vec()
}

/// Each frame's composed power of `samples`.
pub fn powers(ear: &Ear, clip: &Clip, samples: &[f64]) -> Vec<Vec<f64>> {
    let (first, last) = ear.frames(clip.len);
    (first..=last)
        .map(|m| ear.compose(m, &mut |frame| frame.power(&[samples], 0)))
        .collect()
}

/// The reading of `powers`, each times `gain`.
pub fn reading(ear: &Ear, clip: &Clip, powers: &[Vec<f64>], gain: f64) -> Internal {
    let (first, _) = ear.frames(clip.len);
    let outputs = ear.elements(clip.len, &mut |m| {
        let scaled: Vec<f64> = powers[(m - first) as usize]
            .iter()
            .map(|p| p * gain)
            .collect();
        ear.excite(m, &scaled)
    });
    ear.reading(outputs, 0.0)
}

/// `d'^2` at N0 = 1: every element's squared gap over its band's rate, channels summed.
pub fn distance(a: &Internal, b: &Internal) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.channels.iter().zip(&b.channels) {
        for (u, v) in x.bands.iter().zip(&y.bands) {
            for (e, g) in u.elements.iter().zip(&v.elements) {
                sum += (e - g) * (e - g) / u.rate_hz;
            }
        }
    }
    sum
}

/// The smallest double at which `f` reaches `target` above the first doubling of `start` that
/// does, `f(0)` under it: the secant until it stops moving, ulp steps doubling to the other
/// side, then bisection to adjacent doubles. The method only finds that double.
pub fn root(f: &mut dyn FnMut(f64) -> f64, target: f64, start: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, start);
    let mut g_hi = f(hi) - target;
    let mut g_lo = -target;
    while g_hi < 0.0 {
        (lo, g_lo) = (hi, g_hi);
        hi *= 2.0;
        g_hi = f(hi) - target;
    }
    let mut held = |x: f64, lo: &mut f64, hi: &mut f64| {
        let g = f(x) - target;
        match g < 0.0 {
            true => *lo = x,
            false => *hi = x,
        }
        g
    };
    let ((mut a, mut g_a), (mut b, mut g_b)) = ((lo, g_lo), (hi, g_hi));
    while lo.next_up() < hi {
        let secant = b - g_b * (b - a) / (g_b - g_a);
        if secant == b {
            break;
        }
        let x = match secant > lo && secant < hi {
            true => secant,
            false => 0.5 * (lo + hi),
        };
        let g = held(x, &mut lo, &mut hi);
        ((a, g_a), (b, g_b)) = ((b, g_b), (x, g));
    }
    let mut step = 1u64;
    while lo.next_up() < hi {
        let x = match b == hi {
            true => f64::from_bits(b.to_bits().saturating_sub(step)).max(lo),
            false => f64::from_bits(b.to_bits().saturating_add(step)).min(hi),
        };
        if x == lo || x == hi {
            break;
        }
        let moved_lo = b == hi;
        held(x, &mut lo, &mut hi);
        if (moved_lo && lo == x) || (!moved_lo && hi == x) {
            break;
        }
        b = x;
        step *= 2;
    }
    while lo.next_up() < hi {
        let x = f64::from_bits(lo.to_bits() + (hi.to_bits() - lo.to_bits()) / 2);
        held(x, &mut lo, &mut hi);
    }
    hi
}

fn fitted(rate: u32, scale: f64, spl_db: f64, threads: NonZeroUsize) -> Calibration {
    let ear = Ear::of(f64::from(rate), spl_db, scale);
    let clip = clip(rate);
    let ceiling = PSYCHOACOUSTIC_V1.ceiling(rate);
    let heard: Vec<f64> = FREQUENCIES_HZ
        .iter()
        .copied()
        .filter(|hz| *hz < ceiling)
        .collect();
    let floors = each(threads, heard.clone(), |hz| {
        floor_db(&tone(hz, FULL_SCALE_DBFS), &clip, &ear).db
    });
    let grid: Vec<(f64, f64, f64)> = heard
        .iter()
        .zip(&floors)
        .flat_map(|(hz, floor)| SENSATION_LEVELS_DB.map(|sl| (*hz, sl, *floor)))
        .collect();
    let conditions = each(threads, grid, |(hz, sl_db, floor_db)| {
        let mut d = level_step(&ear, &clip, &tone(hz, floor_db + sl_db));
        Condition {
            hz,
            sl_db,
            floor_db,
            unit_dl_db: root(&mut d, 1.0, level_jnd_db(sl_db)),
        }
    });
    let mean = conditions
        .iter()
        .map(|c| libm::log(level_jnd_db(c.sl_db)) - libm::log(c.unit_dl_db))
        .sum::<f64>()
        / conditions.len() as f64;
    let root_n0 = sva_formula::exp(mean);
    Calibration {
        rate,
        scale,
        spl_db,
        n0: root_n0 * root_n0,
        conditions,
    }
}

type Kept = Vec<((u32, u64, u64), Arc<OnceLock<Calibration>>)>;

static KEPT: Mutex<Kept> = Mutex::new(Vec::new());

/// The calibration at `rate`, scale `scale` and `spl_db`, fitted once per process on up to
/// `threads`, which move no bit; a caller asking while another fits it waits for that fit.
pub fn calibration(rate: u32, scale: f64, spl_db: f64, threads: NonZeroUsize) -> Calibration {
    let key = (rate, scale.to_bits(), spl_db.to_bits());
    let cell = {
        let mut kept = KEPT.lock().expect("nothing panics holding it");
        match kept.iter().find(|(k, _)| *k == key) {
            Some((_, cell)) => cell.clone(),
            None => {
                let cell = Arc::new(OnceLock::new());
                kept.push((key, cell.clone()));
                cell
            }
        }
    };
    cell.get_or_init(|| fitted(rate, scale, spl_db, threads))
        .clone()
}

pub fn n0(rate: u32, scale: f64, spl_db: f64, threads: NonZeroUsize) -> f64 {
    calibration(rate, scale, spl_db, threads).n0
}
