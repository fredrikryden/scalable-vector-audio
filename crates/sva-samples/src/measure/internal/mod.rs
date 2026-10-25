// Concern: the reading `internal`: N' per Bark per frame, its modulation, each over its JND, off power per bin | Non-concern: where the power came from | IO: (planes, spl) -> Internal

//! One cell per Bark band per 10 ms frame per element set, each element over its JND, so two
//! readings subtract into `d'` of an ideal observer with a diagonal covariance. The band-passes
//! read what the lowpass leaves, so a steady level reads in none; past the frames' Nyquist no
//! band-pass is held.

pub mod iso;

use sva_formula::C64;

use crate::fft;
use iso::{CORE, Decay, ROWS, THIRDS};

pub const MODEL: &str = "specific loudness N' per Bark every 10 ms after ISO 532-1:2017's \
     time-varying method: its third-octave band responses, their smoothing, core loudness, \
     nonlinear decay at 1/48000 s and upper slopes, the 0.1 Bark rows pooled by mean; each band's \
     mean square is defined as sum_j |H_b(f_j)|^2 P_j off a 20 ms periodic Hann frame's power per \
     bin 2|X|^2/(n sum w^2), the ends mirrored, not the standard's time-domain filter output, so \
     a steady 1 kHz sine or a noise reads within 1 % of the standard's N', a 100 Hz sine about \
     17 % above it and an onset up to half a frame early; a full-scale sine at `spl` dB SPL; each element over its JND: N' as E = \
     10 log10(4 (1 + N'/u)^4 - 3), the level above threshold ISO's core law gives it at (u = \
     0.0635 10^(0.025 LTQ)), in the 1 dB intensity JND (Zwicker and Fastl 2007, 7.1); E through \
     the modulation filterbank of Ewert and Dau (2000, JASA 108) and Dau et al. (1997, JASA \
     102): a first-order 2.5 Hz lowpass in the same 1 dB, and complex first-order band-passes \
     of Q = 1 an octave apart from 4 Hz whose upper edge 1.5 fc is under the frames' Nyquist, \
     fed E less the lowpass, each element twice its magnitude over the -26 dB AM depth JND \
     (Viemeister 1979, JASA 66) in dB, (20/ln 10) 10^(-26/20); every channel averaged. The \
     lowpass's complement feeding the band-passes is this reading's structure, not the papers': \
     a first-order complex band-pass of Q = 1 passes 1/sqrt(5) of a steady level, which would \
     read each 1 dB of level as some 2 JND in every band-pass. Cells count as independent \
     though the window, the decay, the upper slopes and the overlapping filters correlate \
     neighbours. Envelope beats past the frames' 50 Hz Nyquist alias into the \
     band-passes; a 2 ms envelope read them off at some 0.2 JND per cell less on clip1, for \
     3 to 5 times a fit's evaluation, so the frames stay 10 ms";

pub const FRAME_SECS: f64 = 0.01;
pub const WINDOW_SECS: f64 = 0.02;
pub const BARKS: usize = 24;
const LEVEL_JND_DB: f64 = 1.0;
const DEPTH_JND_DB: f64 = -26.0;
const LOWPASS_HZ: f64 = 2.5;
const FIRST_BANDPASS_HZ: f64 = 4.0;
const Q: f64 = 1.0;

/// The frames a reading takes: `window` samples in an `n`-point transform every `hop`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub sr: f64,
    pub n: usize,
    pub window: usize,
    pub hop: usize,
    pub left: usize,
    pub frames: usize,
    pub len: usize,
}

impl Grid {
    pub fn of(sr: f64, len: usize) -> Grid {
        let window = libm::round(WINDOW_SECS * sr) as usize;
        let n = (2 * window).next_power_of_two();
        let hop = libm::round(FRAME_SECS * sr) as usize;
        Grid {
            sr,
            n,
            window,
            hop,
            left: (n - window) / 2,
            frames: len / hop + 1,
            len,
        }
    }

    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    pub fn step_secs(&self) -> f64 {
        self.hop as f64 / self.sr
    }

    /// The sample frame `m`'s window starts at.
    pub fn first(&self, m: usize) -> i64 {
        (m * self.hop) as i64 - (self.n / 2) as i64 + self.left as i64
    }

    pub fn scale(&self) -> f64 {
        let w2: f64 = hann(self.window).iter().map(|x| x * x).sum();
        2.0 / (self.n as f64 * w2)
    }

    /// The sample a read at `s` lands on, the ends mirrored.
    pub fn mirrored(&self, s: i64) -> i64 {
        let len = self.len as i64;
        if len <= 1 {
            return 0;
        }
        let period = 2 * (len - 1);
        let k = s.rem_euclid(period);
        if k >= len { period - k } else { k }
    }
}

pub fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * libm::cos(std::f64::consts::TAU * i as f64 / n as f64))
        .collect()
}

/// `P[m * bins + j]`: each frame's power per bin.
pub fn bin_power(x: &[f64], g: &Grid) -> Vec<f64> {
    let w = hann(g.window);
    let scale = g.scale();
    let twiddles = fft::twiddles(g.n);
    let bins = g.bins();
    let mut p = vec![0.0; g.frames * bins];
    let (mut re, mut im) = (vec![0.0; g.n], vec![0.0; g.n]);
    for m in 0..g.frames {
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for (k, wk) in w.iter().enumerate() {
            re[g.left + k] = wk * x[g.mirrored(g.first(m) + k as i64) as usize];
        }
        fft::fft_with(&mut re, &mut im, &twiddles);
        for j in 0..bins {
            p[m * bins + j] = (re[j] * re[j] + im[j] * im[j]) * scale;
        }
    }
    p
}

/// The Bark of a frequency, Zwicker and Terhardt (1980), JASA 68, 1523-1525.
pub fn bark(hz: f64) -> f64 {
    13.0 * libm::atan(0.00076 * hz) + 3.5 * libm::atan((hz / 7500.0) * (hz / 7500.0))
}

/// The band-pass centres the frames' step holds.
pub fn channels(step_secs: f64) -> Vec<f64> {
    let nyquist = 0.5 / step_secs;
    let mut out = Vec::new();
    let mut fc = FIRST_BANDPASS_HZ;
    while fc * (1.0 + 0.5 / Q) < nyquist {
        out.push(fc);
        fc *= 2.0;
    }
    out
}

pub struct Ear {
    pub grid: Grid,
    pub spl_db: f64,
    /// `|H_b(f_j)|^2`, `[b * bins + j]`.
    response: Vec<f64>,
    smooth: [f64; THIRDS],
    pa2: f64,
    unit: [f64; BARKS],
    lowpass: f64,
    poles: Vec<C64>,
}

pub struct Forward {
    pub frames: usize,
    /// `[(set * frames + m) * BARKS + i]`, each in its JND.
    pub elements: Vec<f64>,
    /// N' in sone per Bark, `[m * BARKS + i]`.
    pub specific: Vec<f64>,
}

impl Ear {
    pub fn of(grid: Grid, spl_db: f64) -> Ear {
        let bins = grid.bins();
        let mut response = vec![0.0; THIRDS * bins];
        for b in 0..THIRDS {
            for j in 0..bins {
                response[b * bins + j] = iso::response(b, j as f64 * grid.sr / grid.n as f64);
            }
        }
        let step = grid.step_secs();
        let smooth = std::array::from_fn(|b| iso::smoothing(b, step));
        let unit = std::array::from_fn(|i| {
            let z = i as f64 + 0.5;
            iso::core_unit(
                (0..CORE)
                    .find(|k| z <= iso::upper_bark(*k))
                    .unwrap_or(CORE - 1),
            )
        });
        let poles = channels(step)
            .iter()
            .map(|fc| {
                let r = sva_formula::exp(-std::f64::consts::PI * fc / Q * step);
                let (s, c) = libm::sincos(std::f64::consts::TAU * fc * step);
                C64::new(r * c, r * s)
            })
            .collect();
        Ear {
            grid,
            spl_db,
            response,
            smooth,
            pa2: 8e-10 * libm::pow(10.0, spl_db / 10.0),
            unit,
            lowpass: sva_formula::exp(-std::f64::consts::TAU * LOWPASS_HZ * step),
            poles,
        }
    }

    pub fn sets(&self) -> usize {
        2 + self.poles.len()
    }

    /// Each frame's third-octave mean squares off its power per bin, `[m * THIRDS + b]`.
    pub fn thirds(&self, p: &[f64]) -> Vec<f64> {
        let bins = self.grid.bins();
        let frames = p.len() / bins;
        let mut out = vec![0.0; frames * THIRDS];
        for m in 0..frames {
            let row = &p[m * bins..(m + 1) * bins];
            for b in 0..THIRDS {
                out[m * THIRDS + b] = self.response[b * bins..(b + 1) * bins]
                    .iter()
                    .zip(row)
                    .map(|(h, x)| h * x)
                    .sum();
            }
        }
        out
    }

    fn depth_unit() -> f64 {
        20.0 / std::f64::consts::LN_10 * libm::pow(10.0, DEPTH_JND_DB / 20.0)
    }

    /// The reading of a transform's power per bin, `p[m * bins + j]`.
    pub fn forward(&self, p: &[f64]) -> Forward {
        let frames = p.len() / self.grid.bins();
        let mut smoothed = self.thirds(p);
        for b in 0..THIRDS {
            let a = self.smooth[b];
            for _ in 0..3 {
                let mut y = 0.0;
                for m in 0..frames {
                    y = a * y + (1.0 - a) * smoothed[m * THIRDS + b];
                    smoothed[m * THIRDS + b] = y;
                }
            }
        }
        let mut core = vec![0.0; frames * CORE];
        for m in 0..frames {
            let level = self.levels(&smoothed[m * THIRDS..(m + 1) * THIRDS]);
            core[m * CORE..(m + 1) * CORE].copy_from_slice(&iso::core(&level));
        }
        Decay::of(self.grid.step_secs()).run(&mut core);
        let mut specific = vec![0.0; frames * BARKS];
        let mut rows = [0.0; ROWS];
        for m in 0..frames {
            let nm: [f64; CORE] = core[m * CORE..(m + 1) * CORE]
                .try_into()
                .expect("a frame's core bands");
            iso::slopes(&nm, &mut rows);
            for i in 0..BARKS {
                specific[m * BARKS + i] = rows[10 * i..10 * i + 10].iter().sum::<f64>() / 10.0;
            }
        }
        let sets = self.sets();
        let mut elements = vec![0.0; sets * frames * BARKS];
        let level_e: Vec<f64> = specific
            .iter()
            .enumerate()
            .map(|(k, n)| equivalent(*n, self.unit[k % BARKS]))
            .collect();
        let at = |s: usize, m: usize, i: usize| (s * frames + m) * BARKS + i;
        let mut low = vec![0.0; frames * BARKS];
        let a = self.lowpass;
        for i in 0..BARKS {
            let mut y = 0.0;
            for m in 0..frames {
                let e = level_e[m * BARKS + i];
                y = a * y + (1.0 - a) * e;
                low[m * BARKS + i] = y;
                elements[at(0, m, i)] = e / LEVEL_JND_DB;
                elements[at(1, m, i)] = y / LEVEL_JND_DB;
            }
        }
        let unit = Ear::depth_unit();
        for (c, p) in self.poles.iter().enumerate() {
            let g = 1.0 - p.abs();
            for i in 0..BARKS {
                let mut y = C64::new(0.0, 0.0);
                for m in 0..frames {
                    let x = level_e[m * BARKS + i] - low[m * BARKS + i];
                    y = *p * y + C64::new(g * x, 0.0);
                    elements[at(2 + c, m, i)] = 2.0 * y.abs() / unit;
                }
            }
        }
        Forward {
            frames,
            elements,
            specific,
        }
    }

    fn levels(&self, smoothed: &[f64]) -> [f64; THIRDS] {
        std::array::from_fn(|b| iso::level(self.pa2 * smoothed[b]))
    }
}

/// The level above threshold ISO 532-1's core law gives `n` at, unit `u`.
fn equivalent(n: f64, u: f64) -> f64 {
    let q = 1.0 + n.max(0.0) / u;
    let q4 = q * q * q * q;
    10.0 * libm::log10(4.0 * q4 - 3.0)
}

/// What `internal` reads off samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Internal {
    pub rate: f64,
    pub start_secs: f64,
    pub spl_db: f64,
    pub frames: usize,
    pub sets: usize,
    pub specific: Vec<f64>,
    pub elements: Vec<f64>,
}

impl Internal {
    /// Frame `m`'s elements of set `s`.
    pub fn row(&self, s: usize, m: usize) -> &[f64] {
        let at = (s * self.frames + m) * BARKS;
        &self.elements[at..at + BARKS]
    }
}

/// Every channel's samples averaged, one silent sample where there are none.
pub fn mono(planes: &[&[f64]]) -> Vec<f64> {
    let len = super::held(planes);
    let count = planes.len() as f64;
    match len {
        0 => vec![0.0],
        _ => (0..len)
            .map(|n| planes.iter().map(|p| p[n]).sum::<f64>() / count)
            .collect(),
    }
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64, spl_db: f64) -> Internal {
    let x = mono(planes);
    let grid = Grid::of(sr, x.len());
    let ear = Ear::of(grid, spl_db);
    let f = ear.forward(&bin_power(&x, &grid));
    Internal {
        rate: sr,
        start_secs: start,
        spl_db,
        frames: f.frames,
        sets: ear.sets(),
        specific: f.specific,
        elements: f.elements,
    }
}
