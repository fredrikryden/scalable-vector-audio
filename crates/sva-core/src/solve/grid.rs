// Concern: the first fit's grid: power pooled to frames, sine and band templates, Cam maps, their distance, phaseless resynthesis | Non-concern: the fit (first.rs) | IO: (Spectrogram) -> Grid, samples

use sva_engine::fft;
use sva_engine::hearing::{CAM_FIRST, cam, cam_hz};

use super::add::{Spectrogram, floor_db, span_hz};
use super::atoms::phase;
use crate::json::{NONE, list, num};

pub const MODEL: &str = "the analysis spectrogram (add.rs: 64 ms periodic Hann, 1 ms hop) \
     pooled to frames a quarter of its window apart (16 ms), the largest hop at which the \
     squared Hann window sums to a constant, so the frames tile time with no gap and no \
     ripple in weight: each frame the mean power of the hops it holds; a unit-power sine at f \
     puts |W(b - f)|^2 / 4 in bin b \
     (W the window's transform, in closed form from three Dirichlet kernels) over its main \
     lobe, the bins add.rs claims for it, its sidelobes (31 dB down and falling) neglected; a \
     band [lo, hi) of unit power puts n sum w^2 / 4 over its bins in all, flat (add.rs's \
     noise level by Parseval); the Cam map sums each whole-Cam band's bins, in that noise \
     convention; a cell is audible where the clip's or the fit's band is over the threshold at \
     the band's centre, its distance |fit - clip| dB with each held at that threshold; \
     phaseless resynthesis: each frame's bins at the square root of their power, every phase \
     drawn, inverse transformed, scaled to the frame's mean square by Parseval and laid at its \
     centre under the Hann window over the root of the windows' summed squares there, so \
     frames whose phases are independent add in power to the frames' (Griffin and Lim 1984, \
     IEEE T-ASSP 32(2) 236-243, their overlap-add, normalised for incoherent frames)";

/// The power of `spectrogram` pooled to frames, as `MODEL` states.
pub struct Grid {
    pub rate: f64,
    pub n: usize,
    pub long: usize,
    pub hop: usize,
    /// The analysis hop the frames pool, in samples.
    pub step: usize,
    pub sum_w: f64,
    pub sum_w2: f64,
    pub power: Vec<Vec<f64>>,
}

impl Grid {
    pub fn of(spectrogram: &Spectrogram) -> Grid {
        let pooled =
            (libm::round(spectrogram.long as f64 / 4.0 / spectrogram.hop as f64) as usize).max(1);
        let power = spectrogram
            .power
            .chunks(pooled)
            .map(|hops| {
                (0..hops[0].len())
                    .map(|b| hops.iter().map(|h| h[b]).sum::<f64>() / hops.len() as f64)
                    .collect()
            })
            .collect();
        Grid {
            rate: spectrogram.rate,
            n: spectrogram.n,
            long: spectrogram.long,
            hop: pooled * spectrogram.hop,
            step: spectrogram.hop,
            sum_w: spectrogram.sum_w,
            sum_w2: spectrogram.sum_w2,
            power,
        }
    }

    pub fn with(&self, power: Vec<Vec<f64>>) -> Grid {
        Grid { power, ..*self }
    }

    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    pub fn hz(&self, bin: usize) -> f64 {
        bin as f64 * self.rate / self.n as f64
    }

    pub fn bin_of(&self, hz: f64) -> f64 {
        hz * self.n as f64 / self.rate
    }

    pub fn frame_secs(&self) -> f64 {
        self.hop as f64 / self.rate
    }

    /// From its first hop's centre to the next frame's.
    pub fn frame_span(&self, f: usize) -> (f64, f64) {
        let start = (f * self.hop) as f64 / self.rate;
        (start, start + self.frame_secs())
    }

    /// The sample its hops' windows centre on, on average.
    pub fn centre(&self, f: usize) -> usize {
        f * self.hop + (self.hop - self.step) / 2
    }

    pub fn centre_secs(&self, f: usize) -> f64 {
        self.centre(f) as f64 / self.rate
    }

    /// The padded bins a Hann main lobe covers either side of its peak.
    pub fn lobe(&self) -> usize {
        (2 * self.n).div_ceil(self.long)
    }

    /// The bin power a sine of unit power (amplitude 1) puts in bin `b`, at `hz`.
    pub fn sine(&self, hz: f64, b: usize) -> f64 {
        let omega = std::f64::consts::TAU * (b as f64 - self.bin_of(hz)) / self.n as f64;
        let step = std::f64::consts::TAU / self.long as f64;
        let l = self.long as f64;
        let dirichlet = |w: f64| {
            let half = libm::sin(w / 2.0);
            let r = match half == 0.0 {
                true => l,
                false => libm::sin(l * w / 2.0) / half,
            };
            let turn = -w * (l - 1.0) / 2.0;
            (r * libm::cos(turn), r * libm::sin(turn))
        };
        let (a, c, e) = (
            dirichlet(omega),
            dirichlet(omega - step),
            dirichlet(omega + step),
        );
        let re = 0.5 * a.0 - 0.25 * (c.0 + e.0);
        let im = 0.5 * a.1 - 0.25 * (c.1 + e.1);
        (re * re + im * im) / 4.0
    }

    /// Each bin of a sine's main lobe at `hz` beside the power it puts there.
    pub fn lobe_of(&self, hz: f64) -> Vec<(usize, f64)> {
        let centre = libm::round(self.bin_of(hz)) as i64;
        let lobe = self.lobe() as i64;
        (centre - lobe..=centre + lobe)
            .filter(|b| *b >= 0 && (*b as usize) < self.bins())
            .map(|b| (b as usize, self.sine(hz, b as usize)))
            .collect()
    }

    pub fn bins_of(&self, (lo, hi): (f64, f64)) -> std::ops::Range<usize> {
        let at = |hz: f64| (libm::ceil(self.bin_of(hz)) as usize).min(self.bins());
        at(lo)..at(hi)
    }

    /// The power a band of unit power puts in each of its `bins`.
    pub fn band(&self, bins: usize) -> f64 {
        self.n as f64 * self.sum_w2 / 4.0 / bins as f64
    }

    /// A band's level in dBFS whose bins hold `power` in all.
    pub fn band_db(&self, power: f64) -> f64 {
        10.0 * libm::log10(4.0 * power / (self.n as f64 * self.sum_w2))
    }

    /// A sine's level in dBFS at `db` as the power its peak bin holds.
    pub fn floor_power(&self, db: f64) -> f64 {
        libm::pow(10.0, db / 10.0) * self.sum_w * self.sum_w / 4.0
    }

    /// The whole-Cam bands from `CAM_FIRST` under the Nyquist, each its Hz.
    pub fn cams(&self) -> Vec<(f64, f64)> {
        let (_, hi) = span_hz(self.rate);
        let edges: Vec<f64> = (0..)
            .map(|z| cam_hz(CAM_FIRST + f64::from(z)))
            .take_while(|hz| *hz <= hi)
            .collect();
        edges
            .windows(2)
            .map(|e| (e[0], e[1]))
            .filter(|e| !self.bins_of(*e).is_empty())
            .collect()
    }

    /// Each whole-Cam band's centre in Hz, as `cam_db` lists them.
    pub fn cam_centres(&self) -> Vec<f64> {
        self.cams()
            .iter()
            .map(|(lo, hi)| cam_hz((cam(*lo) + cam(*hi)) / 2.0))
            .collect()
    }

    /// The threshold at each band's centre, the floor a cell is audible over.
    pub fn cam_floors(&self, spl_db: f64) -> Vec<f64> {
        self.cam_centres()
            .iter()
            .map(|hz| floor_db(*hz, spl_db))
            .collect()
    }

    pub fn cam_db(&self) -> Vec<Vec<f64>> {
        let cams = self.cams();
        self.power
            .iter()
            .map(|frame| {
                cams.iter()
                    .map(|c| self.band_db(frame[self.bins_of(*c)].iter().sum()))
                    .collect()
            })
            .collect()
    }

    /// `len` samples whose frames hold this grid's power, every phase drawn, as `MODEL` states.
    pub fn sounded(&self, len: usize) -> Vec<f64> {
        let twiddle = fft::twiddles(self.n);
        let window: Vec<f64> = (0..self.long)
            .map(|k| 0.5 - 0.5 * libm::cos(std::f64::consts::TAU * k as f64 / self.long as f64))
            .collect();
        let from = |f: usize| self.centre(f) as i64 - (self.long / 2) as i64;
        let mut norm = vec![0.0; len];
        for f in 0..self.power.len() {
            for (k, w) in window.iter().enumerate() {
                let at = from(f) + k as i64;
                if at >= 0 && (at as usize) < len {
                    norm[at as usize] += w * w;
                }
            }
        }
        let scale = libm::sqrt(self.n as f64 / self.sum_w2);
        let mut out = vec![0.0; len];
        for (f, frame) in self.power.iter().enumerate() {
            let (mut re, mut im) = (vec![0.0; self.n], vec![0.0; self.n]);
            for (b, p) in frame.iter().enumerate() {
                let (s, c) = libm::sincos(phase(f as u64, b));
                let m = libm::sqrt(*p);
                (re[b], im[b]) = (m * c, m * s);
                if b > 0 && b < self.n - b {
                    (re[self.n - b], im[self.n - b]) = (m * c, -m * s);
                }
            }
            fft::ifft_with(&mut re, &mut im, &twiddle);
            for (k, w) in window.iter().enumerate() {
                let at = from(f) + k as i64;
                if at >= 0 && (at as usize) < len && norm[at as usize] > 0.0 {
                    out[at as usize] += w * re[k] * scale / libm::sqrt(norm[at as usize]);
                }
            }
        }
        out
    }
}

/// The clip's and the fit's Cam maps, their distance per cell, per frame and over audible
/// cells, as JSON beside that mean, as `MODEL` states.
pub fn compared(clip: &Grid, fit: &Grid, spl_db: f64) -> (String, f64) {
    let floors = clip.cam_floors(spl_db);
    let (heard, fitted) = (clip.cam_db(), fit.cam_db());
    let mut all = (0.0, 0usize);
    let mut per_frame = Vec::new();
    let mut apart = Vec::new();
    for (h, f) in heard.iter().zip(&fitted) {
        let mut frame = (0.0, 0usize);
        let row: Vec<f64> = (0..floors.len())
            .map(|z| {
                let d = f[z].max(floors[z]) - h[z].max(floors[z]);
                if h[z].max(f[z]) > floors[z] {
                    frame = (frame.0 + d.abs(), frame.1 + 1);
                }
                d
            })
            .collect();
        all = (all.0 + frame.0, all.1 + frame.1);
        per_frame.push((frame.1 > 0).then(|| frame.0 / frame.1 as f64));
        apart.push(row);
    }
    let mean = match all.1 {
        0 => 0.0,
        n => all.0 / n as f64,
    };
    let map = |m: &Vec<Vec<f64>>| {
        list(m, |row| {
            list(row, |v| match v.is_finite() {
                true => num(*v),
                false => NONE.to_string(),
            })
        })
    };
    let json = format!(
        "{{ \"rate_hz\": {}, \"frame_secs\": {}, \"cam_hz\": {}, \"floor_db\": {}, \
         \"clip_db\": {}, \"fit_db\": {}, \"fit_less_clip_db\": {}, \"frame_mean_abs_db\": {}, \
         \"mean_abs_db\": {} }}",
        num(clip.rate),
        num(clip.frame_secs()),
        list(&clip.cam_centres(), |v| num(*v)),
        list(&floors, |v| num(*v)),
        map(&heard),
        map(&fitted),
        map(&apart),
        list(&per_frame, |v| v.map_or_else(|| NONE.to_string(), num)),
        num(mean)
    );
    (json, mean)
}
