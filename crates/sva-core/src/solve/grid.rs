// Concern: the coarse grid: octave bands by 48 ms frames, sine and band templates, distance, phaseless resynthesis | Non-concern: the fit (first.rs) | IO: (Spectrogram) -> Grid, samples

use sva_engine::fft;

use super::add::{Spectrogram, floor_db, span_hz};
use super::atoms::phase;
use crate::json::{NONE, list, num};

pub const MODEL: &str = "the coarse grid, the smallest thumbnail of the clip: the analysis \
     spectrogram (add.rs: 64 ms periodic Hann, 1 ms hop) read only at hops whose window lies \
     wholly inside the clip, as zero padding past an edge is a jump the clip does not hold; \
     pooled to frames of three quarter-windows (48 ms at 64 ms), the quarter-window being the \
     hop at which the squared Hann sums to a constant, three the whole count nearest 50 ms: \
     each frame the mean power of the hops it holds; and summed over octave bands, edges at \
     1 kHz times powers of two inside the feasible set (Cam 1.8 to 38.9, under the Nyquist), \
     an end piece under half an octave joining its neighbour, so 8 bands at 16 kHz and above; \
     a unit-amplitude sine at f puts |W(b - f)|^2 / 4 in bin b (W the window's transform, in \
     closed form from three Dirichlet kernels, sidelobes included), summed over each band's \
     bins; a band [lo, hi) of unit power (variance one half, add.rs's noise level) puts its \
     expected periodogram in bin b: the sum over lags m of the window's autocorrelation \
     times the band's, sin(2 pi hi m / fs) - sin(2 pi lo m / fs) over 2 pi (hi - lo) m / fs, \
     halved, times cos(2 pi b m / n), exactly, by one transform of the lags folded mod n, so \
     its leakage over its edges lands in its neighbours as the clip's does; a clip shorter \
     than the window is read at every hop, none lying inside it; a \
     cell is audible where the clip's or the fit's band is over the threshold at the band's \
     geometric centre, its distance |fit - clip| dB with each held at that threshold; \
     phaseless resynthesis: each band's power spread flat over its bins, each bin at the \
     square root of its power, every phase drawn, inverse transformed, scaled to the frame's \
     mean square by Parseval and laid at its centre under the Hann window over the root of \
     the windows' summed squares there, so frames whose phases are independent add in power \
     to the frames' (Griffin and Lim 1984, IEEE T-ASSP 32(2) 236-243, their overlap-add, \
     normalised for incoherent frames)";

/// The quarter-windows one frame pools, as `MODEL` states.
const QUARTERS: usize = 3;

/// The power of `spectrogram` pooled to frames, as `MODEL` states.
pub struct Grid {
    pub rate: f64,
    pub n: usize,
    pub long: usize,
    pub hop: usize,
    /// The analysis hop the frames pool, in samples.
    pub step: usize,
    /// The first analysis hop read, the first whose window lies inside the clip.
    pub first: usize,
    pub sum_w: f64,
    pub sum_w2: f64,
    /// The window's autocorrelation at each lag from nought.
    pub autocorrelation: Vec<f64>,
    /// The octave bands, each its Hz, as `MODEL` states.
    pub bands: Vec<(f64, f64)>,
    pub power: Vec<Vec<f64>>,
}

impl Grid {
    /// `spectrogram` of a clip `len` samples long, as `MODEL` states.
    pub fn of(spectrogram: &Spectrogram, len: usize) -> Grid {
        let step = spectrogram.hop;
        let pooled =
            QUARTERS * (libm::round(spectrogram.long as f64 / 4.0 / step as f64) as usize).max(1);
        let half = spectrogram.long / 2;
        let hops = spectrogram.power.len();
        let inside = half.div_ceil(step)..(len.saturating_sub(half) / step + 1).min(hops);
        let inside = match inside.is_empty() {
            true => 0..hops,
            false => inside,
        };
        let first = inside.start;
        let power = spectrogram.power[inside]
            .chunks(pooled)
            .map(|hops| {
                (0..hops[0].len())
                    .map(|b| hops.iter().map(|h| h[b]).sum::<f64>() / hops.len() as f64)
                    .collect()
            })
            .collect();
        let long = spectrogram.long;
        let window: Vec<f64> = (0..long)
            .map(|k| 0.5 - 0.5 * libm::cos(std::f64::consts::TAU * k as f64 / long as f64))
            .collect();
        let autocorrelation = (0..long)
            .map(|m| (0..long - m).map(|k| window[k] * window[k + m]).sum())
            .collect();
        let mut grid = Grid {
            rate: spectrogram.rate,
            n: spectrogram.n,
            long,
            hop: pooled * step,
            step,
            first,
            sum_w: spectrogram.sum_w,
            sum_w2: spectrogram.sum_w2,
            autocorrelation,
            bands: Vec::new(),
            power,
        };
        grid.bands = grid.octaves();
        grid
    }

    pub fn with(&self, power: Vec<Vec<f64>>) -> Grid {
        Grid {
            autocorrelation: self.autocorrelation.clone(),
            bands: self.bands.clone(),
            power,
            ..*self
        }
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

    /// Half a frame either side of its centre.
    pub fn frame_span(&self, f: usize) -> (f64, f64) {
        let centre = self.centre_secs(f);
        (
            centre - self.frame_secs() / 2.0,
            centre + self.frame_secs() / 2.0,
        )
    }

    /// The sample its hops' windows centre on, on average.
    pub fn centre(&self, f: usize) -> usize {
        self.first * self.step + f * self.hop + (self.hop - self.step) / 2
    }

    pub fn centre_secs(&self, f: usize) -> f64 {
        self.centre(f) as f64 / self.rate
    }

    /// The padded bins a Hann main lobe covers either side of its peak.
    pub fn lobe(&self) -> usize {
        (2 * self.n).div_ceil(self.long)
    }

    /// The bin power a sine of unit amplitude puts in bin `b`, at `hz`.
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

    pub fn bins_of(&self, (lo, hi): (f64, f64)) -> std::ops::Range<usize> {
        let at = |hz: f64| (libm::ceil(self.bin_of(hz)) as usize).min(self.bins());
        at(lo)..at(hi)
    }

    /// The total bin power a band of unit power puts over all bins.
    pub fn band_unit(&self) -> f64 {
        self.n as f64 * self.sum_w2 / 4.0
    }

    /// A band's level in dBFS whose bins hold `power` in all.
    pub fn band_db(&self, power: f64) -> f64 {
        10.0 * libm::log10(power / self.band_unit())
    }

    /// The power a band's bins hold in all at `db` dBFS.
    pub fn band_power(&self, db: f64) -> f64 {
        libm::pow(10.0, db / 10.0) * self.band_unit()
    }

    /// The octave bands, each its Hz.
    pub fn bands(&self) -> Vec<(f64, f64)> {
        self.bands.clone()
    }

    /// The octave bands, as `MODEL` states.
    fn octaves(&self) -> Vec<(f64, f64)> {
        let (lo, hi) = span_hz(self.rate);
        let mut edges = vec![lo];
        edges.extend(
            (-10..10)
                .map(|k| 1000.0 * libm::pow(2.0, f64::from(k)))
                .filter(|e| lo < *e && *e < hi),
        );
        edges.push(hi);
        let octaves = |a: f64, b: f64| libm::log2(b / a);
        if edges.len() > 2 && octaves(edges[0], edges[1]) < 0.5 {
            edges.remove(1);
        }
        let last = edges.len() - 1;
        if edges.len() > 2 && octaves(edges[last - 1], edges[last]) < 0.5 {
            edges.remove(last - 1);
        }
        edges
            .windows(2)
            .map(|e| (e[0], e[1]))
            .filter(|e| !self.bins_of(*e).is_empty())
            .collect()
    }

    /// Each band's geometric centre in Hz.
    pub fn band_centres(&self) -> Vec<f64> {
        self.bands()
            .iter()
            .map(|(lo, hi)| libm::sqrt(lo * hi))
            .collect()
    }

    /// The threshold at each band's centre, the floor a cell is audible over.
    pub fn band_floors(&self, spl_db: f64) -> Vec<f64> {
        self.band_centres()
            .iter()
            .map(|hz| floor_db(*hz, spl_db))
            .collect()
    }

    /// Bin powers `frame` summed over each band.
    pub fn pooled(&self, frame: &[f64]) -> Vec<f64> {
        self.bands
            .iter()
            .map(|b| frame[self.bins_of(*b)].iter().sum())
            .collect()
    }

    /// Band powers `powers` spread flat over each band's bins, nought outside every band.
    pub fn spread(&self, powers: &[f64]) -> Vec<f64> {
        let mut frame = vec![0.0; self.bins()];
        for (band, p) in self.bands.iter().zip(powers) {
            let bins = self.bins_of(*band);
            let each = p / bins.len() as f64;
            frame[bins].iter_mut().for_each(|v| *v = each);
        }
        frame
    }

    /// This grid at the coarse resolution: each frame's band powers spread flat.
    pub fn broad(&self) -> Grid {
        self.with(
            self.power
                .iter()
                .map(|f| self.spread(&self.pooled(f)))
                .collect(),
        )
    }

    /// The power a sine of unit amplitude at `hz` puts in each band, sidelobes included.
    pub fn sine_bands(&self, hz: f64) -> Vec<f64> {
        self.bands
            .iter()
            .map(|b| self.bins_of(*b).map(|k| self.sine(hz, k)).sum())
            .collect()
    }

    /// The power a band `(lo, hi)` of unit power puts in each band, its leakage over its
    /// edges included, as `MODEL` states.
    pub fn noise_bands(&self, (lo, hi): (f64, f64)) -> Vec<f64> {
        let turn = std::f64::consts::TAU / self.rate;
        let mut re = vec![0.0; self.n];
        for (m, r) in self.autocorrelation.iter().enumerate() {
            let rho = match m {
                0 => 0.5,
                m => {
                    let m = m as f64;
                    0.5 * (libm::sin(turn * hi * m) - libm::sin(turn * lo * m))
                        / (turn * (hi - lo) * m)
                }
            };
            re[m % self.n] += r * rho;
            if m > 0 {
                re[(self.n - m % self.n) % self.n] += r * rho;
            }
        }
        let mut im = vec![0.0; self.n];
        fft::fft_with(&mut re, &mut im, &fft::twiddles(self.n));
        let bins: Vec<f64> = re[..self.bins()].iter().map(|p| p.max(0.0)).collect();
        self.pooled(&bins)
    }

    /// Each frame's band levels in dBFS.
    pub fn band_db_map(&self) -> Vec<Vec<f64>> {
        self.power
            .iter()
            .map(|frame| {
                self.pooled(frame)
                    .iter()
                    .map(|p| self.band_db(*p))
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

/// The clip's and the fit's band maps, their distance per cell, per frame and over audible
/// cells, as JSON beside that mean, as `MODEL` states.
pub fn compared(clip: &Grid, fit: &Grid, spl_db: f64) -> (String, f64) {
    let floors = clip.band_floors(spl_db);
    let (heard, fitted) = (clip.band_db_map(), fit.band_db_map());
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
    let edges = |b: &(f64, f64)| format!("[{}, {}]", num(b.0), num(b.1));
    let json = format!(
        "{{ \"rate_hz\": {}, \"frame_secs\": {}, \"first_centre_secs\": {}, \"band_hz\": {}, \
         \"band_edges_hz\": {}, \"floor_db\": {}, \"clip_db\": {}, \"fit_db\": {}, \
         \"fit_less_clip_db\": {}, \"frame_mean_abs_db\": {}, \"mean_abs_db\": {} }}",
        num(clip.rate),
        num(clip.frame_secs()),
        num(clip.centre_secs(0)),
        list(&clip.band_centres(), |v| num(*v)),
        list(&clip.bands(), edges),
        list(&floors, |v| num(*v)),
        map(&heard),
        map(&fitted),
        map(&apart),
        list(&per_frame, |v| v.map_or_else(|| NONE.to_string(), num)),
        num(mean)
    );
    (json, mean)
}
