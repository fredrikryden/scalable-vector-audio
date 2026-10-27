// Concern: an excitation map at scale s sounded back as a noise vocoder, a noise band per channel shaped by its E | Non-concern: reading a map (mod.rs) | IO: (E per frame and channel, len) -> samples

use super::{Ear, decibels};
use crate::fft;
use crate::measure::hearing::{self, CAM_SPACING};

pub const MODEL: &str = "a map of E per channel per frame sounded as a noise vocoder: one white \
     noise, uniform on [-1/2, 1/2) by the seed-0 draw at each sample, split by its DFT, \
     zero-padded to the next power of two, into each channel's own band, the Cam numbers within \
     half the channel spacing of its centre (bins [lo, hi), the bands tiling the axis), each band \
     scaled to a density of 1 per Hz one-sided and shaped by sqrt(E/E_1), E linear in time \
     between frames and held past the first and last, E_1 the excitation a flat density of 1 per \
     Hz puts in the channel, full scale times G(fc) times s ERB(fc), the integral of the \
     level-free roex(p/s) at p = p51 on both sides, so a flat noise sounds back at its own \
     density; the channels summed";

impl Ear {
    /// `excitation`, frame `frames(len).0 + i` at `i`, sounded over `len` samples as `MODEL`
    /// states.
    pub fn vocoded(&self, excitation: &[Vec<f64>], len: usize) -> Vec<f64> {
        let mut out = vec![0.0; len];
        if len == 0 || excitation.is_empty() {
            return out;
        }
        let n = len.next_power_of_two();
        let twiddles = fft::twiddles(n);
        let mut re: Vec<f64> = (0..n)
            .map(|k| sva_formula::draw(0, k as i64) - 0.5)
            .collect();
        let mut im = vec![0.0; n];
        fft::fft_with(&mut re, &mut im, &twiddles);
        let unit = libm::sqrt(self.rate / (2.0 / 12.0));
        let (first, _) = self.frames(len);
        let last = excitation.len() - 1;
        let half = CAM_SPACING * self.scale / 2.0;
        let bin =
            |hz: f64| (libm::ceil(hz * n as f64 / self.rate).max(0.0) as usize).min(n / 2 + 1);
        for (c, channel) in self.channels.iter().enumerate() {
            let z = hearing::cam(channel.hz);
            let (from, to) = (
                bin(hearing::cam_hz(z - half)),
                bin(hearing::cam_hz(z + half)),
            );
            if from >= to {
                continue;
            }
            let (mut band_re, mut band_im) = (vec![0.0; n], vec![0.0; n]);
            for j in from..to {
                band_re[j] = re[j];
                band_im[j] = im[j];
                if j > 0 && j < n - j {
                    band_re[n - j] = re[n - j];
                    band_im[n - j] = im[n - j];
                }
            }
            fft::ifft_with(&mut band_re, &mut band_im, &twiddles);
            let flat = self.full_scale()
                * decibels(hearing::ear_gain_db(channel.hz))
                * self.scale
                * hearing::erb_hz(channel.hz);
            for (k, o) in out.iter_mut().enumerate() {
                let at = (k as f64 / self.rate / self.hop_secs - first as f64).max(0.0);
                let i = (libm::floor(at) as usize).min(last);
                let next = (i + 1).min(last);
                let w = (at - i as f64).clamp(0.0, 1.0);
                let e = excitation[i][c] * (1.0 - w) + excitation[next][c] * w;
                *o += unit * libm::sqrt(e.max(0.0) / flat) * band_re[k];
            }
        }
        out
    }
}
