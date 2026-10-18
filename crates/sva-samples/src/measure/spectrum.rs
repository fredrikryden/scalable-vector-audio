// Concern: resolves every channel's samples into magnitudes, third-octave bands and peaks | Non-concern: naming a peak as a note (pitch.rs), choosing the window | IO: (planes, sample rate) -> Spectrum

use crate::fft::fft;
pub use crate::measure::peaks::{Peak, db, peaks};

pub const MAX_FRAME: usize = 8192;
pub const MIN_FRAME: usize = 64;

/// One FFT's memory, which is all that bounds a caller's own frame.
pub const MAX_PINNED_FRAME: usize = 1 << 20;

/// A caller checks this against [`MAX_PINNED_FRAME`]: a shrunk frame is not a pinned one.
pub fn pinned_frame(secs: f64, sample_rate: f64) -> usize {
    let asked = (secs * sample_rate).round().max(0.0);
    if !asked.is_finite() || asked > MAX_PINNED_FRAME as f64 {
        return MAX_PINNED_FRAME * 2;
    }
    (asked as usize).next_power_of_two().max(MIN_FRAME)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    pub lo_hz: f64,
    pub hi_hz: f64,
    pub db: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum {
    pub frame_size: usize,
    pub frames: usize,
    pub resolution_hz: f64,
    pub rms: f64,
    pub centroid_hz: f64,
    pub rolloff85_hz: f64,
    pub bands: Vec<Band>,
    pub peaks: Vec<Peak>,
}

pub fn analyze(
    planes: &[&[f64]],
    sample_rate: f64,
    max_peaks: usize,
    frame_secs: Option<f64>,
) -> Spectrum {
    let Framed {
        mags,
        bin_hz,
        frame_size,
        frames,
        spanned,
    } = magnitudes_of(planes, sample_rate, frame_secs);
    let power = one_sided(&mags);
    let total: f64 = power.iter().sum();

    let centroid_hz = if total > 0.0 {
        power
            .iter()
            .enumerate()
            .map(|(k, p)| k as f64 * bin_hz * p)
            .sum::<f64>()
            / total
    } else {
        0.0
    };

    let mut running = 0.0;
    let mut rolloff85_hz = 0.0;
    for (k, p) in power.iter().enumerate() {
        running += p;
        if total > 0.0 && running >= 0.85 * total {
            rolloff85_hz = k as f64 * bin_hz;
            break;
        }
    }

    let rms = crate::measure::envelope::level(planes);

    Spectrum {
        frame_size,
        frames,
        resolution_hz: bin_hz,
        rms,
        centroid_hz,
        rolloff85_hz,
        bands: bands(
            &power,
            1.5 * frame_size as f64 / spanned as f64,
            bin_hz,
            sample_rate,
        ),
        peaks: crate::measure::peaks::resolved(&mags, bin_hz, spanned, max_peaks),
    }
}

/// 0 and Nyquist alone have no mirror folded in.
fn one_sided(mags: &[f64]) -> Vec<f64> {
    let last = mags.len() - 1;
    mags.iter()
        .enumerate()
        .map(|(k, m)| match k == 0 || k == last {
            true => m * m / 2.0,
            false => m * m,
        })
        .collect()
}

/// A full-scale sine reads 1.0 at its own bin. Without `frame_secs` the size follows the
/// range, so two ranges land on two bin grids.
pub fn magnitudes(
    samples: &[f64],
    sample_rate: f64,
    frame_secs: Option<f64>,
) -> (Vec<f64>, f64, usize, usize) {
    let f = magnitudes_of(&[samples], sample_rate, frame_secs);
    (f.mags, f.bin_hz, f.frame_size, f.frames)
}

pub struct Framed {
    pub mags: Vec<f64>,
    pub bin_hz: f64,
    pub frame_size: usize,
    pub frames: usize,
    /// The samples the Hann spans, fewer than `frame_size` where the range is.
    pub spanned: usize,
}

pub fn magnitudes_of(planes: &[&[f64]], sample_rate: f64, frame_secs: Option<f64>) -> Framed {
    let len = crate::measure::held(planes);
    let frame_size = match frame_secs {
        Some(secs) => pinned_frame(secs, sample_rate).min(MAX_PINNED_FRAME),
        None => len.next_power_of_two().clamp(MIN_FRAME, MAX_FRAME),
    };
    let bins = frame_size / 2 + 1;
    let spanned = len.clamp(1, frame_size);
    let window = crate::stft::hann_periodic(spanned);

    let starts = starts(len, frame_size);
    let mut acc = vec![0f64; bins];
    for start in &starts {
        for samples in planes {
            let mut re = vec![0f64; frame_size];
            let mut im = vec![0f64; frame_size];
            for (n, w) in window.iter().enumerate() {
                re[n] = samples.get(start + n).copied().unwrap_or(0.0) * w;
            }
            fft(&mut re, &mut im);
            for (k, a) in acc.iter_mut().enumerate() {
                *a += re[k] * re[k] + im[k] * im[k];
            }
        }
    }

    let scale = 4.0 / spanned as f64;
    let averaged = (starts.len() * planes.len()) as f64;
    Framed {
        mags: acc.iter().map(|p| (p / averaged).sqrt() * scale).collect(),
        bin_hz: sample_rate / frame_size as f64,
        frame_size,
        frames: starts.len(),
        spanned,
    }
}

/// Whole frames, first sample to last, evenly at most half a frame apart.
pub(crate) fn starts(len: usize, frame: usize) -> Vec<usize> {
    let Some(past) = len.checked_sub(frame).filter(|past| *past > 0) else {
        return vec![0];
    };
    let gaps = past.div_ceil(frame / 2);
    (0..=gaps)
        .map(|i| (2 * i * past + gaps) / (2 * gaps))
        .collect()
}

/// Third-octave from 20 Hz up.
pub fn third_octave_edges(sample_rate: f64) -> Vec<(f64, f64)> {
    let nyquist = sample_rate / 2.0;
    let ratio = 2f64.powf(1.0 / 3.0);
    let mut out = Vec::new();
    let mut lo = 20.0;
    while lo < nyquist {
        let hi = (lo * ratio).min(nyquist);
        out.push((lo, hi));
        lo = hi;
    }
    out
}

/// Over `enbw`, the noise bandwidth in bins, a band reads as the one sine carrying its power.
fn bands(power: &[f64], enbw: f64, bin_hz: f64, sample_rate: f64) -> Vec<Band> {
    let mut out = Vec::new();
    for (lo, hi) in third_octave_edges(sample_rate) {
        let sum: f64 = power
            .iter()
            .enumerate()
            .filter(|(k, _)| {
                let hz = *k as f64 * bin_hz;
                hz >= lo && hz < hi
            })
            .map(|(_, p)| p)
            .sum();
        out.push(Band {
            lo_hz: lo,
            hi_hz: hi,
            db: db((sum / enbw).sqrt()),
        });
    }
    out
}
