// Concern: measures BS.1770 K-weighted loudness across a node's components | Non-concern: per-component level (envelope.rs), designing a biquad (biquad.rs) | IO: (planes, sample rate) -> Loudness

use crate::biquad::{Coeffs, State};

/// K-weighting as the analog prototypes BS.1770 bilinear-transforms, not the cookbook shapes:
/// stating them as a design is what makes every rate but 48 kHz right too.
const SHELF_HZ: f64 = 1681.97;
const SHELF_Q: f64 = 0.70718;
const SHELF_GAIN_DB: f64 = 3.99984;
const SHELF_MID: f64 = 0.499_666_774_154_541_6;
const HIGHPASS_HZ: f64 = 38.1355;
const HIGHPASS_Q: f64 = 0.50033;

/// At or under it the shelf stands at or past Nyquist.
pub const LEAST_RATE_HZ: f64 = 2.0 * SHELF_HZ;

fn shelf(sr: f64) -> Coeffs {
    let k = (std::f64::consts::PI * SHELF_HZ / sr).tan();
    let high = 10f64.powf(SHELF_GAIN_DB / 20.0);
    let mid = high.powf(SHELF_MID);
    let a0 = 1.0 + k / SHELF_Q + k * k;
    Coeffs {
        b0: (high + mid * k / SHELF_Q + k * k) / a0,
        b1: 2.0 * (k * k - high) / a0,
        b2: (high - mid * k / SHELF_Q + k * k) / a0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / SHELF_Q + k * k) / a0,
    }
}

fn highpass(sr: f64) -> Coeffs {
    let k = (std::f64::consts::PI * HIGHPASS_HZ / sr).tan();
    let a0 = 1.0 + k / HIGHPASS_Q + k * k;
    Coeffs {
        b0: 1.0,
        b1: -2.0,
        b2: 1.0,
        a1: 2.0 * (k * k - 1.0) / a0,
        a2: (1.0 - k / HIGHPASS_Q + k * k) / a0,
    }
}

const OFFSET_DB: f64 = -0.691;
const ABSOLUTE_GATE: f64 = -70.0;
const INTEGRATED_GATE_LU: f64 = -10.0;

const RANGE_GATE_LU: f64 = -20.0;

const MOMENTARY_SECS: f64 = 0.4;
const SHORT_TERM_SECS: f64 = 3.0;

const STEP_SECS: f64 = 0.1;

pub const PEAK_NOTE: &str = "sample peak, not true peak: no oversampling exists here, so an inter-sample peak above \
     this figure is not measured";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoudnessFrame {
    /// The block's START, the global seconds every other representation reports.
    pub t: f64,
    pub lufs: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Loudness {
    pub integrated_lufs: Option<f64>,
    pub range_lu: Option<f64>,
    pub momentary_max_lufs: Option<f64>,
    pub short_term_max_lufs: Option<f64>,
    pub sample_peak: f64,
    pub sample_peak_dbfs: Option<f64>,
    pub peak_note: &'static str,
    pub momentary: Vec<LoudnessFrame>,
    pub short_term: Vec<LoudnessFrame>,
}

/// A rate with no K-weighting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateUnderWeighting {
    pub rate_hz: f64,
    pub least_hz: f64,
}

/// Every component weighs 1.0: BS.1770's 1.41 surround lift needs a channel identity an
/// indexed component does not carry.
pub fn analyze(
    planes: &[&[f64]],
    sr: f64,
    start_secs: f64,
) -> Result<Loudness, RateUnderWeighting> {
    if sr <= LEAST_RATE_HZ {
        return Err(RateUnderWeighting {
            rate_hz: sr,
            least_hz: LEAST_RATE_HZ,
        });
    }
    let sample_peak = planes
        .iter()
        .flat_map(|p| p.iter())
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    let scale = headroom(sample_peak);
    let squares = weighted_squares(planes, sr, scale);
    let at = Blocks {
        sr,
        start_secs,
        lift_db: 20.0 * scale.log10(),
    };

    let momentary = blocks(&squares, &at, MOMENTARY_SECS);
    let short_term = blocks(&squares, &at, SHORT_TERM_SECS);

    Ok(Loudness {
        integrated_lufs: gated_mean(&momentary, INTEGRATED_GATE_LU),
        range_lu: range(&short_term),
        momentary_max_lufs: peak_of(&momentary),
        short_term_max_lufs: peak_of(&short_term),
        sample_peak,
        sample_peak_dbfs: (sample_peak > 0.0).then(|| 20.0 * sample_peak.log10()),
        peak_note: PEAK_NOTE,
        momentary,
        short_term,
    })
}

/// A power of two at the peak: dividing is exact; only 3000 dB under it does a square underflow.
fn headroom(peak: f64) -> f64 {
    match peak > 0.0 && peak.is_finite() {
        true => {
            let exponent = peak
                .log2()
                .ceil()
                .clamp(f64::from(f64::MIN_EXP - 1), f64::from(f64::MAX_EXP - 1));
            2f64.powi(exponent as i32)
        }
        false => 1.0,
    }
}

fn weighted_squares(planes: &[&[f64]], sr: f64, scale: f64) -> Vec<f64> {
    let samples = planes.first().map_or(0, |p| p.len());
    let mut out = vec![0.0; samples];
    let (shelf, cut) = (shelf(sr), highpass(sr));
    for plane in planes {
        let (mut first, mut second) = (State::default(), State::default());
        for (sum, x) in out.iter_mut().zip(plane.iter()) {
            let k = second.step(&cut, first.step(&shelf, *x / scale));
            *sum += k * k;
        }
    }
    out
}

struct Blocks {
    sr: f64,
    start_secs: f64,
    lift_db: f64,
}

/// A block sums its own samples, piece by piece between edges, whatever loud stretch precedes it.
fn blocks(squares: &[f64], at: &Blocks, block_secs: f64) -> Vec<LoudnessFrame> {
    let n = (block_secs * at.sr).round() as usize;
    let step = ((STEP_SECS * at.sr).round() as usize).max(1);
    if n == 0 || squares.len() < n {
        return Vec::new();
    }
    let starts: Vec<usize> = (0..=(squares.len() - n)).step_by(step).collect();
    let mut edges: Vec<usize> = starts.iter().flat_map(|k| [*k, k + n]).collect();
    edges.sort_unstable();
    edges.dedup();
    let pieces: Vec<f64> = edges
        .windows(2)
        .map(|w| squares[w[0]..w[1]].iter().sum())
        .collect();
    let piece = |sample: usize| edges.binary_search(&sample).expect("a block edge");
    starts
        .iter()
        .map(|&k| {
            let power = pieces[piece(k)..piece(k + n)].iter().sum::<f64>() / n as f64;
            LoudnessFrame {
                t: at.start_secs + k as f64 / at.sr,
                lufs: level(power) + at.lift_db,
            }
        })
        .collect()
}

fn level(power: f64) -> f64 {
    if power <= 0.0 {
        return f64::NEG_INFINITY;
    }
    OFFSET_DB + 10.0 * power.log10()
}

/// The mean runs on POWER, not on the levels: averaging decibels is not averaging loudness.
fn mean_above(frames: &[LoudnessFrame], threshold: f64) -> Option<f64> {
    let kept: Vec<f64> = frames
        .iter()
        .map(|f| f.lufs)
        .filter(|l| *l > threshold)
        .collect();
    if kept.is_empty() {
        return None;
    }
    let from = kept.iter().copied().fold(OFFSET_DB, f64::max);
    let power: f64 = kept.iter().map(|l| 10f64.powf((l - from) / 10.0)).sum();
    Some(from + 10.0 * (power / kept.len() as f64).log10())
}

/// Both relative gates measure down from the ABSOLUTE-gated mean, never from each other.
fn gated_mean(frames: &[LoudnessFrame], relative_lu: f64) -> Option<f64> {
    mean_above(frames, mean_above(frames, ABSOLUTE_GATE)? + relative_lu)
}

/// EBU Tech 3342, whose gate is deliberately not the integrated measure's.
fn range(short_term: &[LoudnessFrame]) -> Option<f64> {
    let floor = mean_above(short_term, ABSOLUTE_GATE)? + RANGE_GATE_LU;
    let mut kept: Vec<f64> = short_term
        .iter()
        .map(|f| f.lufs)
        .filter(|l| *l > ABSOLUTE_GATE && *l > floor)
        .collect();
    if kept.is_empty() {
        return None;
    }
    kept.sort_by(f64::total_cmp);
    Some(percentile(&kept, 0.95) - percentile(&kept, 0.10))
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let at = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[at.min(sorted.len() - 1)]
}

fn peak_of(frames: &[LoudnessFrame]) -> Option<f64> {
    frames
        .iter()
        .map(|f| f.lufs)
        .filter(|l| l.is_finite())
        .reduce(f64::max)
}
