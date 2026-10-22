// Concern: the proxy's fixed ear: band thresholds, spread, loudness, sharpness weights, frame taps, a sine's band table | Non-concern: what the axes make of it | IO: (rate) -> Ear; (E) -> N'

use std::sync::{Arc, Mutex, OnceLock};

use sva_samples::measure::heard::{self as measured, BANDS};

use super::tape::{Arr, BandTable, Tape, V};

pub const LEVEL_DB: f64 = 70.0;
/// N' scale: step 1 calibrated 1.87 to 2.05 against mosqito's Zwicker per clip.
pub const LOUDNESS_SCALE: f64 = 2.0;
/// The loudness integration time: step 1 calibrated 30 to 50 ms per clip.
pub const INTEGRATION_SECS: f64 = 0.04;
const PA_REF: f64 = 20e-6;
const TABLE_LO_HZ: f64 = 20.0;
const TABLE_HI_HZ: f64 = 24_000.0;
const TABLE_PER_OCTAVE: f64 = 96.0;

pub struct Ear {
    pub rate: f64,
    pub n: usize,
    pub hop_ms: usize,
    pub taps: Vec<f64>,
    pub quiet: Vec<f64>,
    pub spread: Arc<Arr>,
    pub pool: Arc<Arr>,
    pub sharp: Vec<f64>,
    pub table: Arc<BandTable>,
    pub bin_hz: Vec<f64>,
    pub bin_band: Arc<Arr>,
    pub ceiling_hz: f64,
}

fn threshold_db(hz: f64) -> f64 {
    let f = hz.max(20.0) / 1000.0;
    3.64 * libm::pow(f, -0.8) - 6.5 * sva_formula::exp(-0.6 * (f - 3.3) * (f - 3.3))
        + 1e-3 * libm::pow(f, 4.0)
}

fn build(rate: f64) -> Ear {
    let (n, window, hop) = measured::sizes(rate);
    let w = measured::hann_periodic(window);
    let per_ms = (rate / 1000.0) as usize;
    let mut taps: Vec<f64> = w
        .chunks(per_ms)
        .map(|c| c.iter().map(|x| x * x).sum())
        .collect();
    let total: f64 = taps.iter().sum();
    taps.iter_mut().for_each(|t| *t /= total);
    let (of, centres) = measured::bins(n, rate);
    let quiet = centres
        .iter()
        .map(|c| PA_REF * PA_REF * libm::pow(10.0, threshold_db(*c).clamp(-10.0, 80.0) / 10.0))
        .collect();
    let mut spread = vec![0.0; BANDS * BANDS];
    for i in 0..BANDS {
        for j in 0..BANDS {
            let dz = (i as f64 - j as f64) * 0.5;
            spread[i * BANDS + j] = match dz >= 0.0 {
                true => libm::pow(10.0, -12.0 * dz / 10.0),
                false => libm::pow(10.0, 27.0 * dz / 10.0),
            };
        }
    }
    let mut pool = vec![0.0; BANDS / 2 * BANDS];
    for i in 0..BANDS / 2 {
        pool[i * BANDS + 2 * i] = 0.5;
        pool[i * BANDS + 2 * i + 1] = 0.5;
    }
    let sharp = (0..BANDS)
        .map(|b| {
            let z = b as f64 / 2.0;
            let g = if z < 15.8 {
                1.0
            } else {
                0.066 * sva_formula::exp(0.171 * z)
            };
            g * (z + 0.25)
        })
        .collect();
    let octaves = libm::log2(TABLE_HI_HZ / TABLE_LO_HZ);
    let points = (TABLE_PER_OCTAVE * octaves) as usize + 1;
    let step = octaves / (points - 1) as f64;
    let mut rows = Vec::with_capacity(points * BANDS);
    for i in 0..points {
        let hz = TABLE_LO_HZ * libm::exp2(i as f64 * step);
        rows.extend_from_slice(&measured::unit_sine(hz, rate));
    }
    let ceiling_hz = sva_samples::PSYCHOACOUSTIC_V1.ceiling(rate as u32);
    let table = Arc::new(BandTable::new(
        libm::log2(TABLE_LO_HZ),
        step,
        BANDS,
        rows,
        ceiling_hz,
    ));
    let bins = n / 2 + 1;
    let bin_hz = (0..bins).map(|j| j as f64 * rate / n as f64).collect();
    let mut member = vec![0.0; BANDS * bins];
    for (j, b) in of.iter().enumerate() {
        member[b * bins + j] = 1.0;
    }
    Ear {
        rate,
        n,
        hop_ms: (hop as f64 * 1000.0 / rate).round() as usize,
        taps,
        quiet,
        spread: Arc::new(Arr::new(BANDS, BANDS, spread)),
        pool: Arc::new(Arr::new(BANDS / 2, BANDS, pool)),
        sharp,
        table,
        bin_hz,
        bin_band: Arc::new(Arr::new(BANDS, bins, member)),
        ceiling_hz,
    }
}

pub fn ear(rate: f64) -> Arc<Ear> {
    static EARS: OnceLock<Mutex<Vec<Arc<Ear>>>> = OnceLock::new();
    let ears = EARS.get_or_init(|| Mutex::new(Vec::new()));
    let mut held = ears.lock().expect("the ear table");
    if let Some(e) = held.iter().find(|e| e.rate == rate) {
        return Arc::clone(e);
    }
    let e = Arc::new(build(rate));
    held.push(Arc::clone(&e));
    e
}

impl Ear {
    /// `(N' [48, m], N' per Bark [24, m])` of a pattern in Pa^2.
    pub fn loudness(&self, tape: &mut Tape, e: V) -> (V, V) {
        let es = tape.matl(Arc::clone(&self.spread), e);
        let et = tape.constant(Arr::column(self.quiet.clone()));
        let ratio = tape.div(es, et);
        let half = tape.scale(ratio, 0.5);
        let inner = tape.offset(half, 0.5);
        let p = tape.num(0.23);
        let powed = tape.pow(inner, p);
        let less = tape.offset(powed, -1.0);
        let held = tape.clamp(less, 0.0, f64::INFINITY);
        let gain: Vec<f64> = self
            .quiet
            .iter()
            .map(|q| LOUDNESS_SCALE * 0.08 * libm::pow(q / (PA_REF * PA_REF), 0.23))
            .collect();
        let g = tape.constant(Arr::column(gain));
        let np = tape.mul(held, g);
        let n24 = tape.matl(Arc::clone(&self.pool), np);
        (np, n24)
    }

    pub fn power_scale(rms: f64) -> f64 {
        let s = PA_REF * libm::pow(10.0, LEVEL_DB / 20.0) / rms.max(1e-12);
        s * s
    }
}
