// Concern: the judge proxy's ear and what it measures off samples: half-Bark power per frame, each note's peaks and rise | Non-concern: comparing two, the score itself | IO: (planes, rate) -> Heard

use super::score;
use crate::fft;

pub const MODEL: &str = "power per half-Bark band (48 bands, 2 z each Bark by Zwicker's \
     13 atan(0.00076 f) + 3.5 atan((f/7500)^2), the top band holding the rest) per 10 ms frame, \
     off a 2048-point transform of a 20 ms periodic Hann frame centred on each 10 ms instant, the \
     ends mirrored, each bin's power 2|X|^2/(n sum w^2) so a sine of amplitude a sums to a^2/2; \
     beside it each note of the target's own score reading lasting 80 ms or more: its rms, its 10 \
     strongest peaks under 2.5 kHz within 20 dB of the strongest (a Hann transform of the whole \
     note zero-padded eightfold, each peak the largest within its window's width, interpolated \
     parabolically in dB), and where it starts from quiet (its first 2 ms 15 dB under its first \
     peak within 100 ms) the 10-90 % rise of its 2 ms rms to that peak";

pub const BANDS: usize = 48;
pub const FRAME_SECS: f64 = 0.01;
pub const WINDOW_SECS: f64 = 0.02;
/// A note shorter than this is no segment of its own.
pub const LEAST_NOTE_SECS: f64 = 0.08;
pub const PEAK_TOP: usize = 10;
pub const PEAK_CEILING_HZ: f64 = 2500.0;
const PEAK_FLOOR_HZ: f64 = 30.0;
const PEAK_RANGE_DB: f64 = 20.0;
const RISE_RMS_SECS: f64 = 0.002;
const RISE_WITHIN_SECS: f64 = 0.1;
const RISE_QUIET_DB: f64 = 15.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Peak {
    pub hz: f64,
    pub amplitude: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub at_secs: f64,
    pub next_secs: f64,
    pub new: bool,
    pub rms: f64,
    pub peaks: Vec<Peak>,
    pub rise_secs: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Heard {
    pub rate: f64,
    pub start_secs: f64,
    pub duration_secs: f64,
    pub rms: f64,
    /// `power[b * frames + m]`, mean-square full-scale units.
    pub frames: usize,
    pub power: Vec<f64>,
    pub notes: Vec<Note>,
}

pub fn bark(hz: f64) -> f64 {
    13.0 * libm::atan(0.00076 * hz) + 3.5 * libm::atan((hz / 7500.0) * (hz / 7500.0))
}

/// The band a frequency's power falls in.
pub fn band_of(hz: f64) -> usize {
    ((2.0 * bark(hz)) as usize).min(BANDS - 1)
}

/// The transform length and window length at a rate.
pub fn sizes(sr: f64) -> (usize, usize, usize) {
    let window = libm::round(WINDOW_SECS * sr) as usize;
    let n = (2 * window).next_power_of_two();
    let hop = libm::round(FRAME_SECS * sr) as usize;
    (n, window, hop)
}

pub fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * libm::cos(std::f64::consts::TAU * i as f64 / n as f64))
        .collect()
}

/// Each band's median bin frequency, and each bin's band, at a transform length.
pub fn bins(n: usize, sr: f64) -> (Vec<usize>, Vec<f64>) {
    let of: Vec<usize> = (0..=n / 2)
        .map(|j| band_of(j as f64 * sr / n as f64))
        .collect();
    let centres = (0..BANDS)
        .map(|b| {
            let mut hz: Vec<f64> = (0..=n / 2)
                .filter(|j| of[*j] == b)
                .map(|j| j as f64 * sr / n as f64)
                .collect();
            match hz.len() {
                0 => 1e9,
                k => {
                    hz.sort_by(f64::total_cmp);
                    match k % 2 {
                        1 => hz[k / 2],
                        _ => 0.5 * (hz[k / 2 - 1] + hz[k / 2]),
                    }
                }
            }
        })
        .collect();
    (of, centres)
}

/// The power a unit-amplitude sine at `hz` puts in each band: its window's power spectrum
/// over the bins within 400 Hz, at the transform's own scale.
pub fn unit_sine(hz: f64, sr: f64) -> [f64; BANDS] {
    let (n, window, _) = sizes(sr);
    let w = hann_periodic(window);
    let w2: f64 = w.iter().map(|x| x * x).sum();
    let (of, _) = bins(n, sr);
    let mut out = [0.0; BANDS];
    if hz >= sr / 2.0 - 50.0 {
        return out;
    }
    let lo = (((hz - 400.0) * n as f64 / sr).max(0.0)) as usize;
    let hi = ((((hz + 400.0) * n as f64 / sr) as usize) + 2).min(n / 2 + 1);
    for j in lo..hi {
        let d = j as f64 * sr / n as f64 - hz;
        if d.abs() >= 400.0 {
            continue;
        }
        out[of[j]] += window_power(&w, std::f64::consts::TAU * d / sr);
    }
    let scale = 2.0 / (n as f64 * w2) * 0.25;
    out.iter_mut().for_each(|o| *o *= scale);
    out
}

/// `|sum_n w[n] e^(-i omega n)|^2` of the periodic Hann of `w.len()` points, in closed form:
/// half the rectangle's transform at omega less a quarter of it a bin either side.
fn window_power(w: &[f64], omega: f64) -> f64 {
    let n = w.len() as f64;
    let rect = |o: f64| -> (f64, f64) {
        let (s1, c1) = libm::sincos(o);
        let (den_re, den_im) = (1.0 - c1, s1);
        if den_re * den_re + den_im * den_im < 1e-24 {
            return (n, 0.0);
        }
        let (sn, cn) = libm::sincos(o * n);
        let (num_re, num_im) = (1.0 - cn, sn);
        let d = den_re * den_re + den_im * den_im;
        (
            (num_re * den_re + num_im * den_im) / d,
            (num_im * den_re - num_re * den_im) / d,
        )
    };
    let bin = std::f64::consts::TAU / n;
    let (a, b, c) = (rect(omega), rect(omega - bin), rect(omega + bin));
    let re = 0.5 * a.0 - 0.25 * (b.0 + c.0);
    let im = 0.5 * a.1 - 0.25 * (b.1 + c.1);
    re * re + im * im
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64) -> Heard {
    let len = super::held(planes);
    let count = planes.len() as f64;
    let x: Vec<f64> = (0..len)
        .map(|n| planes.iter().map(|p| p[n]).sum::<f64>() / count)
        .collect();
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / len.max(1) as f64).sqrt();
    let (frames, power) = pattern(&x, sr);
    let read = score::measure(planes, sr, start);
    let notes = read
        .events
        .iter()
        .filter(|e| e.next_secs - e.at_secs >= LEAST_NOTE_SECS)
        .map(|e| {
            let (a, b) = (e.at_secs - start, e.next_secs - start);
            let seg = &x[at(a, sr, len)..at(b, sr, len)];
            Note {
                at_secs: e.at_secs,
                next_secs: e.next_secs,
                new: e.kind == score::Kind::New,
                rms: (seg.iter().map(|v| v * v).sum::<f64>() / seg.len().max(1) as f64).sqrt(),
                peaks: peaks(seg, sr),
                rise_secs: rise(&x, sr, a, b),
            }
        })
        .collect();
    Heard {
        rate: sr,
        start_secs: start,
        duration_secs: len as f64 / sr,
        rms,
        frames,
        power,
        notes,
    }
}

fn at(secs: f64, sr: f64, len: usize) -> usize {
    ((secs * sr).max(0.0) as usize).min(len)
}

/// `(frames, power[b * frames + m])`.
pub fn pattern(x: &[f64], sr: f64) -> (usize, Vec<f64>) {
    let (n, window, hop) = sizes(sr);
    let w = hann_periodic(window);
    let w2: f64 = w.iter().map(|v| v * v).sum();
    let (of, _) = bins(n, sr);
    let frames = x.len() / hop + 1;
    let half = (n / 2) as i64;
    let left = ((n - window) / 2) as i64;
    let twiddles = fft::twiddles(n);
    let mirrored = |i: i64| -> f64 {
        let len = x.len() as i64;
        if len == 1 {
            return x[0];
        }
        let period = 2 * (len - 1);
        let mut k = i.rem_euclid(period);
        if k >= len {
            k = period - k;
        }
        x[k as usize]
    };
    let scale = 2.0 / (n as f64 * w2);
    let mut power = vec![0.0; BANDS * frames];
    let (mut re, mut im) = (vec![0.0; n], vec![0.0; n]);
    for m in 0..frames {
        let first = (m * hop) as i64 - half;
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for (k, wk) in w.iter().enumerate() {
            re[left as usize + k] = wk * mirrored(first + left + k as i64);
        }
        fft::fft_with(&mut re, &mut im, &twiddles);
        for j in 0..=n / 2 {
            power[of[j] * frames + m] += (re[j] * re[j] + im[j] * im[j]) * scale;
        }
    }
    (frames, power)
}

/// The note's strongest peaks under the ceiling, as the judge's `hpeaks` reads them.
pub fn peaks(seg: &[f64], sr: f64) -> Vec<Peak> {
    if seg.len() < 8 {
        return Vec::new();
    }
    let n = (seg.len() * 8).next_power_of_two();
    let m = seg.len();
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for (k, v) in seg.iter().enumerate() {
        let w = match m {
            1 => 1.0,
            _ => 0.5 - 0.5 * libm::cos(std::f64::consts::TAU * k as f64 / (m - 1) as f64),
        };
        re[k] = v * w;
    }
    fft::fft(&mut re, &mut im);
    let bins = n / 2 + 1;
    let hz = |j: usize| j as f64 * sr / n as f64;
    let db: Vec<f64> = (0..bins)
        .map(|j| 20.0 * libm::log10((re[j] * re[j] + im[j] * im[j]).sqrt() + 1e-12))
        .collect();
    let w = (n / m).max(3);
    let top = (0..bins)
        .filter(|j| hz(*j) > PEAK_FLOOR_HZ && hz(*j) < PEAK_CEILING_HZ)
        .map(|j| db[j])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<(f64, f64)> = Vec::new();
    for i in w..bins.saturating_sub(w) {
        if hz(i) > PEAK_CEILING_HZ {
            break;
        }
        if hz(i) <= PEAK_FLOOR_HZ || db[i] <= top - PEAK_RANGE_DB {
            continue;
        }
        let most = db[i - w..=i + w]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if db[i] != most {
            continue;
        }
        let (a, b, c) = (db[i - 1], db[i], db[i + 1]);
        let curve = a - 2.0 * b + c;
        let p = if curve != 0.0 {
            0.5 * (a - c) / curve
        } else {
            0.0
        };
        out.push((hz(i) + p * (sr / n as f64), b));
    }
    out.sort_by(|x, y| y.1.total_cmp(&x.1));
    out.truncate(PEAK_TOP);
    out.into_iter()
        .map(|(hz, db)| Peak {
            hz,
            amplitude: libm::pow(10.0, db / 20.0),
        })
        .collect()
}

/// The 10-90 % rise of the 2 ms rms to its first peak within 100 ms, where the note starts
/// from quiet.
pub fn rise(x: &[f64], sr: f64, a: f64, b: f64) -> Option<f64> {
    let k = libm::round(RISE_RMS_SECS * sr) as usize;
    let seg = &x[at(a, sr, x.len())..at(b.min(a + RISE_WITHIN_SECS), sr, x.len())];
    if seg.len() < 4 * k {
        return None;
    }
    let mut e = Vec::with_capacity(seg.len() - k + 1);
    let mut s: f64 = seg[..k].iter().map(|v| v * v).sum();
    e.push((s / k as f64).max(0.0).sqrt());
    for i in k..seg.len() {
        s += seg[i] * seg[i] - seg[i - k] * seg[i - k];
        e.push((s / k as f64).max(0.0).sqrt());
    }
    rise_of(&e, sr)
}

/// The same rise off an rms envelope already taken.
pub fn rise_of(e: &[f64], sr: f64) -> Option<f64> {
    let pk = e.iter().copied().fold(0.0, f64::max);
    if pk.is_nan() || pk <= 0.0 || e[0] > pk * libm::pow(10.0, -RISE_QUIET_DB / 20.0) {
        return None;
    }
    let p = e
        .iter()
        .position(|v| *v == pk)
        .expect("the peak is in the envelope");
    let on = e[..=p].iter().rposition(|v| *v <= 0.1 * pk).unwrap_or(0);
    let top = on
        + e[on..=p]
            .iter()
            .position(|v| *v >= 0.9 * pk)
            .expect("the peak itself is at 90 % of the peak");
    Some((top - on).max(1) as f64 / sr)
}
