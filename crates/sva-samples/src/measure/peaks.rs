// Concern: picks a Hann spectrum's peaks, those louder peaks' sidelobes cannot explain | Non-concern: computing the spectrum (spectrum.rs), naming a peak (pitch.rs) | IO: (magnitudes, span) -> peaks

pub const FLOOR_DB: f64 = -160.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub hz: f64,
    pub db: f64,
}

pub fn db(amplitude: f64) -> f64 {
    if amplitude <= 0.0 {
        FLOOR_DB
    } else {
        (20.0 * amplitude.log10()).max(FLOOR_DB)
    }
}

/// Local maxima, parabolically interpolated so a peak between two bins reports its real
/// frequency instead of the nearest bin's.
pub fn peaks(mags: &[f64], bin_hz: f64, max_peaks: usize) -> Vec<Peak> {
    let mut found: Vec<Peak> = located(mags, bin_hz).into_iter().map(|(_, p)| p).collect();
    found.truncate(max_peaks);
    found
}

/// Loudest first by amplitude: `db` floors.
fn located(mags: &[f64], bin_hz: f64) -> Vec<(usize, Peak)> {
    let mut found: Vec<(f64, usize, Peak)> = Vec::new();
    for k in 1..mags.len().saturating_sub(1) {
        if mags[k] <= mags[k - 1] || mags[k] < mags[k + 1] || mags[k] <= 0.0 {
            continue;
        }
        let (a, b, c) = (mags[k - 1], mags[k], mags[k + 1]);
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > f64::EPSILON {
            (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let amplitude = b - 0.25 * (a - c) * delta;
        found.push((
            amplitude,
            k,
            Peak {
                hz: (k as f64 + delta) * bin_hz,
                db: db(amplitude),
            },
        ));
    }
    found.sort_by(|x, y| y.0.total_cmp(&x.0));
    found.into_iter().map(|(_, k, peak)| (k, peak)).collect()
}

/// Loudest first, peaks no kept peak's sidelobes explain, nor a DC's at bin 0.
pub fn resolved(mags: &[f64], bin_hz: f64, spanned: usize, max_peaks: usize) -> Vec<Peak> {
    resolved_counting(mags, bin_hz, spanned, max_peaks, |_| true)
}

/// As [`resolved`], until `max_peaks` kept pass `counts`.
pub fn resolved_counting(
    mags: &[f64],
    bin_hz: f64,
    spanned: usize,
    max_peaks: usize,
    counts: impl Fn(&Peak) -> bool,
) -> Vec<Peak> {
    let r = spanned as f64 / (2 * (mags.len() - 1)) as f64;
    let half = r / 2.0;
    let dc = match mags {
        [zero, one, ..] if zero >= one => *zero,
        _ => 0.0,
    };
    let mut kept: Vec<(usize, Peak)> = Vec::new();
    let mut counted = 0;
    for (k, peak) in located(mags, bin_hz) {
        if counted == max_peaks {
            break;
        }
        let leaked: f64 = kept
            .iter()
            .map(|&(q, _)| {
                let near = (k.abs_diff(q) as f64 * r - half).max(0.0);
                let image = (k + q) as f64 * r - half;
                mags[q] / kernel(half) * (leakage(near) + leakage(image))
            })
            .sum::<f64>()
            + dc / kernel(half) * leakage(k as f64 * r - half);
        if mags[k] > leaked {
            counted += usize::from(counts(&peak));
            kept.push((k, peak));
        }
    }
    kept.into_iter().map(|(_, peak)| peak).collect()
}

pub(crate) fn leakage(m: f64) -> f64 {
    match m > 1.0 {
        true => (1.0 / (std::f64::consts::PI * m * (m * m - 1.0))).min(1.0),
        false => 1.0,
    }
}

pub(crate) fn kernel(d: f64) -> f64 {
    let d = d.abs();
    match d {
        0.0 => 1.0,
        _ if d >= 1.0 => 0.5,
        _ => {
            let x = std::f64::consts::PI * d;
            x.sin() / (x * (1.0 - d * d))
        }
    }
}
