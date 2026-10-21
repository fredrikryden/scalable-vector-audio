// Concern: how far each band's level swings about its own trend per sliding window | Non-concern: specific loudness (loudness_pattern.rs) | IO: (planes, rate, start, hop, window, spl) -> Beating

use super::loudness_pattern::{self, LoudnessPattern};
use super::sliding;

pub const MODEL: &str = "per window of `window_secs` centred on each frame of \
                         `loudness_pattern` at the same `frame` and `spl`, clipped to the \
                         target at its ends, `from`/`to` its span: each band's level y = 10 \
                         log2(N' + T) over the window's frames, T the pattern's \
                         `threshold_sone`; the frames within 20 phon of the band's loudest \
                         there, at least 20 of them, fitted by least squares with a quadratic \
                         in time, its trend; the band's swing the rms of y about that trend; \
                         `phon` the bands' swings averaged, each weighted by its mean specific \
                         loudness over the window, a band no frame of the window holds over T \
                         left out, null where no band is left. A quadratic follows a note's \
                         rise and its fall in phon, so what remains moves faster than the \
                         window's length, beating and tremolo, up to the pattern's window \
                         rate; a swing between two partials inside one band reads as the \
                         beating it is heard as. A window holding under 20 frames refuses.";

/// The fewest frames a band's trend is fitted over: a correctness limit, not a cost one, so
/// it has no override; a quadratic through fewer follows the swing it should leave.
pub const LEAST_FRAMES: usize = 20;
/// How far under a band's loudest a frame may fall and still count: below, a note's silence
/// before or after it, not its swing.
const RANGE_PHON: f64 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatingWindow {
    pub t_secs: f64,
    pub from_secs: f64,
    pub to_secs: f64,
    pub phon: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Beating {
    pub frame_secs: f64,
    pub window_secs: f64,
    pub spl_db: f64,
    pub windows: Vec<BeatingWindow>,
}

/// The window holds fewer than `LEAST_FRAMES` frames; `least_secs` is the least that holds
/// them at this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShortWindow {
    pub window_secs: f64,
    pub frame_secs: f64,
    pub least_secs: f64,
}

pub fn measure(
    planes: &[&[f64]],
    sr: f64,
    start: f64,
    (hop_secs, window_secs): (f64, f64),
    spl_db: f64,
) -> Result<Beating, ShortWindow> {
    let pattern = loudness_pattern::measure(planes, sr, start, hop_secs, spl_db);
    let rate = 1.0 / pattern.frame_secs;
    if ((window_secs * rate).round() as usize) < LEAST_FRAMES {
        return Err(ShortWindow {
            window_secs,
            frame_secs: pattern.frame_secs,
            least_secs: LEAST_FRAMES as f64 * pattern.frame_secs,
        });
    }
    let grid = sliding::spans(
        super::held(planes),
        sr,
        start,
        pattern.frame_secs,
        rate,
        window_secs,
        pattern.frames.len(),
    );
    let windows = grid
        .spans
        .iter()
        .map(|s| BeatingWindow {
            t_secs: s.t_secs,
            from_secs: s.from_secs,
            to_secs: s.to_secs,
            phon: swing(&pattern, s.from..s.to),
        })
        .collect();
    Ok(Beating {
        frame_secs: pattern.frame_secs,
        window_secs: grid.window_secs,
        spl_db,
        windows,
    })
}

fn swing(pattern: &LoudnessPattern, frames: std::ops::Range<usize>) -> Option<f64> {
    let held = &pattern.frames[frames];
    let floor = pattern.threshold_sone;
    let (mut sum, mut weights) = (0.0, 0.0);
    for z in 0..pattern.centres_hz.len() {
        if held.iter().all(|f| f.specific[z] <= floor) {
            continue;
        }
        let y: Vec<f64> = held
            .iter()
            .map(|f| 10.0 * libm::log2(f.specific[z] + floor))
            .collect();
        let top = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let kept: Vec<(f64, f64)> = y
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > top - RANGE_PHON)
            .map(|(i, v)| (2.0 * i as f64 / held.len() as f64 - 1.0, *v))
            .collect();
        if kept.len() < LEAST_FRAMES {
            continue;
        }
        let w = held.iter().map(|f| f.specific[z]).sum::<f64>() / held.len() as f64;
        sum += about_quadratic(&kept) * w;
        weights += w;
    }
    (weights > 0.0).then(|| sum / weights)
}

/// The rms of `points` about their least-squares quadratic, `t` on [-1, 1) so the normal
/// equations stay as well conditioned at a minute's frames as at a second's.
fn about_quadratic(points: &[(f64, f64)]) -> f64 {
    let mut s = [0.0; 5];
    let mut r = [0.0; 3];
    for &(t, y) in points {
        let mut p = 1.0;
        for (k, slot) in s.iter_mut().enumerate() {
            *slot += p;
            if k < 3 {
                r[k] += p * y;
            }
            p *= t;
        }
    }
    let m = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let c = solve3(m, r);
    let square: f64 = points
        .iter()
        .map(|&(t, y)| {
            let d = y - (c[0] + c[1] * t + c[2] * t * t);
            d * d
        })
        .sum();
    (square / points.len() as f64).sqrt()
}

/// Cramer's rule.
fn solve3(m: [[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    let mut out = [0.0; 3];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut a = m;
        for row in 0..3 {
            a[row][k] = r[row];
        }
        *slot = det(a) / d;
    }
    out
}
