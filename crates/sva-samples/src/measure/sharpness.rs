// Concern: sharpness per frame, DIN 45692's weighting of the loudness pattern | Non-concern: specific loudness (loudness_pattern.rs) | IO: (planes, rate, start, hop, spl) -> Sharpness

use super::loudness_pattern;

pub const MODEL: &str = "sharpness per frame of `loudness_pattern` at the same `frame` and \
                         `spl`, DIN 45692 (2009)'s weighted first moment: S = 0.11 sum N'(z) \
                         g(z) z / sum N'(z) acum, z each band's centre in Bark by Traunmüller \
                         (1990), z = 26.81 f / (1960 + f) - 0.53, raised by 0.15 (2 - z) \
                         under 2 and by 0.22 (z - 20.1) over 20.1 as he corrects its ends, so \
                         every band stands above 0 Bark, g(z) = 1 to 15.8 Bark and \
                         0.15 e^(0.42 (z - 15.8)) + 0.85 above, N' each Cam band's specific \
                         loudness standing for its stretch of the Bark scale; null where no \
                         band is over the pattern's `threshold_sone`. Measured, not DIN 45692: \
                         the standard weights Zwicker's specific loudness per Bark (ISO 532-1) \
                         and this one Moore and Glasberg's per Cam, so its acum are the \
                         standard's weighting on another loudness model.";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharpnessFrame {
    pub t_secs: f64,
    pub acum: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Sharpness {
    pub frame_secs: f64,
    pub window_secs: f64,
    pub spl_db: f64,
    pub frames: Vec<SharpnessFrame>,
}

pub fn measure(planes: &[&[f64]], sr: f64, start: f64, hop_secs: f64, spl_db: f64) -> Sharpness {
    let pattern = loudness_pattern::measure(planes, sr, start, hop_secs, spl_db);
    let frames = pattern
        .frames
        .iter()
        .map(|f| SharpnessFrame {
            t_secs: f.t_secs,
            acum: f
                .specific
                .iter()
                .any(|n| *n > pattern.threshold_sone)
                .then(|| acum(&pattern.centres_hz, &f.specific)),
        })
        .collect();
    Sharpness {
        frame_secs: pattern.frame_secs,
        window_secs: pattern.window_secs,
        spl_db,
        frames,
    }
}

fn bark(hz: f64) -> f64 {
    let z = 26.81 * hz / (1960.0 + hz) - 0.53;
    match z {
        z if z < 2.0 => z + 0.15 * (2.0 - z),
        z if z > 20.1 => z + 0.22 * (z - 20.1),
        z => z,
    }
}

fn acum(centres_hz: &[f64], specific: &[f64]) -> f64 {
    let moment: f64 = centres_hz
        .iter()
        .zip(specific)
        .map(|(&hz, &n)| {
            let z = bark(hz);
            let g = match z > 15.8 {
                true => 0.15 * sva_formula::exp(0.42 * (z - 15.8)) + 0.85,
                false => 1.0,
            };
            n * g * z
        })
        .sum();
    0.11 * moment / specific.iter().sum::<f64>()
}
