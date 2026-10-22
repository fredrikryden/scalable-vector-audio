// Concern: the starter a priced fit shrinks: one structure per group the clip's readings find, a line per voice, partial and noise | Non-concern: fitting it | IO: (samples, rate) -> files, varies, cost

use sva_samples::measure::partials::{self, Asked, Curve, Kind as TrackKind};
use sva_samples::measure::{internal, score};

/// How large a starter is written: harmonics per voice, lone partials per onset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub harmonics: usize,
    pub lone: usize,
}

pub const LIMITS: Limits = Limits {
    harmonics: 48,
    lone: 8,
};
const LONE_AFTER_SECS: f64 = 0.04;
const SAME_PITCH_CENTS: f64 = 50.0;
const LEVEL_AT_SECS: [f64; 3] = [0.03, 0.06, 0.1];
const HARMONIC_NEAR: f64 = 0.03;
const ABSENT_DB: f64 = -20.0;
const NOISE_LOW_HZ: f64 = 30.0;
const NOISE_HIGH_HZ: f64 = 20_000.0;
const PARTIALS: Asked = Asked {
    frame_secs: 0.1,
    window_secs: 0.5,
    hop_secs: 0.005,
    floor_db: -70.0,
    tracks: 120,
};

pub const COST: &str = "fits/clip";

/// Every number the priced fit varies: its bounds and its resolution, about a JND of what
/// it moves.
pub const VARIES: [&str; 19] = [
    "att=0.0005s..0.3s/1.2214,log",
    "dec=0.005s..30s/1.1618,log",
    "sus=0.0025..1/1.35,log",
    "rel=0.004s..1.5s/1.35,log",
    "tilt=0.001..4/0.15",
    "ev=-30..12/1.5",
    "damp=-2..30/0.3",
    "inh=0..0.004/0.00002",
    "bnd=-0.6..3/0.02",
    "bt=0.003s..0.08s/1.35,log",
    "fg=-30..30/1.5",
    "fc=50hz..16000hz/2,log",
    "bw=0.05..6/2.7,log",
    "h*=-40..20/4",
    "vel=0..10/0.001",
    "len=0.02s..30s/0.1",
    "lo=30hz..22000hz/1.22,log",
    "hi=30hz..22000hz/1.22,log",
    "cut=4000hz..22000hz/1.15,log",
];

pub struct Seeded {
    pub files: Vec<(String, String)>,
    /// What the limits left out, each naming the flag that raises its limit.
    pub dropped: Vec<String>,
    pub varies: Vec<String>,
    pub cost: String,
}

struct Voice {
    tone: &'static str,
    at: f64,
    f0: f64,
    len: f64,
    levels: Vec<f64>,
}

fn curve_at(c: &Curve, t: f64, outside: Option<f64>) -> f64 {
    let (ts, vs) = (&c.times, &c.values);
    if ts.is_empty() {
        return outside.unwrap_or(0.0);
    }
    if t <= ts[0] {
        return outside.unwrap_or(vs[0]);
    }
    if t >= ts[ts.len() - 1] {
        return outside.unwrap_or(vs[vs.len() - 1]);
    }
    let j = ts.partition_point(|x| *x <= t) - 1;
    let u = (t - ts[j]) / (ts[j + 1] - ts[j]);
    vs[j] * (1.0 - u) + vs[j + 1] * u
}

fn g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    let s = format!("{x:.6}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

const ENVELOPE: &str = "vel*min(1, t/att)*(sus + (1 - sus)*exp(0 - t/dec))";
const BENT: &str = "(t + bnd*bt*(1 - exp(0 - t/bt)))";

fn harmonic(harmonics: usize) -> String {
    let mut knots: Vec<String> = (1..=harmonics)
        .map(|k| format!("{k}s, pow(10, h{k}/20)"))
        .collect();
    knots.push(format!("{}s, 1", harmonics + 1));
    let law = format!(
        "pwl(k*1s, {})*pow(k, 0 - tilt)*(1 + (pow(10, ev/20) - 1)*(0.5 + 0.5*cos(pi*k)))\
         *(1 + (pow(10, fg/20) - 1)/(1 + pow(log(k*f0/fc)/bw, 2)))*exp(0 - damp*k*f0*t/1000)\
         *step(20000hz - k*f0)",
        knots.join(", ")
    );
    let mut head = vec![
        "att = 0.01s".to_string(),
        "dec = 0.4s".to_string(),
        "sus = 0.5".to_string(),
        "rel = 0.05s".to_string(),
        "tilt = 1".to_string(),
        "ev = 0".to_string(),
        "damp = 0".to_string(),
        "inh = 0".to_string(),
        "bnd = 0".to_string(),
        "bt = 0.03s".to_string(),
        "fg = 0".to_string(),
        "fc = 1000hz".to_string(),
        format!("bw = {}", g(std::f64::consts::LN_2)),
        "ph = 0".to_string(),
    ];
    head.extend((1..=harmonics).map(|k| format!("h{k} = 0")));
    head.join("\n")
        + &format!(
            "\ncrop({ENVELOPE}*sum(k, 1, {harmonics}, {law}*sin(2*pi*k*f0*sqrt(1 + inh*k*k)*{BENT} + ph)), \
             0s, len, fall=min(rel, len))\n"
        )
}

fn lone() -> String {
    format!(
        "att = 0.01s\ndec = 0.4s\nsus = 0.5\nrel = 0.05s\nbnd = 0\nbt = 0.03s\nph = 0\n\
         crop({ENVELOPE}*sin(2*pi*f0*{BENT} + ph), 0s, len, fall=min(rel, len))\n"
    )
}

fn noisy(seed: u64, (att, dec, sus, rel, lo, hi): (f64, f64, f64, f64, f64, f64)) -> String {
    format!(
        "att = {}s\ndec = {}s\nsus = {}\nrel = {}s\nlo = {}hz\nhi = {}hz\n",
        g(att),
        g(dec),
        g(sus),
        g(rel),
        g(lo),
        g(hi)
    ) + &format!(
        "crop({ENVELOPE}*highpass(lowpass(sample(0.01*noise({seed}, period=8s)), hi, 0.7071), lo, \
         0.7071), 0s, len, fall=min(rel, len))\n"
    )
}

/// The frequency at a Bark value: `internal::bark` inverted by bisection.
fn hz_at(z: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 24_000.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        match internal::bark(mid) < z {
            true => lo = mid,
            false => hi = mid,
        }
    }
    0.5 * (lo + hi)
}

/// The Bark bands holding a span's loudness between its 10th and 90th percentile.
fn band(ear: &internal::Internal, at: f64, len: f64) -> (f64, f64) {
    let frames = ear.frames;
    let first = ((at / internal::FRAME_SECS) as usize).min(frames);
    let last = (((at + len) / internal::FRAME_SECS) as usize).clamp(first + 1, frames.max(1));
    let power: Vec<f64> = (0..internal::BARKS)
        .map(|b| {
            (first..last.min(frames))
                .map(|m| ear.specific[m * internal::BARKS + b])
                .sum()
        })
        .collect();
    let total: f64 = power.iter().sum();
    let edge = |share: f64| {
        let mut held = 0.0;
        (0..internal::BARKS)
            .find(|b| {
                held += power[*b];
                held >= share * total
            })
            .unwrap_or(internal::BARKS - 1)
    };
    if total <= 0.0 {
        return (NOISE_LOW_HZ, NOISE_HIGH_HZ);
    }
    let lo = hz_at(edge(0.1) as f64).clamp(NOISE_LOW_HZ, NOISE_HIGH_HZ);
    let hi = hz_at((edge(0.9) + 1) as f64).clamp(lo * 1.5, NOISE_HIGH_HZ);
    (lo, hi.max(lo))
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + sva_formula::exp(-x))
}

/// `clip` is the recording's path as the cost's `sample` reads it, beside the composition;
/// its channels are heard averaged.
pub fn seed(channels: &[Vec<f32>], rate: f64, clip: &str, limits: Limits) -> Seeded {
    let len = channels.first().map_or(0, Vec::len);
    let x: Vec<f64> = (0..len)
        .map(|i| channels.iter().map(|p| f64::from(p[i])).sum::<f64>() / channels.len() as f64)
        .collect();
    let x = &x[..];
    let planes = [x];
    let read = score::measure(&planes, rate, 0.0);
    let tracks: Vec<partials::Track> = partials::measure(&planes, rate, 0.0, PARTIALS)
        .tracks
        .into_iter()
        .filter(|t| t.kind == TrackKind::Tonal)
        .collect();
    let dur = x.len() as f64 / rate;
    let mut voices: Vec<Voice> = Vec::new();
    for e in &read.events {
        for v in e.voices.iter().filter(|v| v.new) {
            let (mut s, mut w) = (0.0, 0.0);
            for p in v.partials.iter().filter(|p| p.harmonic > 0) {
                let a = libm::pow(10.0, p.db / 20.0);
                s += a * libm::log(p.hz / p.harmonic as f64);
                w += a;
            }
            let f0 = if w > 0.0 {
                sva_formula::exp(s / w)
            } else {
                v.hz
            };
            voices.push(Voice {
                tone: v.tone.key(),
                at: e.at_secs,
                f0,
                len: 0.0,
                levels: Vec::new(),
            });
        }
    }
    let starts: Vec<(f64, f64)> = voices.iter().map(|v| (v.at, v.f0)).collect();
    for v in &mut voices {
        let next = starts
            .iter()
            .filter(|(at, f0)| {
                *at > v.at + 0.02 && (1200.0 * libm::log2(f0 / v.f0)).abs() < SAME_PITCH_CENTS
            })
            .map(|(at, _)| *at)
            .fold(dur, f64::min);
        v.len = next - v.at;
        v.levels = (1..=(20_000.0 / v.f0) as usize)
            .map(|k| {
                let fk = k as f64 * v.f0;
                if fk > 20_000.0 {
                    return 0.0;
                }
                let mut best: f64 = 0.0;
                for t in &tracks {
                    for dt in LEVEL_AT_SECS {
                        let at = v.at + dt;
                        if (curve_at(&t.frequency, at, None) - fk).abs() < HARMONIC_NEAR * fk {
                            best = best.max(curve_at(&t.amplitude, at, Some(0.0)));
                        }
                    }
                }
                best
            })
            .collect();
    }
    let harmonics_past: usize = voices
        .iter_mut()
        .map(|v| {
            let past = v
                .levels
                .iter()
                .skip(limits.harmonics)
                .filter(|l| **l > 0.0)
                .count();
            v.levels.resize(limits.harmonics, 0.0);
            past
        })
        .sum();
    let mut lines: Vec<(f64, String)> = Vec::new();
    let s6 = |x: f64| format!("{}", (x * 1e6).round() / 1e6);
    let mut files = Vec::new();
    let mut tones: Vec<&str> = voices.iter().map(|v| v.tone).collect();
    tones.dedup();
    tones.sort_unstable();
    tones.dedup();
    for tone in &tones {
        files.push((
            format!("instruments/voice-{tone}"),
            harmonic(limits.harmonics),
        ));
    }
    for v in &voices {
        let top = v.levels.iter().copied().fold(0.0, f64::max);
        let mut fields = vec![
            format!("\"at\": \"{}s\"", s6(v.at)),
            format!("\"node\": \"@../instruments/voice-{}\"", v.tone),
            format!("\"f0\": \"{}hz\"", s6(v.f0)),
            format!("\"vel\": {}", g(top)),
            format!("\"len\": \"{}s\"", s6(v.len)),
        ];
        for (k, level) in v.levels.iter().enumerate() {
            let db = match *level > 0.0 && top > 0.0 {
                true => 20.0 * libm::log10(level / top) + 20.0 * libm::log10((k + 1) as f64),
                false => ABSENT_DB,
            };
            fields.push(format!(
                "\"h{}\": {}",
                k + 1,
                g((db.clamp(-40.0, 20.0) * 100.0).round() / 100.0)
            ));
        }
        lines.push((v.at, format!("{{{}}}", fields.join(", "))));
    }
    let mut lone_lines = 0;
    let mut lone_past = 0;
    for e in &read.events {
        let t = e.at_secs + LONE_AFTER_SECS;
        let mut loud: Vec<(f64, f64)> = tracks
            .iter()
            .map(|tr| {
                (
                    curve_at(&tr.amplitude, t, Some(0.0)),
                    curve_at(&tr.frequency, t, None),
                )
            })
            .filter(|(a, _)| *a > 0.0)
            .collect();
        loud.sort_by(|a, b| b.0.total_cmp(&a.0));
        lone_past += loud.len().saturating_sub(limits.lone);
        for (_, hz) in loud.into_iter().take(limits.lone) {
            lone_lines += 1;
            lines.push((
                e.at_secs,
                format!(
                    "{{\"at\": \"{}s\", \"node\": \"@../instruments/partial\", \"f0\": \"{}hz\", \
                     \"vel\": 0, \"len\": \"{}s\"}}",
                    s6(e.at_secs),
                    s6(hz),
                    s6(e.next_secs - e.at_secs)
                ),
            ));
        }
    }
    if lone_lines > 0 {
        files.push(("instruments/partial".to_string(), lone()));
    }
    let ear = internal::measure(&planes, rate, 0.0, sva_engine::DEFAULT_SPL_DB);
    let mut spans: Vec<(&str, f64, f64)> = read
        .events
        .iter()
        .map(|e| (e.sound.key(), e.at_secs, e.next_secs - e.at_secs))
        .collect();
    let mut sounds: Vec<&str> = spans.iter().map(|(k, ..)| *k).collect();
    sounds.sort_unstable();
    sounds.dedup();
    if sounds.is_empty() {
        sounds.push(score::Sound::Noisy.key());
    }
    for sound in &sounds {
        spans.push((sound, 0.0, dur));
    }
    for (n, sound) in sounds.iter().enumerate() {
        let held: Vec<(f64, f64, (f64, f64))> = spans
            .iter()
            .filter(|(k, ..)| k == sound)
            .map(|(_, at, len)| (*at, *len, band(&ear, *at, *len)))
            .collect();
        let centre = |side: fn(&(f64, f64)) -> f64| {
            let mut logs: Vec<f64> = held.iter().map(|(.., b)| libm::log(side(b))).collect();
            logs.sort_by(f64::total_cmp);
            sva_formula::exp(logs[logs.len() / 2])
        };
        let (lo, hi) = (centre(|b| b.0), centre(|b| b.1));
        files.push((
            format!("instruments/noise-{sound}"),
            noisy(7 + n as u64, (0.005, 0.2, sigmoid(-1.0), 0.05, lo, hi)),
        ));
        for (at, len, (lo, hi)) in held {
            lines.push((
                at,
                format!(
                    "{{\"at\": \"{}s\", \"node\": \"@../instruments/noise-{sound}\", \"vel\": 0, \
                     \"len\": \"{}s\", \"lo\": \"{}hz\", \"hi\": \"{}hz\"}}",
                    s6(at),
                    s6(len),
                    g(lo.round()),
                    g(hi.round())
                ),
            ));
        }
    }
    lines.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut score_text = String::new();
    for (_, l) in &lines {
        score_text.push_str(l);
        score_text.push('\n');
    }
    files.push(("score/clip.jsonl".to_string(), score_text));
    files.push((
        "master".to_string(),
        "cut = 16000hz\nlowpass(lowpass(sample(@score/clip), cut, 0.7071), cut, 0.7071)\n"
            .to_string(),
    ));
    let cost = format!(
        "norm(repr.internal(@master) - repr.internal(sample(\"{}\"))) + 0.001*repr.numbers(@master)",
        clip.replace('"', "")
    );
    files.push((COST.to_string(), format!("{cost}\n")));
    let mut dropped = Vec::new();
    if harmonics_past > 0 {
        dropped.push(format!(
            "{harmonics_past} harmonics heard past `--harmonics {}`",
            limits.harmonics
        ));
    }
    if lone_past > 0 {
        dropped.push(format!(
            "{lone_past} partials at onsets past `--lone {}`",
            limits.lone
        ));
    }
    Seeded {
        files,
        dropped,
        varies: VARIES.iter().map(|v| v.to_string()).collect(),
        cost,
    }
}
