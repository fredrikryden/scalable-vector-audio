// Concern: exercises `analyze` over a recording, the one reading path with no graph behind it | Non-concern: rendering one (render_integration.rs) | IO: (a recording) -> an Answer or CliError

use crate::helpers::{fixture, scratch};

use sva_cli::{SampleEncoding, parse_args, write_channels};
use sva_engine::{Buffer, Output, PSYCHOACOUSTIC_V1, Representation, Source, answer_buffer};

fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| (*s).to_string()).collect()
}

fn tone(rate: u32, hz: f64, secs: f64) -> Vec<f32> {
    let len = (f64::from(rate) * secs) as usize;
    (0..len)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            libm::sin(std::f64::consts::TAU * hz * t) as f32 * 0.5
        })
        .collect()
}

fn written(name: &str, rate: u32, planes: &[&[f32]]) -> std::path::PathBuf {
    let dir = scratch(name);
    let path = dir.join("in.wav");
    write_channels(planes, rate, &path, SampleEncoding::Float).expect("a written wav");
    path
}

/// `analyze <path> --representation <readings>` as the binary prints it.
fn printed(path: &std::path::Path, readings: &str) -> serde_json::Value {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .args([
            "analyze",
            &path.display().to_string(),
            "--representation",
            readings,
        ])
        .output()
        .expect("the binary runs");
    serde_json::from_slice(&out.stdout).expect("one envelope")
}

/// Every event's `at_secs` in the `score` a file reads as, in order.
fn event_times(path: &std::path::Path) -> Vec<f64> {
    let read = printed(path, "score");
    read["data"]["representations"]["score"]["value"]["events"]["items"]
        .as_array()
        .unwrap_or_else(|| panic!("an event list: {read}"))
        .iter()
        .map(|e| e["at_secs"].as_f64().expect("a time"))
        .collect()
}

/// A 440 Hz strike with a 1.5 ms rise and a `decay` fall, `secs` long, at each of `struck`.
fn strikes(rate: u32, len_secs: f64, struck: &[f64], (decay, secs): (f64, f64)) -> Vec<f32> {
    let sr = f64::from(rate);
    let mut samples = vec![0.0f32; (sr * len_secs) as usize];
    for at in struck {
        let from = (at * sr) as usize;
        for n in 0..(sr * secs) as usize {
            let t = n as f64 / sr;
            let shape = (1.0 - sva_formula::exp(-t / 0.0015)) * sva_formula::exp(-t / decay);
            let wave = libm::sin(std::f64::consts::TAU * 440.0 * t);
            if let Some(s) = samples.get_mut(from + n) {
                *s += (0.7 * shape * wave) as f32;
            }
        }
    }
    samples
}

fn decoded(path: &std::path::Path) -> Buffer {
    let (planes, rate) = sva_cli::read_recording(path).expect("a readable wav");
    Buffer::of_planes(
        rate,
        planes
            .iter()
            .map(|p| p.iter().map(|v| f64::from(*v)).collect())
            .collect(),
    )
}

/// The file's own rate stands: nothing resamples, and the reading says which rate it ran at.
#[test]
fn an_analysis_reads_the_files_own_rate_and_never_resamples_it() {
    for rate in [22_050u32, 48_000] {
        let path = written("analyze-rate", rate, &[&tone(rate, 440.0, 0.5)]);
        let buffer = decoded(&path);
        assert_eq!(buffer.rate, rate);
        let answer = answer_buffer(
            "in.wav",
            &buffer,
            Representation::Spectrum {
                max_peaks: 4,
                frame_secs: None,
            },
            PSYCHOACOUSTIC_V1.name,
        )
        .expect("a spectrum off a file");
        assert_eq!(answer.rate, Some(rate));
        assert_eq!(answer.source, Source::Measured);
        assert_eq!(answer.profile, "psychoacoustic-v1");
        let Output::Spectrum(spectrum) = answer.value else {
            panic!("expected a spectrum")
        };
        let peak = spectrum.peaks.first().expect("one peak at least");
        assert!(
            (peak.hz - 440.0).abs() < 20.0,
            "{rate} Hz put the 440 Hz tone at {}",
            peak.hz
        );
    }
}

/// A reading that needs the tree a node was built from has no answer off a file, and says
/// so rather than answering something else.
#[test]
fn a_reading_that_needs_a_graph_refuses_off_a_file() {
    let path = written("analyze-refuse", 8_000, &[&tone(8_000, 220.0, 0.1)]);
    let buffer = decoded(&path);
    for representation in [
        Representation::Ledger { depth: 3 },
        Representation::Alias { oversample: 4 },
        Representation::Lines,
    ] {
        let err = answer_buffer("in.wav", &buffer, representation, PSYCHOACOUSTIC_V1.name)
            .err()
            .unwrap_or_else(|| panic!("{} must refuse", representation.name()));
        assert_eq!(err.code(), "engine.observation_needs_a_graph");
    }
}

/// Argv says which readings a file answers at all, before a byte of it is read.
#[test]
fn analyze_takes_only_the_readings_a_buffer_answers() {
    let answered: Vec<&str> = sva_engine::READINGS
        .iter()
        .filter(|row| row.reads.off_samples(true))
        .map(|row| row.name)
        .collect();
    let mut sorted = answered.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        [
            "bands",
            "beating",
            "crest",
            "derivative",
            "dissonance",
            "envelope",
            "formants",
            "inharmonicity",
            "loudness",
            "loudness_pattern",
            "modulation",
            "partials",
            "periodicity",
            "pitch",
            "samples",
            "score",
            "self_similarity",
            "sharpness",
            "spectrum",
            "stereo",
            "tonal_distance",
            "voice_leading",
        ],
        "the readings a buffer answers alone"
    );
    for name in answered {
        assert!(
            parse_args(&argv(&["analyze", "/tmp/a.wav", "--representation", name])).is_ok(),
            "`{name}` is one a buffer answers"
        );
    }
    for name in ["ledger", "alias", "bindings", "lines", "atoms"] {
        assert!(
            parse_args(&argv(&["analyze", "/tmp/a.wav", "--representation", name])).is_err(),
            "`{name}` needs a rendered graph"
        );
    }
}

/// Two components in, two components read: an interleaved file decodes back to the planes
/// it was written from.
#[test]
fn a_two_channel_file_answers_a_stereo_image_a_mono_or_three_channel_one_cannot() {
    let rate = 16_000u32;
    let left = tone(rate, 300.0, 0.4);
    let right: Vec<f32> = left.iter().map(|s| s * 0.25).collect();
    let path = written("analyze-stereo", rate, &[&left, &right]);
    let buffer = decoded(&path);
    assert_eq!(buffer.width(), 2);

    let Output::Stereo(image) = answer_buffer(
        "in.wav",
        &buffer,
        Representation::Stereo { frame_secs: 0.05 },
        PSYCHOACOUSTIC_V1.name,
    )
    .expect("a stereo image")
    .value
    else {
        panic!("expected a stereo image")
    };
    assert_eq!(image.channels, 2);
    assert!(image.overall.balance_db < 0.0, "the left side leads");

    let mono = written("analyze-mono", rate, &[&left]);
    let err = answer_buffer(
        "in.wav",
        &decoded(&mono),
        Representation::Stereo { frame_secs: 0.05 },
        PSYCHOACOUSTIC_V1.name,
    )
    .expect_err("one component is no image");
    assert_eq!(err.code(), "type.width_mismatch");

    let three = written("analyze-three", rate, &[&left, &right, &left]);
    let err = answer_buffer(
        "in.wav",
        &decoded(&three),
        Representation::Stereo { frame_secs: 0.05 },
        PSYCHOACOUSTIC_V1.name,
    )
    .expect_err("three channels are no one pair");
    assert_eq!(err.code(), "type.width_mismatch");
    assert!(
        err.to_string()
            .contains("needs exactly 2 components and the file holds 3"),
        "{err}"
    );
}

/// A `beating` window fits each band's trend over at least 20 frames: one holding fewer
/// refuses, naming the window and the frame that hold them.
#[test]
fn a_beating_window_under_twenty_frames_refuses_naming_what_holds_them() {
    let rate = 48_000u32;
    let path = written("beating-short-window", rate, &[&tone(rate, 220.0, 0.5)]);
    let refused = answer_buffer(
        "in.wav",
        &decoded(&path),
        Representation::Beating {
            frame_secs: 0.01,
            window_secs: 0.1,
            spl_db: 100.0,
        },
        PSYCHOACOUSTIC_V1.name,
    )
    .expect_err("ten frames hold no trend");
    assert_eq!(refused.code(), "read.window_under_frames");
    let said = refused.to_string();
    assert!(said.contains("`beating(window=0.2s)`"), "{said}");
    assert!(
        said.contains("`beating(window=0.1s, frame=0.005s)`"),
        "{said}"
    );
}

/// At 8 kHz `formants` fits 10 poles off 11 lags, 1.375 ms: a frame or file holding fewer
/// refuses, naming the 1.4 ms that holds them, rather than fitting the few it holds.
#[test]
fn formants_under_the_orders_lags_refuse_naming_the_least_frame_and_file() {
    let rate = 8_000u32;
    let refusal = |samples: &[f32], frame_secs: f64| {
        let path = written("formants-under-order", rate, &[samples]);
        answer_buffer(
            "in.wav",
            &decoded(&path),
            Representation::Formants {
                max_formants: 4,
                frame_secs,
            },
            PSYCHOACOUSTIC_V1.name,
        )
        .expect_err("too few lags for the order")
    };
    let tone = tone(rate, 220.0, 0.1);

    let frame = refusal(&tone, 0.001);
    assert_eq!(frame.code(), "read.frame_under_order");
    assert!(
        frame
            .to_string()
            .contains("ask a frame of 0.0014s or longer, as `formants(frame=0.0014s)`"),
        "{frame}"
    );

    let file = refusal(&tone[..10], 0.05);
    assert_eq!(file.code(), "read.target_under_order");
    assert!(
        file.to_string()
            .contains("analyze a file holding at least 0.0014s"),
        "{file}"
    );
}

/// A frame the transform cannot hold refuses on both paths rather than shrinking silently
/// or overflowing the size it was asked for.
#[test]
fn a_frame_past_the_transform_bound_refuses_on_both_paths() {
    for span in ["1e9", "1e300"] {
        let refused = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
            .current_dir(scratch("frame-bound"))
            .args([
                "render",
                "@master",
                "--representation",
                &format!("spectrum(frame={span})"),
            ])
            .output()
            .expect("the binary runs");
        let printed = String::from_utf8_lossy(&refused.stdout);
        assert_eq!(refused.status.code(), Some(3), "{printed}");
        assert!(printed.contains("past the"), "{printed}");
    }

    let rate = 8_000u32;
    let path = written("frame-bound-file", rate, &[&tone(rate, 220.0, 0.1)]);
    let refused = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .args([
            "analyze",
            &path.display().to_string(),
            "--representation",
            "spectrum(frame=1e9)",
        ])
        .output()
        .expect("the binary runs");
    let printed = String::from_utf8_lossy(&refused.stdout);
    assert_eq!(refused.status.code(), Some(3), "{printed}");
    assert!(printed.contains("past the"), "{printed}");
}

/// BRIEF section 9 gates a groove on onsets to 1 ms. A hop says which strike, never when:
/// the novelty names the hop and the samples about it say where the strike rises.
#[test]
fn a_scores_events_land_within_one_millisecond_of_a_grid() {
    let rate = 48_000u32;
    let struck = [0.25f64, 0.5, 0.75];
    let path = written(
        "score-grid",
        rate,
        &[&strikes(rate, 1.0, &struck, (0.02, 0.08))],
    );
    let found = event_times(&path);
    assert_eq!(
        found.len(),
        struck.len(),
        "three strikes, three events: {found:?}"
    );
    for (heard, wanted) in found.iter().zip(struck) {
        assert!(
            (heard - wanted).abs() <= 0.001,
            "{heard} is more than a millisecond from {wanted}"
        );
    }
    let read = printed(&path, "score");
    let score = &read["data"]["representations"]["score"];
    assert_eq!(
        score["value"]["resolution_secs"], 0.005,
        "a reader states the hop its novelty was read at, not the sample period: {score}"
    );
    assert_eq!(score["source"], "measured", "{score}");

    let dest = path.with_file_name("score.json");
    let read = printed(&path, &format!("score={}", dest.display()));
    assert_eq!(
        read["data"]["written"]["items"][0]["path"],
        dest.display().to_string(),
        "a destination is named in `written`: {read}"
    );
    let file: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dest).expect("a written reading"))
            .expect("JSON");
    let events = &file["data"]["representations"]["score"]["value"]["events"];
    assert_eq!(events["pagination"]["count"], struck.len(), "{file}");
}

/// A novelty needs a frame before the one it scores, and the target's first has none: a
/// target that opens at silence and sounds in its first frame saw the strike that started it.
#[test]
fn a_strike_at_zero_is_an_event() {
    let rate = 48_000u32;
    let struck = [0.0f64, 0.25, 0.5];
    let path = written(
        "score-at-zero",
        rate,
        &[&strikes(rate, 0.8, &struck, (0.02, 0.08))],
    );
    let found = event_times(&path);
    assert_eq!(found.len(), struck.len(), "a strike at zero too: {found:?}");
    for (heard, wanted) in found.iter().zip(struck) {
        assert!(
            (heard - wanted).abs() <= 0.001,
            "{heard} is more than a millisecond from {wanted}"
        );
    }
}

/// A tone swelling in is one note: whatever its window's leakage moves between frames of
/// it held after is no strike.
#[test]
fn a_held_tone_is_one_event_where_it_starts() {
    let rate = 48_000u32;
    let held: Vec<f32> = (0..(f64::from(rate) * 2.0) as usize)
        .map(|i| {
            let t = i as f64 / f64::from(rate);
            let swell = (t / 0.25).min(1.0);
            (swell * libm::sin(std::f64::consts::TAU * 440.0 * t) * 0.5) as f32
        })
        .collect();
    let found = event_times(&written("score-held", rate, &[&held]));
    assert!(
        found.len() <= 1 && found.iter().all(|t| *t < 0.25),
        "a held tone strikes nothing past its start: {found:?}"
    );
}

/// The same tone one quarter turn along: no event may come or go, or move past the 1 ms
/// gate, on which phase the target's first sample happens to hold.
#[test]
fn a_held_tones_opening_phase_changes_no_event() {
    let rate = 48_000u32;
    let at_phase = |quarter: f64| {
        let held: Vec<f32> = (0..(f64::from(rate) * 2.0) as usize)
            .map(|i| {
                let t = i as f64 / f64::from(rate);
                let swell = (t / 0.25).min(1.0);
                (swell * libm::sin(std::f64::consts::TAU * 440.0 * t + quarter) * 0.5) as f32
            })
            .collect();
        event_times(&written("score-phase", rate, &[&held]))
    };
    let (zero, quarter) = (at_phase(0.0), at_phase(std::f64::consts::FRAC_PI_2));
    assert_eq!(zero.len(), quarter.len(), "{zero:?} against {quarter:?}");
    for (a, b) in zero.iter().zip(&quarter) {
        assert!((a - b).abs() <= 0.001, "{zero:?} against {quarter:?}");
    }
}

/// Strikes 80 ms apart, past the 60 ms within which two candidates are one strike, are each
/// their own event: nothing about one may fold into its neighbour.
#[test]
fn strikes_past_the_one_strike_span_are_each_their_own_event() {
    let rate = 44_100u32;
    let gap = 0.08f64;
    let struck: Vec<f64> = (0..12).map(|k| f64::from(k) * gap).collect();
    let samples = strikes(rate, gap * 12.0 + 0.2, &struck, (0.012, 0.06));
    let found = event_times(&written("score-dense", rate, &[&samples]));
    assert_eq!(found.len(), struck.len(), "every strike: {found:?}");
}

fn keys(value: &serde_json::Value) -> Vec<&str> {
    let mut held: Vec<&str> = value
        .as_object()
        .unwrap_or_else(|| panic!("an object: {value}"))
        .keys()
        .map(String::as_str)
        .collect();
    held.sort_unstable();
    held
}

/// The JSON a fit reads: each reading's fields, its events', voices', frames' and notes', and
/// the arguments it spells. A consumer outside this repository parses these names.
#[test]
fn score_loudness_pattern_sharpness_and_beating_print_the_fields_a_fit_reads() {
    let rate = 48_000u32;
    let sr = f64::from(rate);
    let chord: Vec<f32> = (0..(sr * 1.2) as usize)
        .map(|i| {
            let t = i as f64 / sr;
            let note = |hz: f64, at: f64| {
                let u = t - at;
                if u < 0.0 {
                    return 0.0;
                }
                (1..=4)
                    .map(|k| {
                        let k = f64::from(k);
                        libm::sin(std::f64::consts::TAU * hz * k * u) / k
                    })
                    .sum::<f64>()
                    * (1.0 - sva_formula::exp(-u / 0.002))
                    * sva_formula::exp(-u / 0.4)
            };
            (0.2 * (note(220.0, 0.05) + note(277.18, 0.05) + note(330.0, 0.6))) as f32
        })
        .collect();
    let path = written("score-shape", rate, &[&chord]);
    let read = printed(&path, "score,loudness_pattern,sharpness,beating");
    let held = &read["data"]["representations"];

    let score = &held["score"];
    assert_eq!(score["args"], serde_json::json!({}), "{score}");
    assert_eq!(
        keys(&score["value"]),
        ["events", "model", "resolution_secs"]
    );
    let event = &score["value"]["events"]["items"][0];
    assert_eq!(
        keys(event),
        [
            "at_secs",
            "attack_secs",
            "cue",
            "end_secs",
            "hit",
            "kind",
            "level_db",
            "main",
            "next_secs",
            "period_hz",
            "periodicity",
            "sound",
            "voices",
        ]
    );
    let voice = &event["voices"]["items"][0];
    assert_eq!(
        keys(voice),
        [
            "bend_cents",
            "coherence",
            "db",
            "hz",
            "new",
            "partials",
            "share",
            "tone"
        ]
    );
    assert_eq!(
        keys(&voice["partials"]["items"][0]),
        ["db", "harmonic", "hz"]
    );
    let events = score["value"]["events"]["items"]
        .as_array()
        .expect("events");
    assert_eq!(events.len(), 2, "two strikes: {score}");
    assert!(
        events[0]["voices"]["items"]
            .as_array()
            .expect("voices")
            .len()
            >= 2,
        "a chord is several voices: {score}"
    );

    let pattern = &held["loudness_pattern"];
    assert_eq!(
        pattern["args"],
        serde_json::json!({ "frame": "10ms", "spl": "100db" }),
        "{pattern}"
    );
    assert_eq!(
        keys(&pattern["value"]),
        [
            "centres_hz",
            "frame_secs",
            "frames",
            "model",
            "spl_db",
            "threshold_sone",
            "window_secs"
        ]
    );
    let frame = &pattern["value"]["frames"]["items"][0];
    assert_eq!(keys(frame), ["specific", "t_secs", "total_sone"]);
    assert_eq!(
        frame["specific"].as_array().expect("bands").len(),
        pattern["value"]["centres_hz"]
            .as_array()
            .expect("centres")
            .len()
    );

    let frames = pattern["value"]["frames"]["items"]
        .as_array()
        .expect("frames")
        .len();

    let sharpness = &held["sharpness"];
    assert_eq!(
        sharpness["args"],
        serde_json::json!({ "frame": "10ms", "spl": "100db" }),
        "{sharpness}"
    );
    assert_eq!(
        keys(&sharpness["value"]),
        ["frame_secs", "frames", "model", "spl_db", "window_secs"]
    );
    let held_frames = sharpness["value"]["frames"]["items"]
        .as_array()
        .expect("frames");
    assert_eq!(held_frames.len(), frames, "a frame per pattern frame");
    assert_eq!(keys(&held_frames[0]), ["acum", "t_secs"]);

    let beating = &held["beating"];
    assert_eq!(
        beating["args"],
        serde_json::json!({ "frame": "10ms", "window": "500ms", "spl": "100db" }),
        "{beating}"
    );
    assert_eq!(
        keys(&beating["value"]),
        ["frame_secs", "model", "spl_db", "window_secs", "windows"]
    );
    let windows = beating["value"]["windows"]["items"]
        .as_array()
        .expect("windows");
    assert_eq!(windows.len(), frames, "a window per pattern frame");
    assert_eq!(
        keys(&windows[0]),
        ["from_secs", "phon", "t_secs", "to_secs"]
    );
}

/// The codes a caller branches on: a reading gone, a value it cannot read, an argument a
/// reading does not take.
#[test]
fn the_score_and_perceptual_readings_refuse_with_stable_codes() {
    for (asked, code) in [
        ("onsets", "query.retired"),
        ("panel", "query.retired"),
        ("sharpness(spl=loud)", "query.bad_value"),
        ("loudness_pattern(spl=90)", "query.bad_value"),
        ("score(frame=10ms)", "query.unknown_argument"),
        ("sharpness(window=1s)", "query.unknown_argument"),
        ("beating(peaks=2)", "query.unknown_argument"),
    ] {
        let refused = parse_args(&argv(&["analyze", "/tmp/a.wav", "--representation", asked]))
            .err()
            .unwrap_or_else(|| panic!("`{asked}` refuses"));
        let codes: Vec<String> = refused.diagnostics().into_iter().map(|d| d.code).collect();
        assert_eq!(codes, [code], "`{asked}`: {refused}");
    }
}

/// A missing file is the caller's typo, exit 24, never `internal_error`.
#[test]
fn a_missing_wav_is_not_found() {
    let absent = scratch("absent-wav").join("never-written.wav");
    let Err(refused) = sva_cli::read_recording(&absent) else {
        panic!("a wav that was never written cannot be read");
    };
    assert_eq!(refused.code(), "not_found");
    assert_eq!(refused.exit_code(), 24);
}

/// Breathy voiced speech whose fundamental is often weaker than its octave, against the median
/// F0 its node files write over its voiced frames (Praat and YIN agree to 5 %). Each frame's
/// fundamental heads its longest series; their median lands within 12 %, not an octave up.
#[test]
fn breathy_speech_reads_its_fundamental_not_its_octave() {
    for (name, known) in [("phrase", 240.0), ("millennials", 240.0), ("dont", 245.0)] {
        let path = fixture(&format!("voice/{name}.wav"));
        let reading = Representation::Pitch {
            max_notes: 16,
            frame_secs: 0.05,
        };
        let found = answer_buffer("in.wav", &decoded(&path), reading, PSYCHOACOUSTIC_V1.name)
            .expect("a pitch reading");
        let Output::Pitch(frames) = found.value else {
            panic!("expected pitch frames");
        };
        let loudest = frames
            .iter()
            .filter_map(|f| f.notes.first().map(|n| n.db))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut fundamentals: Vec<f64> = frames
            .iter()
            .filter(|f| f.notes.first().is_some_and(|n| n.db > loudest - 15.0))
            .filter_map(|f| {
                let head = |mut hz: f64| {
                    while let Some(lower) = f.notes.iter().find(|n| n.hz == hz)?.harmonic_of {
                        hz = lower;
                    }
                    Some(hz)
                };
                let heads: Vec<f64> = f.notes.iter().filter_map(|n| head(n.hz)).collect();
                f.notes
                    .iter()
                    .filter(|n| n.harmonic_of.is_none())
                    .map(|n| (heads.iter().filter(|&&h| h == n.hz).count(), n.hz))
                    .filter(|(members, _)| *members > 1)
                    .max_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)))
                    .map(|(_, hz)| hz)
            })
            .collect();
        fundamentals.sort_by(f64::total_cmp);
        let median = fundamentals[fundamentals.len() / 2];
        assert!(
            (median / known - 1.0).abs() < 0.12,
            "{name}: median fundamental {median} Hz, not {known} Hz, over {fundamentals:?}"
        );
    }
}

/// What a seed reads off `partials`: every argument it ran with, defaults spelled, and per
/// track knot tables a node pastes into `pwl` and `pwl_integral`, with the flag, the
/// measures behind it and the bounds.
#[test]
fn partials_prints_its_arguments_and_tracks_a_node_pastes() {
    let rate = 16_000u32;
    let path = written("analyze-partials", rate, &[&tone(rate, 440.0, 0.6)]);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .args([
            "analyze",
            &path.display().to_string(),
            "--representation",
            "partials",
        ])
        .output()
        .expect("the binary runs");
    let printed: serde_json::Value = serde_json::from_slice(&out.stdout).expect("one envelope");
    let reading = &printed["data"]["representations"]["partials"];
    assert_eq!(
        reading["args"],
        serde_json::json!({
            "frame": "100ms",
            "window": "500ms",
            "hop": "5ms",
            "floor": "-70db",
            "peaks": 120
        }),
        "{reading}"
    );
    let tracks = reading["value"]["tracks"]["items"]
        .as_array()
        .expect("a track list");
    assert_eq!(tracks.len(), 1, "{reading}");
    let track = &tracks[0];
    assert_eq!(track["kind"], "tonal");
    assert!(
        (track["hz"].as_f64().expect("hz") - 440.0).abs() < 0.01,
        "{track}"
    );
    for (curve, unit) in [("frequency", "hz"), ("amplitude", "")] {
        let held = &track[curve];
        let times = held["times"].as_array().expect("times");
        let values = held["values"].as_array().expect("values");
        assert_eq!(times.len(), values.len(), "{held}");
        let knots: Vec<String> = times
            .iter()
            .zip(values)
            .map(|(t, v)| format!("{t}s, {v}{unit}"))
            .collect();
        assert_eq!(held["knots"], knots.join(", "), "{held}");
        assert!(held["error"].as_f64().is_some(), "{held}");
    }
    assert!(track["phase"].as_f64().is_some(), "{track}");
    assert!(track["flutter_db"].as_f64().is_some(), "{track}");
    assert!(track["coherence"].as_f64().is_some(), "{track}");
    assert!(track["bandwidth_hz"].is_null(), "{track}");
    for bound in ["resolution_hz", "resolution_secs", "residual_db"] {
        assert!(track["bound"][bound].as_f64().is_some(), "{track}");
    }
    assert!(
        track["bound"]["unclaimed_db"].is_null(),
        "a lone tone's band is its own everywhere: {track}"
    );
}
