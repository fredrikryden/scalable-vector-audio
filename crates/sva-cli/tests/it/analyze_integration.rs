// Concern: exercises `analyze` over a recording, the one reading path with no graph behind it | Non-concern: rendering one (render_integration.rs) | IO: (a recording) -> an Answer or CliError

use crate::helpers::{fixture, scratch};

use sva_cli::{SampleEncoding, parse_args, write_channels};
use sva_engine::{Buffer, Output, PSYCHOACOUSTIC_V1, Representation, answer_buffer};

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

/// The JSON a fit reads: each reading's fields, its hits', frames' and windows', and the
/// arguments it spells. A consumer outside this repository parses these names.
#[test]
fn hits_loudness_pattern_sharpness_and_beating_print_the_fields_a_fit_reads() {
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
    let path = written("hits-shape", rate, &[&chord]);
    let read = printed(&path, "hits,loudness_pattern,sharpness,beating");
    let held = &read["data"]["representations"];

    let hits = &held["hits"];
    assert_eq!(hits["args"], serde_json::json!({}), "{hits}");
    assert_eq!(
        keys(&hits["value"]),
        ["bands_hz", "hits", "model", "resolution_secs"]
    );
    assert_eq!(
        keys(&hits["value"]["hits"]["items"][0]),
        [
            "at_secs",
            "attack_secs",
            "band_db",
            "cue",
            "end_secs",
            "kind",
            "level_db",
            "next_secs",
            "strength",
        ]
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

/// `internal`'s fields, each channel's and each band's: a band holds one element per
/// `1 / rate_hz` from the clip's start to its last frame.
#[test]
fn internal_prints_each_channels_bands_and_their_elements() {
    let rate = 16_000u32;
    let path = written("internal-shape", rate, &[&tone(rate, 1000.0, 0.3)]);
    let read = printed(&path, "internal");
    let internal = &read["data"]["representations"]["internal"];
    assert_eq!(
        internal["args"],
        serde_json::json!({ "spl": "100db" }),
        "{internal}"
    );
    let value = &internal["value"];
    assert_eq!(
        keys(value),
        [
            "channels",
            "hop_secs",
            "model",
            "rate_hz",
            "scale",
            "spl_db",
            "start_secs"
        ]
    );
    let channels = value["channels"]["items"].as_array().expect("channels");
    assert_eq!(channels.len(), 149);
    assert_eq!(keys(&channels[0]), ["bands", "hz", "window_secs"]);
    let frames = 300.0;
    for channel in channels {
        for band in channel["bands"]["items"].as_array().expect("bands") {
            assert_eq!(
                keys(band),
                [
                    "bandwidth_hz",
                    "element",
                    "elements",
                    "hz",
                    "rate_bound_hz",
                    "rate_hz"
                ]
            );
            let per_frame = band["rate_hz"].as_f64().expect("a rate") * 0.001;
            assert_eq!(
                band["elements"].as_array().expect("elements").len(),
                libm::floor(frames * per_frame) as usize + 1,
                "{band}"
            );
        }
    }
}

/// The codes a caller branches on: a value a reading cannot read, an argument it does not take.
#[test]
fn the_hits_and_perceptual_readings_refuse_with_stable_codes() {
    for (asked, code) in [
        ("sharpness(spl=loud)", "query.bad_value"),
        ("loudness_pattern(spl=90)", "query.bad_value"),
        ("hits(frame=10ms)", "query.unknown_argument"),
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
