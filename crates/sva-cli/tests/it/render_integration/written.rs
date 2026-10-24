// Concern: the readings that leave the object, as audio or as a file | Non-concern: what a render answers in the envelope (the parent suite) | IO: (fixtures/*) -> a written file

use std::path::{Path, PathBuf};

use crate::helpers::{buffer, fixture, printed, put, scratch};
use sva_cli::{SampleEncoding, write_channels};
use sva_core::{PROBE, probe};
use sva_engine::Representation;

#[test]
fn a_stereo_render_writes_an_interleaved_two_channel_wav() {
    let rendered = probe(&fixture("stereo"), "@master([0, 1s])").unwrap();
    let dir = scratch("stereo-wav");
    let path = dir.join("out.wav");
    let held = buffer(&rendered, PROBE);
    let owned: Vec<Vec<f32>> = (0..held.width()).map(|c| held.as_f32(c)).collect();
    let planes: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
    write_channels(&planes, rendered.config.rate, &path, SampleEncoding::Float).unwrap();

    let mut reader = hound::WavReader::open(&path).unwrap();
    assert_eq!(reader.spec().channels, 2);
    let pcm: Vec<f32> = reader.samples::<f32>().map(Result::unwrap).collect();
    assert_eq!(pcm.len(), held.len() * 2);
    let loudest = pcm
        .chunks_exact(2)
        .max_by(|a, b| a[0].abs().total_cmp(&b[0].abs()))
        .unwrap();
    assert!(
        loudest[0].abs() > loudest[1].abs(),
        "the left component leads in every frame, so interleaving kept the order"
    );
}

/// The destination is the one place a reading leaves the object it would have printed in.
#[test]
fn a_json_destination_carries_every_sample_where_stdout_caps_them() {
    let dir = scratch("destination");
    let out = scratch("destination-out");
    put(&dir, "master", "crop(sin(2*pi*220*t), 0s, 0.2s)\n");
    let rendered = probe(&dir, "@master").unwrap();
    let answer = rendered.answer(PROBE, Representation::Samples).unwrap();
    let path: PathBuf = out.join("s.json");
    sva_cli::write_destination(
        &printed("samples", answer, rendered.config.rate),
        &path,
        &sva_cli::Framing {
            target: "@master".to_string(),
            rate: rendered.config.rate,
            bits: Some(rendered.config.profile.precision_bits),
            interval: None,
            profile: rendered.config.profile.name,
            encoding: SampleEncoding::Float,
            replace: true,
        },
    )
    .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.contains("\"count\": 8820"));
    assert!(text.contains("\"has_more\": false"));

    let inside = dir.join("out.wav");
    assert!(
        sva_cli::refuse_inside(&dir, &inside).is_err(),
        "a destination inside the composition would be read as a node"
    );
    assert!(sva_cli::refuse_inside(&dir, Path::new("/tmp/out.wav")).is_ok());
}

/// A reading that refuses must not leave an earlier reading's file behind: every answer is
/// taken before any destination is opened.
#[test]
fn a_later_reading_refusing_writes_no_earlier_destination() {
    let dir = scratch("partial-write");
    let out = scratch("partial-write-out");
    put(&dir, "master", "crop(sin(2*pi*220*t), 0s, 0.05s)\n");
    let path = out.join("out.wav");

    let run = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(&dir)
        .args([
            "render",
            "@master",
            "--representation",
            &format!("samples={},stereo", path.display()),
        ])
        .output()
        .expect("the binary runs");
    let printed = String::from_utf8_lossy(&run.stdout);
    assert!(printed.contains("\"status\": \"error\""), "{printed}");
    assert!(printed.contains("type.width_mismatch"), "{printed}");
    assert!(
        !path.exists(),
        "a reading that refused left {} behind",
        path.display()
    );
}

/// A path already taken is the one a retry repeats: until the caller says so, the file on
/// disk is the one that was there, and no other destination is opened either.
#[test]
fn a_destination_that_already_holds_a_file_refuses_until_the_caller_confirms() {
    let dir = scratch("replace");
    let out = scratch("replace-out");
    put(&dir, "master", "sin(2*pi*220*t)\n");
    let taken = out.join("taken.json");
    let fresh = out.join("fresh.json");
    std::fs::write(&taken, "the file that was there\n").unwrap();

    let render = |extra: &[&str]| {
        let mut args = vec![
            "render".to_string(),
            "@master".to_string(),
            "--representation".to_string(),
            format!("lines={},atoms={}", fresh.display(), taken.display()),
        ];
        args.extend(extra.iter().map(|a| (*a).to_string()));
        std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
            .current_dir(&dir)
            .args(&args)
            .output()
            .expect("the binary runs")
    };

    let refused = render(&[]);
    let printed = String::from_utf8_lossy(&refused.stdout);
    assert!(printed.contains("\"code\": \"conflict\""), "{printed}");
    assert_eq!(refused.status.code(), Some(4), "{printed}");
    assert_eq!(
        std::fs::read_to_string(&taken).unwrap(),
        "the file that was there\n",
        "nothing replaced it"
    );
    assert!(
        !fresh.exists(),
        "the destination before it was never opened either"
    );

    let confirmed = render(&["--confirm"]);
    let printed = String::from_utf8_lossy(&confirmed.stdout);
    assert!(printed.contains("\"status\": \"success\""), "{printed}");
    assert!(fresh.exists() && taken.exists(), "both were written");
    assert_ne!(
        std::fs::read_to_string(&taken).unwrap(),
        "the file that was there\n",
        "the caller said so"
    );
}

/// A sample 32-bit floats cannot hold, or one past full scale in integer PCM, refuses by code
/// and leaves no file behind, rather than writing an infinity or a clipped sample.
#[test]
fn a_sample_the_encoding_cannot_hold_refuses_and_writes_nothing() {
    let dir = scratch("unrepresentable");
    let out = scratch("unrepresentable-out");
    put(&dir, "master", "crop(sample(exp(t*440)), 0s, 1s)\n");
    put(&dir, "loud", "crop(2*sin(2*pi*220*t), 0s, 0.05s)\n");
    let written = |target: &str, bits: &str, file: &str| {
        let path = out.join(file);
        let run = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"))
            .current_dir(&dir)
            .args([
                "render",
                target,
                "--bits",
                bits,
                "--representation",
                &format!("samples={}", path.display()),
            ])
            .output()
            .expect("the binary runs");
        (String::from_utf8_lossy(&run.stdout).to_string(), path)
    };
    for (target, bits, file) in [("@master", "32", "float.wav"), ("@loud", "16", "pcm.wav")] {
        let (printed, path) = written(target, bits, file);
        assert!(
            printed.contains("render.unrepresentable_sample"),
            "{printed}"
        );
        assert!(!path.exists(), "{} was written", path.display());
    }
    let (printed, path) = written("@loud", "32", "held.wav");
    assert!(printed.contains("\"status\": \"success\""), "{printed}");
    assert!(path.exists());
}
