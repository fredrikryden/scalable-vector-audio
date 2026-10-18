// Concern: proves a score's degrees sound in the tonic and scale `variables/key` names, and refusals name the line | Non-concern: a score's JSON (sva-ast) | IO: (a key, a scale) -> its lines
use std::path::PathBuf;

use crate::helpers::{put, scratch};
use sva_core::{CliError, PROBE, probe};
use sva_engine::{Output, Representation};

const SCORE: &str = "{\"at\": \"0b\", \"sound\": \"@voice\", \"deg\": 1}\n\
                     {\"at\": \"0.5b\", \"sound\": \"@voice\", \"deg\": 3}\n";

fn composition(name: &str, key: &str, score: &str) -> PathBuf {
    let dir = scratch(&format!("score-{name}"));
    let doc = |models: &str| {
        format!("; Models: {models} | Neglects: dynamics | IO: none -> x | Tags: t\n")
    };
    put(&dir, "variables/bpm", &format!("{}120\n", doc("the tempo")));
    put(
        &dir,
        "variables/meter",
        &format!("{}4/4\n", doc("the meter")),
    );
    put(
        &dir,
        "variables/key",
        &format!("{}{key}\n", doc("the tonic")),
    );
    put(
        &dir,
        "scales/major",
        &format!("{}200\n400\n500\n700\n900\n1100\n1200\n", doc("major")),
    );
    put(
        &dir,
        "scales/minor",
        &format!("{}200\n300\n500\n700\n800\n1000\n1200\n", doc("minor")),
    );
    put(
        &dir,
        "voice",
        &format!("{}f0 = 440\nsin(2*pi*f0*t)\n", doc("a sine")),
    );
    put(&dir, "phrase", score);
    dir
}

fn lines(name: &str, key: &str) -> Vec<f64> {
    let dir = composition(name, key, SCORE);
    let rendered = probe(&dir, "@phrase([0, 1s])").unwrap_or_else(|e| panic!("{key}: {e}"));
    let Output::Lines(lines) = rendered
        .answer(PROBE, Representation::Lines)
        .unwrap_or_else(|e| panic!("{key}: {e}"))
        .value
    else {
        panic!("{key}: expected a line list");
    };
    let mut hz: Vec<f64> = lines
        .iter()
        .map(|l| l.hz.abs())
        .filter(|h| *h > 1.0)
        .collect();
    hz.sort_by(f64::total_cmp);
    hz.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    hz
}

#[test]
fn the_key_transposes_a_score_and_its_scale_line_changes_mode() {
    let major = lines("d3-major", "scale = @../scales/major\nD3");
    assert_eq!(major.len(), 2, "two degrees, two lines: {major:?}");
    let third = |hz: &[f64]| 1200.0 * (hz[1] / hz[0]).log2();
    assert!((third(&major) - 400.0).abs() < 1e-9, "{major:?}");

    let up = lines("a3-major", "scale = @../scales/major\nA3");
    let fifth = 2f64.powf(7.0 / 12.0);
    assert!(
        (up[0] / major[0] - fifth).abs() < 1e-12,
        "{up:?} against {major:?}"
    );

    let minor = lines("d3-minor", "scale = @../scales/minor\nD3");
    assert!((minor[0] - major[0]).abs() < 1e-9, "the tonic stays");
    assert!((third(&minor) - 300.0).abs() < 1e-9, "{minor:?}");
}

#[test]
fn a_refusal_names_the_score_line() {
    let bad = "{\"at\": \"0b\", \"sound\": \"@voice\", \"deg\": 1}\n\
               {\"at\": \"0.5b\", \"sound\": \"@voice\", \"cutoff\": 900}\n";
    let dir = composition("unread-key", "scale = @../scales/major\nD3", bad);
    let Err(CliError::Refusals(refused)) = probe(&dir, "@phrase([0, 1s])") else {
        panic!("a key the sound does not read refuses");
    };
    assert_eq!(refused[0].at.path, "phrase");
    assert!(
        refused[0].reason.starts_with("line 2, column 35: `cutoff`"),
        "{}",
        refused[0].reason
    );
}
