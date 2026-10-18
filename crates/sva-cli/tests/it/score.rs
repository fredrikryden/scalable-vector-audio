// Concern: proves a score's degrees sound in variables/key and variables/scale, written or set, and refusals name the line | Non-concern: a score's JSON (sva-ast) | IO: (a key, a scale) -> its lines
use std::path::PathBuf;

use crate::helpers::{put, scratch};
use sva_core::{CliError, Job, PROBE, execute, overlaid, probe};
use sva_engine::{Output, Representation, Tier};

const SCORE: &str = "{\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 1}\n\
                     {\"at\": \"0.5b\", \"node\": \"@voice\", \"deg\": 3}\n";

/// `tonic` in the scale file `scales/<scale>`.
fn composition(name: &str, (tonic, scale): (&str, &str), score: &str) -> PathBuf {
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
        &format!("{}{tonic}\n", doc("the tonic")),
    );
    put(
        &dir,
        "variables/scale",
        &format!("{}@../scales/{scale}\n", doc("the scale")),
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

fn lines(name: &str, key: (&str, &str)) -> Vec<f64> {
    set_lines(&composition(name, key, SCORE), &[])
}

/// The phrase's lines with each of `sets` read in place of its variable's file.
fn set_lines(dir: &std::path::Path, sets: &[(&str, &str)]) -> Vec<f64> {
    let sets: Vec<(String, String)> = sets
        .iter()
        .map(|(path, text)| (path.to_string(), text.to_string()))
        .collect();
    let files = sva_ast::Dir::at(dir);
    let source = overlaid(&files, &sets).unwrap_or_else(|e| panic!("{sets:?}: {e}"));
    let rendered = execute(Job::over(&source, "@phrase([0, 1s])"), &Tier::default())
        .unwrap_or_else(|e| panic!("{sets:?}: {e}"));
    let Output::Lines(lines) = rendered
        .answer(PROBE, Representation::Lines)
        .unwrap_or_else(|e| panic!("{sets:?}: {e}"))
        .value
    else {
        panic!("{sets:?}: expected a line list");
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
fn the_key_transposes_a_score_and_the_scale_changes_mode() {
    let major = lines("d3-major", ("D3", "major"));
    assert_eq!(major.len(), 2, "two degrees, two lines: {major:?}");
    let third = |hz: &[f64]| 1200.0 * (hz[1] / hz[0]).log2();
    assert!((third(&major) - 400.0).abs() < 1e-9, "{major:?}");

    let up = lines("a3-major", ("A3", "major"));
    let fifth = 2f64.powf(7.0 / 12.0);
    assert!(
        (up[0] / major[0] - fifth).abs() < 1e-12,
        "{up:?} against {major:?}"
    );

    let minor = lines("d3-minor", ("D3", "minor"));
    assert!((minor[0] - major[0]).abs() < 1e-9, "the tonic stays");
    assert!((third(&minor) - 300.0).abs() < 1e-9, "{minor:?}");
}

/// A variable set for one render is read as its file would be: the same lines, bit for bit,
/// as a composition whose file holds that text.
#[test]
fn a_set_variable_renders_as_a_file_holding_its_text_would() {
    let dir = composition("d3-major-set", ("D3", "major"), SCORE);
    assert_eq!(
        set_lines(&dir, &[("key", "A3")]),
        lines("a3-held", ("A3", "major"))
    );
    assert_eq!(
        set_lines(&dir, &[("scale", "@../scales/minor")]),
        lines("d3-minor-held", ("D3", "minor"))
    );
    assert_eq!(
        set_lines(&dir, &[("scale", "minor")]),
        lines("d3-minor-named", ("D3", "minor")),
        "a built-in sounds as the file holding its cents"
    );
    let files = sva_ast::Dir::at(&dir);
    for sets in [
        vec![("voice".to_string(), "1".to_string())],
        vec![
            ("key".to_string(), "A3".to_string()),
            ("key".to_string(), "B3".to_string()),
        ],
    ] {
        assert!(
            matches!(overlaid(&files, &sets), Err(CliError::Usage { .. })),
            "{sets:?}"
        );
    }
    let path = [("variables/key".to_string(), "A3".to_string())];
    let Err(CliError::Usage { code, message }) = overlaid(&files, &path) else {
        panic!("a set names the variable, not its file");
    };
    assert_eq!(code, "render.set_path");
    assert!(message.contains("`key=A3`"), "{message}");
}

/// A scale the language names is the scale its file of the same cents would be, line for line.
#[test]
fn a_built_in_scale_sounds_as_its_file_would() {
    for (name, ..) in sva_ast::BUILTIN_SCALES {
        let dir = composition(&format!("named-{name}"), ("D3", "major"), SCORE);
        put(&dir, "variables/scale", &format!("{name}\n"));
        let cents = sva_ast::builtin_scale(name).expect("a built-in").cents;
        let rows: Vec<String> = cents.iter().map(f64::to_string).collect();
        put(&dir, "scales/held", &format!("{}\n", rows.join("\n")));
        assert_eq!(
            set_lines(&dir, &[]),
            set_lines(&dir, &[("scale", "@../scales/held")]),
            "{name}"
        );
    }
}

#[test]
fn a_refusal_names_the_score_line() {
    let bad = "{\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 1}\n\
               {\"at\": \"0.5b\", \"node\": \"@voice\", \"cutoff\": 900}\n";
    let dir = composition("unread-key", ("D3", "major"), bad);
    let Err(CliError::Refusals(refused)) = probe(&dir, "@phrase([0, 1s])") else {
        panic!("a key the sound does not read refuses");
    };
    assert_eq!(refused[0].at.path, "phrase");
    assert!(
        refused[0].reason.starts_with("line 2, column 34: `cutoff`"),
        "{}",
        refused[0].reason
    );
}
