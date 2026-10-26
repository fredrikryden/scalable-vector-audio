// Concern: freezes `solve`'s refusals, its stages, progress lines and the levels it writes, and `new --from`'s stub | Non-concern: the atoms a level finds | IO: (argv, a clip) -> envelope, stderr, files

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

use crate::helpers::{put, scratch};

fn sva(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("sva-cli runs")
}

fn json(out: &Output) -> Value {
    serde_json::from_slice(&out.stdout).expect("one JSON envelope on stdout")
}

fn items(v: &Value) -> &[Value] {
    v["items"].as_array().map_or(&[], Vec::as_slice)
}

const TONE: &str = "0.1*exp(0 - 6*t)*sin(2*pi*440*t)";

/// A short clip of `body` rendered by sva-cli itself at a low rate, so a solve is quick.
fn clip(name: &str, body: &str) -> (PathBuf, String) {
    let root = scratch(name);
    let made = root.join("made");
    put(
        &made,
        "tone",
        &format!(
            "; Models: a fixture clip | Neglects: all | IO: (t) -> amplitude | Tags: fixture\n{body}\n"
        ),
    );
    let wav = root.join("clip.wav");
    let samples = format!("samples={}", wav.display());
    let out = sva(
        &made,
        &[
            "render",
            "@tone([0, 0.3s])",
            "--rate",
            "8000",
            "--bits",
            "32",
            "--cache",
            "none",
            "--representation",
            &samples,
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    (root, wav.display().to_string())
}

/// `argv` exits `exit` refusing as `code`; the envelope, for more.
fn refuses(dir: &Path, argv: &[&str], code: &str, exit: i32) -> Value {
    let out = sva(dir, argv);
    let answer = json(&out);
    assert_eq!(out.status.code(), Some(exit), "{argv:?}: {answer:?}");
    let codes = items(&answer["error"]["details"]["codes"]);
    assert!(
        codes.iter().any(|c| c == code),
        "{argv:?} refuses as `{code}`: {answer:?}"
    );
    answer
}

/// A solve answers one stage per level, coarsest first, each heard on stderr as it finishes
/// with its terms, and writes each level's composition and wav as it does; every activity is
/// named, its comparisons never falling, each level's composition renders, and N0 is kept
/// under `--cache`.
#[test]
fn a_solve_answers_and_writes_a_stage_per_level() {
    let (root, wav) = clip("solve-levels", TONE);
    let n0 = root.join("n0").display().to_string();
    let argv = [
        "solve", &wav, "--levels", "2", "--out", "fit", "--cache", &n0,
    ];
    let out = sva(&root, &argv);
    let answer = json(&out);
    assert_eq!(answer["status"], "success", "{answer}");
    let data = &answer["data"];
    assert_eq!(data["continuation"]["scheme"], "pyramid", "{data}");
    let stages = items(&data["stages"]);
    let labels: Vec<&str> = stages.iter().filter_map(|s| s["label"].as_str()).collect();
    assert_eq!(labels, ["level 1 of 2", "level 2 of 2"], "{data}");
    assert!(data["chosen"].is_object(), "{data}");
    let lines: Vec<Value> = String::from_utf8_lossy(&out.stderr)
        .lines()
        .map(|l| serde_json::from_str(l).expect("each progress line is JSON"))
        .collect();
    let names = [
        "reading_reference",
        "ranking_proposals",
        "descending",
        "measuring",
    ];
    let mut done = 0;
    for line in lines.iter().filter(|l| l.get("action").is_some()) {
        let action = line["action"].as_str().unwrap_or_default();
        assert!(names.contains(&action), "an action by name: {line}");
        let work = line["work"]["comparisons"].as_u64().expect("a count");
        assert!(work >= done, "the work never falls, from {done}: {line}");
        done = work;
    }
    let finished: Vec<&Value> = lines.iter().filter(|l| l["finished"] == true).collect();
    let heard: Vec<&str> = finished
        .iter()
        .filter_map(|l| l["stage"].as_str())
        .collect();
    assert_eq!(heard, labels, "a finished line per stage");
    for line in &finished {
        for key in ["residual_bits", "bits", "numbers", "length_bits"] {
            assert!(
                line[key].is_number(),
                "a finished stage hears `{key}`: {line}"
            );
        }
    }
    for (k, stage) in stages.iter().enumerate() {
        let level = root.join(format!("fit/level-{}", k + 1));
        assert_eq!(
            stage["directory"].as_str(),
            Some(format!("fit/level-{}", k + 1).as_str())
        );
        assert!(root.join(format!("fit/level-{}.wav", k + 1)).is_file());
        let rendered = json(&sva(
            &level,
            &[
                "render",
                "@master([0, 0.3s])",
                "--cache",
                "none",
                "--representation",
                "loudness",
            ],
        ));
        assert_eq!(rendered["status"], "success", "{rendered}");
    }
    let kept = |dir: &Path| {
        let mut files = Vec::new();
        let mut open = vec![dir.to_path_buf()];
        while let Some(d) = open.pop() {
            for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                match entry.path().is_dir() {
                    true => open.push(entry.path()),
                    false => files.push(entry.path()),
                }
            }
        }
        files
    };
    assert_eq!(kept(&root.join("n0")).len(), 2, "N0 kept per level's scale");
    refuses(
        &root,
        &[
            "solve", &wav, "--levels", "1", "--out", "fit", "--cache", &n0,
        ],
        "destination.already_exists",
        4,
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// Each argv mistake refuses under its own code before anything is read or written.
#[test]
fn a_solve_refuses_each_mistake_under_its_code() {
    let root = scratch("solve-refusals");
    let cases: [(&[&str], &str, i32); 5] = [
        (&["solve"], "cli.missing_argument", 3),
        (&["solve", "clip.wav", "--levels", "0"], "cli.bad_value", 3),
        (&["solve", "clip.wav", "--spl", "loud"], "cli.bad_value", 3),
        (
            &["solve", "clip.wav", "--vary", "gain=0..1"],
            "cli.unknown_argument",
            3,
        ),
        (&["solve", "nowhere.wav"], "io.no_such_path", 24),
    ];
    for (argv, code, exit) in cases {
        refuses(&root, argv, code, exit);
    }
    assert!(!root.join("fit").exists(), "nothing is written");
    let _ = std::fs::remove_dir_all(&root);
}

/// `new --from` refuses under its stub code and writes nothing; a node reading `repr.` is
/// told nothing takes it now, naming what does.
#[test]
fn new_from_refuses_as_unbuilt_and_a_node_reading_repr_is_told_why() {
    let dir = scratch("solve-unbuilt");
    let out = sva(&dir, &["new", "fit", "--from", "clip.wav"]);
    let answer = json(&out);
    assert_eq!(out.status.code(), Some(3), "{answer}");
    let codes = items(&answer["error"]["details"]["codes"]);
    assert!(codes.iter().any(|c| c == "seed.unbuilt"), "{answer}");
    assert!(!dir.join("fit").exists(), "nothing is written");
    put(
        &dir,
        "tone",
        "; Models: a tone | Neglects: all | IO: (t) -> amplitude | Tags: fixture\nsin(2*pi*220*t)\n",
    );
    put(
        &dir,
        "reads-repr",
        "; Models: a node reading a representation | Neglects: all | IO: (t) -> amplitude | Tags: fixture\nrepr.envelope(@tone)\n",
    );
    let answer = json(&sva(
        &dir,
        &[
            "render",
            "@reads-repr",
            "--cache",
            "none",
            "--representation",
            "loudness",
        ],
    ));
    let helps = items(&answer["data"]["diagnostics"]).iter().any(|d| {
        d["help"]
            .as_str()
            .is_some_and(|h| h.contains("sva-cli solve <recording>"))
    });
    assert!(helps, "{answer:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
