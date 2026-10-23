// Concern: freezes `new --from` and `solve --method prox`: the starter, the answer's stages, the write, each refusal code | Non-concern: which model a stage finds | IO: (argv, a clip) -> envelope

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

use crate::helpers::{put, scratch};

fn sva(dir: &Path, args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("sva-cli runs")
}

fn json(out: &Output) -> Value {
    serde_json::from_slice(&out.stdout).expect("one JSON envelope on stdout")
}

fn argv(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

/// A short two-note clip rendered by sva-cli itself, and the directory beside it.
fn clip(name: &str) -> (PathBuf, PathBuf) {
    let root = scratch(name);
    let made = root.join("made");
    put(
        &made,
        "tone",
        "; Models: a plucked tone | Neglects: all | IO: (t) -> amplitude | Tags: fixture\n\
         0.1*exp(0 - 6*t)*(sin(2*pi*220*t) + 0.5*sin(2*pi*440*t))\n",
    );
    let wav = root.join("clip.wav");
    let out = sva(
        &made,
        &argv(&[
            "render",
            "@tone([0, 0.3s])",
            "--rate",
            "48000",
            "--bits",
            "32",
            "--cache",
            "none",
            "--representation",
            &format!("samples={}", wav.display()),
        ]),
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    (root, wav)
}

/// The answer's stages, each stage's label.
fn labels(data: &Value) -> Vec<String> {
    data["stages"]["items"]
        .as_array()
        .expect("stages")
        .iter()
        .map(|s| s["label"].as_str().expect("a label").to_string())
        .collect()
}

/// `new --from` answers the starter's files, here two harmonics a voice to keep each render
/// short, and a `solve` it may run as written. Through `path`
/// that solve answers a stage a phase and chooses the least predicted length, it alone measured;
/// without a continuation it answers one stage at the cost's price and writes it when confirmed.
#[test]
fn a_priced_solve_answers_its_stages_and_writes_its_choice() {
    let (root, wav) = clip("prox-path");
    let made = json(&sva(
        &root,
        &argv(&[
            "new",
            "fit",
            "--from",
            &wav.display().to_string(),
            "--harmonics",
            "2",
        ]),
    ));
    assert_eq!(made["status"], "success", "{made}");
    let files: Vec<&str> = made["data"]["files"]["items"]
        .as_array()
        .expect("files")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for want in ["master", "score/clip.jsonl", "fits/clip", "clip.wav"] {
        assert!(
            files.contains(&want),
            "the starter writes `{want}`: {files:?}"
        );
    }
    let next = made["data"]["next"]["items"][0]["command"]
        .as_str()
        .expect("a next command");
    let dir = root.join("fit");
    let mut words: Vec<String> = next
        .split(" --")
        .skip(1)
        .flat_map(|flag| {
            let (key, value) = flag.split_once(' ').unwrap_or((flag, ""));
            let mut out = vec![format!("--{key}")];
            if !value.is_empty() {
                out.push(value.trim_matches('\'').to_string());
            }
            out
        })
        .filter(|w| w != "--write" && w != "--confirm")
        .collect();
    let at = words
        .iter()
        .position(|w| w == "--continuation")
        .expect("a continuation");
    assert!(words[at + 1].starts_with("path:"), "{next}");
    let mut alone = argv(&["solve", "@fits/clip", "--steps", "3", "--cache", "none"]);
    alone.extend(words[..at].iter().chain(&words[at + 2..]).cloned());
    words[at + 1] = "path:0.012,0.1".to_string();
    let mut solve = argv(&["solve", "@fits/clip", "--steps", "3", "--cache", "none"]);
    solve.extend(words);
    let answered = json(&sva(&dir, &solve));
    assert_eq!(answered["status"], "success", "{answered}");
    let data = &answered["data"];
    assert_eq!(data["method"], "prox");
    assert_eq!(data["continuation"]["scheme"], "path", "{data}");
    let phases: Vec<String> = labels(data)
        .into_iter()
        .filter(|l| !l.ends_with(", restructured"))
        .collect();
    assert_eq!(phases, ["start", "settle", "price 0.012", "price 0.1"]);
    let stages = data["stages"]["items"].as_array().expect("stages");
    for key in ["label", "price", "numbers", "bits", "predicted"] {
        assert!(
            !stages[0][key].is_null(),
            "a stage answers `{key}`: {}",
            stages[0]
        );
    }
    let least = stages
        .iter()
        .map(|s| s["length_bits"].as_f64().expect("a length"))
        .fold(f64::INFINITY, f64::min);
    assert_eq!(
        data["chosen"]["length_bits"].as_f64(),
        Some(least),
        "the chosen stage is the least predicted length: {data}"
    );
    for key in ["measured", "error_bits"] {
        assert!(
            !data["chosen"][key].is_null(),
            "the chosen stage answers `{key}`: {data}"
        );
    }
    let measured = stages.iter().filter(|s| !s["measured"].is_null()).count();
    assert_eq!(measured, 1, "the chosen stage alone is measured: {data}");
    let master = std::fs::read_to_string(dir.join("master")).expect("master");

    alone.push("--write".to_string());
    let refused = sva(&dir, &alone);
    assert_eq!(refused.status.code(), Some(4), "a write needs `--confirm`");
    let data = &json(&refused)["data"];
    assert!(data["would_write"].is_object(), "{data}");
    assert!(data["continuation"].is_null(), "{data}");
    let stages = data["stages"]["items"].as_array().expect("stages");
    assert_eq!(stages.len(), 1, "no continuation solves once: {data}");
    assert!(
        !stages[0]["measured"].is_null(),
        "its one stage is measured: {data}"
    );
    let cost = data["cost_expression"].as_str().expect("the cost");
    assert!(cost.ends_with("+ repr.numbers(@master)"), "{cost}");
    assert_eq!(
        stages[0]["price"], 1.0,
        "at the cost's own price, 1: {cost}"
    );
    alone.push("--confirm".to_string());
    let written = json(&sva(&dir, &alone));
    assert_eq!(written["status"], "success", "{written}");
    assert_ne!(
        std::fs::read_to_string(dir.join("master")).expect("master"),
        master,
        "the chosen stage's numbers are written in place"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// A starter whose limits drop what the clip holds names `--continuation grow` next; run as
/// written, it answers the composition as given solved, then each level it grew, chooses the
/// least predicted length and writes it under `--confirm`.
#[test]
fn a_coarse_starter_grows_and_writes_its_choice() {
    let (root, wav) = clip("prox-grow");
    let made = json(&sva(
        &root,
        &argv(&[
            "new",
            "fit",
            "--from",
            &wav.display().to_string(),
            "--harmonics",
            "1",
            "--lone",
            "0",
        ]),
    ));
    assert_eq!(made["status"], "success", "{made}");
    let next = made["data"]["next"]["items"][0]["command"]
        .as_str()
        .expect("a next command");
    assert!(next.contains(" --continuation grow "), "{next}");
    let dir = root.join("fit");
    let mut solve = argv(&["solve", "@fits/clip", "--steps", "3", "--cache", "none"]);
    solve.extend(next.split(" --").skip(1).flat_map(|flag| {
        let (key, value) = flag.split_once(' ').unwrap_or((flag, ""));
        let mut out = vec![format!("--{key}")];
        if !value.is_empty() {
            out.push(value.trim_matches('\'').to_string());
        }
        out
    }));
    let master = std::fs::read_to_string(dir.join("master")).expect("master");
    let answered = json(&sva(&dir, &solve));
    assert_eq!(answered["status"], "success", "{answered}");
    let data = &answered["data"];
    assert_eq!(data["continuation"]["scheme"], "grow", "{data}");
    let labels = labels(data);
    assert_eq!(labels[..2], ["start", "settle"], "{labels:?}");
    assert!(
        labels[2..].iter().all(|l| l.starts_with("grow h=")),
        "{labels:?}"
    );
    assert!(data["chosen"].is_object(), "{data}");
    assert!(data["written"].is_object(), "{data}");
    assert_ne!(
        std::fs::read_to_string(dir.join("master")).expect("master"),
        master,
        "the chosen stage is written in place"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// `--harmonics` and `--lone` size the starter, and the answer names what each left out.
#[test]
fn a_starter_names_what_its_limits_leave_out() {
    let (root, wav) = clip("prox-limits");
    let made = json(&sva(
        &root,
        &argv(&[
            "new",
            "small",
            "--from",
            &wav.display().to_string(),
            "--harmonics",
            "1",
            "--lone",
            "1",
        ]),
    ));
    assert_eq!(made["status"], "success", "{made}");
    let dropped: Vec<&str> = made["data"]["dropped"]["items"]
        .as_array()
        .expect("dropped")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    for flag in ["`--harmonics 1`", "`--lone 1`"] {
        assert!(
            dropped.iter().any(|d| d.contains(flag)),
            "the answer names {flag}: {dropped:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// Each misuse of the priced method refuses under its own code.
#[test]
fn a_priced_solve_refuses_each_mistake_under_its_code() {
    let (root, wav) = clip("prox-refusals");
    let reference = wav.display().to_string();
    put(
        &root,
        "song/tone",
        "; Models: a tone | Neglects: all | IO: (t) -> amplitude | Tags: fixture\ngain = 0.1\n\
         gain*sin(2*pi*220*t)\n",
    );
    let dir = root.join("song");
    let heard = format!("norm(repr.internal(@tone) - repr.internal(sample(\"{reference}\")))");
    let priced = format!("{heard} + 0.012*repr.numbers(@tone)");
    let cases: [(Vec<String>, &str); 8] = [
        (
            argv(&[
                "solve",
                &priced,
                "--method",
                "prox",
                "--vary",
                "gain=0..1/0.1",
                "--rate",
                "44100",
            ]),
            "solve.prox_rate",
        ),
        (
            argv(&[
                "solve",
                &heard,
                "--vary",
                "gain=0..1/0.1",
                "--continuation",
                "path:0.1",
            ]),
            "cli.prox_flag",
        ),
        (
            argv(&[
                "solve",
                &priced,
                "--method",
                "prox",
                "--vary",
                "gain=0..1/0.1",
                "--continuation",
                "spiral",
            ]),
            "solve.bad_continuation",
        ),
        (
            argv(&[
                "solve",
                &priced,
                "--method",
                "prox",
                "--vary",
                "gain=0..1/0.1",
                "--continuation",
                "grow:4",
            ]),
            "solve.bad_continuation",
        ),
        (
            argv(&[
                "solve",
                &heard,
                "--method",
                "prox",
                "--vary",
                "gain=0..1/0.1",
            ]),
            "solve.prox_cost",
        ),
        (
            argv(&["solve", &priced, "--method", "prox", "--vary", "gain=0..1"]),
            "solve.prox_needs_step",
        ),
        (
            argv(&["solve", &heard, "--vary", "tone:gain=0..1/0.1"]),
            "solve.scoped_variable",
        ),
        (
            argv(&[
                "solve",
                &priced,
                "--method",
                "prox",
                "--rate",
                "48000",
                "--vary",
                "other:gain=0..1/0.1",
            ]),
            "solve.reaches_nothing",
        ),
    ];
    for (args, code) in cases {
        let out = json(&sva(&dir, &args));
        assert_eq!(
            out["data"]["diagnostics"]["items"][0]["code"], code,
            "{args:?}: {out}"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}
