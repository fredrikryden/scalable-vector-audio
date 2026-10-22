// Concern: freezes `new --from` and `solve --method prox`: the starter, the answer's fields, the write, each refusal code | Non-concern: which model the path finds | IO: (argv, a clip) -> envelope

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

/// `new --from` answers the starter's files and a `solve` it may run as written; that solve
/// answers one point a price with its axes, the chosen one, and writes it only when confirmed.
#[test]
fn a_priced_solve_answers_its_path_and_writes_its_choice() {
    let (root, wav) = clip("prox-path");
    let made = json(&sva(
        &root,
        &argv(&["new", "fit", "--from", &wav.display().to_string()]),
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
    let at = words.iter().position(|w| w == "--path").expect("a path");
    words[at + 1] = "0.012".to_string();
    let mut solve = argv(&["solve", "@fits/clip", "--steps", "3", "--cache", "none"]);
    solve.extend(words);
    let answered = json(&sva(&dir, &solve));
    assert_eq!(answered["status"], "success", "{answered}");
    let data = &answered["data"];
    assert_eq!(data["method"], "prox");
    let path = data["path"]["items"].as_array().expect("a path");
    assert_eq!(path.len(), 1, "one point a price");
    for key in ["price", "numbers", "bits", "heard", "predicted"] {
        assert!(
            !path[0][key].is_null(),
            "a point answers `{key}`: {}",
            path[0]
        );
    }
    assert!(data["chosen"].is_object(), "{data}");
    let master = std::fs::read_to_string(dir.join("master")).expect("master");

    solve.push("--write".to_string());
    let refused = sva(&dir, &solve);
    assert_eq!(refused.status.code(), Some(4), "a write needs `--confirm`");
    assert!(json(&refused)["data"]["would_write"].is_object());
    solve.push("--confirm".to_string());
    let written = json(&sva(&dir, &solve));
    assert_eq!(written["status"], "success", "{written}");
    assert_ne!(
        std::fs::read_to_string(dir.join("master")).expect("master"),
        master,
        "the chosen point's numbers are written in place"
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
    let heard = format!("sum(repr.heard(@tone) - repr.heard(sample(\"{reference}\")))");
    let priced = format!("{heard} + 0.012*repr.numbers(@tone)");
    let cases: [(Vec<String>, &str); 6] = [
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
            argv(&["solve", &heard, "--vary", "gain=0..1/0.1", "--path", "0.1"]),
            "cli.prox_flag",
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
