// Concern: freezes `new --from` and `solve --method prox`: the starter, its stages, the write, a zero-step score, each refusal code | Non-concern: the model a stage finds | IO: (argv, clip) -> envelope

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

const TONE: &str = "0.1*exp(0 - 6*t)*(sin(2*pi*220*t) + 0.5*sin(2*pi*440*t))";

/// A short clip of `body` rendered by sva-cli itself, and the directory beside it.
fn clip(name: &str, body: &str) -> (PathBuf, PathBuf) {
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

/// A `next` command's flags as argv words, each quoted value unquoted.
fn flags(next: &str) -> Vec<String> {
    next.split(" --")
        .skip(1)
        .flat_map(|flag| {
            let (key, value) = flag.split_once(' ').unwrap_or((flag, ""));
            let mut out = vec![format!("--{key}")];
            if !value.is_empty() {
                out.push(value.trim_matches('\'').to_string());
            }
            out
        })
        .collect()
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

/// `new --from` answers the starter's files, two harmonics a voice, and a `solve` to run.
/// Through `path` that solve answers a stage a phase, each phase ending no higher than it
/// started, and chooses the least predicted length, it alone measured;
/// without a continuation it answers one stage at the cost's price and writes it when confirmed.
#[test]
#[ignore = "spec for the solve rewrite"]
fn a_priced_solve_answers_its_stages_and_writes_its_choice() {
    let (root, wav) = clip("prox-path", TONE);
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
    let mut words: Vec<String> = flags(next)
        .into_iter()
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
    let out = sva(&dir, &solve);
    let answered = json(&out);
    assert_eq!(answered["status"], "success", "{answered}");
    let mut started = std::collections::HashMap::new();
    let mut finished = Vec::new();
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        let Ok(p) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if p.get("action").is_some() {
            continue;
        }
        if p["finished"] == true {
            let files = p["files"]["items"].as_array().expect("files");
            assert!(
                files.iter().all(|f| f["text"].is_null()),
                "no texts: {line}"
            );
            finished.push(p["stage"].as_str().expect("a stage").to_string());
            continue;
        }
        let objective = p["residual_bits"].as_f64().expect("a residual")
            + p["price"].as_f64().expect("a price") * p["bits"].as_f64().expect("bits");
        let phase = (
            p["stage"].to_string(),
            p["phase"].to_string(),
            p["draws"].to_string(),
        );
        if p["step"] == 0 {
            started.insert(phase.clone(), objective);
        }
        if !p["ended"].is_null() {
            let start = started[&phase];
            assert!(
                objective <= start,
                "a phase ends no higher than it starts, {start}: {line}"
            );
        }
    }
    assert!(!started.is_empty(), "each phase prints its start");
    let data = &answered["data"];
    assert_eq!(data["method"], "prox");
    assert_eq!(data["continuation"]["scheme"], "path", "{data}");
    let phases: Vec<String> = labels(data)
        .into_iter()
        .filter(|l| !l.ends_with(", restructured"))
        .collect();
    assert_eq!(phases, ["start", "settle", "price 0.012", "price 0.1"]);
    assert_eq!(
        finished,
        labels(data),
        "a stage finished line per listed stage"
    );
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
/// least predicted length and writes it under `--confirm`. Stderr names every activity, its
/// comparisons never falling, and each finished stage's terms.
#[test]
#[ignore = "spec for the solve rewrite"]
fn a_coarse_starter_grows_and_writes_its_choice() {
    let (root, wav) = clip("prox-grow", TONE);
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
    let mut solve = argv(&[
        "solve",
        "@fits/clip",
        "--steps",
        "3",
        "--cache",
        "none",
        "--heartbeat",
        "0",
    ]);
    solve.extend(flags(next));
    let master = std::fs::read_to_string(dir.join("master")).expect("master");
    let out = sva(&dir, &solve);
    let answered = json(&out);
    assert_eq!(answered["status"], "success", "{answered}");
    let lines: Vec<Value> = String::from_utf8_lossy(&out.stderr)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let names = [
        "seeding",
        "reading_reference",
        "rendering_noise",
        "descending",
        "debiasing",
        "ranking_proposals",
        "trying_proposal",
        "evaluating_at_cap",
        "measuring",
    ];
    let mut done = 0;
    let mut heard = std::collections::BTreeSet::new();
    for line in lines.iter().filter(|l| l.get("action").is_some()) {
        let action = line["action"].as_str().unwrap_or_default();
        assert!(names.contains(&action), "an action by name: {line}");
        heard.insert(action);
        let work = line["work"]["comparisons"].as_u64().expect("a count");
        assert!(work >= done, "the work never falls, from {done}: {line}");
        done = work;
    }
    for want in names.iter().filter(|n| **n != "rendering_noise") {
        assert!(heard.contains(want), "`{want}` is heard: {heard:?}");
    }
    assert!(done > 0, "the solve compared cells");
    let finished: Vec<&Value> = lines.iter().filter(|l| l["finished"] == true).collect();
    assert_eq!(finished.len(), labels(&answered["data"]).len());
    for line in finished {
        for key in ["residual_bits", "bits", "numbers", "predicted_at_cap"] {
            assert!(
                !line[key].is_null(),
                "a finished stage hears `{key}`: {line}"
            );
        }
    }
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

/// A clip with no pitch seeds no voice node, and its `next` varies only what the starter
/// reads: none of them is refused as reaching nothing.
#[test]
#[ignore = "spec for the solve rewrite"]
fn an_unpitched_starter_varies_only_what_it_reads() {
    let (root, wav) = clip(
        "prox-unpitched",
        "sample(0.1*noise(7, period=8s))*exp(0 - 30*t)",
    );
    let made = json(&sva(
        &root,
        &argv(&["new", "fit", "--from", &wav.display().to_string()]),
    ));
    assert_eq!(made["status"], "success", "{made}");
    let next = made["data"]["next"]["items"][0]["command"]
        .as_str()
        .expect("a next command");
    assert!(
        !next.contains("--vary 'tilt="),
        "no voice node reads `tilt`: {next}"
    );
    let mut solve = argv(&["solve", "@fits/clip", "--steps", "1", "--cache", "none"]);
    solve.extend(
        flags(next)
            .into_iter()
            .filter(|w| w != "--write" && w != "--confirm"),
    );
    let answered = json(&sva(&root.join("fit"), &solve));
    let codes: Vec<&Value> = answered["data"]["diagnostics"]["items"]
        .as_array()
        .map(|d| d.iter().map(|x| &x["code"]).collect())
        .unwrap_or_default();
    assert!(
        !codes.iter().any(|c| *c == "solve.reaches_nothing"),
        "{answered}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// A starter whose every score line is silent writes a score of no events: it solves at no
/// steps and at some, and the score it writes renders silence.
#[test]
#[ignore = "spec for the solve rewrite"]
fn an_all_silent_fit_writes_a_score_that_renders_silence() {
    let (root, wav) = clip(
        "prox-silent",
        "sample(0.1*noise(7, period=8s))*exp(0 - 30*t)",
    );
    let made = json(&sva(
        &root,
        &argv(&["new", "fit", "--from", &wav.display().to_string()]),
    ));
    assert_eq!(made["status"], "success", "{made}");
    let dir = root.join("fit");
    let score = dir.join("score/clip.jsonl");
    let starter = std::fs::read_to_string(&score).expect("the starter's score");
    assert!(
        starter.lines().all(|l| l.contains("\"vel\": 0,")),
        "every line is silent: {starter}"
    );
    let next = made["data"]["next"]["items"][0]["command"]
        .as_str()
        .expect("a next command");
    let varies: Vec<String> = flags(next)
        .chunks(2)
        .filter(|f| f[0] == "--vary")
        .flatten()
        .cloned()
        .collect();
    for extra in [
        &["--steps", "0"][..],
        &["--steps", "2", "--write", "--confirm"][..],
    ] {
        let mut words = argv(&["solve", "@fits/clip", "--method", "prox", "--cache", "none"]);
        words.extend(argv(extra));
        words.extend(varies.iter().cloned());
        let answered = json(&sva(&dir, &words));
        assert_eq!(answered["status"], "success", "{extra:?}: {answered}");
    }
    assert_eq!(
        std::fs::read_to_string(&score).expect("the written score"),
        "",
        "no line sounds, so none is written"
    );
    let out = root.join("silent.wav");
    let rendered = json(&sva(
        &dir,
        &argv(&[
            "render",
            "@score/clip([0, 0.3s])",
            "--cache",
            "none",
            "--representation",
            &format!("samples={}", out.display()),
        ]),
    ));
    assert_eq!(rendered["status"], "success", "{rendered}");
    let bytes = std::fs::read(&out).expect("the rendered wav");
    let data = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("a data chunk")
        + 8;
    assert!(
        bytes.len() > data && bytes[data..].iter().all(|b| *b == 0),
        "an empty score renders silence"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// A signal during a trial that builds a later stage stops the run there: exit 130, and the
/// last of `stages` is that trial under its progress lines' label, `stopped` with the descents
/// it ran, no length at the cap's draws and not chosen.
#[cfg(unix)]
#[test]
#[ignore = "spec for the solve rewrite"]
fn a_signal_in_a_stage_lists_it_stopped() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    let (root, wav) = clip("prox-signal", TONE);
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
    let next = made["data"]["next"]["items"][0]["command"]
        .as_str()
        .expect("a next command");
    let mut child = Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(root.join("fit"))
        .args(argv(&[
            "solve",
            "@fits/clip",
            "--steps",
            "30",
            "--cache",
            "none",
        ]))
        .args(
            flags(next)
                .into_iter()
                .filter(|w| w != "--write" && w != "--confirm"),
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("sva-cli runs");
    let mut lines = BufReader::new(child.stderr.take().expect("stderr"));
    let trial = loop {
        let mut line = String::new();
        assert_ne!(
            lines.read_line(&mut line).expect("a line"),
            0,
            "a trial runs"
        );
        let Ok(p) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if p.get("action").is_some() {
            continue;
        }
        let stage = p["stage"].as_str().expect("a stage").to_string();
        if stage.contains(" trial ") {
            break stage;
        }
    };
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("kill runs");
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut lines, &mut rest).expect("the progress after it");
    let out = child.wait_with_output().expect("it exits");
    let answer = json(&out);
    assert_eq!(out.status.code(), Some(130), "{answer}");
    assert_eq!(answer["error"]["code"], "cancelled", "{answer}");
    let data = &answer["data"];
    let stages = data["stages"]["items"].as_array().expect("stages");
    let (stopped, finished) = stages.split_last().expect("a stage");
    assert_eq!(stopped["label"], trial.as_str(), "{data}");
    assert_eq!(stopped["stopped"], true, "{stopped}");
    assert!(stopped["length_bits"].is_null(), "{stopped}");
    let descents = stopped["descents"]["items"].as_array().expect("descents");
    assert_eq!(
        descents.last().expect("a descent")["ended"],
        "signal",
        "{stopped}"
    );
    assert!(
        finished.iter().all(|s| s["stopped"] == false),
        "the stages before it finished: {data}"
    );
    assert_ne!(data["chosen"]["label"], trial.as_str(), "{data}");
    let _ = std::fs::remove_dir_all(&root);
}

/// Every file under `dir` and its text.
fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).expect("a directory") {
        let path = entry.expect("an entry").path();
        match path.is_dir() {
            true => out.extend(snapshot(&path)),
            false => out.push((path.clone(), std::fs::read(&path).expect("a file"))),
        }
    }
    out.sort();
    out
}

/// `--steps 0` scores a composition as it is: a fit written under `--confirm`, solved again with
/// no steps, answers one stage measured as the fit measured what it wrote, changes no file and
/// lists none; with `--write` or a continuation it refuses instead.
#[test]
#[ignore = "spec for the solve rewrite"]
fn zero_steps_measure_the_composition_as_written() {
    let (root, wav) = clip("prox-zero", TONE);
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
    let next = made["data"]["next"]["items"][0]["command"]
        .as_str()
        .expect("a next command");
    let mut varies: Vec<Vec<String>> = flags(next)
        .chunks(2)
        .filter(|f| f[0] == "--vary")
        .map(<[String]>::to_vec)
        .collect();
    let dir = root.join("fit");
    let solve = |extra: &[&str], varies: &[Vec<String>]| {
        let mut words = argv(&["solve", "@fits/clip", "--method", "prox", "--cache", "none"]);
        words.extend(argv(extra));
        words.extend(varies.iter().flatten().cloned());
        json(&sva(&dir, &words))
    };
    let fitted = solve(&["--steps", "3", "--write", "--confirm"], &varies);
    assert_eq!(fitted["status"], "success", "{fitted}");
    let written = snapshot(&dir);

    // The write drops the silent lines and the numbers only they read; so do their `--vary`s.
    let scored = loop {
        let scored = solve(&["--steps", "0"], &varies);
        let refusal = &scored["data"]["diagnostics"]["items"][0];
        if refusal["code"] != "solve.reaches_nothing" {
            break scored;
        }
        let message = refusal["message"].as_str().expect("a message");
        let name = message
            .strip_prefix("`--vary ")
            .and_then(|m| m.split_once('`'))
            .expect("the refusal names its `--vary`")
            .0;
        let before = varies.len();
        varies.retain(|v| v[1].split_once('=').expect("a vary").0 != name);
        assert!(
            varies.len() < before,
            "`{name}` is a `--vary` given: {message}"
        );
    };
    assert_eq!(scored["status"], "success", "{scored}");
    let data = &scored["data"];
    let stages = data["stages"]["items"].as_array().expect("stages");
    assert_eq!(stages.len(), 1, "no scheme answers one stage: {data}");
    let descents = data["chosen"]["descents"]["items"]
        .as_array()
        .expect("descents");
    assert!(
        descents.iter().all(|d| d["steps"] == 0),
        "no step is taken: {descents:?}"
    );
    assert!(!data["chosen"]["predicted"].is_null(), "{data}");
    assert_eq!(
        data["chosen"]["measured"], fitted["data"]["chosen"]["measured"],
        "the composition as written measures as the fit that wrote it measured it"
    );
    assert_eq!(data["files"]["items"], serde_json::json!([]), "{data}");
    assert!(
        snapshot(&dir) == written,
        "no file under the composition changes"
    );

    for (extra, code) in [
        (["--write", "--confirm"], "solve.prox_write"),
        (["--continuation", "path:0.1"], "solve.prox_steps"),
    ] {
        let mut words = vec!["--steps", "0"];
        words.extend(extra);
        let refused = solve(&words, &varies);
        assert_eq!(
            refused["data"]["diagnostics"]["items"][0]["code"], code,
            "{extra:?}: {refused}"
        );
    }
    assert!(snapshot(&dir) == written, "a refusal changes no file");
    let _ = std::fs::remove_dir_all(&root);
}

/// Each misuse of the priced method refuses under its own code.
#[test]
#[ignore = "spec for the solve rewrite"]
fn a_priced_solve_refuses_each_mistake_under_its_code() {
    let (root, wav) = clip("prox-refusals", TONE);
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
    let cropped = format!(
        "norm(repr.internal(@tone([0, 0.1s])) - repr.internal(sample(\"{reference}\")([0, 0.2s]))) \
         + 0.012*repr.numbers(@tone)"
    );
    let cases: [(Vec<String>, &str); 9] = [
        (
            argv(&[
                "solve",
                &cropped,
                "--method",
                "prox",
                "--vary",
                "gain=0..1/0.1",
            ]),
            "solve.prox_cost",
        ),
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
