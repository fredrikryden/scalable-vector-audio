// Concern: freezes `solve`'s refusals, exits and envelope, `load_json`, `@fits/` and `sample("<path>")` | Non-concern: the search's path or render count | IO: (argv, a fixture copy) -> envelope, exit

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::Value;

use crate::helpers::{fixture, put, scratch};

fn sva(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(dir)
        .args(args)
        .arg("--cache")
        .arg("none")
        .output()
        .expect("sva-cli runs")
}

/// A subcommand that keeps no store, so takes no `--cache`.
fn storeless(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("sva-cli runs")
}

fn json(out: &Output) -> Value {
    serde_json::from_slice(&out.stdout).expect("one JSON envelope on stdout")
}

fn at<'v>(v: &'v Value, path: &[&str]) -> &'v Value {
    path.iter().fold(v, |v, key| &v[*key])
}

fn items(v: &Value) -> &[Value] {
    v.as_array().map_or(&[], Vec::as_slice)
}

/// A copy of the `solve` fixture, beside a reference rendered from it at `gain=0.8`.
fn composition(name: &str) -> (PathBuf, String) {
    let root = scratch(name);
    let dir = root.join("song");
    std::fs::create_dir_all(dir.join("variables")).expect("a copy");
    for rel in [
        "tone",
        "variables/gain",
        "variables/idle",
        "variables/tonic",
    ] {
        std::fs::copy(fixture("solve").join(rel), dir.join(rel)).expect("a fixture file");
    }
    let reference = root.join("ref.wav");
    let samples = format!("samples={}", reference.display());
    let made = sva(
        &dir,
        &[
            "render",
            "@tone",
            "--set",
            "gain=0.8",
            "--bits",
            "32",
            "--representation",
            &samples,
        ],
    );
    assert!(
        made.status.success(),
        "{}",
        String::from_utf8_lossy(&made.stdout)
    );
    (dir, reference.display().to_string())
}

fn envelope_cost(reference: &str) -> String {
    format!("norm(repr.envelope(@tone) - repr.envelope(sample(\"{reference}\")))")
}

/// A well-formed solve exits 0; each mistake refuses under its own code and the exit its
/// standard code names, and a write it may not make exits 4.
#[test]
fn a_solve_answers_each_exit_and_each_mistake_its_code() {
    let (dir, reference) = composition("solve-refusals");
    put(
        &dir,
        "reads-repr",
        "; Models: a node reading a representation | Neglects: all | IO: (t) -> amplitude | Tags: fixture\nrepr.envelope(@tone)\n",
    );
    let cost = envelope_cost(&reference);
    let missing = envelope_cost("/nowhere/at/all.wav");
    let cases: [(Vec<&str>, &str, i32); 10] = [
        (vec!["solve", &cost, "--vary", "gain=0..1/0.25"], "", 0),
        (
            vec!["solve", &cost, "--vary", "gain=1..0"],
            "solve.bad_vary",
            3,
        ),
        (
            vec!["solve", &cost, "--vary", "gain:0..1"],
            "solve.bad_vary",
            3,
        ),
        (vec!["solve", &cost], "solve.no_vary", 3),
        (
            vec!["solve", &cost, "--vary", "tonic=0ms..10ms"],
            "solve.reaches_no_number",
            3,
        ),
        (
            vec!["solve", &cost, "--vary", "idle=0..2"],
            "solve.reaches_nothing",
            3,
        ),
        (
            vec![
                "solve",
                "repr.envelope(@tone) - repr.envelope(@tone)",
                "--vary",
                "gain=0..1",
            ],
            "solve.vector_cost",
            3,
        ),
        (
            vec!["solve", &missing, "--vary", "gain=0..1"],
            "io.no_such_path",
            24,
        ),
        (
            vec!["render", "@reads-repr", "--representation", "loudness"],
            "engine.unbound_variable",
            3,
        ),
        (
            vec![
                "solve",
                &cost,
                "--vary",
                "gain=0..1/0.25",
                "--max-renders",
                "2",
                "--write",
            ],
            "",
            4,
        ),
    ];
    for (argv, code, exit) in cases {
        refuses(&dir, &argv, code, exit);
    }
    let answer = json(&sva(
        &dir,
        &["render", "@reads-repr", "--representation", "loudness"],
    ));
    let helps = items(at(&answer, &["data", "diagnostics", "items"]))
        .iter()
        .any(|d| {
            d["help"]
                .as_str()
                .is_some_and(|h| h.contains("sva-cli solve"))
        });
    assert!(
        helps,
        "a node writing `repr.` is told only a cost takes it: {answer:?}"
    );
}

/// `argv` exits `exit`, refusing as `code` where one is named; the envelope, for more.
fn refuses(dir: &Path, argv: &[&str], code: &str, exit: i32) -> Value {
    let out = sva(dir, argv);
    let answer = json(&out);
    assert_eq!(out.status.code(), Some(exit), "{argv:?}: {answer:?}");
    if !code.is_empty() {
        let codes = items(at(&answer, &["error", "details", "codes", "items"]));
        assert!(
            codes.iter().any(|c| c == code),
            "{argv:?} refuses as `{code}`: {answer:?}"
        );
    }
    answer
}

/// `analyze` prints each reading's `representation` and every one of its `args`; `load_json`
/// reads that `data` back as the reading its leaf subtracts, a number or numbers as weights,
/// and each mismatch refuses under its own code before anything renders.
#[test]
fn load_json_reads_a_reading_a_weight_or_refuses_under_its_code() {
    let (dir, reference) = composition("solve-load-json");
    let root = dir.parent().expect("a parent");
    let analyzed = root.join("envelope.json");
    let written = format!("envelope(frame=50ms)={}", analyzed.display());
    let out = storeless(&dir, &["analyze", &reference, "--representation", &written]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let held: Value =
        serde_json::from_slice(&std::fs::read(&analyzed).expect("written")).expect("JSON");
    let envelope = at(&held, &["data", "representations", "envelope"]);
    assert_eq!(
        (&envelope["representation"], &envelope["args"]),
        (
            &Value::from("envelope"),
            &serde_json::json!({ "frame": "50ms" })
        )
    );
    let mut short = held.clone();
    short["data"]["representations"]["envelope"]["value"]["pagination"]["has_more"] =
        Value::Bool(true);
    std::fs::write(root.join("short.json"), short.to_string()).expect("written");
    let mut without_args = held.clone();
    let read = without_args["data"]["representations"]["envelope"]
        .as_object_mut()
        .expect("an object");
    read.remove("representation");
    read.remove("args");
    std::fs::write(root.join("without_args.json"), without_args.to_string()).expect("written");
    std::fs::write(root.join("w.json"), r#"{"w": 2, "ws": [1, 0.5]}"#).expect("written");
    std::fs::write(root.join("x.json.gz"), [0x1f, 0x8b, 8, 0]).expect("written");

    let live = format!("repr.envelope(sample(\"{reference}\"), frame=50ms)");
    let tick = "0*norm(repr.envelope(@tone) - repr.envelope(@tone))";
    let against = |file: &str, path: &str| {
        format!("norm({live} - load_json(\"../{file}\", \"{path}\")) + {tick}")
    };
    let answer = refuses(
        &dir,
        &[
            "solve",
            &format!(
                "load_json(\"../w.json\", \".w\")*{} + dot(load_json(\"../w.json\", \".ws\"), [2, 4])",
                against("envelope.json", ".data")
            ),
            "--vary",
            "gain=0..1/0.5",
            "--max-renders",
            "1",
        ],
        "",
        0,
    );
    let terms = items(at(&answer, &["data", "cost", "terms", "items"]));
    assert_eq!(
        terms[0]["value"], 0.0,
        "a reading read back costs nothing: {answer:?}"
    );
    assert_eq!(terms[2]["value"], 4.0, "loaded weights weigh: {answer:?}");

    let cases: [(String, &[&str], &str, i32); 9] = [
        (against("nowhere.json", ".data"), &[], "io.no_such_path", 24),
        (against("x.json.gz", ".data"), &[], "solve.bad_json", 3),
        (
            against("envelope.json", ".data | .x"),
            &[],
            "solve.bad_path",
            3,
        ),
        (
            against("envelope.json", ".dat"),
            &[],
            "solve.no_such_key",
            3,
        ),
        (
            format!("norm(load_json(\"../envelope.json\", \".data\")) + {tick}"),
            &[],
            "solve.unpaired_load",
            3,
        ),
        (
            against("envelope.json", ".data"),
            &["--rate", "48000"],
            "solve.mismatched_load",
            3,
        ),
        (against("short.json", ".data"), &[], "solve.cut_short", 3),
        (
            against("without_args.json", ".data"),
            &[],
            "solve.load_without_args",
            3,
        ),
        (
            against("envelope.json", ".data.representations.envelope"),
            &[],
            "solve.bad_load",
            3,
        ),
    ];
    for (cost, extra, code, exit) in &cases {
        let mut argv = vec!["solve", cost.as_str(), "--vary", "gain=0..1/0.5"];
        argv.extend(extra.iter());
        let out = sva(&dir, &argv);
        assert!(out.stderr.is_empty(), "{cost}: refused before any render");
        refuses(&dir, &argv, code, *exit);
    }
}

/// `@fits/<name>` is the cost a file under `fits/` holds, over many lines; a typo refuses at
/// its line in that file, a missing one as not found, and `lint` reads no node there.
#[test]
fn a_cost_file_is_read_from_fits_and_refuses_at_its_line() {
    let (dir, reference) = composition("solve-fits");
    std::fs::create_dir_all(dir.join("fits")).expect("a directory");
    let head = "; Models: the tone against its reference | Neglects: all but level | IO: (gain) -> cost | Tags: fit\n";
    put(
        &dir,
        "fits/tone",
        &format!(
            "{head}norm(repr.envelope(@tone)   ; the render\n  - repr.envelope(sample(\"{reference}\")))\n"
        ),
    );
    put(
        &dir,
        "fits/typo",
        &format!("{head}norm(repr.envelope(@tone)\n  - nrm(1))\n"),
    );
    refuses(
        &dir,
        &[
            "solve",
            "@fits/tone",
            "--vary",
            "gain=0..1/0.25",
            "--max-renders",
            "2",
        ],
        "",
        0,
    );
    let answer = refuses(
        &dir,
        &["solve", "@fits/typo", "--vary", "gain=0..1/0.25"],
        "solve.bad_cost",
        3,
    );
    let location = at(&answer, &["data", "diagnostics", "items"])[0]["location"].clone();
    assert_eq!(location["file"], "fits/typo", "{answer:?}");
    assert_eq!(location["start"]["line"], 3, "{answer:?}");
    refuses(
        &dir,
        &["solve", "@fits/none", "--vary", "gain=0..1/0.25"],
        "io.no_such_path",
        24,
    );
    let out = storeless(&dir, &["lint"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// The fields an agent reads off a solve: every variable's value and text, the cost's total
/// and each term, the render count, why it stopped and each render in the trace.
#[test]
fn a_solve_answers_values_cost_terms_renders_stopped_by_and_trace() {
    let (dir, reference) = composition("solve-envelope");
    let cost = format!(
        "{} + 0.5*norm(repr.loudness(@tone) - repr.loudness(sample(\"{reference}\")))",
        envelope_cost(&reference)
    );
    let out = sva(
        &dir,
        &[
            "solve",
            &cost,
            "--vary",
            "gain=0..1/0.25",
            "--max-renders",
            "4",
            "--threads",
            "1",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let data = json(&out)["data"].clone();
    let gain = at(&data, &["values", "gain"]);
    assert!(gain["value"].is_f64() || gain["value"].is_u64(), "{data:?}");
    assert!(
        gain["text"].as_str().is_some_and(|t| !t.is_empty()),
        "{data:?}"
    );
    let terms = items(at(&data, &["cost", "terms", "items"]));
    assert_eq!(terms.len(), 2, "one per top-level term: {data:?}");
    assert!(at(&data, &["cost", "total"]).is_number(), "{data:?}");
    let renders = data["renders"].as_u64().expect("a count") as usize;
    assert!(renders >= 1, "{data:?}");
    assert_eq!(
        items(at(&data, &["trace", "items"])).len(),
        renders,
        "{data:?}"
    );
    let stopped = data["stopped_by"].as_str().unwrap_or_default();
    assert!(["converged", "budget"].contains(&stopped), "{data:?}");
    let lines = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        lines.lines().count(),
        renders,
        "one progress line per render: {lines}"
    );
}

/// A signal stops a solve between renders: exit 130, the best so far under `data`.
#[cfg(unix)]
#[test]
fn a_signal_stops_a_solve_with_the_best_so_far() {
    use std::io::{BufRead, BufReader};
    let (dir, reference) = composition("solve-signal");
    let cost = envelope_cost(&reference);
    let mut child = Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(&dir)
        .args([
            "solve",
            &cost,
            "--vary",
            "gain=0..1/0.001",
            "--cache",
            "none",
        ])
        .args(["--max-renders", "100000", "--step-tol", "0.0000001"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("sva-cli runs");
    let mut lines = BufReader::new(child.stderr.take().expect("stderr"));
    let mut first = String::new();
    lines.read_line(&mut first).expect("a first progress line");
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("kill runs");
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut lines, &mut rest).expect("the progress after it");
    let out = child.wait_with_output().expect("it exits");
    let answer = json(&out);
    assert_eq!(out.status.code(), Some(130), "{answer:?}");
    assert_eq!(answer["error"]["code"], "cancelled", "{answer:?}");
    assert_eq!(answer["data"]["stopped_by"], "cancelled", "{answer:?}");
    assert!(at(&answer, &["data", "values", "gain", "value"]).is_number());
}

/// `sample("<path>")` passes a recording at the render's rate through bit for bit, states
/// the bound it reads another rate under, and takes a path nowhere but as its argument.
#[test]
fn sample_of_a_path_is_the_recording_at_the_render_rate() {
    let (dir, reference) = composition("solve-sample");
    let back = dir.parent().expect("a parent").join("back.wav");
    let samples = format!("samples={}", back.display());
    let read = format!("sample(\"{reference}\")");
    let out = sva(
        &dir,
        &[
            "render",
            &read,
            "--bits",
            "32",
            "--representation",
            &samples,
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let planes = |path: &str| sva_cli::read_channels(Path::new(path)).expect("a wav").0;
    assert_eq!(planes(&reference), planes(&back.display().to_string()));

    let out = sva(
        &dir,
        &[
            "render",
            &read,
            "--rate",
            "48000",
            "--representation",
            "loudness",
        ],
    );
    let answer = json(&out);
    let one = items(at(&answer, &["data", "label", "resampled", "items"]))
        .first()
        .expect("a resampling stated");
    assert_eq!(one["from_hz"], 44_100);
    assert_eq!(one["to_hz"], 48_000);
    for key in ["passband_hz", "stopband_hz", "ripple_db", "stopband_db"] {
        assert!(one[key].as_f64().is_some_and(|v| v > 0.0), "{key}: {one:?}");
    }

    let out = sva(
        &dir,
        &["render", "0.5*\"x.wav\"", "--representation", "loudness"],
    );
    assert_eq!(out.status.code(), Some(3));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("sample("),
        "names what works"
    );
}
