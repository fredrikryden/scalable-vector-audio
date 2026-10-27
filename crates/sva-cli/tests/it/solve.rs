// Concern: freezes `solve`'s refusals, progress lines, files written, a resume, `new --from`'s stub | Non-concern: the atoms found (sva-core's solve_fit) | IO: (argv, a clip) -> envelope, stderr, files

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

/// A solve fits the clip and writes a render per pass, its composition, its render and both
/// grids sounded under `--out`; stderr hears a start line with the grid's shape and then one
/// line per pass, each an add or an adjust, the description length falling and each line's
/// change its difference; the composition renders; a held destination refuses.
#[test]
fn a_solve_writes_its_fit_and_hears_a_line_per_pass() {
    let (root, wav) = clip("solve-fit", TONE);
    let out = sva(&root, &["solve", &wav, "--out", "fit"]);
    let answer = json(&out);
    assert_eq!(answer["status"], "success", "{answer}");
    let fit = &answer["data"]["fit"];
    assert!(fit["partials"].as_u64().is_some_and(|n| n >= 1), "{fit}");
    for key in [
        "dl_bits",
        "atom_bits",
        "residual_bits",
        "gain_db",
        "secs",
        "tolerance",
    ] {
        assert!(fit[key].is_number(), "`{key}`: {fit}");
    }
    assert_eq!(fit["from_pass"], 0, "{fit}");
    assert!(
        ["converged", "tolerance"].contains(&fit["ended"].as_str().unwrap_or("")),
        "{fit}"
    );
    let heard: Vec<Value> = String::from_utf8_lossy(&out.stderr)
        .lines()
        .map(|l| serde_json::from_str(l).expect("each progress line is JSON"))
        .collect();
    let lines = heard;
    let passes = fit["passes"].as_u64().expect("a count");
    assert_eq!(Some(lines.len() as u64 - 1), Some(passes), "{fit}");
    assert_eq!(
        fit["adds"]
            .as_u64()
            .zip(fit["adjusts"].as_u64())
            .map(|(a, b)| a + b),
        Some(passes),
        "{fit}"
    );
    assert_eq!(lines[0]["action"], "start", "{}", lines[0]);
    assert!(lines[0]["atom"].is_null());
    for key in ["rate_hz", "frame_secs", "band_edges_hz", "band_floor_db"] {
        assert!(!lines[0][key].is_null(), "the start line holds `{key}`");
    }
    let bands = lines[0]["band_edges_hz"].as_array().map_or(0, Vec::len);
    let mut dl = f64::INFINITY;
    for (k, line) in lines.iter().enumerate() {
        assert_eq!(line["pass"].as_u64(), Some(k as u64), "{line}");
        let total = line["dl_bits"].as_f64().expect("a total");
        let parts = ["atom_bits", "residual_bits"].map(|p| line[p].as_f64().expect("a part"));
        assert!(
            (total - parts[0] - parts[1]).abs() <= 1e-6 * total.abs(),
            "{line}"
        );
        if k > 0 {
            assert!(["add", "adjust"].contains(&line["action"].as_str().unwrap_or("")));
            assert!(
                line.get("from_db").is_some() && line.get("to_db").is_some(),
                "{line}"
            );
            let wav = root.join("fit").join(format!("pass-{k:03}.wav"));
            assert!(wav.is_file(), "{} written", wav.display());
            assert!(line["atom"]["kind"].is_string(), "{line}");
            let delta = line["delta_dl_bits"].as_f64().expect("a change");
            assert!((delta - (total - dl)).abs() <= 1e-6 * dl.abs(), "{line}");
            assert!(total < dl, "the description length falls: {line}");
            assert_eq!(line["drop_bits"].as_f64(), Some(-delta), "{line}");
            let start = lines[0]["dl_bits"].as_f64().expect("a total");
            let so_far = line["total_drop_bits"].as_f64().expect("a drop");
            assert!(
                (start - so_far - total).abs() <= 1e-9 * start.abs(),
                "{line}"
            );
            let share = line["share"].as_f64().expect("a share");
            assert!(0.0 < share && share <= 1.0, "{line}");
            assert!(
                line["move"]
                    .as_str()
                    .is_some_and(|m| m.ends_with(line["atom"]["kind"].as_str().unwrap_or("?"))),
                "{line}"
            );
            assert!(line["pass_secs"].is_number(), "{line}");
        }
        dl = total;
        for row in line["residual_db"].as_array().into_iter().flatten() {
            assert_eq!(row.as_array().map_or(0, Vec::len), bands, "{line}");
        }
        assert!(line["envelopes"]["items"].is_array(), "{line}");
        assert!(line["skipped"]["items"].is_array(), "{line}");
    }
    for name in [
        "master",
        "score/clip.jsonl",
        "final.wav",
        "clip-grid.wav",
        "fit-grid.wav",
    ] {
        assert!(root.join("fit").join(name).is_file(), "{name} written");
    }
    let rendered = json(&sva(
        &root.join("fit"),
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
    refuses(
        &root,
        &["solve", &wav, "--out", "fit"],
        "destination.already_exists",
        4,
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// Each argv mistake refuses under its own code before anything is read or written.
#[test]
fn a_solve_refuses_each_mistake_under_its_code() {
    let root = scratch("solve-refusals");
    let cases: [(&[&str], &str, i32); 8] = [
        (&["solve"], "cli.missing_argument", 3),
        (&["solve", "clip.wav", "--spl", "loud"], "cli.bad_value", 3),
        (
            &["solve", "clip.wav", "--tolerance", "-1"],
            "cli.bad_value",
            3,
        ),
        (&["solve", "clip.wav", "--from"], "cli.missing_value", 3),
        (&["solve", "clip.wav", "--out"], "cli.missing_value", 3),
        (
            &["solve", "clip.wav", "--levels", "2"],
            "cli.unknown_argument",
            3,
        ),
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

/// A solve resumed from the fit it wrote reads that composition, numbers its passes on from
/// the last render there, keeps those renders, and passes on till no move lowers the DL; a
/// directory holding no fit refuses as unread.
#[test]
fn a_solve_resumes_from_the_fit_it_wrote() {
    let (root, wav) = clip("solve-resume", TONE);
    let first = json(&sva(&root, &["solve", &wav, "--out", "fit"]));
    assert_eq!(first["status"], "success", "{first}");
    let made = first["data"]["fit"]["passes"].as_u64().expect("a count");
    let out = sva(
        &root,
        &["solve", &wav, "--from", "fit", "--out", "fit", "--confirm"],
    );
    let answer = json(&out);
    assert_eq!(answer["status"], "success", "{answer}");
    let fit = &answer["data"]["fit"];
    assert_eq!(fit["from_pass"].as_u64(), Some(made), "{fit}");
    assert_eq!(fit["tolerance"], 0.0, "{fit}");
    assert_eq!(fit["ended"], "converged", "{fit}");
    let lines: Vec<Value> = String::from_utf8_lossy(&out.stderr)
        .lines()
        .map(|l| serde_json::from_str(l).expect("each progress line is JSON"))
        .collect();
    assert_eq!(lines[0]["action"], "start", "{}", lines[0]);
    assert_eq!(lines[0]["pass"].as_u64(), Some(made), "{}", lines[0]);
    let before = first["data"]["fit"]["dl_bits"].as_f64().expect("a total");
    let resumed = lines[0]["dl_bits"].as_f64().expect("a total");
    assert!(
        (resumed - before).abs() <= 1e-9 * before.abs(),
        "{resumed} {before}"
    );
    for (k, line) in lines.iter().enumerate().skip(1) {
        assert_eq!(line["pass"].as_u64(), Some(made + k as u64), "{line}");
    }
    assert_eq!(
        fit["passes"].as_u64(),
        Some(made + lines.len() as u64 - 1),
        "{fit}"
    );
    for k in 1..=made + lines.len() as u64 - 1 {
        let wav = root.join("fit").join(format!("pass-{k:03}.wav"));
        assert!(wav.is_file(), "{} kept or written", wav.display());
    }
    std::fs::create_dir_all(root.join("empty")).expect("a directory");
    refuses(
        &root,
        &["solve", &wav, "--from", "empty"],
        "solve.from_unread",
        3,
    );
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
