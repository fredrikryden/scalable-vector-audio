// Concern: proves lint checks each file under its rules, and a target under its render's types | Non-concern: the per-node text checks (sva-cli lint.rs) | IO: (a composition) -> a report or a refusal

use crate::helpers::{doc, scratch};
use sva_cli::lint;
use sva_core::LintCode;

fn composition(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = scratch(name);
    for (rel, body) in files {
        std::fs::write(dir.join(rel), format!("{}{body}", doc(rel))).expect("a node file");
    }
    dir
}

fn codes(report: &sva_cli::LintReport) -> String {
    report
        .findings
        .iter()
        .map(|f| format!("{} on {}", f.code.code_str(), f.subject))
        .collect::<Vec<_>>()
        .join(", ")
}

/// A check that knows the line it found something on carries it into the diagnostic, so a
/// caller reading the response can open the file at it.
#[test]
fn a_lint_finding_names_its_line() {
    let dir = scratch("lint-line");
    let block = format!(";{}\n", "x".repeat(1_100));
    std::fs::write(
        dir.join("master"),
        format!(
            "{}crop(sin(2*pi*220*t), 0s, 1s)\n{block}",
            doc("a tone under a long note")
        ),
    )
    .expect("a node file");

    let Err(refused) = lint(&dir, None) else {
        panic!("a comment block over the threshold refuses")
    };
    let found = refused.diagnostics();
    let [held] = &found[..] else {
        panic!("one block, one diagnostic, got {}", found.len())
    };
    assert_eq!(held.code, "lint.long_comment_block");
    assert_eq!(
        held.file.as_deref(),
        Some("master"),
        "the node it was found in"
    );
    assert_eq!(
        held.start,
        Some((3, 1)),
        "the line the run it found starts on"
    );
}

/// A rate is what a render chooses, so a node written against one sounds different at every
/// other. The refusal names the `sp` spelling that holds at any of them.
#[test]
fn a_literal_sample_rate_refuses() {
    let dir = composition(
        "literal-rate",
        &[
            ("acc", "self[idx(t) - 1] + sample(sin(2*pi*220*t))/44100\n"),
            ("clean", "self[idx(t) - 1] + sample(sin(2*pi*220*t))*1sp\n"),
        ],
    );
    let Err(sva_core::CliError::LintRefused(found)) = lint(&dir, None) else {
        panic!("a literal rate refuses")
    };
    let refused: Vec<&str> = found
        .iter()
        .filter(|f| f.code == LintCode::LiteralSampleRate)
        .map(|f| f.subject.as_str())
        .collect();
    assert_eq!(refused, vec!["acc"], "only the node that wrote one");
    assert!(found[0].message.contains("1sp"), "the repair is named");
}

/// A random source is a function of time, so a draw with no key is refused rather than
/// read as one constant, at lint as at render.
#[test]
fn a_draw_with_no_key_refuses() {
    let dir = composition(
        "keyless-draw",
        &[
            ("still", "crop(2*rand(seed=3) - 1, 0s, 1s)\n"),
            ("moving", "crop(2*rand(t, seed=3) - 1, 0s, 1s)\n"),
        ],
    );
    let Err(sva_core::CliError::LintRefused(found)) = lint(&dir, None) else {
        panic!("a keyless draw refuses")
    };
    let refused: Vec<&str> = found
        .iter()
        .filter(|f| f.code == LintCode::Arity)
        .map(|f| f.subject.as_str())
        .collect();
    assert_eq!(refused, vec!["still"], "only the node that drew no key");
    assert!(
        found[0].message.contains("rand(t, seed=k)"),
        "the repair is named"
    );
    let rendered = sva_core::execute(
        sva_core::Job::over(&sva_ast::Dir::at(&dir), "@still"),
        &sva_core::Tier::default(),
    );
    let Err(refused) = rendered else {
        panic!("a render of the keyless draw refuses too")
    };
    let codes: Vec<String> = refused.diagnostics().into_iter().map(|d| d.code).collect();
    assert_eq!(codes, vec!["grammar.arity"], "{codes:?}");
}

/// A reserved `variables/` node is read by name, so no ref walk reaches one.
#[test]
fn lint_of_one_node_sees_the_tempo() {
    let dir = composition(
        "one-node-tempo",
        &[("kick", "crop(sin(2*pi*50*t), 0s, 2b)\n")],
    );
    std::fs::create_dir_all(dir.join("variables")).expect("a variables directory");
    for (name, body) in [("bpm", "120\n"), ("meter", "4/4\n")] {
        std::fs::write(
            dir.join("variables").join(name),
            format!("{}{body}", doc(name)),
        )
        .expect("a variable");
    }

    let report = lint(&dir, Some("@kick")).expect("one node lints under its own composition");
    assert_eq!(codes(&report), "", "nothing to report: {}", codes(&report));
}

/// A target names the instance its defaults resolved to, as `render` does.
#[test]
fn lint_of_a_node_with_a_default_resolves() {
    let dir = composition(
        "defaulted-node",
        &[
            ("plain", "crop(sin(2*pi*440*t), 0s, 1s)\n"),
            ("voice", "f0 = 440\ncrop(sin(2*pi*f0*t), 0s, 1s)\n"),
        ],
    );

    assert_eq!(
        codes(&lint(&dir, Some("@plain([0, 1s])")).expect("a plain node")),
        ""
    );
    let report = lint(&dir, Some("@voice([0, 1s])")).expect("a node that declares a default");
    assert_eq!(codes(&report), "", "{}", codes(&report));
    assert!(report.nodes > 0, "one node was checked, so one is reported");
}

const ANNOTATION: &str = "{\"annotation\": \"Models: a kick on every line | Neglects: dynamics | IO: (t) -> amplitude | Tags: score\"}\n";

/// A score is JSON lines: its annotation is its comment, and it has no body to measure, however
/// long.
#[test]
fn a_long_score_lints_clean() {
    let line = "{\"at\": \"0b\", \"node\": \"@kick\"}\n";
    let dir = composition(
        "long-score",
        &[("kick", "crop(sin(2*pi*50*t), 0s, 0.1s)\n")],
    );
    let score = format!("{ANNOTATION}{}", line.repeat(400));
    std::fs::write(dir.join("hits.jsonl"), score).expect("a score");
    let report = lint(&dir, None).unwrap_or_else(|e| panic!("{}", e.message()));
    assert_eq!(codes(&report), "", "{}", codes(&report));
}

/// Line 1 of a score is its annotation, held to the doc comment's codes.
#[test]
fn a_score_without_its_annotation_refuses_as_a_node_without_its_comment() {
    let line = "{\"at\": \"0b\", \"node\": \"@kick\"}\n";
    let dir = composition(
        "bare-score",
        &[("kick", "crop(sin(2*pi*50*t), 0s, 0.1s)\n")],
    );
    for (score, code) in [
        (line.to_string(), "lint.missing_comment"),
        (
            format!("{{\"annotation\": \"Models: a kick\"}}\n{line}"),
            "lint.malformed_comment",
        ),
    ] {
        std::fs::write(dir.join("hits.jsonl"), score).expect("a score");
        let Err(refused) = lint(&dir, None) else {
            panic!("{code} refuses");
        };
        let found = refused.diagnostics();
        assert_eq!(found[0].code, code, "{}", found[0].message);
        assert_eq!(found[0].file.as_deref(), Some("hits.jsonl"));
        assert_eq!(found[0].start, Some((1, 1)));
    }
}

/// A typo'd target refuses under the code `render` answers a ref to it with.
#[test]
fn lint_of_an_undefined_target_refuses_like_render() {
    let dir = composition("undefined-target", &[("master", "sin(2*pi*300*t)\n")]);
    let Err(linted) = lint(&dir, Some("@drums/kik")) else {
        panic!("a target nothing defines refuses");
    };
    let Err(rendered) = sva_core::probe(&dir, "@drums/kik") else {
        panic!("render refuses the ref that names it");
    };
    assert_eq!(linted.code(), rendered.code());
    assert_eq!(linted.exit_code(), rendered.exit_code());
}
