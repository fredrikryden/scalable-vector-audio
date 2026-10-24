// Concern: proves lint checks each file under its rules, and a target under its render's types | Non-concern: the per-node text checks (sva-cli lint.rs) | IO: (a composition) -> a report or a refusal

use crate::helpers::{doc, scratch};
use sva_cli::lint;

fn composition(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = scratch(name);
    for (rel, body) in files {
        std::fs::write(dir.join(rel), format!("{}{body}", doc(rel))).expect("a node file");
    }
    dir
}

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
