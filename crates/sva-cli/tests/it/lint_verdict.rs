// Concern: proves a lint verdict answers every finding the scan made | Non-concern: which node earns which finding (the rest of the suite) | IO: (a composition) -> a verdict + diagnostics

use crate::helpers::{doc, scratch};
use sva_cli::lint;

/// The refusal used to raise before every other pass, so one malformed comment hid the rest.
#[test]
fn a_refusing_lint_still_lists_its_warnings() {
    let dir = scratch("refusing-with-warnings");
    std::fs::write(dir.join("master"), "@tone*0.5\n").expect("a node with no doc comment");
    std::fs::write(
        dir.join("tone"),
        format!("{}sin(2*pi*110*t)\n", doc("tone")),
    )
    .expect("a documented node");
    std::fs::create_dir_all(dir.join("variables")).expect("a variables directory");
    std::fs::write(dir.join("variables/key"), format!("{}sin(t)\n", doc("key")))
        .expect("a key that is no pitch");

    let Err(refused) = lint(&dir, None) else {
        panic!("a node with no `;`-comment refuses");
    };
    let answered = refused.diagnostics();
    let found: Vec<(&str, sva_core::Severity)> = answered
        .iter()
        .map(|d| (d.code.as_str(), d.severity))
        .collect();
    assert!(
        found.contains(&("lint.missing_comment", sva_core::Severity::Error)),
        "{found:?}"
    );
    assert!(
        found.contains(&("lint.key_is_not_a_pitch", sva_core::Severity::Warning)),
        "the warning is answered beside the refusal: {found:?}"
    );
    assert_eq!(
        refused.message(),
        "1 lint violation(s)",
        "the verdict counts what refused, not what was warned"
    );
}
