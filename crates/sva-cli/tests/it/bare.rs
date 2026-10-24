// Concern: proves lint warns of a node a bare `@path` cannot play: a parameter with no default, a support with no end | Non-concern: the other lint checks | IO: (a composition) -> findings

use crate::helpers::{doc, scratch};
use sva_cli::lint;
use sva_core::{LintCode, Severity};

fn composition(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = scratch(name);
    for (rel, body) in files {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().expect("inside the scratch dir")).expect("a dir");
        std::fs::write(path, format!("{}{body}", doc(rel))).expect("a node file");
    }
    dir
}

/// Each `(subject, message)` lint found under `code`, every one a warning.
fn found(dir: &std::path::Path, code: LintCode) -> Vec<(String, String)> {
    let report = lint(dir, None).unwrap_or_else(|e| panic!("lint refused: {}", e.message()));
    report
        .findings
        .iter()
        .filter(|f| f.code == code)
        .inspect(|f| assert_eq!(f.severity, Severity::Warning, "{}", f.message))
        .map(|f| (f.subject.clone(), f.message.clone()))
        .collect()
}

#[test]
fn a_parameter_with_no_default_line_is_named_with_its_file() {
    let dir = composition("no-default", &[("tone", "crop(sin(2*pi*f0*t), 0s, 1s)\n")]);
    let held = found(&dir, LintCode::ParameterHasNoDefault);
    let [(subject, message)] = held.as_slice() else {
        panic!("one parameter, one finding: {held:?}");
    };
    assert_eq!(subject, "tone");
    assert!(message.contains("`f0`"), "{message}");
    assert!(
        message.contains("f0 = "),
        "names the line to write: {message}"
    );

    let defaulted = composition(
        "with-default",
        &[("tone", "f0 = C4\ncrop(sin(2*pi*f0*t), 0s, 1s)\n")],
    );
    assert_eq!(found(&defaulted, LintCode::ParameterHasNoDefault), []);
}

#[test]
fn a_node_whose_support_never_ends_is_warned_of() {
    let dir = composition("endless", &[("drone", "sin(2*pi*110*t)\n")]);
    let held = found(&dir, LintCode::SupportNeverEnds);
    assert_eq!(held.len(), 1, "{held:?}");
    assert_eq!(held[0].0, "drone");

    let cropped = composition("cropped", &[("drone", "crop(sin(2*pi*110*t), 0s, 2s)\n")]);
    assert_eq!(found(&cropped, LintCode::SupportNeverEnds), []);
}
