// Concern: proves a reading may name the file a composer wrote, not only the instance it expanded into | Non-concern: which instances exist (instantiate.rs) | IO: (a file name) -> a reading

use crate::fixtures::graph_of;
use sva_engine::{Ask, EngineError, Output, RenderConfig, Representation, Tier, answer, render};

/// One instance under another name still answers to the name on disk.
#[test]
fn a_node_with_a_default_line_renders_by_name() {
    let g = graph_of(
        "defaulted",
        &[
            ("voice", "f0 = A4\nsin(2*pi*f0*t)\n"),
            ("master", "@voice\n"),
        ],
    );
    let held = render(
        &g,
        "voice",
        RenderConfig::seconds(44_100, 0.25).asking(vec![Ask {
            node: "voice".to_string(),
            representation: Representation::Lines,
        }]),
        &Tier::default(),
    )
    .expect("a node that declares defaults renders");
    let id = held.id("voice").expect("the file names its one instance");
    let Output::Lines(lines) = answer(&held, id, Representation::Lines)
        .expect("lines")
        .value
    else {
        panic!("expected a line list");
    };
    assert!(
        lines.iter().any(|l| (l.hz - 440.0).abs() < 1e-9),
        "the default sounds: {lines:?}"
    );
}

/// A file two tuples share is named by neither, and the refusal says which two.
#[test]
fn a_file_two_invocations_share_is_named_by_neither() {
    let g = graph_of(
        "shared",
        &[
            ("voice", "f0 = A4\nsin(2*pi*f0*t)\n"),
            ("master", "@voice + @voice(t, f0=220)\n"),
        ],
    );
    let held = render(
        &g,
        "master",
        RenderConfig::seconds(44_100, 0.05),
        &Tier::default(),
    )
    .expect("two voices");
    let refused = held.node("voice").expect_err("the file names neither");
    assert!(
        matches!(&refused, EngineError::AmbiguousNode(file, names) if file == "voice" && names.len() == 2),
        "{refused:?}"
    );
    assert!(held.node("master").is_ok(), "the root still answers");
}

/// An instance written in another spelling than its own still refuses, never resolving by
/// guess, and names every instance of its file as a reading must spell it.
#[test]
fn an_instance_spelt_otherwise_refuses_naming_each_spelling_that_resolves() {
    let g = graph_of(
        "spelt",
        &[
            ("voice", "f0 = A4\ng = 1\nsin(2*pi*f0*t)*g\n"),
            ("master", "@voice(t, g=0.5, f0=220)\n"),
        ],
    );
    let held = render(
        &g,
        "master",
        RenderConfig::seconds(44_100, 0.05),
        &Tier::default(),
    )
    .expect("one voice");
    for written in ["voice(g=0.5,f0=220)", "voice(f0=220)"] {
        let refused = held.node(written).expect_err("no instance is spelt so");
        assert_eq!(refused.code(), "engine.unknown_node");
        let EngineError::UnknownInstance(asked, names) = &refused else {
            panic!("{refused:?}")
        };
        assert_eq!(asked, written);
        let [name] = names.as_slice() else {
            panic!("one instance: {names:?}")
        };
        assert!(held.node(name).is_ok(), "`{name}` resolves as listed");
        assert!(
            refused.to_string().contains(&format!("`@{name}`")),
            "{refused}"
        );
    }
    assert!(
        matches!(held.node("nobody(f0=1)"), Err(EngineError::UnknownNode(_))),
        "a file with no instance has none to list"
    );
}
