// Concern: proves a reading may name the file a composer wrote, not only the instance it expanded into | Non-concern: which instances exist (instantiate.rs) | IO: (a file name) -> a reading

use crate::fixtures::graph_of;
use sva_engine::{Ask, Output, RenderConfig, Representation, Tier, answer, render};

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
    let spelt = "voice(f0=220, g=0.5)";
    assert!(held.node(spelt).is_ok(), "`{spelt}` resolves");
    for written in ["voice(g=0.5,f0=220)", "voice(f0=220)", "nobody(f0=1)"] {
        let refused = held.node(written).expect_err("no instance is spelt so");
        assert_eq!(refused.code(), "engine.unknown_node");
        let lists = refused.to_string().contains(&format!("`@{spelt}`"));
        assert_eq!(lists, written.starts_with("voice"), "{refused}");
    }
}
