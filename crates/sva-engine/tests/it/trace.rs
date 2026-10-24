// Concern: proves a trace names what a node is built from and who reads it | Non-concern: measuring it, which needs audio | IO: (a composition, target) -> Traced

use crate::fixtures::graph_of;
use sva_engine::trace;

const CHORD: &str = "sin(2*pi*256*t) + sin(2*pi*512*t)\n";

#[test]
fn a_trace_names_what_a_node_reads_and_who_reads_it() {
    let g = graph_of(
        "structure",
        &[
            ("chord", CHORD),
            ("voiced", "@chord*0.5\n"),
            ("master", "@voiced + @chord\n"),
        ],
    );
    let traced = trace(&g, &["master".to_string()], "voiced").expect("a trace");
    assert_eq!(traced.node, "voiced");
    assert_eq!(traced.down, vec!["chord".to_string()]);
    assert_eq!(
        traced
            .up
            .iter()
            .map(|u| u.node.as_str())
            .collect::<Vec<_>>(),
        vec!["master"]
    );
    assert_eq!(traced.entry, vec!["master".to_string()]);
    assert!(traced.cycle.is_none());
}
