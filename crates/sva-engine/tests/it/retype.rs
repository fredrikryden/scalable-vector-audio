// Concern: proves each render in a session is a render of its own, and a refusal leaves it as it was | Non-concern: memory's answers, a stream's changes | IO: (compositions, a session) -> bits

use std::collections::BTreeSet;

use crate::fixtures::{Now, graph_of};
use sva_ast::Graph;
use sva_engine::{Never, Render, RenderConfig, Session, Tier, identity, render_in, render_over};

const RATE: u32 = 8_000;

fn config() -> RenderConfig {
    RenderConfig::seconds(RATE, 0.05)
}

/// `x` and `y` under a sum `p`, which a master samples; `x` at `hz`.
fn mixed(name: &str, hz: u32) -> Graph {
    let x = format!("sample(crop(sin(2*pi*{hz}*t), 0s, 0.05s))\n");
    graph_of(
        name,
        &[
            ("x", &x),
            ("y", "sample(crop(sin(2*pi*330*t), 0s, 0.05s))\n"),
            ("p", "@x*0.5 + @y\n"),
            ("master", "sample(@p*0.5)\n"),
        ],
    )
}

fn rendered(session: &mut Session, graph: &Graph, tier: &Tier) -> Render {
    let held = render_in(session, graph, "master", config(), tier, (&Never, None)).now();
    held.unwrap_or_else(|e| panic!("{e}"))
}

fn typed(render: &Render) -> BTreeSet<String> {
    let stats = render.cache_stats.as_ref().expect("a render over memory");
    stats.typed.iter().cloned().collect()
}

fn bits(render: &Render) -> Vec<Vec<u64>> {
    let root = render.output(render.root).expect("the root's samples");
    let planes = root.planes.iter();
    planes
        .map(|p| p.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// What a render of `graph` alone writes and names its root, to hold a session's render to.
fn alone(graph: &Graph) -> Render {
    let held = render_over(graph, "master", config(), &Tier::default()).now();
    held.unwrap_or_else(|e| panic!("{e}"))
}

fn same_as_alone(render: &Render, graph: &Graph) {
    let fresh = alone(graph);
    assert_eq!(bits(render), bits(&fresh));
    for path in ["x", "y", "p", "master"] {
        let (mine, theirs) = (
            render.id(path).expect("held"),
            fresh.id(path).expect("held"),
        );
        assert_eq!(
            identity(&render.tys, mine).expect("named"),
            identity(&fresh.tys, theirs).expect("named"),
            "`{path}` is the value a render of its own names"
        );
    }
}

/// A render keeps the version it was rendered from while its session renders an edit: each
/// names and reads its own composition.
#[test]
fn a_render_keeps_its_version_while_its_session_renders_an_edit() {
    let mut session = Session::default();
    let tier = Tier::default();
    let (before, edited) = (mixed("retype-kept", 220), mixed("retype-kept-edit", 440));
    let first = rendered(&mut session, &before, &tier);
    let second = rendered(&mut session, &edited, &tier);
    same_as_alone(&first, &before);
    same_as_alone(&second, &edited);
}

/// A render refused while typing leaves the session as it was: the next types only its change.
#[test]
fn a_refused_render_leaves_the_session_as_it_was() {
    let mut session = Session::default();
    let tier = Tier::default();
    rendered(&mut session, &mixed("retype-held", 220), &tier);
    let broken = graph_of(
        "retype-broken",
        &[
            ("x", "sample(crop(sin(2*pi*440*t), 0s, 0.05s))\n"),
            ("y", "sample(crop(sin(2*pi*330*t), 0s, 0.05s))\n"),
            ("p", "@x*0.5 + @y\n"),
            ("master", "sample(@p*0.5) + lowpass(1)\n"),
        ],
    );
    let refused = render_in(
        &mut session,
        &broken,
        "master",
        config(),
        &tier,
        (&Never, None),
    )
    .now();
    assert!(refused.is_err(), "a lowpass of no cutoff refuses");
    let edited = mixed("retype-mended", 440);
    let after = rendered(&mut session, &edited, &tier);
    let share: BTreeSet<String> = ["x", "p", "master"].map(String::from).into();
    assert_eq!(typed(&after), share);
    same_as_alone(&after, &edited);
}

/// A session answers each render as a render of its own would: another target of the same
/// composition renders its bits, and a node the composition no longer defines refuses.
#[test]
fn a_session_renders_each_target_and_each_composition_as_a_render_of_its_own() {
    let mut session = Session::default();
    let tier = Tier::default();
    let graph = mixed("retype-targets", 220);
    for target in ["master", "p", "master", "x"] {
        let held = render_in(
            &mut session,
            &graph,
            target,
            config(),
            &tier,
            (&Never, None),
        )
        .now();
        let held = held.unwrap_or_else(|e| panic!("{target}: {e}"));
        let alone = render_over(&graph, target, config(), &Tier::default()).now();
        assert_eq!(bits(&held), bits(&alone.expect("a render")), "{target}");
    }
    let mut lacking = graph.clone();
    lacking.set("x", None);
    let refused = render_in(
        &mut session,
        &lacking,
        "master",
        config(),
        &tier,
        (&Never, None),
    )
    .now();
    let alone = render_over(&lacking, "master", config(), &Tier::default()).now();
    assert_eq!(
        refused.err().map(|e| e.to_string()),
        alone.err().map(|e| e.to_string())
    );
}
