// Concern: proves one instance exists per argument tuple and names what a binding can be refused for | Non-concern: ordering the instances (refs.rs) | IO: (a composition) -> Instances or a refusal

use crate::fixtures::graph_of;
use sva_engine::instantiate::{Instances, instantiate};
use sva_engine::{BindingFault, DEFAULT_SAMPLE_RATE, EngineError};

fn fault_of(e: EngineError) -> BindingFault {
    match e {
        EngineError::Binding { fault, .. } => fault,
        other => panic!("expected a binding refusal, got {other:?}"),
    }
}

fn named(i: &Instances) -> Vec<String> {
    i.paths().map(str::to_string).collect()
}

#[test]
fn each_argument_tuple_is_its_own_instance_and_equal_tuples_share_one() {
    let g = graph_of(
        "tuples",
        &[
            ("src", "sin(t)\n"),
            ("gain", "x*k\n"),
            (
                "song",
                "@gain(t, x=@src, k=2) + @gain(t, x=@src, k=3) + @gain(t, k=2, x=@src)\n",
            ),
        ],
    );
    let i = instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap();
    assert_eq!(
        named(&i),
        vec!["gain(k=2, x=@src)", "gain(k=3, x=@src)", "song", "src"],
        "two tuples, three call sites: the repeat shares, and argument order does not matter"
    );
}

#[test]
fn an_unbound_variable_and_an_unused_argument_both_refuse() {
    let g = graph_of(
        "unbound",
        &[
            ("f", "x*k\n"),
            ("src", "sin(t)\n"),
            ("song", "@f(t, x=@src)\n"),
        ],
    );
    assert_eq!(
        fault_of(instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap_err()),
        BindingFault::Unbound("f".to_string(), "k".to_string())
    );

    let g = graph_of(
        "unused",
        &[
            ("f", "x*2\n"),
            ("src", "sin(t)\n"),
            ("song", "@f(t, x=@src, gain=3)\n"),
        ],
    );
    assert_eq!(
        fault_of(instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap_err()),
        BindingFault::Unused("f".to_string(), "gain".to_string())
    );
}

/// Where the fix is: the caller omitted the argument, so the caller's file and the span of
/// the invocation are what a reader needs, not the callee's body.
#[test]
fn a_missing_argument_is_located_at_the_invocation_rather_than_at_the_free_variable() {
    let g = graph_of(
        "located",
        &[
            ("src", "sin(t)\n"),
            ("fx/gain", "x*k\n"),
            ("song", "0.5 + @fx/gain(t, x=@src)\n"),
        ],
    );
    let EngineError::Binding { node, span, fault } =
        instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap_err()
    else {
        panic!("expected a binding refusal")
    };
    assert_eq!(node, "song");
    assert_eq!(span.map(|s| (s.start, s.end)), Some((6, 14)));
    assert_eq!(
        fault,
        BindingFault::Unbound("fx/gain".to_string(), "k".to_string())
    );
}

#[test]
fn a_parameter_may_not_take_a_name_the_language_already_binds() {
    let g = graph_of(
        "reserved",
        &[("f", "x*2\n"), ("song", "@f(t, x=1, sin=2)\n")],
    );
    assert_eq!(
        fault_of(instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap_err()),
        BindingFault::Reserved("sin".to_string())
    );
}

/// A parameter's expression is never copied, so what a walk sees has to be assembled: this
/// pins the assembly against the substituted tree it replaces, shifts and all.
#[test]
fn a_resolved_walk_reads_as_the_substituted_copy_did() {
    let g = graph_of(
        "resolved",
        &[
            ("src", "sin(t)\n"),
            ("inner", "w(t - 0.02s)*2\n"),
            ("outer", "@inner(t, w=z + 1)\n"),
            ("song", "@outer(t, z=@src(t)*3)\n"),
        ],
    );
    let i = instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap();
    let copies = i.exprs();
    let printed: Vec<String> = copies
        .iter()
        .map(|(path, e)| format!("{path} = {}", sva_ast::render_expr(e)))
        .collect();
    assert_eq!(
        printed,
        vec![
            "inner(w=@src*3 + 1) = (@src(t - 0.02s)*3 + 1)*2",
            "outer(z=@src*3) = @inner(w=@src*3 + 1)",
            "song = @outer(z=@src*3)",
            "src = sin(t)",
        ],
        "a shifted parameter read moves every `t` under it, however many scopes deep"
    );
}

/// The names above a default are part of its arithmetic, resolved in declaration order
/// against what the caller bound.
#[test]
fn a_default_may_read_a_bound_default() {
    let files = &[
        ("motif", "key = 200\nthird = key*2\nsin(2*pi*third*t)\n"),
        ("master", "@motif(t) + @motif(t, key=300)\n"),
    ];
    let g = graph_of("chained-defaults", files);
    let held =
        instantiate(&g, "master", DEFAULT_SAMPLE_RATE).expect("a default reads the line above it");

    let mut named: Vec<String> = held
        .paths()
        .filter(|p| p.starts_with("motif("))
        .map(str::to_string)
        .collect();
    named.sort();
    assert_eq!(
        named,
        vec![
            "motif(key=200, third=200*2)".to_string(),
            "motif(key=300, third=300*2)".to_string()
        ],
        "the caller's key reaches the default written off it, one instance each"
    );
}

/// A `sum` whose index reaches a binding is one instance per index, each the node its own
/// number binds, so it renders as the refs written out by hand do.
#[test]
fn a_sum_index_in_a_binding_is_one_instance_per_index() {
    let files = |root: &'static str| [("band", "sin(2*pi*fc*t)\n"), ("root", root)];
    let g = graph_of(
        "index-binds",
        &files("sum(k, 1, 3, @band(t, fc=100*k)/k)\n"),
    );
    let held = instantiate(&g, "root", DEFAULT_SAMPLE_RATE).expect("one instance per index");
    let bands: Vec<String> = named(&held)
        .into_iter()
        .filter(|p| p.starts_with("band("))
        .collect();
    assert_eq!(bands.len(), 3, "{bands:?}");

    let config = || sva_engine::RenderConfig::seconds(8_000, 0.01);
    let rendered = |g| {
        let held = sva_engine::render(&g, "root", config(), &sva_engine::Tier::default())
            .expect("a render");
        let root = held.id("root").expect("the root");
        held.output(root).expect("samples").plane(0).to_vec()
    };
    let by_hand = graph_of(
        "index-by-hand",
        &files("@band(t, fc=100*1)/1 + @band(t, fc=100*2)/2 + @band(t, fc=100*3)/3\n"),
    );
    assert_eq!(rendered(g), rendered(by_hand), "the sum is its terms");
}

/// A default is a number each instance knows, so a sum written out one node per index may take
/// its bound from one, and a caller binding that parameter writes out its own count.
#[test]
fn a_sum_bounded_by_a_parameter_writes_out_each_instance_s_own_count() {
    let partials = |song: &str| {
        let g = graph_of(
            "bounded",
            &[
                ("partial", "k = 1\nsin(2*pi*110*k*t)/k\n"),
                ("bank", "n = 3\nsum(k, 1, n, @partial(t, k=k))\n"),
                ("song", song),
            ],
        );
        let i = instantiate(&g, "song", DEFAULT_SAMPLE_RATE).expect("instances");
        named(&i)
            .into_iter()
            .filter(|n| i.origin(n) == Some("partial"))
            .count()
    };
    assert_eq!(partials("@bank\n"), 3);
    assert_eq!(partials("@bank(t, n=5)\n"), 5);
    assert_eq!(partials("@bank(t, n=2) + @bank(t, n=4)\n"), 4);

    let g = graph_of(
        "fractional",
        &[
            ("partial", "k = 1\nsin(2*pi*110*k*t)/k\n"),
            ("bank", "n = 2.5\nsum(k, 1, n, @partial(t, k=k))\n"),
        ],
    );
    let refused = instantiate(&g, "bank", DEFAULT_SAMPLE_RATE).unwrap_err();
    assert_eq!(refused.code(), "engine.unwritten_series", "{refused}");

    let g = graph_of(
        "spent",
        &[
            ("partial", "k = 1\nsin(2*pi*110*k*t)/k\n"),
            ("bank", "n = 3\nsum(k, 1, n, @partial(t, k=k))\n"),
            ("unbounded", "sum(k, 1, q, @partial(t, k=k))\n"),
            ("song", "@bank(t, n=2, gain=3)\n"),
        ],
    );
    assert_eq!(
        fault_of(instantiate(&g, "song", DEFAULT_SAMPLE_RATE).unwrap_err()),
        BindingFault::Unused("bank".to_string(), "gain".to_string())
    );
    let refused = instantiate(&g, "unbounded", DEFAULT_SAMPLE_RATE).unwrap_err();
    assert_eq!(refused.code(), "engine.unwritten_series", "{refused}");
}
