// Concern: proves a score lowers against variables/scale, and again wherever that scale changes | Non-concern: one line's JSON (score.rs) | IO: (a composition, an edit) -> each line's f0
use sva_ast::{BinOp, Composition, Expr, Graph, Literal, Refusal, load, load_beside};

const KEY: &str = "variables/key";
const SCALE: &str = "variables/scale";

/// Every reserved variable lowered, and every score line checked, as a render settles one.
fn settle(graph: &mut Graph, beside: Option<&Graph>) -> Result<(), Vec<Refusal>> {
    graph.lower_reserved(beside)?;
    graph.check_scores(beside)
}

fn composition(scale: &str) -> Composition {
    with(scale, [])
}

/// D3 at 120 bpm, in the scale `scale` refs; an empty `scale` names none.
fn with<const N: usize>(scale: &str, more: [(&str, &str); N]) -> Composition {
    let mut composition = Composition::from_iter([
        (KEY, "D3\n"),
        ("variables/bpm", "120\n"),
        ("variables/meter", "4/4\n"),
        ("scales/major", "200\n400\n500\n700\n900\n1100\n1200\n"),
        ("scales/minor", "200\n300\n500\n700\n800\n1000\n1200\n"),
        ("voice", "f0 = 440\nsin(2*pi*f0*t)\n"),
        (
            "phrase",
            "{\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 3}\n",
        ),
    ]);
    if !scale.is_empty() {
        composition.insert(SCALE, scale);
    }
    for (path, text) in more {
        composition.insert(path, text);
    }
    composition
}

/// Each `f0` a score's lines bind, as cents above the tonic, in written order.
fn cents(e: &Expr) -> Vec<f64> {
    match e {
        Expr::Bin(BinOp::Add, l, r) => [cents(l), cents(r)].concat(),
        Expr::Ref { binds, .. } => binds
            .iter()
            .filter(|(name, _)| name == "f0")
            .map(|(_, f0)| ratio(f0))
            .collect(),
        other => panic!("a score is a sum of refs: {other:?}"),
    }
}

/// The tonic times literals, as cents.
fn ratio(e: &Expr) -> f64 {
    let mut product = 1.0;
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        match e {
            Expr::Bin(BinOp::Mul, l, r) => stack.extend([&**l, &**r]),
            Expr::Bin(BinOp::Div, l, r) => {
                stack.push(l);
                let Expr::Lit(Literal::Num(n)) = **r else {
                    panic!("divides by a literal: {r:?}")
                };
                product /= n;
            }
            Expr::Lit(Literal::Num(n)) => product *= n,
            Expr::Ref { .. } => {}
            other => panic!("the tonic times literals: {other:?}"),
        }
    }
    (1200.0 * product.log2() * 1e6).round() / 1e6
}

/// The ratio the one line's `f0` multiplies the tonic by.
fn third(e: &Expr) -> f64 {
    let Expr::Ref { binds, .. } = e else {
        panic!("one line is one ref: {e:?}")
    };
    match &binds[0].1 {
        Expr::Bin(BinOp::Mul, _, ratio) => match **ratio {
            Expr::Lit(Literal::Num(r)) => (1200.0 * r.log2() * 1e6).round() / 1e6,
            ref other => panic!("a lowered degree is a number: {other:?}"),
        },
        other => panic!("`f0` is the tonic times a ratio: {other:?}"),
    }
}

#[test]
fn an_edit_to_the_scale_lowers_every_held_score_again() {
    let mut source = composition("@../scales/major\n");
    let mut held = load(&source).expect("a composition");
    settle(&mut held, None).expect("every line settles");
    assert_eq!(third(held.expr("phrase").unwrap()), 400.0);

    source.insert(SCALE, "@../scales/minor\n");
    let mut delta = load_beside(&source, &["phrase"], &held).expect("the edit loads");
    assert!(
        !delta.defines("phrase"),
        "the score's own text did not change"
    );
    settle(&mut delta, Some(&held)).expect("every line settles");
    assert_eq!(third(delta.expr("phrase").unwrap()), 300.0);
}

#[test]
fn a_degree_with_no_scale_refuses_on_its_line() {
    let mut held = load(&composition("")).expect("a composition");
    let refused = settle(&mut held, None).unwrap_err();
    assert!(
        refused[0].reason.starts_with("line 1, column 39: `deg`"),
        "{}",
        refused[0].reason
    );
}

#[test]
fn a_chromatic_note_and_an_octave_follow_a_key_or_scale_change() {
    let phrase = "; Models: the raised seventh | Neglects: rhythm | IO: (t) -> amplitude | Tags: score\n\
                  {\"at\": \"0b\", \"node\": \"@voice\", \"f0\": \"deg(7)*1st\"}\n\
                  {\"at\": \"2b + 1b/3\", \"node\": \"@voice\", \"f0\": \"deg(5)/2\"}\n";
    let lead = "f0 = deg(3)\nsin(2*pi*f0*t)\n";
    let mut source = with("@../scales/minor\n", [("phrase", phrase), ("lead", lead)]);
    let mut held = load(&source).expect("a composition");
    settle(&mut held, None).expect("every degree settles");
    assert_eq!(cents(held.expr("phrase").unwrap()), [1100.0, -500.0]);
    assert_eq!(ratio(&held.defaults("lead")[0].1), 300.0);

    source.insert(SCALE, "@../scales/major\n");
    let mut delta = load_beside(&source, &["phrase", "lead"], &held).expect("the edit loads");
    settle(&mut delta, Some(&held)).expect("every degree settles");
    assert_eq!(cents(delta.expr("phrase").unwrap()), [1200.0, -500.0]);
    assert_eq!(ratio(&delta.defaults("lead")[0].1), 400.0);
}

#[test]
fn a_chord_lowers_as_its_lines_would() {
    let chord = "{\"at\": \"0b\", \"node\": \"@voice\", \"deg\": [1, 3, 5]}\n";
    let lines = "{\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 1}\n\
                 {\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 3}\n\
                 {\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 5}\n";
    let source = with("@../scales/major\n", [("chord", chord), ("lines", lines)]);
    let mut held = load(&source).expect("a composition");
    settle(&mut held, None).expect("every degree settles");
    assert_eq!(held.expr("chord"), held.expr("lines"));
}

const KEYS: &str = "f0 = 440\nvel = 1\nlen = 1s\nvel*crop(sin(2*pi*f0*t), 0s, len)\n";

#[test]
fn a_broadcast_line_lowers_as_its_lines_would() {
    let one = "{\"at\": [\"0b\", \"0b\", \"1b/3\"], \"node\": \"@keys\", \"deg\": [1, 3, 5], \
               \"vel\": [0.8, 0.5, 0.6], \"len\": [\"1b\", \"0.5b\", \"1b\"]}\n";
    let lines = "{\"at\": \"0b\", \"node\": \"@keys\", \"deg\": 1, \"vel\": 0.8, \"len\": \"1b\"}\n\
                 {\"at\": \"0b\", \"node\": \"@keys\", \"deg\": 3, \"vel\": 0.5, \"len\": \"0.5b\"}\n\
                 {\"at\": \"1b/3\", \"node\": \"@keys\", \"deg\": 5, \"vel\": 0.6, \"len\": \"1b\"}\n";
    let source = with(
        "@../scales/major\n",
        [("keys", KEYS), ("one", one), ("lines", lines)],
    );
    let mut held = load(&source).expect("a composition");
    settle(&mut held, None).expect("every degree settles");
    assert_eq!(held.expr("one"), held.expr("lines"));
}

#[test]
fn each_node_a_line_lists_is_checked_against_its_own_parameters() {
    let both = "{\"at\": \"0b\", \"node\": [\"@keys\", \"@voice\"], \"vel\": 0.5}\n";
    let source = with("@../scales/major\n", [("keys", KEYS), ("both", both)]);
    let mut held = load(&source).expect("a composition");
    let refused = settle(&mut held, None).unwrap_err();
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert!(
        refused[0]
            .reason
            .contains("`vel` is no parameter `@voice` reads"),
        "{}",
        refused[0].reason
    );
}

#[test]
fn a_degree_in_an_equation_with_no_whole_number_refuses_where_written() {
    let source = with("@../scales/major\n", [("lead", "sin(2*pi*deg(1.5)*t)\n")]);
    let mut held = load(&source).expect("a composition");
    let refused = settle(&mut held, None).unwrap_err();
    assert_eq!(refused[0].at.path, "lead");
    assert_eq!(refused[0].at.span.start, 9);
    assert!(
        refused[0].reason.contains("deg(7)*1st"),
        "{}",
        refused[0].reason
    );
}
