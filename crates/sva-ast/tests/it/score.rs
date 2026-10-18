// Concern: proves a score lowers against the key's scale, and again wherever that scale changes | Non-concern: one line's JSON (score.rs) | IO: (a composition, an edit) -> each line's f0
use sva_ast::{BinOp, Composition, Expr, KEY, Literal, load, load_beside};

fn composition(key: &str) -> Composition {
    Composition::from_iter([
        (KEY, key),
        ("scales/major", "200\n400\n500\n700\n900\n1100\n1200\n"),
        ("scales/minor", "200\n300\n500\n700\n800\n1000\n1200\n"),
        ("voice", "f0 = 440\nsin(2*pi*f0*t)\n"),
        (
            "phrase",
            "{\"at\": \"0b\", \"sound\": \"@voice\", \"deg\": 3}\n",
        ),
    ])
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
fn an_edit_to_the_scale_line_lowers_every_held_score_again() {
    let mut source = composition("scale = @../scales/major\nD3\n");
    let mut held = load(&source).expect("a composition");
    held.lower_scores(None).expect("every line settles");
    assert_eq!(third(held.expr("phrase").unwrap()), 400.0);

    source.insert(KEY, "scale = @../scales/minor\nD3\n");
    let mut delta = load_beside(&source, &["phrase"], &held).expect("the edit loads");
    assert!(
        !delta.defines("phrase"),
        "the score's own text did not change"
    );
    delta.lower_scores(Some(&held)).expect("every line settles");
    assert_eq!(third(delta.expr("phrase").unwrap()), 300.0);
}

#[test]
fn a_degree_with_no_scale_line_refuses_on_its_line() {
    let mut held = load(&composition("D3\n")).expect("a composition");
    let refused = held.lower_scores(None).unwrap_err();
    assert!(
        refused[0].reason.starts_with("line 1, column 33: `deg`"),
        "{}",
        refused[0].reason
    );
}
