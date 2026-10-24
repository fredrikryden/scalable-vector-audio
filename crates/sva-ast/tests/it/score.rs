// Concern: proves a score's degrees lower against variables/scale as a song reaches them, whatever the scale | Non-concern: one line's JSON, lanes | IO: (a composition) -> each line's f0
use sva_ast::{Address, BinOp, ByteSpan, Composition, Expr, Graph, Literal, Refusal, load};

const KEY: &str = "variables/key";
const SCALE: &str = "variables/scale";

/// Loaded, every reserved variable lowered and `scores` lowered as one song reaches them.
fn settle(source: &Composition, scores: &[&str]) -> Result<Graph, Vec<Refusal>> {
    let mut graph = load(source)?;
    graph.lower_reserved(None)?;
    let reads = scores.iter().map(|path| Expr::Ref {
        path: path.to_string(),
        arg: Box::new(Expr::Var("t".to_string())),
        binds: Vec::new(),
        address: Address::Time,
        span: ByteSpan::at(0),
    });
    let root = reads
        .reduce(|a, b| Expr::Bin(BinOp::Add, Box::new(a), Box::new(b)))
        .expect("a score");
    graph.define_settled(None, "song", root)?;
    let mut song = graph.song("song")?;
    let lowered = song.lowered(&graph, &[])?;
    graph.apply(lowered);
    Ok(graph)
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
            "phrase.jsonl",
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
    (1200.0 * libm::log2(product) * 1e6).round() / 1e6
}

#[test]
fn a_degree_with_no_scale_refuses_on_its_line() {
    let refused = settle(&composition(""), &["phrase"]).unwrap_err();
    assert_eq!(refused[0].at.path, "phrase.jsonl");
    assert!(
        refused[0].reason.starts_with("line 1, column 39: `deg`"),
        "{}",
        refused[0].reason
    );
}

#[test]
fn a_chromatic_note_and_an_octave_count_in_the_scale_too() {
    let phrase = "{\"annotation\": \"Models: the raised seventh | Neglects: rhythm | IO: (t) -> amplitude | Tags: score\"}\n\
                  {\"at\": \"0b\", \"node\": \"@voice\", \"f0\": \"deg(7)*1st\"}\n\
                  {\"at\": \"2b + 1b/3\", \"node\": \"@voice\", \"f0\": \"deg(5)/2\"}\n";
    let lead = "f0 = deg(3)\nsin(2*pi*f0*t)\n";
    let minor = with(
        "@../scales/minor\n",
        [("phrase.jsonl", phrase), ("lead", lead)],
    );
    let held = settle(&minor, &["phrase", "lead"]).expect("every degree settles");
    assert_eq!(cents(held.expr("phrase").unwrap()), [1100.0, -500.0]);
    assert_eq!(ratio(&held.defaults("lead")[0].1), 300.0);

    let major = with(
        "@../scales/major\n",
        [("phrase.jsonl", phrase), ("lead", lead)],
    );
    let held = settle(&major, &["phrase", "lead"]).expect("every degree settles");
    assert_eq!(cents(held.expr("phrase").unwrap()), [1200.0, -500.0]);
    assert_eq!(ratio(&held.defaults("lead")[0].1), 400.0);
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
        [("keys", KEYS), ("one.jsonl", one), ("lines.jsonl", lines)],
    );
    let held = settle(&source, &["one", "lines"]).expect("every degree settles");
    assert_eq!(held.expr("one"), held.expr("lines"));
}
