// Concern: writes an expression's outline, a score's lines or a song's derivation as the `data` object | Non-concern: building any of them (sva-ast) | IO: (text) or (source, target) -> JSON

use sva_ast::outline::{Argument, Form};
use sva_ast::score::{EXTENSION, Item, spelled};
use sva_ast::song::{Curve, Held, Home, LaneId, Pos};
use sva_ast::{Address, BinOp, ByteSpan, Literal, Outline, Refusal, SpanUnit};

use crate::CliError;
use crate::json::{NONE, escape, list, num};

/// `{ "text", "outline" }`, or the parser's own refusal, located in the text. Score text, a
/// JSON line, has no one tree: it answers `{ "text", "score" }`, one item per placed item.
pub fn outline_data(text: &str) -> Result<String, CliError> {
    if text.trim_start().starts_with('{') {
        return score_data(text);
    }
    let tree = sva_ast::outline(text).map_err(|d| {
        CliError::Refusals(vec![Refusal::new("expression", d.span, d.code, d.message)])
    })?;
    Ok(format!(
        "{{ \"text\": \"{}\", \"outline\": {} }}",
        escape(text),
        node(&tree)
    ))
}

fn time(amount: f64, unit: SpanUnit) -> String {
    format!("\"{}\"", spelled(amount, unit))
}

fn score_data(text: &str) -> Result<String, CliError> {
    let score = sva_ast::parse_score(text)
        .map_err(|d| CliError::Refusals(vec![Refusal::new("score", d.span, d.code, d.message)]))?;
    let unit = score.unit;
    let mut items = Vec::new();
    for line in &score.lines {
        for item in &line.items {
            let fields = match item {
                Item::Event(e) => format!(
                    ", \"node\": \"@{}\", \"deg\": {}, \"len\": {}, \"args\": {}",
                    escape(&e.node.name),
                    e.deg
                        .map_or_else(|| NONE.to_string(), |(d, _)| d.to_string()),
                    e.len
                        .map_or_else(|| NONE.to_string(), |l| time(l.amount.to_f64(), unit)),
                    list(&e.binds, |b| format!(
                        "{{ \"name\": \"{}\", \"value\": \"{}\" }}",
                        escape(&b.name),
                        escape(&sva_ast::render_expr(&b.value))
                    ))
                ),
                Item::Mark(m) => format!(", \"mark\": \"{}\"", escape(&m.name)),
                Item::Comment(_) => String::new(),
                Item::Move(m) => format!(
                    ", \"node\": {}, \"lane\": \"{}\", \"to\": \"{}\", \"len\": {}, \"shape\": {}",
                    m.node
                        .as_ref()
                        .map_or_else(|| NONE.to_string(), |n| format!("\"@{}\"", escape(&n.name))),
                    escape(&m.lane.name),
                    escape(&m.to.text),
                    m.ramp
                        .map_or_else(|| NONE.to_string(), |r| time(r.len.amount.to_f64(), unit)),
                    m.ramp
                        .map_or_else(|| NONE.to_string(), |r| format!("\"{}\"", r.shape.name())),
                ),
            };
            items.push(format!(
                "{{ \"kind\": \"{}\", \"line\": {}, \"span\": {}, \"at\": {}{fields} }}",
                item.kind(),
                line.number,
                span(line.span),
                time(item.at().amount.to_f64(), unit)
            ));
        }
    }
    Ok(format!(
        "{{ \"text\": \"{}\", \"score\": {} }}",
        escape(text),
        list(&items, String::clone)
    ))
}

/// A position as bars and seconds.
fn placed(at: Pos, curve: &Curve) -> String {
    format!(
        "{{ \"bars\": {}, \"seconds\": {} }}",
        num(at.bars(curve)),
        num(at.seconds(curve))
    )
}

/// `sva-cli outline @score/song`: the score as the song a render of it is, its marks, tempo
/// curve, lanes, events and size.
pub fn song_data(source: &dyn sva_ast::Source, target: &str) -> Result<String, CliError> {
    let (graph, song) = crate::song_of(source, target)?;
    let curve = song.curve();
    let marks = list(song.marks(), |m| {
        format!(
            "{{ \"name\": \"{}\", \"at\": {}, \"file\": \"{}{EXTENSION}\", \"line\": {} }}",
            escape(&m.name),
            placed(m.at, curve),
            escape(&m.score),
            m.line
        )
    });
    let tempo = list(&curve.pieces(), |(from, to, start, end, a, b)| {
        format!(
            "{{ \"bars\": [{}, {}], \"seconds\": [{}, {}], \"bpm\": [{}, {}] }}",
            num(*from),
            num(*to),
            num(*start),
            num(*end),
            num(*a),
            num(*b)
        )
    });
    let instances = song.instances();
    let lanes = list(song.lanes(), |lane| {
        let reaching = instances
            .iter()
            .filter(|i| i.lanes.contains(&lane.id))
            .count();
        let moving = song.passed().iter().filter(|(_, l)| *l == lane.id).count();
        let home = match &lane.home {
            Home::Number(v, _) => num(*v),
            Home::Scale(_) => "\"a scale\"".to_string(),
            Home::Default(text) => format!("\"{}\"", escape(text)),
            Home::None => NONE.to_string(),
        };
        let held = |h: &Option<Held>| match h {
            Some(Held::Number(v)) => num(*v),
            Some(Held::Scale(s)) => format!("\"{:?}\"", s.cents),
            None => NONE.to_string(),
        };
        let moves = list(&lane.moves, |m| {
            let r = &m.reached;
            let line = graph.score(&r.score).map_or(0, |s| s.lines[r.line].number);
            format!(
                "{{ \"from\": {}, \"to\": {}, \"start\": {}, \"end\": {}, \"shape\": {}, \"file\": \"{}{EXTENSION}\", \"line\": {line}, \"chain\": \"{}\" }}",
                held(&m.from),
                held(&Some(m.to.clone())),
                placed(r.start, curve),
                placed(r.end.unwrap_or(r.start), curve),
                m.shape
                    .map_or_else(|| NONE.to_string(), |s| format!("\"{}\"", s.name())),
                escape(&r.score),
                escape(&r.chain.text(curve))
            )
        });
        format!(
            "{{ \"lane\": \"{}\", \"home\": {home}, \"moves\": {moves}, \"instances\": {}, \"moving\": {moving} }}",
            escape(&lane.id.name()),
            reaching
        )
    });
    let events = events(&graph, &song);
    let typed = sva_engine::types(&graph, crate::PROBE).map_err(CliError::Engine)?;
    Ok(format!(
        "{{ \"song\": \"{}\", \"marks\": {marks}, \"tempo\": {tempo}, \"lanes\": {lanes}, \"events\": {}, \"size\": {{ \"events\": {}, \"instances\": {}, \"limit\": {} }} }}",
        escape(target),
        list(&events, String::clone),
        events.len(),
        typed.paths().count(),
        sva_engine::instantiate::MAX_INSTANCES
    ))
}

/// Each note `song` places, once however many reads reach it: its song time, when it first
/// sounds, how long its line holds it, its node, pitch, line and where each parameter comes from.
fn events(graph: &sva_ast::Graph, song: &sva_ast::song::Song) -> Vec<String> {
    let curve = song.curve();
    let source = graph.sourced();
    song.instances()
        .iter()
        .filter_map(|i| Some((i, i.line.as_ref()?, i.at?, i.sounds(curve)?)))
        .map(|(i, (score, li, ii), at, sounds)| {
            let Some(Item::Event(e)) = graph.score(score).map(|s| &s.lines[*li].items[*ii]) else {
                return String::new();
            };
            let hz = match e.deg {
                Some((n, _)) => song
                    .key_at(at.seconds(curve))
                    .zip(song.scale_at(at.seconds(curve)).and_then(|s| s.ratio(n)))
                    .map(|(k, r)| k * r),
                None => e
                    .binds
                    .iter()
                    .find(|b| b.name == "f0")
                    .and_then(|b| sva_ast::fold_number(&b.value, &|n| sva_ast::note_hz(n))),
            };
            let mut params = sva_ast::song::parameters(&source, &i.node);
            params.sort();
            let source = |p: &String| {
                if i.lanes.contains(&LaneId::Sound(i.node.clone(), p.clone())) {
                    "lane"
                } else if i.bound.contains(p) {
                    "line"
                } else {
                    "default"
                }
            };
            let len = song.ends(graph, i).map_or_else(
                || NONE.to_string(),
                |end| {
                    format!(
                        "{{ \"bars\": {}, \"seconds\": {} }}",
                        num(end.bars(curve) - at.bars(curve)),
                        num(end.seconds(curve) - at.seconds(curve))
                    )
                },
            );
            format!(
                "{{ \"at\": {}, \"sounds\": {}, \"len\": {len}, \"node\": \"{}\", \"hz\": {}, \"file\": \"{}{EXTENSION}\", \"line\": {}, \"params\": {} }}",
                placed(at, curve),
                placed(sounds, curve),
                escape(&i.node),
                hz.map_or_else(|| NONE.to_string(), num),
                escape(score),
                graph.score(score).map_or(0, |s| s.lines[*li].number),
                list(&params, |p| format!(
                    "{{ \"name\": \"{}\", \"from\": \"{}\" }}",
                    escape(p),
                    source(p)
                ))
            )
        })
        .collect()
}

/// `{ "song", "events" }`: the events `song_data` lists, without its marks, tempo, lanes and
/// size, whose count types the whole graph on the engine; what a page draws a roll from.
pub fn events_data(source: &dyn sva_ast::Source, target: &str) -> Result<String, CliError> {
    let (graph, song) = crate::song_of(source, target)?;
    Ok(format!(
        "{{ \"song\": \"{}\", \"events\": {} }}",
        escape(target),
        list(&events(&graph, &song), String::clone)
    ))
}

fn span(at: ByteSpan) -> String {
    format!("{{ \"start\": {}, \"end\": {} }}", at.start, at.end)
}

fn node(o: &Outline) -> String {
    let (kind, fields) = match &o.form {
        Form::Literal(l) => ("literal", literal(l)),
        Form::Name(name) => ("name", format!("\"name\": \"{}\"", escape(name))),
        Form::Operator {
            op,
            at,
            left,
            right,
        } => (
            "operator",
            format!(
                "\"op\": \"{}\", \"at\": {}, \"left\": {}, \"right\": {}",
                symbol(*op),
                span(*at),
                node(left),
                node(right)
            ),
        ),
        Form::Call { name, at, args } => (
            "call",
            format!(
                "\"name\": \"{}\", \"at\": {}, \"args\": {}",
                escape(name),
                span(*at),
                list(args, argument)
            ),
        ),
        Form::Ref {
            path,
            at,
            arg,
            binds,
            address,
        } => (
            "ref",
            format!(
                "\"path\": \"{}\", \"at\": {}, \"read\": \"{}\", \"arg\": {}, \"binds\": {}",
                escape(path),
                span(*at),
                read(*address),
                node(arg),
                list(binds, argument)
            ),
        ),
        Form::SelfRef { at, arg, address } => (
            "self",
            format!(
                "\"at\": {}, \"read\": \"{}\", \"arg\": {}",
                span(*at),
                read(*address),
                node(arg)
            ),
        ),
        Form::Indexed { name, at, arg } => (
            "parameter",
            format!(
                "\"name\": \"{}\", \"at\": {}, \"read\": \"{}\", \"arg\": {}",
                escape(name),
                span(*at),
                read(Address::Index),
                node(arg)
            ),
        ),
    };
    format!(
        "{{ \"kind\": \"{kind}\", \"span\": {}, \"written\": {}, {fields} }}",
        span(o.span),
        o.written
    )
}

/// `time` for `x(e)`, `index` for `x[i]`.
fn read(address: Address) -> &'static str {
    match address {
        Address::Time => "time",
        Address::Index => "index",
    }
}

fn argument(a: &Argument) -> String {
    match a {
        Argument::Positional(value) => {
            format!("{{ \"kind\": \"positional\", \"value\": {} }}", node(value))
        }
        Argument::Named { name, at, value } => format!(
            "{{ \"kind\": \"named\", \"name\": \"{}\", \"at\": {}, \"value\": {} }}",
            escape(name),
            span(*at),
            node(value)
        ),
    }
}

/// A unit the lexer resolves, `s` or `db`, is a number already; bars and samples wait for a
/// tempo and a rate.
fn literal(l: &Literal) -> String {
    let (unit, value) = match l {
        Literal::Num(v) => ("number", num(*v)),
        Literal::Bars(v) => ("bars", num(*v)),
        Literal::Samples(v) => ("samples", num(*v)),
        Literal::Str(s) => ("text", format!("\"{}\"", escape(s))),
        Literal::Path(path) => ("path", format!("\"{}\"", escape(path))),
        Literal::Recorded(recording) => ("path", format!("\"{}\"", escape(&recording.path))),
    };
    format!("\"unit\": \"{unit}\", \"value\": {value}")
}

fn symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
    }
}
