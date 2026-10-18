// Concern: derives a target's song over the engine: the spans its folds ask, and an engine refusal of a moving value as bad-lane | Non-concern: the derivation (sva-ast song/) | IO: (Graph, root) -> Song

use sva_ast::song::{LaneId, Song, Span};
use sva_ast::{ByteSpan, DiagCode, Graph, Refusal};
use sva_engine::{BindingFault, Diagnostic, EngineError, Extent, Located};

use crate::cli_error::CliError;

/// `root`'s song lowered into `graph`: each instance a lane may move under is asked of the
/// engine for its span first, over the nodes no score sits under already lowered.
pub fn sung(graph: &mut Graph, root: &str, rate: u32) -> Result<Song, CliError> {
    let mut song = graph.song(root).map_err(CliError::Refusals)?;
    let asked = song.asked();
    let spans = match asked.is_empty() {
        true => Vec::new(),
        false => spans(graph, &song, asked, rate)?,
    };
    let lowered = song.lowered(graph, &spans).map_err(CliError::Refusals)?;
    graph.apply(lowered);
    Ok(song)
}

fn spans(
    graph: &Graph,
    song: &Song,
    asked: Vec<sva_ast::Expr>,
    rate: u32,
) -> Result<Vec<Span>, CliError> {
    let mut probing = graph.clone();
    probing.apply(sva_ast::song::Lowered {
        nodes: song.instruments().to_vec(),
    });
    let mut roots = Vec::with_capacity(asked.len());
    for (k, expr) in asked.into_iter().enumerate() {
        let name = format!("#span-{k}");
        probing
            .define_settled(None, &name, expr)
            .map_err(CliError::Refusals)?;
        roots.push(name);
    }
    let held = sva_engine::supports(&probing, &roots, rate).map_err(CliError::Engine)?;
    let seconds = |n: i64| f64::from(rate).recip() * n as f64;
    Ok(held
        .into_iter()
        .map(|Extent { start, end }| Span {
            start: (start != i64::MIN).then(|| seconds(start)),
            end: (end != i64::MAX).then(|| seconds(end - 1)),
        })
        .collect())
}

fn refused_file(e: &EngineError) -> Option<(String, Option<ByteSpan>)> {
    let (node, span) = match e {
        EngineError::Refused(d) => (d.location.node.clone(), d.location.span),
        EngineError::Binding { node, span, .. } => (node.clone(), *span),
        _ => return None,
    };
    let file = node.split(['(', '#']).next().unwrap_or(&node).to_string();
    Some((file, span))
}

/// The instance limit refusal, naming the placement chain of the line whose instance crossed
/// it.
fn crossed(
    song: &Song,
    graph: &Graph,
    (node, file, span): (&str, &str, Option<ByteSpan>),
    e: &EngineError,
) -> EngineError {
    let placed = song.instances().iter().rev().find(|i| {
        i.line.as_ref().is_some_and(|(score, li, _)| {
            score == file
                && graph
                    .score(score)
                    .is_some_and(|s| Some(s.lines[*li].span) == span)
        })
    });
    let chain = placed.map_or(String::new(), |i| {
        format!(" ({})", i.chain.line_text(song.curve()))
    });
    EngineError::Refused(Box::new(Diagnostic {
        code: e.code().to_string(),
        message: format!("{e}{chain}"),
        location: match graph.score(file) {
            Some(_) => Located::at(format!("{file}{}", sva_ast::score::EXTENSION), span),
            None => Located::at(node, span),
        },
        help: "place fewer distinct instances: a line whose values repeat shares one".to_string(),
    }))
}

/// An engine refusal inside what a moving lane reaches is the lane's: reported at the move,
/// with the engine's code, message and help, the lane and the instance's placement chain.
pub fn as_lane(song: &Song, graph: &Graph, e: EngineError) -> CliError {
    let Some((file, span)) = refused_file(&e) else {
        return CliError::Engine(e);
    };
    if let EngineError::Binding {
        fault: BindingFault::TooManyInstances(_),
        node,
        ..
    } = &e
    {
        return CliError::Engine(crossed(song, graph, (node, &file, span), &e));
    }
    let passed = song.passed();
    let reaches = |from: &str| from == file || graph.reaching(&[from.to_string()]).contains(&file);
    let candidates: Vec<&(usize, LaneId)> = passed
        .iter()
        .filter(|(k, _)| reaches(&song.instances()[*k].node))
        .collect();
    let (code, message, help) = match &e {
        EngineError::Refused(d) => (d.code.clone(), d.message.clone(), d.help.clone()),
        EngineError::Binding { fault, .. } => {
            (fault.code().to_string(), fault.to_string(), String::new())
        }
        other => (other.code().to_string(), other.to_string(), String::new()),
    };
    let written = span.and_then(|s| {
        let text = graph.text(&file)?;
        text.get(s.start..s.end.max(s.start)).map(str::to_string)
    });
    let names = |lane: &LaneId, text: &str| match lane {
        LaneId::Sound(_, param) => text
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|word| word == param),
        LaneId::Variable(_) | LaneId::Degree(_) => text.contains(&lane.name()),
    };
    let chosen = candidates
        .iter()
        .find(|(_, lane)| names(lane, &message))
        .or_else(|| {
            candidates
                .iter()
                .find(|(_, lane)| written.as_deref().is_some_and(|w| names(lane, w)))
        })
        .or(candidates.first())
        .copied();
    let Some((k, lane)) = chosen else {
        return CliError::Engine(e);
    };
    let instance = &song.instances()[*k];
    match song.moved_under(graph, *k, lane) {
        Some((file, at, text)) => {
            let (line, column) = sva_ast::score::line_column(text, at);
            CliError::Refusals(vec![Refusal::new(
                file,
                ByteSpan::at(at),
                DiagCode::BadLane,
                format!(
                    "line {line}, column {column}: the lane `{}` moves under `@{}` ({}), and the \
                     engine refuses a moving value there: {code}: {message}; help: {help}; pass \
                     a constant on the line, or move the lane between notes",
                    lane.name(),
                    instance.node,
                    instance.chain.text(song.curve()),
                ),
            )
            .placed((line, column))])
        }
        None => CliError::Engine(e),
    }
}
