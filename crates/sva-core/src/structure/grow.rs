// Concern: the growths a finer starter offers: a voice kind's harmonics raised, lone partials added at an onset | Non-concern: reading the clip, scoring | IO: (Model, Readings, Limits) -> Vec<Proposal>

use sva_ast::{Arg, Binds, Expr};

use super::{Edit, Proposal, alive, lines_of, played, varied};
use crate::cli_error::CliError;
use crate::model::{Model, constant, header, relative};
use crate::seed::{self, Limits, Readings, SAME_PITCH_CENTS};
use crate::solve::Vary;

/// The growths of a starter written at `from` to `to`: one per voice kind when harmonics
/// rise, one per onset with lone partials left to add, each added line's `vel` its track's
/// amplitude just after the onset, so it sounds as heard and the descent prices it from there.
pub fn proposals(
    model: &Model,
    texts: &dyn Fn(&str) -> Option<String>,
    varies: &[Vary],
    r: &Readings,
    from: Limits,
    to: Limits,
) -> Result<Vec<Proposal>, CliError> {
    let Some(score) = model.score.as_deref() else {
        return Ok(Vec::new());
    };
    if varied(varies, score, "at").is_some() {
        return Err(CliError::usage(
            "structure.grow_at",
            "`--vary` names `at`; growth finds a line's reading by its onset as written, so \
             leave `at` fixed",
        ));
    }
    let file = format!("{score}.jsonl");
    let text = texts(&file).ok_or_else(|| CliError::NotFound(format!("no score `{file}`")))?;
    let rows: Vec<&str> = text.lines().collect();
    let onsets: Vec<Option<String>> = model
        .lines
        .iter()
        .map(|l| {
            let row: serde_json::Value =
                serde_json::from_str(rows[l.number - 1]).expect("a score line the graph read");
            row["at"].as_str().map(str::to_string)
        })
        .collect();
    let mut out = Vec::new();
    if to.harmonics > from.harmonics {
        let mut tones: Vec<&str> = r.voices.iter().map(|v| v.tone).collect();
        tones.sort_unstable();
        tones.dedup();
        for tone in tones {
            let base = seed::voice(tone);
            let mut edits = Vec::new();
            for node in model.nodes.keys().filter(|n| of_kind(n, &base)) {
                let current =
                    texts(node).ok_or_else(|| CliError::NotFound(format!("no node `{node}`")))?;
                let fields = lines_of(model, node)
                    .filter(|l| alive(model, varies, *l))
                    .map(|l| (l, graft(model, r, tone, onsets[l].as_deref(), l, from, to)))
                    .collect();
                edits.push(Edit::Rewrite {
                    node: node.clone(),
                    text: grown(&current, to.harmonics),
                    fields,
                });
            }
            if !edits.is_empty() {
                out.push(Proposal {
                    label: format!("grow {base} to {} harmonics", to.harmonics),
                    edits,
                });
            }
        }
    }
    let node = format!("@{}", relative(score, seed::PARTIAL));
    for e in &r.events {
        let at = format!("{}s", seed::s6(e.at_secs));
        let held: Vec<f64> = (0..model.lines.len())
            .filter(|l| model.lines[*l].node == seed::PARTIAL && onsets[*l] == Some(at.clone()))
            .filter_map(|l| played(model, l, "f0"))
            .collect();
        let rows: Vec<String> = seed::loud(r, e)
            .into_iter()
            .take(to.lone)
            .skip(from.lone)
            .filter(|(hz, _)| held.iter().all(|f| cents(*hz, *f) >= SAME_PITCH_CENTS))
            .map(|(hz, level)| seed::lone_line(&node, e, hz, level))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let mut edits = Vec::new();
        if !model.nodes.contains_key(seed::PARTIAL) {
            edits.push(Edit::Rewrite {
                node: seed::PARTIAL.to_string(),
                text: seed::lone(),
                fields: Vec::new(),
            });
        }
        let at_secs: f64 = seed::s6(e.at_secs).parse().expect("a number");
        let after = (0..model.lines.len())
            .filter(|l| onsets[*l] == Some(at.clone()) || model.lines[*l].at < at_secs)
            .map(|l| model.lines[l].number)
            .max()
            .unwrap_or(0);
        edits.push(Edit::Add { after, rows });
        out.push(Proposal {
            label: format!("lone partials at {at}"),
            edits,
        });
    }
    Ok(out)
}

/// The limits a starter was written at, as its composition still holds them: the most
/// harmonics a voice node sums, the most lone partials one onset places.
pub fn written(model: &Model) -> Limits {
    let harmonics = model
        .nodes
        .iter()
        .filter(|(n, _)| n.starts_with(&seed::voice("")))
        .filter_map(|(_, node)| summed(&node.body))
        .max()
        .unwrap_or(0);
    let mut onsets: Vec<u64> = model
        .lines
        .iter()
        .filter(|l| l.node == seed::PARTIAL)
        .map(|l| l.at.to_bits())
        .collect();
    onsets.sort_unstable();
    let lone = onsets
        .chunk_by(|a, b| a == b)
        .map(<[u64]>::len)
        .max()
        .unwrap_or(0);
    Limits { harmonics, lone }
}

/// The upper bound of the series `e` sums.
fn summed(e: &Expr) -> Option<usize> {
    if let Expr::Call { name, args, .. } = e
        && name == sva_ast::SERIES
        && let Some(Arg::Pos(top)) = args.get(2)
    {
        return constant(top).map(|n| n as usize);
    }
    sva_ast::children(e, Binds::Keep)
        .into_iter()
        .find_map(summed)
}

/// `node` is `base` or a split's copy of it, `<base>-<k>`.
fn of_kind(node: &str, base: &str) -> bool {
    node.strip_prefix(base).is_some_and(|k| {
        k.is_empty()
            || k.strip_prefix('-')
                .is_some_and(|k| !k.is_empty() && k.chars().all(|c| c.is_ascii_digit()))
    })
}

fn cents(a: f64, b: f64) -> f64 {
    (1200.0 * libm::log2(a / b)).abs()
}

/// The seed's node at `harmonics`, each default `current` writes kept as it writes it.
fn grown(current: &str, harmonics: usize) -> String {
    let written = |name: &str| current.lines().find(|l| header(l) == Some(name));
    let mut out = String::new();
    for line in seed::harmonic(harmonics).lines() {
        out.push_str(header(line).and_then(written).unwrap_or(line));
        out.push('\n');
    }
    out
}

/// Line `line`'s `h` past `from`'s harmonics up to `to`'s, off the nearest voice of `tone`
/// heard at its onset, re-referenced to its `vel` and its node's `tilt`; unheard where none is.
fn graft(
    m: &Model,
    r: &Readings,
    tone: &str,
    onset: Option<&str>,
    line: usize,
    from: Limits,
    to: Limits,
) -> Vec<(String, f64)> {
    let f0 = played(m, line, "f0").expect("a voice line's f0");
    let heard = r
        .voices
        .iter()
        .filter(|v| v.tone == tone && onset == Some(format!("{}s", seed::s6(v.at)).as_str()))
        .map(|v| (cents(f0, v.f0), v))
        .filter(|(c, _)| *c < SAME_PITCH_CENTS)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, v)| v);
    let vel = played(m, line, "vel").expect("a voice line's vel");
    let tilt = played(m, line, "tilt").expect("a voice node's tilt");
    (from.harmonics + 1..=to.harmonics)
        .map(|k| {
            let level = heard.and_then(|v| v.levels.get(k - 1)).copied();
            let db = seed::level_db(level.unwrap_or(0.0), vel, tilt, k);
            (format!("h{k}"), db)
        })
        .collect()
}
