// Concern: one edit and the files it rewrites over a composition, every value a moved line plays kept exact | Non-concern: proposing edits (assign, split, merge) | IO: (Proposal, texts) -> files

pub mod assign;
pub mod merge;
pub mod split;

use std::collections::{BTreeMap, BTreeSet};

use crate::cli_error::CliError;
use crate::model::{Model, header, relative};
use crate::solve::Vary;

#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Lines moved to `to`, each keeping every value it plays.
    Move { lines: Vec<usize>, to: String },
    /// A new node at `path` copying `from`'s text.
    Copy { from: String, path: String },
    /// Node defaults set; lines' fields re-expressed so each line plays the same.
    Recentre {
        node: String,
        defaults: Vec<(String, f64)>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub label: String,
    pub edits: Vec<Edit>,
}

impl Edit {
    fn apply(&self, m: &mut Model) {
        match self {
            Edit::Copy { from, path } => {
                let node = m.nodes[from].clone();
                m.nodes.insert(path.clone(), node);
            }
            Edit::Move { lines, to } => {
                for l in lines {
                    let kept = playing(m, *l);
                    m.lines[*l].node = to.clone();
                    restated(m, *l, &kept);
                }
            }
            Edit::Recentre { node, defaults } => {
                let lines: Vec<usize> = lines_of(m, node).collect();
                let kept: Vec<_> = lines.iter().map(|l| playing(m, *l)).collect();
                let held = &mut m.nodes.get_mut(node).expect("a node").defaults;
                for (name, v) in defaults {
                    let d = held.iter_mut().find(|(d, _)| d == name).expect("a default");
                    d.1 = *v;
                }
                for (l, k) in lines.into_iter().zip(kept) {
                    restated(m, l, &k);
                }
            }
        }
    }
}

/// `e` applied to `m`, and kept.
fn push(m: &mut Model, edits: &mut Vec<Edit>, e: Edit) {
    e.apply(m);
    edits.push(e);
}

fn lines_of<'a>(m: &'a Model, node: &'a str) -> impl Iterator<Item = usize> + 'a {
    (0..m.lines.len()).filter(move |l| m.lines[*l].node == node)
}

/// The value line `line` plays for `name`: its own field, else its node's default.
fn played(m: &Model, line: usize, name: &str) -> Option<f64> {
    let l = &m.lines[line];
    let field = l.fields.iter().find(|(f, _)| f == name);
    let default = || m.nodes[&l.node].defaults.iter().find(|(d, _)| d == name);
    field.or_else(default).map(|(_, v)| *v)
}

/// Each default line `line`'s node reads, at the value the line plays.
fn playing(m: &Model, line: usize) -> Vec<(String, f64)> {
    let node = &m.nodes[&m.lines[line].node];
    let named = node.defaults.iter().map(|(name, _)| name);
    named
        .map(|name| (name.clone(), played(m, line, name).expect("a default")))
        .collect()
}

/// Line `line`'s fields for `kept`, against its node: a value its default restates dropped.
fn restated(m: &mut Model, line: usize, kept: &[(String, f64)]) {
    let node = &m.nodes[&m.lines[line].node];
    let default = |name: &str| {
        node.defaults
            .iter()
            .find(|(d, _)| d == name)
            .map(|(_, v)| *v)
    };
    let restates: Vec<bool> = kept.iter().map(|(n, v)| default(n) == Some(*v)).collect();
    let fields = &mut m.lines[line].fields;
    for ((name, v), restates) in kept.iter().zip(restates) {
        match (fields.iter().position(|(f, _)| f == name), restates) {
            (Some(i), true) => {
                fields.remove(i);
            }
            (Some(i), false) => fields[i].1 = *v,
            (None, true) => {}
            (None, false) => fields.push((name.clone(), *v)),
        }
    }
}

/// The `--vary` naming `name` in the file at `path`, the first as the solver reads it.
fn varied<'a>(varies: &'a [Vary], path: &str, name: &str) -> Option<&'a Vary> {
    varies.iter().find(|v| v.names(path, name))
}

/// Whether line `line` sounds: its first own field varied up from 0 is above 0, if it has one.
fn alive(m: &Model, varies: &[Vary], line: usize) -> bool {
    let score = m.score.as_deref().unwrap_or("");
    let l = &m.lines[line];
    let defaults = &m.nodes[&l.node].defaults;
    l.fields
        .iter()
        .filter(|(f, _)| !defaults.iter().any(|(d, _)| d == f))
        .find(|(f, _)| varied(varies, score, f).is_some_and(|v| !v.log && v.bounds().0 == 0.0))
        .is_none_or(|(_, x)| *x > 0.0)
}

/// `v` written exactly, in `--vary`'s rendered unit where one names it.
fn exact(varies: &[Vary], path: &str, name: &str, v: f64) -> (f64, &'static str) {
    let v = if v == 0.0 { 0.0 } else { v };
    (
        v,
        varied(varies, path, name).map_or("", Vary::rendered_unit),
    )
}

fn node_text(
    text: &str,
    path: &str,
    was: &[(String, f64)],
    now: &[(String, f64)],
    varies: &[Vary],
) -> String {
    let mut out = String::new();
    for line in text.lines() {
        let set = header(line)
            .and_then(|name| now.iter().find(|(n, _)| n == name))
            .filter(|(name, v)| !was.iter().any(|(n, w)| n == name && w == v));
        match set {
            Some((name, v)) => {
                let (v, unit) = exact(varies, path, name, *v);
                out.push_str(&format!("{name} = {v}{unit}\n"));
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

fn score_text(
    was: &Model,
    now: &Model,
    moved: &BTreeSet<usize>,
    text: &str,
    varies: &[Vary],
) -> Result<String, CliError> {
    let score = now.score.as_deref().expect("a moved line is a score's");
    let mut out = String::new();
    for (i, row) in text.lines().enumerate() {
        let here: Vec<usize> = (0..now.lines.len())
            .filter(|l| now.lines[*l].number == i + 1)
            .collect();
        let line = match here.as_slice() {
            [one] if moved.contains(one) => *one,
            many if many.len() > 1 && many.iter().any(|l| moved.contains(l)) => {
                return Err(CliError::usage(
                    "structure.list_line",
                    format!(
                        "score line {} places {} events from lists; a structure edit moves one \
                         event a line, so write each on its own",
                        i + 1,
                        many.len()
                    ),
                ));
            }
            _ => {
                out.push_str(row);
                out.push('\n');
                continue;
            }
        };
        let mut written: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(row).expect("a score line the graph read is an object");
        let node = &now.lines[line].node;
        written.insert(
            "node".to_string(),
            serde_json::Value::String(format!("@{}", relative(score, node))),
        );
        let field = |m: &Model, name: &str| {
            let fields = &m.lines[line].fields;
            fields.iter().find(|(f, _)| f == name).map(|(_, v)| *v)
        };
        for (name, _) in &now.nodes[node].defaults {
            let set = field(now, name);
            if set == field(was, name) {
                continue;
            }
            match set {
                None => {
                    written.remove(name);
                }
                Some(v) => {
                    let value = match exact(varies, score, name, v) {
                        (v, "") => serde_json::Value::from(v),
                        (v, unit) => serde_json::Value::String(format!("{v}{unit}")),
                    };
                    written.insert(name.clone(), value);
                }
            }
        }
        out.push_str(&serde_json::Value::Object(written).to_string());
        out.push('\n');
    }
    Ok(out)
}

/// The files `p` rewrites over the composition `model` reads, under the paths it reads them at.
pub fn applied(
    model: &Model,
    texts: &dyn Fn(&str) -> Option<String>,
    varies: &[Vary],
    p: &Proposal,
) -> Result<Vec<(String, String)>, CliError> {
    let mut m = model.clone();
    let mut from: BTreeMap<String, String> = BTreeMap::new();
    let mut moved = BTreeSet::new();
    for e in &p.edits {
        match e {
            Edit::Copy { from: f, path } => {
                let source = from.get(f).cloned().unwrap_or_else(|| f.clone());
                from.insert(path.clone(), source);
            }
            Edit::Recentre { node, .. } => {
                from.entry(node.clone()).or_insert_with(|| node.clone());
                moved.extend(lines_of(&m, node));
            }
            Edit::Move { lines, .. } => moved.extend(lines.iter().copied()),
        }
        e.apply(&mut m);
    }
    let mut out = Vec::new();
    for (path, source) in &from {
        let text =
            texts(source).ok_or_else(|| CliError::NotFound(format!("no node `{source}`")))?;
        let (was, now) = (&model.nodes[source].defaults, &m.nodes[path].defaults);
        if path != source || was != now {
            out.push((path.clone(), node_text(&text, path, was, now, varies)));
        }
    }
    if let (false, Some(score)) = (moved.is_empty(), &m.score) {
        let file = format!("{score}.jsonl");
        let text = texts(&file).ok_or_else(|| CliError::NotFound(format!("no score `{file}`")))?;
        out.push((file, score_text(model, &m, &moved, &text, varies)?));
    }
    Ok(out)
}
