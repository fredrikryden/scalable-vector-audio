// Concern: a fitted state as the composition's text: node defaults, score lines bar the silent ones and restated defaults | Non-concern: whether to write | IO: (State, texts) -> (path, text)

use std::collections::BTreeSet;

use sva_ast::{Arg, Expr};

use super::model::constant;

use super::price::{reads, simplify};
use super::state::{Slot, State};
use crate::cli_error::CliError;

fn header(line: &str) -> Option<&str> {
    let (name, _) = line.split_once('=')?;
    let name = name.trim();
    (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
        .then_some(name)
}

/// The defaults a node's lines all leave at a zero that silences them: written out of its body.
fn silent(s: &State, node: usize) -> BTreeSet<String> {
    let lines: Vec<usize> = (0..s.model.lines.len())
        .filter(|l| s.alive(*l) && s.line_node(*l) == node)
        .collect();
    s.centre
        .iter()
        .filter(|((n, name), c)| {
            *n == node
                && s.vars[s.var_of[**c]].zero_free()
                && s.x[**c] == 0.0
                && lines.iter().all(|l| {
                    s.dev
                        .get(&(*l, name.clone()))
                        .is_none_or(|d| s.x[*d] == 0.0)
                })
        })
        .map(|((_, name), _)| name.clone())
        .collect()
}

/// `e` with every `pwl` knot dropped that sits on the straight line through its constant
/// neighbours, so the table reads the same at every instant.
fn compact(e: &Expr) -> Expr {
    let mapped = sva_ast::map_children(e, sva_ast::Binds::Keep, |c| {
        Ok::<_, std::convert::Infallible>(compact(c))
    });
    let Ok(inner) = mapped;
    let Expr::Call { name, args, span } = &inner else {
        return inner;
    };
    if name != sva_ast::PWL || args.len() < 7 {
        return inner;
    }
    let value = |a: &Arg| match a {
        Arg::Pos(x) => constant(x),
        Arg::Named(..) => None,
    };
    let knot = |j: usize| (value(&args[1 + 2 * j]), value(&args[2 + 2 * j]));
    let knots = (args.len() - 1) / 2;
    let mut kept = vec![0];
    for j in 1..knots.saturating_sub(1) {
        let on_line = match (knot(kept[kept.len() - 1]), knot(j), knot(j + 1)) {
            ((Some(tp), Some(vp)), (Some(tj), Some(vj)), (Some(tn), Some(vn))) if tn > tp => {
                vp + (vn - vp) * (tj - tp) / (tn - tp) == vj
            }
            _ => false,
        };
        if !on_line {
            kept.push(j);
        }
    }
    kept.push(knots - 1);
    let mut out = vec![args[0].clone()];
    for j in kept {
        out.push(args[1 + 2 * j].clone());
        out.push(args[2 + 2 * j].clone());
    }
    Expr::Call {
        name: name.clone(),
        args: out,
        span: *span,
    }
}

/// The node's body with its silent defaults folded out, and the names it still reads.
fn folded(s: &State, node: usize) -> (Expr, BTreeSet<String>) {
    let body = compact(&simplify(
        &s.model.nodes[&s.nodes[node]].body,
        &silent(s, node),
    ));
    let read = reads(&body, &BTreeSet::new());
    (body, read)
}

fn node_text(s: &State, node: usize, text: &str) -> String {
    let (body, read) = folded(s, node);
    let original = &s.model.nodes[&s.nodes[node]].body;
    let rewritten = !silent(s, node).is_empty() || compact(original) != *original;
    let defaults = &s.model.nodes[&s.nodes[node]].defaults;
    let lines: Vec<&str> = text.lines().collect();
    let last = lines
        .iter()
        .rposition(|l| !l.trim().is_empty() && !l.trim_start().starts_with(';'));
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if Some(i) == last && rewritten {
            out.push_str(&sva_ast::render_expr(&body));
            out.push('\n');
            continue;
        }
        match header(line) {
            Some(name) if defaults.iter().any(|(d, _)| d == name) && !read.contains(name) => {}
            Some(name) => match s.centre.get(&(node, name.to_string())) {
                Some(c) => {
                    let var = &s.vars[s.var_of[*c]];
                    let v = var.value(s.x[*c].clamp(var.lo, var.hi));
                    out.push_str(&format!("{name} = {}\n", var.text(v)));
                }
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            },
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

fn field(name: &str, text: String) -> String {
    format!("\"{}\": {text}", crate::json::escape(name))
}

fn score_text(s: &State, text: &str) -> Result<String, CliError> {
    let mut out = String::new();
    for (i, row) in text.lines().enumerate() {
        let lines: Vec<usize> = (0..s.model.lines.len())
            .filter(|l| s.model.lines[*l].number == i + 1)
            .collect();
        let line = match lines.as_slice() {
            [] => {
                out.push_str(row);
                out.push('\n');
                continue;
            }
            [one] => *one,
            _ => {
                return Err(CliError::usage(
                    "solve.prox_list_line",
                    format!(
                        "score line {} places {} events from lists; a priced fit writes one event \
                         a line, so write each on its own",
                        i + 1,
                        lines.len()
                    ),
                ));
            }
        };
        if !s.alive(line) {
            continue;
        }
        let written: serde_json::Map<String, serde_json::Value> = serde_json::from_str(row)
            .map_err(|e| {
                CliError::usage("solve.prox_score", format!("score line {}: {e}", i + 1))
            })?;
        let l = &s.model.lines[line];
        let mut parts = vec![
            field("at", written["at"].to_string()),
            field(
                "node",
                format!(
                    "\"@{}\"",
                    relative(s.model.score.as_deref().unwrap_or(""), &l.node)
                ),
            ),
        ];
        let read = folded(s, s.line_node(line)).1;
        for (name, slot) in s.slots(line) {
            if !read.contains(&name) {
                continue;
            }
            let text = match slot {
                Slot::Fused { dev, .. } if s.x[dev] == 0.0 => None,
                Slot::Fixed(_) => written.get(&name).map(|v| v.to_string()),
                _ => {
                    let var = s.var_of_slot(slot).expect("a varied slot");
                    let t = var.text(s.value(slot));
                    Some(match var.unit {
                        "" => t,
                        _ => format!("\"{t}\""),
                    })
                }
            };
            if let Some(t) = text {
                parts.push(field(&name, t));
            }
        }
        out.push_str(&format!("{{{}}}\n", parts.join(", ")));
    }
    Ok(out)
}

/// `to` as a ref from the score at `from`.
fn relative(from: &str, to: &str) -> String {
    let depth = from.matches('/').count();
    format!("{}{to}", "../".repeat(depth))
}

/// Every file the state changes, under the paths the composition reads them at.
pub fn files(
    s: &State,
    text: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, CliError> {
    let mut out = Vec::new();
    for (n, path) in s.nodes.iter().enumerate() {
        match s.copied.get(path) {
            Some(source) if super::price::used(s, n) => {
                let t = text(source)
                    .ok_or_else(|| CliError::NotFound(format!("no node `{source}`")))?;
                out.push((path.clone(), node_text(s, n, &t)));
            }
            Some(_) => {}
            None => {
                let Some(t) = text(path) else {
                    continue;
                };
                let new = node_text(s, n, &t);
                if new != t {
                    out.push((path.clone(), new));
                }
            }
        }
    }
    let Some(score) = &s.model.score else {
        return Ok(out);
    };
    let score = format!("{score}.jsonl");
    let t = text(&score).ok_or_else(|| CliError::NotFound(format!("no score `{score}`")))?;
    let new = score_text(s, &t)?;
    if new != t {
        out.push((score, new));
    }
    Ok(out)
}
