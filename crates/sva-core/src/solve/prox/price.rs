// Concern: what a state's composition costs in bits, each number its `--vary`'s, and each number's step weight | Non-concern: stepping, what it sounds like | IO: (State) -> (numbers, bits), weights

use std::collections::BTreeSet;

use sva_ast::{Arg, BinOp, Expr, Literal};

use super::state::{Role, Slot, State};
use crate::model::constant;

fn num(n: f64) -> Expr {
    Expr::Lit(Literal::Num(n))
}

/// `e` with every name in `zeros` at 0, folded where that leaves constants.
pub fn simplify(e: &Expr, zeros: &BTreeSet<String>) -> Expr {
    let is = |x: &Expr, v: f64| constant(x) == Some(v);
    let out = match e {
        Expr::Var(v) if zeros.contains(v) => return num(0.0),
        Expr::Bin(op, a, b) => {
            let (a, b) = (simplify(a, zeros), simplify(b, zeros));
            match op {
                BinOp::Mul if is(&a, 0.0) || is(&b, 0.0) => num(0.0),
                BinOp::Mul if is(&a, 1.0) => b,
                BinOp::Mul if is(&b, 1.0) => a,
                BinOp::Add if is(&a, 0.0) => b,
                BinOp::Add | BinOp::Sub if is(&b, 0.0) => a,
                BinOp::Div if is(&a, 0.0) => num(0.0),
                BinOp::Div if is(&b, 1.0) => a,
                _ => Expr::Bin(*op, Box::new(a), Box::new(b)),
            }
        }
        Expr::Call { name, args, span } => Expr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| match a {
                    Arg::Pos(x) => Arg::Pos(simplify(x, zeros)),
                    Arg::Named(k, x) => Arg::Named(k.clone(), simplify(x, zeros)),
                })
                .collect(),
            span: *span,
        },
        other => other.clone(),
    };
    match constant(&out) {
        Some(c) if !matches!(out, Expr::Lit(_)) => num(c),
        _ => out,
    }
}

fn free(e: &Expr, bound: &mut Vec<String>, out: &mut BTreeSet<String>) {
    match e {
        Expr::Var(v) if v != "t" && v != "pi" && !bound.contains(v) => {
            out.insert(v.clone());
        }
        Expr::Call { name, args, .. } if name == "sum" => {
            if let [Arg::Pos(Expr::Var(k)), rest @ ..] = args.as_slice() {
                bound.push(k.clone());
                for a in rest {
                    let (Arg::Pos(x) | Arg::Named(_, x)) = a;
                    free(x, bound, out);
                }
                bound.pop();
            }
        }
        _ => {
            for c in sva_ast::children(e, sva_ast::Binds::Keep) {
                free(c, bound, out);
            }
        }
    }
}

/// The names `body` still reads once `zeros` are 0.
pub fn reads(body: &Expr, zeros: &BTreeSet<String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    free(&simplify(body, zeros), &mut Vec::new(), &mut out);
    out
}

fn node_zeros(s: &State, node: usize, but: Option<&str>) -> BTreeSet<String> {
    s.centre
        .iter()
        .filter(|((n, name), c)| {
            *n == node
                && (s.vars[s.var_of[**c]].zero_free() && s.x[**c] == 0.0
                    || Some(name.as_str()) == but)
        })
        .map(|((_, name), _)| name.clone())
        .collect()
}

pub fn used(s: &State, node: usize) -> bool {
    s.nodes[node] == s.model.root
        || (0..s.model.lines.len()).any(|l| s.alive(l) && s.line_node(l) == node)
}

/// The bits a default of `node` costs: its `--vary`'s, nothing where it varies nowhere.
fn centre_bits(s: &State, node: usize, name: &str) -> f64 {
    s.centre
        .get(&(node, name.to_string()))
        .map_or(0.0, |c| s.vars[s.var_of[*c]].bits())
}

/// What a used node writes, `(numbers, bits)`: each default the body still reads.
pub fn node_numbers(s: &State, node: usize) -> (usize, f64) {
    let path = &s.nodes[node];
    let def = &s.model.nodes[path];
    let live = reads(&def.body, &node_zeros(s, node, None));
    let read: Vec<&String> = def
        .defaults
        .iter()
        .map(|(name, _)| name)
        .filter(|name| live.contains(*name))
        .collect();
    let bits = read.iter().map(|n| centre_bits(s, node, n)).sum();
    (read.len(), bits)
}

/// What a line writes, `(numbers, bits)`, none where no score holds it: its onset, each field its node has no default
/// for, and each deviation its node's body still reads.
pub fn line_numbers(s: &State, line: usize) -> (usize, f64) {
    if s.model.score.is_none() {
        return (0, 0.0);
    }
    let l = &s.model.lines[line];
    let def = &s.model.nodes[&l.node];
    let slots = s.slots(line);
    let zeros: BTreeSet<String> = slots
        .iter()
        .filter(|(_, slot)| {
            matches!(slot, Slot::Fused { .. })
                && s.var_of_slot(*slot).is_some_and(|v| v.zero_free())
                && s.coord_of(*slot) == Some(0.0)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let live = reads(&def.body, &zeros);
    let (mut n, mut bits) = (1, 0.0);
    for (name, slot) in &slots {
        let defaulted = def.defaults.iter().any(|(d, _)| d == name);
        let written = match slot {
            Slot::Fused { dev, .. } => s.x[*dev] != 0.0 && live.contains(name),
            Slot::Fixed(_) if defaulted => l.fields.iter().any(|(f, _)| f == name),
            _ => true,
        };
        if written {
            n += 1;
            bits += s.var_of_slot(*slot).map_or(0.0, |v| v.bits());
        }
    }
    (n, bits)
}

/// The numbers the state's composition writes and their bits.
pub fn numbers(s: &State) -> (usize, f64) {
    let mut out = (0, 0.0);
    let mut add = |(n, b): (usize, f64)| {
        out.0 += n;
        out.1 += b;
    };
    (0..s.nodes.len())
        .filter(|n| used(s, *n))
        .for_each(|n| add(node_numbers(s, n)));
    (0..s.model.lines.len())
        .filter(|l| s.alive(*l))
        .for_each(|l| add(line_numbers(s, l)));
    out
}

/// A zero-free default's price in bits: its own and every default's it silences at 0.
fn gate(s: &State, node: usize, name: &str) -> f64 {
    let body = &s.model.nodes[&s.nodes[node]].body;
    let before = reads(body, &node_zeros(s, node, None));
    let after = reads(body, &node_zeros(s, node, Some(name)));
    let freed: f64 = before
        .difference(&after)
        .filter(|n| n.as_str() != name)
        .map(|n| centre_bits(s, node, n))
        .sum();
    centre_bits(s, node, name) + freed
}

/// The bits a line writes besides its deviations, all gone with its gain.
fn line_price(s: &State, line: usize) -> f64 {
    let def = &s.model.nodes[&s.model.lines[line].node];
    s.slots(line)
        .iter()
        .filter(|(name, _)| !def.defaults.iter().any(|(d, _)| d == name))
        .map(|(_, slot)| s.var_of_slot(*slot).map_or(0.0, |v| v.bits()))
        .sum()
}

/// Each coordinate's L1 weight in bits from an unpriced fit `x0`; 0 where unpriced.
pub fn weights(s: &State, scales: &[f64]) -> Vec<f64> {
    s.roles
        .iter()
        .enumerate()
        .map(|(i, role)| {
            let var = &s.vars[s.var_of[i]];
            let x0 = s.x[i];
            match role {
                Role::Centre { node, name } if var.zero_free() => {
                    gate(s, *node, name) / x0.abs().max(var.res)
                }
                Role::Centre { .. } => 0.0,
                Role::Dev { .. } => var.bits() / x0.abs().max(var.res),
                Role::Own {
                    line, root: true, ..
                } => line_price(s, *line) / x0.max(0.1 * scales[i]).max(1e-300),
                Role::Own { .. } => 0.0,
            }
        })
        .collect()
}
