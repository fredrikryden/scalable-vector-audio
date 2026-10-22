// Concern: a composition's structure as a priced fit reads it: root, node bodies and defaults, score lines | Non-concern: what it sounds like (played.rs) | IO: (Graph, root) -> Model

use std::collections::BTreeMap;

use sva_ast::{Arg, BinOp, Expr, Graph, Literal, resolve_ref_path};

use crate::cli_error::CliError;

#[derive(Clone, Debug)]
pub struct Node {
    pub body: Expr,
    pub defaults: Vec<(String, f64)>,
}

#[derive(Clone, Debug)]
pub struct Line {
    /// The score file's line it was read off, counted from 1.
    pub number: usize,
    pub at: f64,
    pub node: String,
    pub fields: Vec<(String, f64)>,
}

/// A composition as the lowering reads it: its root, its nodes' bodies and defaults, and the
/// lines of the one score the root places, or the root alone as one line where it places none.
#[derive(Clone, Debug)]
pub struct Model {
    pub root: String,
    pub score: Option<String>,
    pub nodes: BTreeMap<String, Node>,
    pub lines: Vec<Line>,
}

fn unsupported(what: String) -> CliError {
    CliError::usage(
        "solve.prox_model",
        format!(
            "{what}; a priced fit reads node defaults and score fields bound to numbers, and one score the root places"
        ),
    )
}

/// The number a constant expression folds to.
pub fn constant(e: &Expr) -> Option<f64> {
    Some(match e {
        Expr::Lit(Literal::Num(n)) => *n,
        Expr::Var(v) if v == "pi" => std::f64::consts::PI,
        Expr::Bin(op, a, b) => {
            let (a, b) = (constant(a)?, constant(b)?);
            match op {
                BinOp::Add => a + b,
                BinOp::Sub => a - b,
                BinOp::Mul => a * b,
                BinOp::Div => a / b,
                BinOp::Mod => return None,
            }
        }
        Expr::Call { name, args, .. } => {
            let xs: Vec<f64> = args
                .iter()
                .map(|a| match a {
                    Arg::Pos(x) => constant(x),
                    Arg::Named(..) => None,
                })
                .collect::<Option<_>>()?;
            match (name.as_str(), xs.as_slice()) {
                ("pow", [a, b]) => libm::pow(*a, *b),
                ("exp", [a]) => sva_formula::exp(*a),
                ("log", [a]) => libm::log(*a),
                ("sqrt", [a]) => a.sqrt(),
                _ => return None,
            }
        }
        _ => return None,
    })
}

impl Model {
    pub fn of(graph: &Graph, root: &str) -> Result<Model, CliError> {
        let mut nodes = BTreeMap::new();
        let node = |path: &str| -> Result<Node, CliError> {
            let defined = graph
                .defined(path)
                .ok_or_else(|| CliError::NotFound(format!("no node `{path}`")))?;
            let defaults = defined
                .defaults
                .iter()
                .map(|(k, v)| {
                    constant(v).map(|n| (k.clone(), n)).ok_or_else(|| {
                        unsupported(format!("`{path}` defaults `{k}` to no constant"))
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(Node {
                body: defined.body.clone(),
                defaults,
            })
        };
        let root_node = node(root)?;
        let mut scores = Vec::new();
        refs(&root_node.body, &mut scores);
        let score = match scores.as_slice() {
            [] => {
                let line = Line {
                    number: 0,
                    at: 0.0,
                    node: root.to_string(),
                    fields: Vec::new(),
                };
                nodes.insert(root.to_string(), root_node);
                return Ok(Model {
                    root: root.to_string(),
                    score: None,
                    nodes,
                    lines: vec![line],
                });
            }
            [score] => resolve_ref_path(root, score)
                .ok_or_else(|| unsupported(format!("`{root}` reads `@{score}`")))?,
            more => {
                return Err(unsupported(format!(
                    "`{root}` places {} scores",
                    more.len()
                )));
            }
        };
        let read = graph
            .score(&score)
            .ok_or_else(|| unsupported(format!("`@{score}` is no score")))?;
        nodes.insert(root.to_string(), root_node);
        let mut lines = Vec::new();
        for (row, item) in read.items() {
            let sva_ast::score::Item::Event(event) = item else {
                continue;
            };
            let path = resolve_ref_path(&score, &event.node.name)
                .ok_or_else(|| unsupported(format!("`{}` names no node", event.node.name)))?;
            if !nodes.contains_key(&path) {
                nodes.insert(path.clone(), node(&path)?);
            }
            let mut fields = Vec::new();
            for b in &event.binds {
                let v = constant(&b.value).ok_or_else(|| {
                    unsupported(format!("a line binds `{}` to no constant", b.name))
                })?;
                fields.push((b.name.clone(), v));
            }
            if let Some(len) = event.len {
                fields.push(("len".to_string(), len.amount.to_f64()));
            }
            lines.push(Line {
                number: row.number,
                at: event.at.amount.to_f64(),
                node: path,
                fields,
            });
        }
        Ok(Model {
            root: root.to_string(),
            score: Some(score),
            nodes,
            lines,
        })
    }
}

fn refs(e: &Expr, out: &mut Vec<String>) {
    if let Expr::Ref { path, .. } = e {
        out.push(path.clone());
    }
    for c in sva_ast::children(e, sva_ast::Binds::Keep) {
        refs(c, out);
    }
}
