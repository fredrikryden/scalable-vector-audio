// Concern: the numbers a composition writes below a root: each literal of a reached node's body and defaults, each field of its scores' lines | Non-concern: pricing them | IO: (&Graph, root) -> Written

use sva_ast::score::Item;
use sva_ast::{Expr, Graph, Literal};

/// One number as written: where, and the name it binds where it binds one.
#[derive(Clone, Debug, PartialEq)]
pub struct WrittenNumber {
    pub path: String,
    pub name: Option<String>,
    pub value: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Written {
    pub numbers: Vec<WrittenNumber>,
}

fn literals(e: &Expr, out: &mut Vec<f64>) {
    match e {
        Expr::Lit(Literal::Num(n) | Literal::Samples(n) | Literal::Bars(n)) => out.push(*n),
        _ => {
            for c in sva_ast::children(e, sva_ast::Binds::Substitute) {
                literals(c, out);
            }
        }
    }
}

/// What every node `root` reaches writes, in path order.
pub fn written(graph: &Graph, root: &str) -> Written {
    let mut numbers = Vec::new();
    let mut push = |path: &str, name: Option<&str>, e: &Expr| {
        let mut held = Vec::new();
        literals(e, &mut held);
        numbers.extend(held.into_iter().map(|value| WrittenNumber {
            path: path.to_string(),
            name: name.map(str::to_string),
            value,
        }));
    };
    for path in graph.reaching(&[root.to_string()]) {
        if let Some(score) = graph.score(&path) {
            for (_, item) in score.items() {
                let Item::Event(event) = item else {
                    continue;
                };
                let at = Expr::Lit(Literal::Num(event.at.amount.to_f64()));
                push(&path, Some("at"), &at);
                if let Some(len) = event.len {
                    push(
                        &path,
                        Some("len"),
                        &Expr::Lit(Literal::Num(len.amount.to_f64())),
                    );
                }
                for b in &event.binds {
                    push(&path, Some(&b.name), &b.value);
                }
            }
            continue;
        }
        let Some(defined) = graph.defined(&path) else {
            continue;
        };
        for (name, value) in &defined.defaults {
            push(&path, Some(name), value);
        }
        push(&path, None, &defined.body);
    }
    Written { numbers }
}
