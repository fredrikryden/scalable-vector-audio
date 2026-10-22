// Concern: writes an Expr back out as source text | Non-concern: parsing it (parser.rs), what a node's text names (sva-engine) | IO: (&Expr) -> String

use crate::expr::{Address, Arg, BinOp, Expr, INDEX, JOIN, Literal, PWL, PWL_INTEGRAL};

/// 2 for a literal, 1 for a product holding one, else 0.
fn timed_factor(e: &Expr) -> u8 {
    match e {
        Expr::Lit(Literal::Num(_)) => 2,
        Expr::Bin(BinOp::Mul | BinOp::Div | BinOp::Mod, l, r) => {
            u8::from(timed_factor(l).max(timed_factor(r)) > 0)
        }
        _ => 0,
    }
}

/// Parenthesizes by precedence, so a printed argument stays short enough to read.
pub fn render(e: &Expr) -> String {
    write(e, 0, false)
}

fn op_text(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => " + ",
        BinOp::Sub => " - ",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => " % ",
    }
}

fn precedence(op: BinOp) -> u8 {
    match op {
        BinOp::Add | BinOp::Sub => 1,
        BinOp::Mul | BinOp::Div | BinOp::Mod => 2,
    }
}

/// `time` marks a slot the parser reads back as a duration, where a bare number refuses. It
/// reaches through `+` and `-` and through a `join`, and stops where the parser's check does:
/// an index is a count and never takes it.
fn write(e: &Expr, outer: u8, time: bool) -> String {
    match e {
        Expr::Lit(Literal::Num(n)) if time => format!("{n}s"),
        Expr::Lit(Literal::Num(n)) => format!("{n}"),
        Expr::Lit(Literal::Bars(n)) => format!("{n}b"),
        Expr::Lit(Literal::Samples(n)) => format!("{n}sp"),
        Expr::Lit(Literal::Str(s)) => s.clone(),
        Expr::Lit(Literal::Path(path)) => format!("\"{path}\""),
        Expr::Lit(Literal::Recorded(recording)) => format!("\"{}\"", recording.path),
        Expr::Var(name) => name.clone(),
        Expr::Bin(op, l, r) => {
            let p = precedence(*op);
            let (tl, tr) = match op {
                BinOp::Add | BinOp::Sub | BinOp::Mod => (time, time),
                BinOp::Mul => {
                    let right = timed_factor(r) >= timed_factor(l);
                    (time && !right, time && right)
                }
                BinOp::Div => (time, false),
            };
            let body = format!("{}{}{}", write(l, p, tl), op_text(*op), write(r, p + 1, tr));
            if p < outer { format!("({body})") } else { body }
        }
        Expr::Call { name, args, .. } => format!("{name}({})", args_text(name, args, time)),
        Expr::Ref {
            path,
            arg,
            binds,
            address,
            ..
        } => {
            let mut inner = write(arg, 0, *address == Address::Time);
            for (k, v) in binds {
                inner.push_str(&format!(", {k}={}", write(v, 0, false)));
            }
            match (&**arg, binds.is_empty(), address) {
                (Expr::Var(name), true, Address::Time) if name == "t" => format!("@{path}"),
                (.., Address::Time) => format!("@{path}({inner})"),
                (.., Address::Index) => format!("@{path}[{inner}]"),
            }
        }
        Expr::SelfRef {
            arg,
            address: Address::Time,
            ..
        } => format!("self({})", write(arg, 0, true)),
        Expr::SelfRef {
            arg,
            address: Address::Index,
            ..
        } => format!("self[{}]", write(arg, 0, false)),
        Expr::Indexed { name, arg, .. } => format!("{name}[{}]", write(arg, 0, false)),
    }
}

fn args_text(name: &str, args: &[Arg], time: bool) -> String {
    let joined = time && name == JOIN;
    let mut at = 0usize;
    args.iter()
        .map(|a| match a {
            Arg::Pos(e) => {
                at += 1;
                let timed = match name {
                    "crop" => at == 2 || at == 3,
                    INDEX => at == 1,
                    PWL | PWL_INTEGRAL => at == 1 || at.is_multiple_of(2),
                    _ => false,
                };
                write(e, 0, joined || timed)
            }
            Arg::Named(k, e) => format!(
                "{k}={}",
                write(
                    e,
                    0,
                    name == "crop" && matches!(k.as_str(), "start" | "end" | "rise" | "fall")
                )
            ),
        })
        .collect::<Vec<_>>()
        .join(", ")
}
