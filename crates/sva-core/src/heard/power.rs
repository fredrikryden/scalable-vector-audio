// Concern: a composition's expected power per band and frame, off its closed forms, per score line on the tape | Non-concern: comparing it, which numbers vary | IO: (Model, line, scope) -> Played

use std::collections::BTreeMap;
use std::sync::Arc;

use sva_ast::{Arg, BinOp, Expr, Graph, Literal, resolve_ref_path};
use sva_formula::filter::Shape;

use super::ear::Ear;
use super::tape::{Arr, Conv, Pad, Tape, V};
use crate::cli_error::CliError;

/// The builtins the lowering reads; any other call in a played node refuses naming it.
pub const LOWERED: &str = "crop, sum, sin, cos, exp, log, pow, sqrt, min, max, step, pwl, \
     sample, noise, lowpass, highpass, bandpass";

/// A line of white `noise(seed)` holds this power a hertz.
const NOISE_DENSITY: f64 = 0.5;
const SUPPORT_MARGIN_SECS: f64 = 0.01;

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
        "heard.unlowered",
        format!(
            "{what}; the expected-power lowering reads {LOWERED}, products of a sound by \
             factors free of sounds, and one score the root places"
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

/// `d e / d t`, symbolically.
fn ddt(e: &Expr) -> Result<Expr, CliError> {
    let num = |n: f64| Expr::Lit(Literal::Num(n));
    let bin = |op, a: Expr, b: Expr| Expr::Bin(op, Box::new(a), Box::new(b));
    let zero = |x: &Expr| matches!(x, Expr::Lit(Literal::Num(n)) if *n == 0.0);
    let mul = |a: Expr, b: Expr| {
        if zero(&a) || zero(&b) {
            num(0.0)
        } else {
            bin(BinOp::Mul, a, b)
        }
    };
    let add = |a: Expr, b: Expr| match (zero(&a), zero(&b)) {
        (true, _) => b,
        (_, true) => a,
        _ => bin(BinOp::Add, a, b),
    };
    let call = |name: &str, args: Vec<Expr>| Expr::Call {
        name: name.to_string(),
        args: args.into_iter().map(Arg::Pos).collect(),
        span: sva_ast::ByteSpan::new(0, 0),
    };
    Ok(match e {
        Expr::Lit(_) => num(0.0),
        Expr::Var(v) => num(if v == "t" { 1.0 } else { 0.0 }),
        Expr::Bin(op, a, b) => {
            let (da, db) = (ddt(a)?, ddt(b)?);
            match op {
                BinOp::Add => add(da, db),
                BinOp::Sub => match zero(&db) {
                    true => da,
                    false => bin(BinOp::Sub, da, db),
                },
                BinOp::Mul => add(mul(da, (**b).clone()), mul((**a).clone(), db)),
                BinOp::Div => {
                    let first = match zero(&da) {
                        true => num(0.0),
                        false => bin(BinOp::Div, da, (**b).clone()),
                    };
                    let second = match zero(&db) {
                        true => num(0.0),
                        false => bin(
                            BinOp::Div,
                            mul((**a).clone(), db),
                            mul((**b).clone(), (**b).clone()),
                        ),
                    };
                    match zero(&second) {
                        true => first,
                        false => bin(BinOp::Sub, first, second),
                    }
                }
                BinOp::Mod => return Err(unsupported("a phase reads `%`".to_string())),
            }
        }
        Expr::Call { name, args, .. } => {
            let x = match args.as_slice() {
                [Arg::Pos(x)] => x,
                [Arg::Pos(x), Arg::Pos(p)] if name == "pow" => {
                    let dx = ddt(x)?;
                    if !zero(&ddt(p)?) {
                        return Err(unsupported("a phase raises to a power moving in t".into()));
                    }
                    let lower = bin(BinOp::Sub, p.clone(), num(1.0));
                    return Ok(mul(mul(p.clone(), call("pow", vec![x.clone(), lower])), dx));
                }
                _ => return Err(unsupported(format!("a phase calls `{name}`"))),
            };
            let dx = ddt(x)?;
            if zero(&dx) {
                return Ok(num(0.0));
            }
            match name.as_str() {
                "exp" => mul(e.clone(), dx),
                "sin" => mul(call("cos", vec![x.clone()]), dx),
                "cos" => mul(bin(BinOp::Sub, num(0.0), call("sin", vec![x.clone()])), dx),
                "log" => bin(BinOp::Div, dx, x.clone()),
                "sqrt" => bin(BinOp::Div, dx, mul(num(2.0), e.clone())),
                _ => return Err(unsupported(format!("a phase calls `{name}`"))),
            }
        }
        _ => return Err(unsupported("a phase reads a ref".to_string())),
    })
}

#[derive(Clone)]
struct Filter {
    shape: Shape,
    cutoff: V,
    q: V,
}

enum Comp {
    /// `sqrt(amp2) sin(...)` at `hz`, `amp2` the product of its terms; `hz` `[k, 1]` or `[k, t]`.
    Lines {
        amp2: Vec<(V, bool)>,
        hz: V,
        filters: Vec<Filter>,
    },
    /// `sqrt(amp2)` times white noise of `density` a hertz; `amp2` over `t` alone.
    Noise {
        amp2: Vec<(V, bool)>,
        density: V,
        filters: Vec<Filter>,
    },
}

pub type Scope = BTreeMap<String, V>;

/// The ms grid one line's support is read on, and the frames it touches.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    /// Global ms index of the first ms.
    pub g0: i64,
    pub ms: usize,
    pub m_lo: usize,
    pub frames: usize,
}

/// What one score line plays, as the distance reads it.
pub struct Played {
    pub grid: Grid,
    /// Power `[48, frames]` from frame `m_lo`.
    pub power: V,
    /// Per line of `sum`, its frequency at its last frame and its power `[lines, frames]`.
    pub line_hz: Vec<f64>,
    pub line_power: Option<V>,
    /// Power per ms from `g0`, `[1, ms]`.
    pub power_ms: V,
}

struct Lowering<'a> {
    model: &'a Model,
    line: usize,
    line_scope: &'a Scope,
    grid: Option<Grid>,
    t: Option<V>,
}

fn named<'e>(args: &'e [Arg], name: &str, pos: usize) -> Option<&'e Expr> {
    args.iter()
        .find_map(|a| match a {
            Arg::Named(k, x) if k == name => Some(x),
            _ => None,
        })
        .or_else(|| {
            args.iter()
                .filter_map(|a| match a {
                    Arg::Pos(x) => Some(x),
                    Arg::Named(..) => None,
                })
                .nth(pos)
        })
}

fn positional(args: &[Arg]) -> Vec<&Expr> {
    args.iter()
        .filter_map(|a| match a {
            Arg::Pos(x) => Some(x),
            Arg::Named(..) => None,
        })
        .collect()
}

/// Whether `e` sounds: an oscillator in t, noise, or a ref placing sound.
fn sounds(e: &Expr) -> bool {
    match e {
        Expr::Call { name, args, .. } => {
            let here = match name.as_str() {
                "noise" => true,
                "sin" | "cos" => positional(args).first().is_some_and(|x| reads_t(x)),
                _ => false,
            };
            here || positional(args).iter().any(|x| sounds(x))
        }
        Expr::Ref { .. } => true,
        Expr::Bin(_, a, b) => sounds(a) || sounds(b),
        _ => false,
    }
}

fn reads_t(e: &Expr) -> bool {
    match e {
        Expr::Var(v) => v == "t",
        _ => sva_ast::children(e, sva_ast::Binds::Keep)
            .into_iter()
            .any(reads_t),
    }
}

/// `|H(e^(i w))|^2` of the sampled RBJ biquad, at every frequency `hz` holds.
fn response(tape: &mut Tape, rate: f64, f: &Filter, hz: V) -> Result<V, CliError> {
    let tau = std::f64::consts::TAU;
    let hi = 0.45 * rate;
    let cutoff = tape.clamp(f.cutoff, hi.min(1.0), hi);
    let w0 = tape.scale(cutoff, tau / rate);
    let c = tape.cos(w0);
    let s = tape.sin(w0);
    let q2 = tape.scale(f.q, 2.0);
    let al = tape.div(s, q2);
    let one = tape.num(1.0);
    let omc = tape.sub(one, c);
    let opc = tape.add(one, c);
    let (b0, b1, b2) = match f.shape {
        Shape::Lowpass => {
            let h = tape.scale(omc, 0.5);
            (h, omc, h)
        }
        Shape::Highpass => {
            let h = tape.scale(opc, 0.5);
            let n = tape.neg(opc);
            (h, n, h)
        }
        Shape::Bandpass => {
            let z = tape.num(0.0);
            let n = tape.neg(al);
            (al, z, n)
        }
        other => {
            return Err(unsupported(format!(
                "a `{}` filter",
                sva_engine::shape_name(other)
            )));
        }
    };
    let a0 = tape.add(one, al);
    let a1 = tape.scale(c, -2.0);
    let a2 = tape.sub(one, al);
    let w = tape.scale(hz, tau / rate);
    let cw = tape.cos(w);
    let w2 = tape.scale(w, 2.0);
    let c2w = tape.cos(w2);
    let sw = tape.sin(w);
    let s2w = tape.sin(w2);
    let part = |tape: &mut Tape, x0: V, x1: V, x2: V| {
        let r1 = tape.mul(x1, cw);
        let r2 = tape.mul(x2, c2w);
        let re = tape.add(x0, r1);
        let re = tape.add(re, r2);
        let i1 = tape.mul(x1, sw);
        let i2 = tape.mul(x2, s2w);
        let im = tape.add(i1, i2);
        let re2 = tape.square(re);
        let im2 = tape.square(im);
        tape.add(re2, im2)
    };
    let num = part(tape, b0, b1, b2);
    let den = part(tape, a0, a1, a2);
    Ok(tape.div(num, den))
}

impl Lowering<'_> {
    fn scalar_of(&self, tape: &mut Tape, e: &Expr, scope: &Scope) -> Result<V, CliError> {
        if reads_t(e) {
            return Err(unsupported(
                "a value read as a constant moves in t".to_string(),
            ));
        }
        self.factor(tape, e, scope)
    }

    /// Each multiplicand of `e` on the axes it reads, flagged where it divides.
    fn terms_of(
        &self,
        tape: &mut Tape,
        e: &Expr,
        inverse: bool,
        scope: &Scope,
    ) -> Result<Vec<(V, bool)>, CliError> {
        let mut ts = Vec::new();
        terms(e, inverse, &mut ts);
        ts.iter()
            .map(|(x, inv)| Ok((self.factor(tape, x, scope)?, *inv)))
            .collect()
    }

    /// `e` free of sounds, on the axes it reads.
    fn factor(&self, tape: &mut Tape, e: &Expr, scope: &Scope) -> Result<V, CliError> {
        Ok(match e {
            Expr::Lit(Literal::Num(n)) => tape.num(*n),
            Expr::Lit(other) => return Err(unsupported(format!("the literal {other:?}"))),
            Expr::Var(v) if v == "pi" => tape.num(std::f64::consts::PI),
            Expr::Var(v) => match scope.get(v) {
                Some(x) => *x,
                None if v == "t" => self
                    .t
                    .ok_or_else(|| unsupported("t above the score".into()))?,
                None => return Err(unsupported(format!("`{v}`, which nothing binds"))),
            },
            Expr::Bin(BinOp::Mul | BinOp::Div, ..) => {
                let ts = self.terms_of(tape, e, false, scope)?;
                tape.product(&ts)
            }
            Expr::Bin(BinOp::Sub, a, b) if constant(a) == Some(0.0) => {
                let y = self.factor(tape, b, scope)?;
                tape.neg(y)
            }
            Expr::Bin(op, a, b) => {
                let (x, y) = (self.factor(tape, a, scope)?, self.factor(tape, b, scope)?);
                match op {
                    BinOp::Add => tape.add(x, y),
                    BinOp::Sub => tape.sub(x, y),
                    _ => return Err(unsupported("`%`".to_string())),
                }
            }
            Expr::Call { name, args, .. } => self.call_factor(tape, name, args, scope)?,
            _ => return Err(unsupported("a ref inside a factor".to_string())),
        })
    }

    fn call_factor(
        &self,
        tape: &mut Tape,
        name: &str,
        args: &[Arg],
        scope: &Scope,
    ) -> Result<V, CliError> {
        let pos = positional(args);
        let one = |tape: &mut Tape, s: &Self| -> Result<V, CliError> {
            let x = pos
                .first()
                .ok_or_else(|| unsupported(format!("`{name}()`")))?;
            s.factor(tape, x, scope)
        };
        Ok(match name {
            "sin" => {
                let x = one(tape, self)?;
                tape.sin(x)
            }
            "cos" => {
                let x = one(tape, self)?;
                tape.cos(x)
            }
            "exp" => {
                let x = one(tape, self)?;
                tape.exp(x)
            }
            "log" => {
                let x = one(tape, self)?;
                tape.ln(x)
            }
            "sqrt" => {
                let x = one(tape, self)?;
                tape.sqrt(x)
            }
            "step" => {
                let x = one(tape, self)?;
                let v = tape.val(x);
                let held = Arr::new(
                    v.r,
                    v.c,
                    v.v.iter().map(|y| f64::from(u8::from(*y >= 0.0))).collect(),
                );
                tape.constant(held)
            }
            "pow" | "min" | "max" => {
                let [a, b] = pos.as_slice() else {
                    return Err(unsupported(format!("`{name}` of {} values", pos.len())));
                };
                let (x, y) = (self.factor(tape, a, scope)?, self.factor(tape, b, scope)?);
                match name {
                    "pow" => tape.pow(x, y),
                    "min" => tape.min(x, y),
                    _ => tape.max(x, y),
                }
            }
            "pwl" => {
                let x = self.factor(tape, pos[0], scope)?;
                let mut xs = Vec::new();
                let mut vals = Vec::new();
                for pair in pos[1..].chunks(2) {
                    let [k, v] = pair else {
                        return Err(unsupported("a `pwl` knot without a value".to_string()));
                    };
                    xs.push(
                        constant(k)
                            .ok_or_else(|| unsupported("a `pwl` knot at no constant".into()))?,
                    );
                    vals.push(self.scalar_of(tape, v, scope)?);
                }
                tape.pwl(x, Arc::new(xs), vals)
            }
            "crop" => {
                let x = self.factor(tape, pos[0], scope)?;
                let w = self.window(tape, args, scope)?;
                tape.mul(x, w)
            }
            other => return Err(unsupported(format!("`{other}`"))),
        })
    }

    /// `crop`'s window on the line's grid.
    fn window(&self, tape: &mut Tape, args: &[Arg], scope: &Scope) -> Result<V, CliError> {
        let t = self
            .t
            .ok_or_else(|| unsupported("a crop above the score".into()))?;
        let mut side = |name: &str, pos: usize| -> Result<Option<V>, CliError> {
            named(args, name, pos)
                .map(|e| self.scalar_of(tape, e, scope))
                .transpose()
        };
        let a = side("start", 1)?.ok_or_else(|| unsupported("a crop without a start".into()))?;
        let b = side("end", 2)?.ok_or_else(|| unsupported("a crop without an end".into()))?;
        let rise = side("rise", usize::MAX)?;
        let fall = side("fall", usize::MAX)?;
        let tv = tape.val(t).v.clone();
        let (av, bv) = (tape.scalar_of(a), tape.scalar_of(b));
        let inside = tape.constant(Arr::row(
            tv.iter()
                .map(|x| f64::from(u8::from(*x >= av && *x < bv)))
                .collect(),
        ));
        let mut w = inside;
        let pi = std::f64::consts::PI;
        if let Some(r) = rise {
            let r = tape.clamp(r, 1e-6, f64::INFINITY);
            let u = tape.sub(t, a);
            let u = tape.div(u, r);
            let u = tape.clamp(u, 0.0, 1.0);
            let c = tape.scale(u, pi);
            let c = tape.cos(c);
            let c = tape.scale(c, -0.5);
            let c = tape.offset(c, 0.5);
            w = tape.mul(w, c);
        }
        if let Some(f) = fall {
            let f = tape.clamp(f, 1e-6, f64::INFINITY);
            let start = tape.sub(b, f);
            let u = tape.sub(t, start);
            let u = tape.div(u, f);
            let u = tape.clamp(u, 0.0, 1.0);
            let c = tape.scale(u, pi);
            let c = tape.cos(c);
            let c = tape.scale(c, 0.5);
            let c = tape.offset(c, 0.5);
            w = tape.mul(w, c);
        }
        Ok(w)
    }

    fn sound(&mut self, tape: &mut Tape, e: &Expr, scope: &Scope) -> Result<Vec<Comp>, CliError> {
        match e {
            Expr::Bin(op, a, b) => {
                let (sa, sb) = (sounds(a), sounds(b));
                match (op, sa, sb) {
                    (BinOp::Add | BinOp::Sub, true, true) => {
                        let mut x = self.sound(tape, a, scope)?;
                        x.extend(self.sound(tape, b, scope)?);
                        Ok(x)
                    }
                    (BinOp::Sub, false, true) if constant(a) == Some(0.0) => {
                        self.sound(tape, b, scope)
                    }
                    (BinOp::Mul, true, false) | (BinOp::Mul, false, true) => {
                        let (s, f) = if sa { (a, b) } else { (b, a) };
                        let fs = self.terms_of(tape, f, false, scope)?;
                        let comps = self.sound(tape, s, scope)?;
                        Ok(scaled(tape, comps, &fs))
                    }
                    (BinOp::Div, true, false) => {
                        let fs = self.terms_of(tape, b, true, scope)?;
                        let comps = self.sound(tape, a, scope)?;
                        Ok(scaled(tape, comps, &fs))
                    }
                    _ => Err(unsupported(
                        "a sound meets a sound other than by `+` or `-`".to_string(),
                    )),
                }
            }
            Expr::Call { name, args, .. } => self.call_sound(tape, name, args, scope),
            Expr::Ref { path, .. } => {
                let resolved = resolve_ref_path(&self.model.root, path);
                match resolved.is_some() && resolved == self.model.score {
                    true => self.placed(tape),
                    false => Err(unsupported(format!("the ref `@{path}` beside the score"))),
                }
            }
            _ => Err(unsupported("a sound of no known form".to_string())),
        }
    }

    fn placed(&mut self, tape: &mut Tape) -> Result<Vec<Comp>, CliError> {
        let line = &self.model.lines[self.line];
        let node = &self.model.nodes[&line.node];
        let grid = self
            .grid
            .expect("the grid is laid before the score is read");
        let at = line.at;
        let t: Vec<f64> = (0..grid.ms)
            .map(|j| (grid.g0 as f64 + j as f64 + 0.5) / 1000.0 - at)
            .collect();
        self.t = Some(tape.constant(Arr::row(t)));
        let scope = self.line_scope.clone();
        let out = self.sound(tape, &node.body, &scope);
        self.t = None;
        out
    }

    fn call_sound(
        &mut self,
        tape: &mut Tape,
        name: &str,
        args: &[Arg],
        scope: &Scope,
    ) -> Result<Vec<Comp>, CliError> {
        let pos = positional(args);
        match name {
            "sample" => self.sound(tape, pos[0], scope),
            "crop" => {
                let comps = self.sound(tape, pos[0], scope)?;
                let w = self.window(tape, args, scope)?;
                Ok(scaled(tape, comps, &[(w, false)]))
            }
            "sin" | "cos" => {
                let phase = pos[0];
                let d = ddt(phase)?;
                let hz = self.factor(tape, &d, scope)?;
                let hz = tape.scale(hz, 1.0 / std::f64::consts::TAU);
                Ok(vec![Comp::Lines {
                    amp2: Vec::new(),
                    hz,
                    filters: Vec::new(),
                }])
            }
            "noise" => {
                if let Some(c) = named(args, "color", usize::MAX)
                    && constant(c) != Some(1.0)
                {
                    return Err(unsupported("coloured `noise`".to_string()));
                }
                let density = tape.num(NOISE_DENSITY);
                Ok(vec![Comp::Noise {
                    amp2: Vec::new(),
                    density,
                    filters: Vec::new(),
                }])
            }
            "sum" => {
                let [Expr::Var(k), lo, hi, body] = pos.as_slice() else {
                    return Err(unsupported("`sum` of no index".to_string()));
                };
                let (lo, hi) = (
                    constant(lo).ok_or_else(|| unsupported("a `sum` from no constant".into()))?,
                    constant(hi).ok_or_else(|| unsupported("a `sum` to no constant".into()))?,
                );
                let ks: Vec<f64> = (lo as i64..=hi as i64).map(|x| x as f64).collect();
                let mut inner = scope.clone();
                inner.insert(k.clone(), tape.constant(Arr::column(ks)));
                self.sound(tape, body, &inner)
            }
            other => match sva_engine::shape(other) {
                Some(shape) => {
                    let mut comps = self.sound(tape, pos[0], scope)?;
                    let cutoff = named(args, "cutoff", 1)
                        .ok_or_else(|| unsupported(format!("`{other}` with no cutoff")))?;
                    let cutoff = self.scalar_of(tape, cutoff, scope)?;
                    let q = match named(args, "q", 2) {
                        Some(q) => self.scalar_of(tape, q, scope)?,
                        None => tape.num(shape.default_q()),
                    };
                    let f = Filter { shape, cutoff, q };
                    for c in &mut comps {
                        match c {
                            Comp::Lines { filters, .. } | Comp::Noise { filters, .. } => {
                                filters.push(f.clone())
                            }
                        }
                    }
                    Ok(comps)
                }
                None => Err(unsupported(format!("`{other}` as a sound"))),
            },
        }
    }

    /// Where the line's sound can be heard: its node's outermost crop, else to the clip's end.
    fn support(&self, tape: &mut Tape, e: &Expr, scope: &Scope) -> Result<(f64, f64), CliError> {
        if let Expr::Call { name, args, .. } = e
            && name == "crop"
        {
            let (a, b) = (named(args, "start", 1), named(args, "end", 2));
            if let (Some(a), Some(b)) = (a, b) {
                let (a, b) = (
                    self.scalar_of(tape, a, scope)?,
                    self.scalar_of(tape, b, scope)?,
                );
                return Ok((tape.scalar_of(a), tape.scalar_of(b)));
            }
        }
        Ok((0.0, f64::INFINITY))
    }
}

/// Each component's power times the square of every term, divided where flagged.
fn scaled(tape: &mut Tape, comps: Vec<Comp>, terms: &[(V, bool)]) -> Vec<Comp> {
    let squared: Vec<(V, bool)> = terms
        .iter()
        .map(|(v, inv)| (tape.square(*v), *inv))
        .collect();
    comps
        .into_iter()
        .map(|c| match c {
            Comp::Lines {
                mut amp2,
                hz,
                filters,
            } => {
                amp2.extend_from_slice(&squared);
                Comp::Lines { amp2, hz, filters }
            }
            Comp::Noise {
                mut amp2,
                density,
                filters,
            } => {
                amp2.extend_from_slice(&squared);
                Comp::Noise {
                    amp2,
                    density,
                    filters,
                }
            }
        })
        .collect()
}

/// The multiplicands of a product or quotient, each flagged where it divides.
fn terms(e: &Expr, inverse: bool, out: &mut Vec<(Expr, bool)>) {
    match e {
        Expr::Bin(BinOp::Mul, a, b) => {
            terms(a, inverse, out);
            terms(b, inverse, out);
        }
        Expr::Bin(BinOp::Div, a, b) => {
            terms(a, inverse, out);
            terms(b, !inverse, out);
        }
        other => out.push((other.clone(), inverse)),
    }
}

/// `x` with a column each ms of the grid, where it held one value for all.
fn over_grid(tape: &mut Tape, x: V, ms: usize) -> V {
    match tape.val(x).c == ms {
        true => x,
        false => {
            let ones = tape.constant(Arr::row(vec![1.0; ms]));
            tape.mul(x, ones)
        }
    }
}

/// The grid of ms and frames a support `[a, b)` seconds touches, inside `frames` frames.
pub fn grid_of(ear: &Ear, a: f64, b: f64, frames: usize) -> Option<Grid> {
    let hop = ear.hop_ms as i64;
    let first_ms = libm::floor(a * 1000.0) as i64;
    let m_lo = ((first_ms - hop + hop - 1).div_euclid(hop)).max(0) as usize;
    let end_ms = libm::ceil((b + SUPPORT_MARGIN_SECS) * 1000.0).min(1e12) as i64;
    let m_hi = ((end_ms + hop).div_euclid(hop) as usize + 1).min(frames);
    (m_hi > m_lo).then(|| Grid {
        g0: hop * m_lo as i64 - hop,
        ms: hop as usize * (m_hi - m_lo) + hop as usize,
        m_lo,
        frames: m_hi - m_lo,
    })
}

/// One score line's expected power; none where it sounds inside no frame.
pub fn played(
    tape: &mut Tape,
    model: &Model,
    ear: &Ear,
    line: usize,
    root_scope: &Scope,
    line_scope: &Scope,
    frames: usize,
) -> Result<Option<Played>, CliError> {
    let mut low = Lowering {
        model,
        line,
        line_scope,
        grid: None,
        t: None,
    };
    let l = &model.lines[line];
    let body = &model.nodes[&l.node].body;
    let (a, b) = low.support(tape, body, line_scope)?;
    let Some(grid) = grid_of(ear, l.at + a.max(0.0), l.at + b, frames) else {
        return Ok(None);
    };
    low.grid = Some(grid);
    let root = &model.nodes[&model.root].body;
    let comps = match model.score {
        Some(_) => low.sound(tape, root, root_scope)?,
        None => low.placed(tape)?,
    };
    let hop = ear.hop_ms;
    let average = Arc::new(Conv {
        k: ear.taps.clone(),
        stride: hop,
        off: 0,
        out: grid.frames,
        pad: Pad::Zero,
    });
    let centre = Arc::new(Conv {
        k: vec![1.0],
        stride: hop,
        off: hop as i64,
        out: grid.frames,
        pad: Pad::Edge,
    });
    let rate = ear.rate;
    let mut power: Option<V> = None;
    let mut ms_power: Option<V> = None;
    let mut line_hz = Vec::new();
    let mut line_power: Vec<V> = Vec::new();
    let accumulate = |tape: &mut Tape, held: &mut Option<V>, x: V| {
        *held = Some(match held.take() {
            Some(h) => tape.add(h, x),
            None => x,
        });
    };
    for c in comps {
        match c {
            Comp::Lines { amp2, hz, filters } => {
                let amp2 = tape.product(&amp2);
                let k = tape.val(amp2).r.max(tape.val(hz).r);
                let amp2 = over_grid(tape, amp2, grid.ms);
                let p = tape.conv(Arc::clone(&average), amp2);
                let hz = match tape.val(hz).c {
                    1 => hz,
                    _ => tape.conv(Arc::clone(&centre), hz),
                };
                let mut gain: Option<V> = None;
                for f in &filters {
                    let h = response(tape, rate, f, hz)?;
                    gain = Some(match gain {
                        Some(x) => tape.mul(x, h),
                        None => h,
                    });
                }
                let settled: Vec<f64> = {
                    let hv = tape.val(hz);
                    (0..k)
                        .map(|i| hv.v[i.min(hv.r - 1) * hv.c + hv.c - 1])
                        .collect()
                };
                let open: Vec<f64> = settled
                    .iter()
                    .map(|f| f64::from(u8::from(*f > 0.0 && *f < ear.ceiling_hz)))
                    .collect();
                let first_gain: Vec<f64> = match gain {
                    Some(g) => {
                        let gv = tape.val(g);
                        (0..k).map(|i| gv.v[i.min(gv.r - 1) * gv.c]).collect()
                    }
                    None => vec![1.0; k],
                };
                let mask = tape.constant(Arr::column(open.clone()));
                let p = match gain {
                    Some(g) => tape.mul(p, g),
                    None => p,
                };
                let p = tape.mul(p, mask);
                let hz = match tape.val(hz).r == k {
                    true => hz,
                    false => {
                        let ones = tape.constant(Arr::new(k, 1, vec![1.0; k]));
                        tape.mul(hz, ones)
                    }
                };
                let e = tape.bands(p, hz, Arc::clone(&ear.table));
                accumulate(tape, &mut power, e);
                let w: Vec<f64> = first_gain
                    .iter()
                    .zip(&open)
                    .map(|(g, o)| 0.5 * g * o)
                    .collect();
                let s = match tape.val(amp2).r {
                    1 => tape.scale(amp2, w.iter().sum()),
                    _ => tape.matl(Arc::new(Arr::row(w)), amp2),
                };
                accumulate(tape, &mut ms_power, s);
                line_hz.extend(settled);
                line_power.push(p);
            }
            Comp::Noise {
                amp2,
                density,
                filters,
            } => {
                let amp2 = tape.product(&amp2);
                if tape.val(amp2).r != 1 {
                    return Err(unsupported("noise read per harmonic".to_string()));
                }
                let amp2 = over_grid(tape, amp2, grid.ms);
                let p = tape.conv(Arc::clone(&average), amp2);
                let bins = tape.constant(Arr::column(ear.bin_hz.clone()));
                let open: Vec<f64> = ear
                    .bin_hz
                    .iter()
                    .map(|f| f64::from(u8::from(*f > 0.0 && *f < ear.ceiling_hz)))
                    .collect();
                let mut shape = tape.constant(Arr::column(open));
                for f in &filters {
                    let h = response(tape, rate, f, bins)?;
                    shape = tape.mul(shape, h);
                }
                let per_bin = tape.scale(density, rate / ear.n as f64);
                let shape = tape.mul(shape, per_bin);
                let total = tape.sum(shape);
                let col = tape.matl(Arc::clone(&ear.bin_band), shape);
                let e = tape.mul(col, p);
                accumulate(tape, &mut power, e);
                let s = tape.mul(amp2, total);
                accumulate(tape, &mut ms_power, s);
            }
        }
    }
    let Some(power) = power else {
        return Ok(None);
    };
    let ms_power = ms_power.expect("power came with its ms power");
    let line_power = match line_power.is_empty() {
        true => None,
        false => Some(stack(tape, line_power)),
    };
    Ok(Some(Played {
        grid,
        power,
        line_hz,
        line_power,
        power_ms: ms_power,
    }))
}

/// Rows of several arrays of one column count, one array.
fn stack(tape: &mut Tape, parts: Vec<V>) -> V {
    if parts.len() == 1 {
        return parts[0];
    }
    let total: usize = parts.iter().map(|p| tape.val(*p).r).sum();
    let mut out: Option<V> = None;
    let mut at = 0;
    for p in parts {
        let r = tape.val(p).r;
        let mut place = vec![0.0; total * r];
        for i in 0..r {
            place[(at + i) * r + i] = 1.0;
        }
        let m = Arc::new(Arr::new(total, r, place));
        let y = tape.matl(m, p);
        out = Some(match out {
            Some(h) => tape.add(h, y),
            None => y,
        });
        at += r;
    }
    out.expect("one part at least")
}
