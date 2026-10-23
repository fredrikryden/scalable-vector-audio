// Concern: a state's composition as played: exact text, each line lowered by the render with tangents, its noise rendered | Non-concern: the transform (lines.rs) | IO: (State) -> lowered lines

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use sva_ast::{Arg, Expr, Literal};
use sva_engine::Varied;
use sva_formula::Tangents;

use super::draws::{self, seed};
use super::lines::{self, Filter, Line, Realisation};
use super::state::{Slot, State};
use crate::cli_error::CliError;
use crate::model::{constant, header, relative};
use sva_samples::measure::internal::Grid;

/// The local index a root default's tangent seeds, past every line's own.
pub const ROOT_SEED: u32 = 1 << 30;

fn refused(what: String) -> CliError {
    CliError::usage("solve.prox_played", what)
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

/// A filter as written: its shape, cutoff and q.
#[derive(Clone, Debug)]
struct Written {
    shape: sva_formula::Shape,
    cutoff: Expr,
    q: Option<Expr>,
}

/// `e`'s chain of filters down to what it filters.
fn filters(e: &Expr) -> Option<(Vec<Written>, &Expr)> {
    let Expr::Call { name, args, .. } = e else {
        return None;
    };
    let shape = sva_engine::shape(name)?;
    let inner = named(args, "x", 0)?;
    let cutoff = named(args, "cutoff", 1)?.clone();
    let q = named(args, "q", 2).cloned();
    let (mut chain, source) = filters(inner).unwrap_or((Vec::new(), inner));
    chain.push(Written { shape, cutoff, q });
    Some((chain, source))
}

fn holds_noise(e: &Expr) -> bool {
    match e {
        Expr::Call { name, .. } if name == "noise" => true,
        _ => sva_ast::children(e, sva_ast::Binds::Keep)
            .into_iter()
            .any(holds_noise),
    }
}

/// A node's noise: what its filters read, and the filters.
#[derive(Clone, Debug)]
struct Noise {
    source: Expr,
    filters: Vec<Written>,
}

/// The names `e` reads, `t` aside.
fn names(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Var(v) if v != "t" && !out.contains(v) => out.push(v.clone()),
        _ => sva_ast::children(e, sva_ast::Binds::Keep)
            .into_iter()
            .for_each(|c| names(c, out)),
    }
}

/// `e` with its one filtered noise cut out to `1`, the filters' numbers and the noise's own
/// names (its seed) read at no weight so the node still reads every default it declares.
fn cut(e: &Expr, out: &mut Option<Noise>) -> Expr {
    if let Some((chain, source)) = filters(e)
        && holds_noise(source)
        && out.is_none()
    {
        let num = |x: f64| Expr::Lit(Literal::Num(x));
        let bin = |op, a: Expr, b: Expr| Expr::Bin(op, Box::new(a), Box::new(b));
        let read = chain
            .iter()
            .flat_map(|w| std::iter::once(w.cutoff.clone()).chain(w.q.clone()))
            .fold(num(0.0), |acc, x| bin(sva_ast::BinOp::Add, acc, x));
        let mut own = Vec::new();
        names(source, &mut own);
        let read = own
            .into_iter()
            .fold(read, |acc, v| bin(sva_ast::BinOp::Add, acc, Expr::Var(v)));
        *out = Some(Noise {
            source: source.clone(),
            filters: chain,
        });
        return bin(
            sva_ast::BinOp::Add,
            bin(sva_ast::BinOp::Mul, num(0.0), read),
            num(1.0),
        );
    }
    let mapped = sva_ast::map_children(e, sva_ast::Binds::Keep, |c| {
        Ok::<_, std::convert::Infallible>(cut(c, out))
    });
    let Ok(x) = mapped;
    x
}

/// How far the played value moves a unit of its coordinate: none where held at a bound.
pub fn slope(s: &State, slot: Slot) -> f64 {
    match slot {
        Slot::Fixed(_) => 0.0,
        _ => {
            let var = s.var_of_slot(slot).expect("a varied slot");
            let c = s.coord_of(slot).expect("a coordinate");
            match c >= var.lo && c <= var.hi {
                true if var.log => var.value(c),
                true => 1.0,
                false => 0.0,
            }
        }
    }
}

fn exact(s: &State, slot: Slot) -> String {
    let v = s.value(slot);
    let unit = s.var_of_slot(slot).map_or("", |var| var.unit);
    match unit {
        "" => format!("{v}"),
        u => format!("\"{v}{u}\""),
    }
}

/// The composition the state plays, every line and number as it stands, each node's noise cut
/// out; and each noisy node's noise.
struct Text {
    files: Vec<(String, String)>,
    noise: BTreeMap<String, Noise>,
}

fn text(s: &State, source: &dyn Fn(&str) -> Option<String>) -> Result<Text, CliError> {
    let mut files = Vec::new();
    let mut noise = BTreeMap::new();
    for (n, path) in s.nodes.iter().enumerate() {
        if *path != s.model.root && !(0..s.model.lines.len()).any(|l| s.line_node(l) == n) {
            continue;
        }
        let t = source(path).ok_or_else(|| CliError::NotFound(format!("no node `{path}`")))?;
        let body = &s.model.nodes[path].body;
        let mut held = None;
        let cut_body = match *path == s.model.root {
            true => body.clone(),
            false => cut(body, &mut held),
        };
        let lines: Vec<&str> = t.lines().collect();
        let last = lines
            .iter()
            .rposition(|l| !l.trim().is_empty() && !l.trim_start().starts_with(';'));
        let mut out = String::new();
        for (i, line) in lines.iter().enumerate() {
            if Some(i) == last && held.is_some() {
                out.push_str(&sva_ast::render_expr(&cut_body));
                out.push('\n');
                continue;
            }
            match header(line)
                .and_then(|name| s.centre.get(&(n, name.to_string())).map(|c| (name, c)))
            {
                Some((name, c)) => out.push_str(&format!(
                    "{name} = {}\n",
                    exact(s, Slot::Own(*c)).trim_matches('"')
                )),
                None => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        if let Some(h) = held {
            noise.insert(path.clone(), h);
        }
        files.push((path.clone(), out));
    }
    if let Some(score) = &s.model.score {
        let file = format!("{score}.jsonl");
        let t = source(&file).ok_or_else(|| CliError::NotFound(format!("no score `{file}`")))?;
        let mut out = String::new();
        for (i, row) in t.lines().enumerate() {
            let Some(line) = (0..s.model.lines.len()).find(|l| s.model.lines[*l].number == i + 1)
            else {
                out.push_str(row);
                out.push('\n');
                continue;
            };
            let mut written: serde_json::Map<String, serde_json::Value> = serde_json::from_str(row)
                .map_err(|e| refused(format!("score line {}: {e}", i + 1)))?;
            written.insert(
                "node".to_string(),
                serde_json::Value::String(format!(
                    "@{}",
                    relative(score, &s.model.lines[line].node)
                )),
            );
            for (name, slot) in s.slots(line) {
                if matches!(slot, Slot::Fixed(_)) {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(&exact(s, slot))
                    .map_err(|e| refused(format!("score line {}: {e}", i + 1)))?;
                written.insert(name, v);
            }
            out.push_str(&serde_json::Value::Object(written).to_string());
            out.push('\n');
        }
        files.push((file, out));
    }
    Ok(Text { files, noise })
}

type Rendered = BTreeMap<(String, u64), Arc<Realisation>>;

/// Each noise rendered once at the clip's samples, by its text, seed and onset. No number a fit
/// varies moves an onset or a noise's text, so it holds one render per noisy line and seed.
#[derive(Default)]
pub struct Renders(Mutex<Rendered>);

fn rendered(
    renders: &Renders,
    source: &Expr,
    seed: Option<u64>,
    at: f64,
    g: &Grid,
) -> Result<Arc<Realisation>, CliError> {
    let text = match seed {
        Some(n) => format!("seed = {n}\n{}", sva_ast::render_expr(source)),
        None => sva_ast::render_expr(source),
    };
    let key = (text.clone(), at.to_bits());
    if let Some(x) = renders.0.lock().expect("the noise renders").get(&key) {
        return Ok(Arc::clone(x));
    }
    let mut c = sva_ast::Composition::new();
    c.insert("noise-source".to_string(), format!("{text}\n"));
    let placed = match at == 0.0 {
        true => "@noise-source\n".to_string(),
        false => format!("@noise-source(t - {at}s)\n"),
    };
    c.insert("noise-placed".to_string(), placed);
    let graph = sva_ast::load(&c).map_err(CliError::Refusals)?;
    let rate = g.sr as u32;
    let held = sva_engine::render(
        &graph,
        "noise-placed",
        sva_engine::RenderConfig::seconds(rate, g.len as f64 / g.sr),
        &sva_engine::Tier::default(),
    )
    .map_err(CliError::Engine)?;
    let id = held.id("noise-placed").expect("the root it rendered");
    let mut x = held
        .output(id)
        .expect("a rendered buffer")
        .plane(0)
        .to_vec();
    x.resize(g.len, 0.0);
    let x = Arc::new(Realisation::of(x, g));
    renders
        .0
        .lock()
        .expect("the noise renders")
        .insert(key, Arc::clone(&x));
    Ok(x)
}

/// A name's played value, and the local index its tangent seeds where it varies.
type Names<'a> = dyn Fn(&str) -> Option<(f64, Option<u32>)> + 'a;

/// A number a filter reads: a literal, or a name the line or root binds.
fn number(e: &Expr, names: &Names) -> Result<(f64, Option<u32>), CliError> {
    if let Some(v) = constant(e) {
        return Ok((v, None));
    }
    match e {
        Expr::Var(v) => {
            names(v).ok_or_else(|| refused(format!("a filter reads `{v}`, which nothing binds")))
        }
        other => Err(refused(format!(
            "a filter reads `{}`; a priced fit reads a filter's cutoff and q as a number or a name",
            sva_ast::render_expr(other)
        ))),
    }
}

fn chain(written: &[Written], names: &Names, sr: f64) -> Result<Vec<Filter>, CliError> {
    written
        .iter()
        .map(|w| {
            let (cutoff, seed) = number(&w.cutoff, names)?;
            let q = match &w.q {
                Some(q) => number(q, names)?.0,
                None => w.shape.default_q(),
            };
            Ok(Filter::new(w.shape, cutoff, q, seed, sr))
        })
        .collect()
}

/// What one evaluation reads: each line lowered and read, the master's chain.
pub struct Lowered {
    pub lines: Vec<Lined>,
    pub master: Vec<Filter>,
}

/// One score line read, and where its local indices land.
pub struct Lined {
    pub line: Line,
    pub slots: Vec<Slot>,
}

/// The lines of the played composition's score, in written order: each instance and onset.
fn placed(inst: &sva_engine::instantiate::Instances, score: &str) -> Vec<(String, f64)> {
    use sva_engine::instantiate::Node;
    fn walk(
        inst: &sva_engine::instantiate::Instances,
        e: &Expr,
        cx: sva_engine::instantiate::Cx,
        out: &mut Vec<(String, f64)>,
    ) {
        match inst.node(e, cx) {
            Node::Read { path, arg, .. } => {
                let at = match arg {
                    Expr::Bin(sva_ast::BinOp::Sub, _, d) => constant(d).unwrap_or(0.0),
                    Expr::Bin(sva_ast::BinOp::Add, _, d) => -constant(d).unwrap_or(0.0),
                    _ => 0.0,
                };
                out.push((path.to_string(), at));
            }
            Node::Bin(_, l, r) => {
                walk(inst, l, cx, out);
                walk(inst, r, cx, out);
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    if let Some((e, cx)) = inst.at(score) {
        walk(inst, e, cx, &mut out);
    }
    out
}

/// Every line of the state's composition lowered with tangents and read at its frames.
pub fn lowered(
    s: &State,
    source: &dyn Fn(&str) -> Option<String>,
    g: &Grid,
    renders: &Renders,
    draws: usize,
    threads: std::num::NonZeroUsize,
) -> Result<Lowered, CliError> {
    let text = text(s, source)?;
    let mut c = sva_ast::Composition::new();
    for (p, t) in &text.files {
        c.insert(p.clone(), t.clone());
    }
    let mut graph = sva_ast::load(&c).map_err(CliError::Refusals)?;
    let rate = g.sr as u32;
    let root = s.model.root.clone();
    crate::sung(&mut graph, &root, rate)?;
    let inst =
        sva_engine::instantiate::instantiate(&graph, &root, rate).map_err(CliError::Engine)?;
    let at: Vec<(String, f64)> = match &s.model.score {
        Some(score) => placed(&inst, score),
        None => vec![(inst.instance_of(&root).map_err(CliError::Engine)?, 0.0)],
    };
    if at.len() != s.model.lines.len() {
        return Err(refused(format!(
            "the score plays {} lines where {} are written; a priced fit reads one event a line",
            at.len(),
            s.model.lines.len()
        )));
    }
    let root_slots = s.root_slots();
    let root_names = |name: &str| -> Option<(f64, Option<u32>)> {
        let i = root_slots.iter().position(|(n, _)| n == name)?;
        let slot = root_slots[i].1;
        Some((
            s.value(slot),
            (!matches!(slot, Slot::Fixed(_))).then_some(ROOT_SEED + i as u32),
        ))
    };
    let master = match s.model.score.is_some() {
        true => match filters(&s.model.nodes[&root].body) {
            Some((chain_written, _)) => chain(&chain_written, &root_names, g.sr)?,
            None => Vec::new(),
        },
        false => Vec::new(),
    };
    let keys = draws::keys(s);
    let tasks: Vec<usize> = (0..s.model.lines.len()).collect();
    let lines = sva_engine::each(threads, tasks, |i| {
        let (path, onset) = &at[i];
        let node = &s.model.lines[i].node;
        let slots = s.slots(i);
        let seeds: Vec<[(u32, f64); 1]> = (0..slots.len()).map(|k| [(k as u32, 1.0)]).collect();
        let varied: Vec<Varied> = slots
            .iter()
            .zip(&seeds)
            .filter(|((_, slot), _)| !matches!(slot, Slot::Fixed(_)))
            .map(|((name, _), t)| Varied {
                instance: path,
                name,
                tangent: t,
            })
            .collect();
        let noise = text.noise.get(node);
        let names = |name: &str| -> Option<(f64, Option<u32>)> {
            let k = slots.iter().position(|(n, _)| n == name)?;
            let slot = slots[k].1;
            Some((
                s.value(slot),
                (!matches!(slot, Slot::Fixed(_))).then_some(k as u32),
            ))
        };
        let noise_filters = match noise {
            Some(n) => chain(&n.filters, &names, g.sr)?,
            None => Vec::new(),
        };
        let noise_samples = match noise {
            Some(n) if super::write::line_reads(s, i, "seed") => (0..draws)
                .map(|k| rendered(renders, &n.source, Some(seed(keys[i], k)), *onset, g))
                .collect::<Result<Vec<_>, _>>()?,
            Some(n) => vec![rendered(renders, &n.source, None, *onset, g)?],
            None => Vec::new(),
        };
        let (read, tangents) = Tangents::default().active(|| -> Result<_, CliError> {
            let form = sva_engine::tangents(&inst, path, &varied).map_err(CliError::Engine)?;
            let split = lines::split(&form.body, noise.is_some()).map_err(|why| {
                refused(format!(
                    "score line {} (`{node}`): {why}",
                    s.model.lines[i].number
                ))
            })?;
            let corners = sva_formula::corners(&form.body);
            let (instants, frames) = lines::layout(*onset, split.l.v, split.r.v, &corners, g);
            let leaves = lines::read_leaves(&split.leaves, &instants).map_err(|why| {
                refused(format!(
                    "score line {} (`{node}`): {why}",
                    s.model.lines[i].number
                ))
            })?;
            Ok((split, instants, frames, leaves))
        });
        let (split, instants, frames, leaves) = read?;
        Ok::<_, CliError>(Lined {
            line: Line {
                at: *onset,
                l: split.l,
                r: split.r,
                rise: split.rise,
                fall: split.fall,
                terms: split.terms,
                leaves,
                instants,
                frames,
                tangents,
                noise: noise_samples,
                noise_filters,
            },
            slots: slots.into_iter().map(|(_, slot)| slot).collect(),
        })
    });
    let lines = lines.into_iter().collect::<Result<Vec<_>, _>>()?;
    Ok(Lowered { lines, master })
}
