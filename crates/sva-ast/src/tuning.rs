// Concern: reads each `deg` call off the key's scale, and checks each score line against its sound | Non-concern: reading scores or scales (score.rs, scale.rs) | IO: (&mut Graph[, beside]) -> Refusals

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::{Address, Arg, BinOp, Binds, DEGREE, Expr, Literal, map_children_ok};
use crate::graph::{Graph, resolve_ref_path};
use crate::ingest::{Data, KEY, body_at};
use crate::parameters::free_parameters;
use crate::refusal::Refusal;
use crate::scale::Scale;
use crate::score::{Event, Score, refused};

/// `variables/key` as a node at `path` spells a ref to it.
pub(crate) fn key_ref(path: &str) -> String {
    format!("{}{KEY}", "../".repeat(path.matches('/').count()))
}

/// Each `deg(n)` as the tonic times `deg(n)`, written at parse so a load follows the key. The
/// tuning pass turns the call into its ratio; no builtin answers it, so a degree no tuning
/// pass read refuses rather than plays.
pub(crate) fn keyed(e: &Expr, key: &str) -> Expr {
    match e {
        Expr::Call { name, span, .. } if name == DEGREE => {
            let tonic = Expr::Ref {
                path: key.to_string(),
                arg: Box::new(Expr::Var("t".to_string())),
                binds: Vec::new(),
                address: Address::Time,
                span: *span,
            };
            Expr::Bin(BinOp::Mul, Box::new(tonic), Box::new(e.clone()))
        }
        _ => map_children_ok(e, Binds::Substitute, |c| keyed(c, key)),
    }
}

fn calls_degree(e: &Expr) -> bool {
    match e {
        Expr::Call { name, .. } if name == DEGREE => true,
        _ => crate::expr::children(e, Binds::Substitute)
            .into_iter()
            .any(calls_degree),
    }
}

/// The one whole degree a `deg` call names, folded from literals.
fn degree(args: &[Arg]) -> Result<i64, String> {
    let whole = match args {
        [Arg::Pos(x)] => crate::series::folded(x, &|_| None)
            .filter(|d| d.fract() == 0.0 && d.abs() < 2f64.powi(53)),
        _ => None,
    };
    whole.map(|d| d as i64).ok_or_else(|| {
        "`deg` takes one whole scale degree folded from literals, as deg(5) or deg(-6); a \
         chromatic note is a degree times a ratio, as deg(7)*1st"
            .to_string()
    })
}

type Tuning = Option<(String, Scale)>;

impl Graph {
    /// Every `deg` call here, in a score or an equation, read off the scale here or in
    /// `beside`, and every score line checked against its sound. A key or scale this graph
    /// changes reads each of `beside`'s nodes that calls `deg` again from its text.
    pub fn lower_scores(&mut self, beside: Option<&Graph>) -> Result<(), Vec<Refusal>> {
        let tuning = self.tuning(beside).map_err(|r| vec![r])?;
        let retuned =
            self.defines(KEY) || tuning.as_ref().is_some_and(|(path, _)| self.defines(path));
        let mut refusals = Vec::new();
        if let Some(beside) = beside.filter(|_| retuned) {
            let degrees: Vec<String> = beside
                .paths()
                .filter(|path| !self.defines(path) && beside.reads_directly(path, KEY))
                .filter(|path| beside.text(path).is_some_and(|text| text.contains(DEGREE)))
                .map(str::to_string)
                .collect();
            for path in degrees {
                if let Err(refused) = self.reread(&path, beside) {
                    refusals.push(refused);
                }
            }
        }
        let scale = tuning.as_ref().map(|(_, scale)| scale);
        let paths: Vec<String> = self.paths().map(str::to_string).collect();
        for path in paths {
            if let Some(score) = self.score(&path) {
                let checked = self.checked(&path, score, beside);
                refusals.extend(checked);
            }
            if let Some((body, defaults)) = self.tuned(&path, scale, &mut refusals) {
                self.rewrite(&path, body, defaults);
            }
        }
        match refusals.is_empty() {
            true => Ok(()),
            false => Err(refusals),
        }
    }

    /// A node defined after the pass, as a probe is: its `deg` calls read off the scale.
    pub(crate) fn tune_defined(
        &mut self,
        path: &str,
        beside: Option<&Graph>,
    ) -> Result<(), Refusal> {
        let defined = self.defined(path).expect("a node this graph holds");
        if !calls_degree(&defined.body) {
            return Ok(());
        }
        let tuning = self.tuning(beside)?;
        let mut refusals = Vec::new();
        let scale = tuning.as_ref().map(|(_, scale)| scale);
        if let Some((body, defaults)) = self.tuned(path, scale, &mut refusals) {
            self.rewrite(path, body, defaults);
        }
        refusals.into_iter().next().map_or(Ok(()), Err)
    }

    fn tuning(&self, beside: Option<&Graph>) -> Result<Tuning, Refusal> {
        let keyed = match self.defines(KEY) {
            true => self.data(KEY),
            false => beside.and_then(|b| b.data(KEY)),
        };
        let Some(Data::Key(line)) = keyed else {
            return Ok(None);
        };
        let target = resolve_ref_path(KEY, &line.path);
        let found = target.as_deref().and_then(|target| {
            self.scale(target)
                .or_else(|| beside.and_then(|b| b.scale(target)))
        });
        match (target, found) {
            (Some(target), Some(scale)) => Ok(Some((target, scale.clone()))),
            (target, _) => Err(Refusal::new(
                KEY,
                line.span,
                DiagCode::BadScale,
                format!(
                    "`scale = @{}` names {}, and a scale is a file of rows of cents above the \
                     tonic, the last the period, as `scales/major` holds them",
                    line.path,
                    match target {
                        Some(target) if self.defines(&target) => format!("`{target}`, no scale"),
                        Some(target) => format!("`{target}`, which no file holds"),
                        None => "a path above the composition root".to_string(),
                    }
                ),
            )),
        }
    }

    /// The node at `path` with each `deg` call its ratio, `None` where it calls none. A
    /// refusal in a score names its line and column.
    fn tuned(
        &self,
        path: &str,
        scale: Option<&Scale>,
        refusals: &mut Vec<Refusal>,
    ) -> Option<(Expr, Vec<(String, Expr)>)> {
        let defined = self.defined(path)?;
        let calls =
            calls_degree(&defined.body) || defined.defaults.iter().any(|(_, v)| calls_degree(v));
        if !calls {
            return None;
        }
        let text = self.text(path).expect("every node holds its text");
        let score = self.score(path).is_some();
        let refuse = |span: ByteSpan, message: String| match score {
            true => {
                let before = &text[..span.start];
                let line = before.matches('\n').count() + 1;
                let column = before.len() - before.rfind('\n').map_or(0, |at| at + 1) + 1;
                let diag = refused(span.start, (line, column), &message);
                Refusal::new(path, diag.span, diag.code, diag.message)
            }
            false => Refusal::new(path, span, DiagCode::BadScale, message),
        };
        let from = if score { 0 } else { body_at(text) };
        let body = tune(
            &defined.body,
            scale,
            &|span, message| refuse(ByteSpan::new(span.start + from, span.end + from), message),
            refusals,
        );
        let defaults = defined
            .defaults
            .iter()
            .map(|(name, value)| (name.clone(), tune(value, scale, &refuse, refusals)))
            .collect();
        Some((body, defaults))
    }

    /// Each line's first event against its sound: a chord's events bind alike.
    fn checked(&self, path: &str, score: &Score, beside: Option<&Graph>) -> Vec<Refusal> {
        let text = self.text(path).expect("every node holds its text");
        let mut refusals = Vec::new();
        let mut last = None;
        for event in &score.events {
            if last.replace(event.line) == Some(event.line) {
                continue;
            }
            let Some((key, message)) = self.unread(path, event, beside) else {
                continue;
            };
            let line = &text[event.span.start..event.span.end];
            let column = line.find(&format!("\"{key}\"")).map_or(1, |at| at + 1);
            let diag = refused(
                event.span.start + column - 1,
                (event.line, column),
                &message,
            );
            refusals.push(Refusal::new(path, diag.span, diag.code, diag.message));
        }
        refusals
    }

    /// The first key this line binds that its sound does not read, or the first parameter the
    /// sound reads that neither the line nor a default of the sound binds.
    fn unread(
        &self,
        path: &str,
        event: &Event,
        beside: Option<&Graph>,
    ) -> Option<(String, String)> {
        let sound = resolve_ref_path(path, &event.sound)?;
        let defined = self
            .defined(&sound)
            .or_else(|| beside.and_then(|b| b.defined(&sound)))?;
        let is_node = |name: &str| {
            resolve_ref_path(&sound, name)
                .is_some_and(|n| self.defines(&n) || beside.is_some_and(|b| b.defines(&n)))
        };
        let mut reads: Vec<String> = defined.defaults.iter().map(|(k, _)| k.clone()).collect();
        reads.extend(free_parameters(
            &defined.body,
            &defined.defaults,
            &[],
            is_node,
        ));
        let mut given: Vec<(&str, &str)> = event
            .args
            .iter()
            .map(|(k, _)| (k.as_str(), k.as_str()))
            .collect();
        if event.deg.is_some() {
            given.push(("f0", "deg"));
        }
        let reads_list = reads.join("`, `");
        if let Some((name, written)) = given
            .iter()
            .find(|(name, _)| !reads.iter().any(|r| r == name))
        {
            let set = match name == written {
                true => format!("`{name}`"),
                false => format!("`{written}`, which sets `{name}`,"),
            };
            return Some((
                written.to_string(),
                format!(
                    "{set} is no parameter `@{}` reads; it reads `{reads_list}`",
                    event.sound
                ),
            ));
        }
        let names: Vec<&str> = given.iter().map(|(name, _)| *name).collect();
        let unbound = free_parameters(&defined.body, &defined.defaults, &names, is_node);
        let first = unbound.first()?;
        Some((
            "sound".to_string(),
            format!(
                "`@{}` reads `{first}`, which neither this line nor a default line of it binds",
                event.sound
            ),
        ))
    }
}

/// `e` with each `deg(n)` the ratio of degree `n` in `scale`; a call it cannot read is refused
/// through `refuse` and left as written.
fn tune(
    e: &Expr,
    scale: Option<&Scale>,
    refuse: &dyn Fn(ByteSpan, String) -> Refusal,
    refusals: &mut Vec<Refusal>,
) -> Expr {
    let Expr::Call { name, args, span } = e else {
        return map_children_ok(e, Binds::Substitute, |c| tune(c, scale, refuse, refusals));
    };
    if name != DEGREE {
        return map_children_ok(e, Binds::Substitute, |c| tune(c, scale, refuse, refusals));
    }
    let ratio = degree(args).and_then(|deg| match scale {
        None => Err(format!(
            "`deg` counts in the scale `{KEY}` names, and it names none; write `scale = \
             @../scales/major` above its tonic"
        )),
        Some(scale) => scale.ratio(deg).ok_or_else(|| {
            format!("`deg` {deg} lies so far from the tonic its ratio is no number")
        }),
    });
    match ratio {
        Ok(ratio) => Expr::Lit(Literal::Num(ratio)),
        Err(message) => {
            refusals.push(refuse(*span, message));
            e.clone()
        }
    }
}
