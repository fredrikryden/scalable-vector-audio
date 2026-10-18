// Concern: settles each score against its sounds and the key's scale, then lowers it | Non-concern: reading scores or scales (score.rs, scale.rs) | IO: (&mut Graph[, beside]) -> Refusals

use crate::diag::DiagCode;
use crate::expr::{Address, Arg, BinOp, Expr, Literal};
use crate::graph::{Graph, resolve_ref_path};
use crate::ingest::{Data, KEY};
use crate::parameters::free_parameters;
use crate::refusal::Refusal;
use crate::scale::Scale;
use crate::score::{Event, Score, refused};

/// `variables/key` as a score at `path` spells a ref to it.
pub(crate) fn key_ref(path: &str) -> String {
    format!("{}{KEY}", "../".repeat(path.matches('/').count()))
}

/// The tonic times the degree's ratio. With no scale known yet the ratio is a `deg` call,
/// which no builtin answers, so a score no tuning pass lowered refuses rather than plays.
pub(crate) fn pitch(key: &str, event: &Event, deg: i64, ratio: Option<f64>) -> Expr {
    let tonic = Expr::Ref {
        path: key.to_string(),
        arg: Box::new(Expr::Var("t".to_string())),
        binds: Vec::new(),
        address: Address::Time,
        span: event.span,
    };
    let step = match ratio {
        Some(ratio) => Expr::Lit(Literal::Num(ratio)),
        None => Expr::Call {
            name: "deg".to_string(),
            args: vec![Arg::Pos(Expr::Lit(Literal::Num(deg as f64)))],
            span: event.span,
        },
    };
    Expr::Bin(BinOp::Mul, Box::new(tonic), Box::new(step))
}

type Tuning = Option<(String, Scale)>;

impl Graph {
    /// Every score here lowered against the sounds and scale here or in `beside`; a key or
    /// scale this graph changes re-lowers each of `beside`'s scores that counts degrees.
    pub fn lower_scores(&mut self, beside: Option<&Graph>) -> Result<(), Vec<Refusal>> {
        let tuning = self.tuning(beside).map_err(|r| vec![r])?;
        let retuned =
            self.defines(KEY) || tuning.as_ref().is_some_and(|(path, _)| self.defines(path));
        if let Some(beside) = beside.filter(|_| retuned) {
            let counting: Vec<String> = beside
                .paths()
                .filter(|path| !self.defines(path))
                .filter(|path| beside.score(path).is_some_and(counts_degrees))
                .map(str::to_string)
                .collect();
            for path in counting {
                self.set(&path, beside.held(&path));
            }
        }
        let scores: Vec<(String, Score)> = self
            .paths()
            .filter_map(|path| Some((path.to_string(), self.score(path)?.clone())))
            .collect();
        let mut refusals = Vec::new();
        for (path, score) in scores {
            match self.settled(&path, &score, &tuning, beside) {
                Ok(lowered) => self.rewrite(&path, lowered),
                Err(found) => refusals.extend(found),
            }
        }
        match refusals.is_empty() {
            true => Ok(()),
            false => Err(refusals),
        }
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

    fn settled(
        &self,
        path: &str,
        score: &Score,
        tuning: &Tuning,
        beside: Option<&Graph>,
    ) -> Result<Expr, Vec<Refusal>> {
        let text = self.text(path).expect("every node holds its text");
        let mut refusals = Vec::new();
        for event in &score.events {
            let line = &text[event.span.start..event.span.end];
            let at = |key: &str, message: String| {
                let column = line.find(&format!("\"{key}\"")).map_or(1, |at| at + 1);
                let diag = refused(
                    event.span.start + column - 1,
                    (event.line, column),
                    &message,
                );
                Refusal::new(path, diag.span, diag.code, diag.message)
            };
            if let Some(found) = self.unread(path, event, beside) {
                let (key, message) = found;
                refusals.push(at(&key, message));
            }
            match (event.deg, tuning) {
                (Some(_), None) => refusals.push(at(
                    "deg",
                    format!(
                        "`deg` counts in the scale `{KEY}` names, and it names none; write \
                         `scale = @../scales/major` above its tonic"
                    ),
                )),
                (Some(deg), Some((_, scale))) if scale.ratio(deg).is_none() => refusals.push(at(
                    "deg",
                    format!("`deg` {deg} lies so far from the tonic its ratio is no number"),
                )),
                _ => {}
            }
        }
        if !refusals.is_empty() {
            return Err(refusals);
        }
        let key = key_ref(path);
        let ratio = |deg: i64| tuning.as_ref().and_then(|(_, scale)| scale.ratio(deg));
        Ok(score.lowered(&|event, deg| pitch(&key, event, deg, ratio(deg))))
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

fn counts_degrees(score: &Score) -> bool {
    score.events.iter().any(|e| e.deg.is_some())
}
