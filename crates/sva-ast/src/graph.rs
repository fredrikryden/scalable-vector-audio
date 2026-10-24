// Concern: folds a source's node texts into a graph, a song-lowered node kept beside its body | Non-concern: where the texts come from (source.rs), lowering a song (song/) | IO: (&dyn Source) -> Graph

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::{Binds, Expr, children, map_children_ok};
use crate::filename::FileSpan;
use crate::ingest::{Data, Parsed, base_name, parse_file, parse_score_file};
use crate::refusal::Refusal;
use crate::reserved::{Values, variable};
use crate::score::EXTENSION;
use crate::skipped::{Skip, Skipped};
use crate::source::Source;

/// The one reserved folder name; its contents are ordinary nodes, so `@variables/bpm` refs.
pub const VARIABLES: &str = "variables";

#[derive(Clone, Debug)]
pub struct Graph {
    nodes: BTreeMap<String, Arc<Defined>>,
    data: HashMap<String, Data>,
    spans: HashMap<String, Option<FileSpan>>,
    /// The reserved variables each node's own text reads, and the values they were lowered to.
    reserved: HashMap<String, Vec<&'static str>>,
    lowered: HashMap<String, Values>,
    skipped: Vec<Skipped>,
    /// Each node's text as its source answered it, or as a defined one prints.
    texts: HashMap<String, String>,
    origins: HashMap<String, Origin>,
    /// The edit of its source each node's text was read at, where the source counts them.
    generations: HashMap<String, u64>,
}

/// A node's body and the default lines written above it, held whole so a reader may keep it.
#[derive(Clone, Debug, PartialEq)]
pub struct Defined {
    pub body: Expr,
    pub defaults: Vec<(String, Expr)>,
}

#[derive(Clone, Debug)]
enum Origin {
    Lowered(Arc<Defined>),
    Made,
}

/// A node's parse with its text, span and data, as one graph hands it to another.
#[derive(Clone, Debug)]
pub struct Held {
    defined: Arc<Defined>,
    origin: Option<Origin>,
    text: String,
    generation: Option<u64>,
    span: Option<FileSpan>,
    data: Option<Data>,
    reserved: Vec<&'static str>,
    lowered: Option<Values>,
}

impl Held {
    pub fn body(&self) -> &Expr {
        &self.defined.body
    }
}

fn rebind(e: &Expr, to: &dyn Fn(&str) -> Option<Expr>) -> Expr {
    match e {
        Expr::Ref {
            path,
            arg,
            binds,
            address,
            span,
        } => Expr::Ref {
            path: path.clone(),
            arg: Box::new(rebind(arg, to)),
            binds: binds
                .iter()
                .map(|(k, v)| (k.clone(), to(k).unwrap_or_else(|| rebind(v, to))))
                .collect(),
            address: *address,
            span: *span,
        },
        _ => map_children_ok(e, Binds::Substitute, |c| rebind(c, to)),
    }
}

impl Graph {
    /// Every binding of a parameter `to` answers, by a ref or a default, bound to that answer.
    pub fn rebound(&self, to: &dyn Fn(&str) -> Option<Expr>) -> Graph {
        let mut out = self.clone();
        for defined in out.nodes.values_mut() {
            let defaults = defined.defaults.iter();
            let defaults =
                defaults.map(|(k, v)| (k.clone(), to(k).unwrap_or_else(|| rebind(v, to))));
            *defined = Arc::new(Defined {
                body: rebind(&defined.body, to),
                defaults: defaults.collect(),
            });
        }
        out
    }

    pub fn defines(&self, path: &str) -> bool {
        self.nodes.contains_key(path)
    }

    pub fn skipped(&self) -> &[Skipped] {
        &self.skipped
    }

    pub fn expr(&self, path: &str) -> Option<&Expr> {
        self.nodes.get(path).map(|defined| &defined.body)
    }

    pub fn defined(&self, path: &str) -> Option<&Arc<Defined>> {
        self.nodes.get(path)
    }

    pub fn source(&self, path: &str) -> Option<&Arc<Defined>> {
        match self.origins.get(path) {
            Some(Origin::Lowered(source)) => Some(source),
            Some(Origin::Made) => None,
            None => self.nodes.get(path),
        }
    }

    pub fn source_expr(&self, path: &str) -> Option<Expr> {
        self.source(path).map(|d| d.body.clone())
    }

    /// Each node as its file holds it, what only a song made gone.
    pub fn sourced(&self) -> Graph {
        let mut out = self.clone();
        for (path, origin) in std::mem::take(&mut out.origins) {
            match origin {
                Origin::Lowered(source) => {
                    out.nodes.insert(path, source);
                }
                Origin::Made => {
                    out.set(&path, None);
                }
            }
        }
        out
    }

    pub(crate) fn derive(&mut self, path: &str, defined: Defined) {
        match self.nodes.get(path) {
            Some(held) => {
                if !self.origins.contains_key(path) {
                    let source = Origin::Lowered(Arc::clone(held));
                    self.origins.insert(path.to_string(), source);
                }
                self.nodes.insert(path.to_string(), Arc::new(defined));
            }
            None => {
                let mut held = self.defining(defined.body);
                held.defined = Arc::new(Defined {
                    body: held.defined.body.clone(),
                    defaults: defined.defaults,
                });
                held.origin = Some(Origin::Made);
                self.set(path, Some(held));
            }
        }
    }

    pub fn text(&self, path: &str) -> Option<&str> {
        self.texts.get(path).map(String::as_str)
    }

    pub fn generation(&self, path: &str) -> Option<u64> {
        self.generations.get(path).copied()
    }

    /// What an invocation binding nothing for a name gets, in written order. A caller's own
    /// binding wins, so this is only ever consulted for names the invocation left out.
    pub fn defaults(&self, path: &str) -> &[(String, Expr)] {
        self.nodes
            .get(path)
            .map_or(&[], |defined| defined.defaults.as_slice())
    }

    pub fn span(&self, path: &str) -> Option<FileSpan> {
        self.spans.get(path).copied().flatten()
    }

    /// The events a score file holds, `None` for every node an equation or a number writes.
    pub fn score(&self, path: &str) -> Option<&crate::score::Score> {
        match self.data.get(path)? {
            Data::Score(score) => Some(score),
            _ => None,
        }
    }

    /// The cents a scale file holds: data a score's degrees read, never a signal.
    pub fn scale(&self, path: &str) -> Option<&crate::scale::Scale> {
        match self.data.get(path)? {
            Data::Scale(scale) => Some(scale),
            _ => None,
        }
    }

    pub(crate) fn reserved_reads(&self, path: &str) -> &[&'static str] {
        self.reserved.get(path).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn lowered_under(&self, path: &str) -> Option<&Values> {
        self.lowered.get(path)
    }

    pub(crate) fn lower(&mut self, path: &str, defined: Defined, under: &Values) {
        let held = self.nodes.get_mut(path).expect("a node this graph holds");
        *Arc::make_mut(held) = defined;
        self.lowered.insert(path.to_string(), under.clone());
    }

    /// `from`'s node at `path`, parsed again from its own text, before any pass rewrote it.
    pub(crate) fn reread(&mut self, path: &str, from: &Graph) -> Result<(), Refusal> {
        let mut held = from.held(path).expect("a node `from` holds");
        let parsed = match held.data {
            Some(Data::Score(_)) => parse_score_file(path, &held.text),
            _ => parse_file(&held.text),
        };
        let parsed = parsed.map_err(|d| Refusal::new(path, d.span, d.code, d.message))?;
        held.span = crate::filename::parse_filename(base_name(path)).1;
        held.reserved = crate::reserved::parsed_reads(&parsed, held.span);
        held.lowered = None;
        held.defined = Arc::new(Defined {
            body: parsed.expr,
            defaults: parsed.defaults,
        });
        held.data = parsed.data;
        self.set(path, Some(held));
        Ok(())
    }

    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(String::as_str)
    }

    /// What `variables/bpm` and `variables/meter` hold, where this composition declares them.
    pub fn tempo(&self) -> Option<crate::reserved::Tempo> {
        self.values(None)
            .expect("a lowered graph's reserved variables hold values")
            .tempo
    }

    /// Adds a node no file backs — an ad-hoc expression read in this graph's namespace.
    /// `false` when `path` is taken, so a probe can never shadow a composition's own file.
    pub fn define(&mut self, path: &str, expr: Expr) -> bool {
        if self.nodes.contains_key(path) {
            return false;
        }
        let held = self.defining(expr);
        self.set(path, Some(held));
        true
    }

    /// A node no file backs, its reads of reserved variables already lowered.
    pub fn defining(&self, body: Expr) -> Held {
        Held {
            origin: None,
            text: crate::print::render(&body),
            generation: None,
            defined: Arc::new(Defined {
                body,
                defaults: Vec::new(),
            }),
            span: None,
            data: None,
            reserved: Vec::new(),
            lowered: None,
        }
    }

    pub fn held(&self, path: &str) -> Option<Held> {
        Some(Held {
            defined: Arc::clone(self.nodes.get(path)?),
            origin: self.origins.get(path).cloned(),
            text: self
                .texts
                .get(path)
                .cloned()
                .expect("every node holds its text"),
            generation: self.generations.get(path).copied(),
            span: self.spans.get(path).copied().flatten(),
            data: self.data.get(path).cloned(),
            reserved: self.reserved_reads(path).to_vec(),
            lowered: self.lowered.get(path).cloned(),
        })
    }

    pub fn set(&mut self, path: &str, held: Option<Held>) -> Option<Held> {
        let before = self.held(path);
        match held {
            Some(held) => {
                self.nodes.insert(path.to_string(), held.defined);
                self.texts.insert(path.to_string(), held.text);
                match held.generation {
                    Some(at) => self.generations.insert(path.to_string(), at),
                    None => self.generations.remove(path),
                };
                self.spans.insert(path.to_string(), held.span);
                match held.data {
                    Some(data) => self.data.insert(path.to_string(), data),
                    None => self.data.remove(path),
                };
                match held.reserved.is_empty() {
                    true => self.reserved.remove(path),
                    false => self.reserved.insert(path.to_string(), held.reserved),
                };
                match held.lowered {
                    Some(values) => self.lowered.insert(path.to_string(), values),
                    None => self.lowered.remove(path),
                };
                match held.origin {
                    Some(origin) => self.origins.insert(path.to_string(), origin),
                    None => self.origins.remove(path),
                };
            }
            None => {
                self.nodes.remove(path);
                self.texts.remove(path);
                self.generations.remove(path);
                self.spans.remove(path);
                self.data.remove(path);
                self.reserved.remove(path);
                self.lowered.remove(path);
                self.origins.remove(path);
            }
        }
        before
    }

    pub fn holds_as(&self, path: &str, other: &Graph) -> bool {
        match (self.nodes.get(path), other.nodes.get(path)) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) || a == b,
            (None, None) => true,
            _ => false,
        }
    }

    pub fn read_as(&self, path: &str, other: &Graph) -> bool {
        self.texts.get(path) == other.texts.get(path)
            && self.generations.get(path) == other.generations.get(path)
    }

    /// Each node `path` reads directly, by its body, a default or a reserved variable.
    fn reaching_from(&self, path: &str) -> Vec<String> {
        let Some(defined) = self.nodes.get(path) else {
            return Vec::new();
        };
        let mut out = reads_of(path, &defined.body);
        for (_, value) in &defined.defaults {
            out.extend(reads_of(path, value));
        }
        out.extend(self.reserved_reads(path).iter().map(|name| variable(name)));
        out
    }

    pub fn reaching(&self, roots: &[String]) -> BTreeSet<String> {
        let (mut seen, mut open) = (BTreeSet::new(), roots.to_vec());
        while let Some(path) = open.pop() {
            if self.defines(&path) && seen.insert(path.clone()) {
                open.extend(self.reaching_from(&path));
            }
        }
        seen
    }

    pub fn adopt(&mut self, from: &Graph, roots: &[String]) -> Vec<String> {
        let lacking: Vec<String> = from
            .reaching(roots)
            .into_iter()
            .filter(|path| !self.defines(path))
            .collect();
        for path in &lacking {
            self.set(path, from.held(path));
        }
        lacking
    }

    pub fn define_settled(
        &mut self,
        beside: Option<&Graph>,
        path: &str,
        expr: Expr,
    ) -> Result<bool, Vec<Refusal>> {
        if self.defines(path) {
            return Ok(false);
        }
        let written = crate::series::written_out(&expr)
            .map_err(|d| vec![Refusal::new(path, d.span, d.code, d.message)])?;
        let mut defined = Defined {
            body: written,
            defaults: Vec::new(),
        };
        if !crate::reserved::reads(&defined.body, &[], None).is_empty() {
            let values = self.values(beside)?;
            defined = self.lowered(path, &defined, &values)?;
        }
        Ok(self.define(path, defined.body))
    }
}

pub(crate) fn variable_text<'g>(graph: &'g Graph, name: &str) -> Option<&'g str> {
    let text = graph.text(&format!("{VARIABLES}/{name}"))?;
    Some(&text[crate::ingest::body_at(text)..])
}

/// Every node the source names. What a whole composition can be told about itself — `lint`, an
/// entry-point set — needs all of them, so this is what a directory load is.
pub fn load(source: &dyn Source) -> Result<Graph, Vec<Refusal>> {
    let mut loading = Loading::new(source.name());
    let listing = source
        .paths()
        .map_err(|reason| vec![loading.directory(reason)])?;
    for path in &listing.found {
        let node = path.strip_suffix(EXTENSION).unwrap_or(path);
        match names_a_node(node) {
            true if !loading.holds(node) => {
                loading.pull(source, node);
            }
            true => {}
            false => loading.skipped.push(Skipped {
                path: path.clone(),
                reason: Skip::Unnameable,
            }),
        }
    }
    for path in &listing.unreadable {
        loading.skipped.push(Skipped {
            path: path.clone(),
            reason: Skip::Special,
        });
    }
    loading.finish(None)
}

/// Whether anything names this path: a ref path, or one an instance's binds follow, which
/// a render still addresses by name though no `@ref` spells it.
pub fn names_a_node(path: &str) -> bool {
    let bare = match path.split_once('(') {
        Some((before, binds)) if binds.ends_with(')') => before,
        _ => path,
    };
    !bare.is_empty() && crate::lexer::whole_ref_path(bare)
}

/// The root set [`load_reaching`] follows, for a caller whose expression is not in the source
/// yet. A bareword call names a file too, so every call name is named speculatively.
pub fn reads_of(referencing: &str, expr: &Expr) -> Vec<String> {
    let mut named = Vec::new();
    collect_named(expr, &mut named);
    named
        .iter()
        .filter_map(|name| resolve_ref_path(referencing, name))
        .collect()
}

/// Only what `roots` reach, following each parsed node's own reads: an unreferenced node is
/// never asked for, and one the source cannot answer for is a located dangling-ref refusal.
pub fn load_reaching(source: &dyn Source, roots: &[&str]) -> Result<Graph, Vec<Refusal>> {
    let mut loading = Loading::new(source.name());
    let mut work: Vec<String> = roots.iter().rev().map(|r| (*r).to_string()).collect();
    while let Some(path) = work.pop() {
        if loading.holds(&path) || !loading.pull(source, &path) {
            continue;
        }
        work.extend(loading.reads(&path));
    }
    loading.finish(None)
}

/// What `roots` reach that `base` does not hold, or holds at an earlier edit of `source`:
/// each node read and parsed once, a ref into `base` resolving there. A node `base` holds as
/// `source` still does is walked through, never read.
pub fn load_beside(
    source: &dyn Source,
    roots: &[&str],
    base: &Graph,
) -> Result<Graph, Vec<Refusal>> {
    let mut loading = Loading::new(source.name());
    let mut walked = BTreeSet::new();
    let mut work: Vec<String> = roots.iter().rev().map(|r| (*r).to_string()).collect();
    while let Some(path) = work.pop() {
        let current = base.defines(&path)
            && generation(source, &path).is_none_or(|at| base.generation(&path) == Some(at));
        if current {
            if walked.insert(path.clone()) {
                work.extend(base.reaching_from(&path));
            }
            continue;
        }
        if loading.holds(&path) || !loading.pull(source, &path) {
            continue;
        }
        work.extend(loading.reads(&path));
    }
    loading.finish(Some(base))
}

/// A body or default reading `sample("<path>")` holds the recording the source answers for.
fn recorded(mut parsed: Parsed, source: &dyn Source) -> Result<Parsed, crate::diag::Diag> {
    let unheld = |e: crate::recording::Unheld| {
        let code = match e.why {
            None => DiagCode::MissingRecording,
            Some(_) => DiagCode::Io,
        };
        crate::diag::Diag::new(code, ByteSpan::at(0), e.message())
    };
    parsed.expr = crate::recording::resolve_recordings(&parsed.expr, source).map_err(unheld)?;
    for (_, value) in &mut parsed.defaults {
        *value = crate::recording::resolve_recordings(value, source).map_err(unheld)?;
    }
    Ok(parsed)
}

fn generation(source: &dyn Source, path: &str) -> Option<u64> {
    source
        .generation(path)
        .or_else(|| source.generation(&format!("{path}{EXTENSION}")))
}

/// One parsed node at a time, so pulling everything and pulling a closure are the same walk.
struct Loading {
    name: String,
    nodes: BTreeMap<String, Expr>,
    defaults: BTreeMap<String, Vec<(String, Expr)>>,
    data: HashMap<String, Data>,
    spans: HashMap<String, Option<FileSpan>>,
    missing: BTreeSet<String>,
    refusals: Vec<Refusal>,
    skipped: Vec<Skipped>,
    texts: HashMap<String, String>,
    generations: HashMap<String, u64>,
    reserved: HashMap<String, Vec<&'static str>>,
}

impl Loading {
    fn new(name: String) -> Loading {
        Loading {
            name,
            nodes: BTreeMap::new(),
            defaults: BTreeMap::new(),
            data: HashMap::new(),
            spans: HashMap::new(),
            missing: BTreeSet::new(),
            refusals: Vec::new(),
            skipped: Vec::new(),
            texts: HashMap::new(),
            generations: HashMap::new(),
            reserved: HashMap::new(),
        }
    }

    fn directory(&self, reason: String) -> Refusal {
        Refusal::new(self.name.clone(), ByteSpan::at(0), DiagCode::Io, reason)
    }

    /// A miss counts as settled: a builtin name every node calls would otherwise cost the
    /// source one failed read per referencing node.
    fn holds(&self, path: &str) -> bool {
        self.nodes.contains_key(path)
            || self.spans.contains_key(path)
            || self.missing.contains(path)
    }

    /// `false` where nothing was added: the source does not hold this path, or its text is not
    /// a node — and the second is already recorded as its own located refusal.
    fn pull(&mut self, source: &dyn Source, path: &str) -> bool {
        let file = format!("{path}{EXTENSION}");
        let read = |at: &str| {
            source
                .get(at)
                .map(|text| text.map(|text| text.into_owned()))
        };
        let (text, score) = match (read(path), read(&file)) {
            (Ok(Some(_)), Ok(Some(_))) => {
                self.refusals.push(
                    Refusal::new(
                        file.as_str(),
                        ByteSpan::at(0),
                        DiagCode::BadScore,
                        format!(
                            "line 1, column 1: `@{path}` matches two files, `{path}` and \
                             `{file}`; a ref names one file, so rename or remove one of them"
                        ),
                    )
                    .placed((1, 1)),
                );
                self.spans.insert(path.to_string(), None);
                return false;
            }
            (Ok(Some(text)), Ok(None)) => (text, false),
            (Ok(None), Ok(Some(text))) => (text, true),
            (Ok(None), Ok(None)) => {
                self.missing.insert(path.to_string());
                return false;
            }
            (Err(reason), _) | (_, Err(reason)) => {
                self.refusals
                    .push(Refusal::new(path, ByteSpan::at(0), DiagCode::Io, reason));
                self.spans.insert(path.to_string(), None);
                return false;
            }
        };
        let (_, span) = crate::filename::parse_filename(base_name(path));
        self.spans.insert(path.to_string(), span);
        self.texts.insert(path.to_string(), text.clone());
        let held = if score { file.as_str() } else { path };
        if let Some(at) = generation(source, path) {
            self.generations.insert(path.to_string(), at);
        }
        let parsed = match score {
            true => parse_score_file(path, &text),
            false => parse_file(&text).and_then(|parsed| recorded(parsed, source)),
        };
        match parsed {
            Ok(parsed) => {
                let reads = crate::reserved::parsed_reads(&parsed, span);
                if !reads.is_empty() {
                    self.reserved.insert(path.to_string(), reads);
                }
                self.nodes.insert(path.to_string(), parsed.expr);
                if !parsed.defaults.is_empty() {
                    self.defaults.insert(path.to_string(), parsed.defaults);
                }
                if let Some(data) = parsed.data {
                    self.data.insert(path.to_string(), data);
                }
                true
            }
            Err(d) => {
                let refusal = Refusal::new(held, d.span, d.code, d.message);
                self.refusals.push(match score {
                    true => refusal.placed(crate::score::line_column(&text, d.span.start)),
                    false => refusal,
                });
                false
            }
        }
    }

    fn reads(&self, path: &str) -> Vec<String> {
        let mut out = match self.nodes.get(path) {
            Some(expr) => reads_of(path, expr),
            None => Vec::new(),
        };
        let reserved = self.reserved.get(path).into_iter().flatten();
        out.extend(reserved.map(|name| variable(name)));
        out
    }

    /// A refusal in a score names its file, line and column.
    fn on_its_line(&self, refusal: Refusal) -> Refusal {
        let path = refusal.at.path.as_str();
        match (self.data.get(path), self.texts.get(path)) {
            (Some(Data::Score(_)), Some(text)) => crate::score::refused(
                format!("{path}{EXTENSION}"),
                text,
                refusal.at.span.start,
                crate::diag::DiagCode::BadScore,
                &refusal.reason,
            ),
            _ => refusal,
        }
    }

    /// Every ref resolving here or in `base`.
    fn finish(self, base: Option<&Graph>) -> Result<Graph, Vec<Refusal>> {
        if !self.refusals.is_empty() {
            return Err(self.refusals);
        }
        let held =
            |path: &str| self.nodes.contains_key(path) || base.is_some_and(|b| b.defines(path));
        let is_score = |path: &str| matches!(self.data.get(path), Some(Data::Score(_)));
        let dangling: Vec<Refusal> = check_refs((&self.nodes, &held), &self.texts, &self.name)
            .into_iter()
            .filter(|r| !(is_score(&r.at.path) && r.reason.contains(&format!("`{VARIABLES}/"))))
            .collect();
        if !dangling.is_empty() {
            return Err(dangling.into_iter().map(|r| self.on_its_line(r)).collect());
        }
        let cycles = check_cycles(&self.nodes);
        if !cycles.is_empty() {
            let placed = |r: Refusal| match is_score(&r.at.path) {
                true => Refusal {
                    reason: format!(
                        "a score places itself, directly or through others: `{}` places what \
                         places it back ({})",
                        r.at.path, r.reason
                    ),
                    ..r
                },
                false => r,
            };
            return Err(cycles
                .into_iter()
                .map(|r| self.on_its_line(placed(r)))
                .collect());
        }
        let mut defaults = self.defaults;
        let nodes = self.nodes.into_iter().map(|(path, body)| {
            let defaults = defaults.remove(&path).unwrap_or_default();
            (path, Arc::new(Defined { body, defaults }))
        });
        Ok(Graph {
            nodes: nodes.collect(),
            data: self.data,
            spans: self.spans,
            reserved: self.reserved,
            lowered: HashMap::new(),
            skipped: self.skipped,
            texts: self.texts,
            generations: self.generations,
            origins: HashMap::new(),
        })
    }
}

/// Resolves against the REFERENCING file's own directory, `./` and a bare path alike. A `.`
/// or `..` normalizes anywhere, so `@a/x/../b` and `@a/b` are one node. `None` above the root,
/// and for an absolute path, which only a command line resolves.
pub fn resolve_ref_path(referencing: &str, ref_path: &str) -> Option<String> {
    if ref_path.starts_with('/') {
        return None;
    }
    let dir = match referencing.rsplit_once('/') {
        Some((dir, _)) => dir,
        None => "",
    };
    let mut segments: Vec<&str> = dir.split('/').filter(|s| !s.is_empty()).collect();
    for segment in ref_path.split('/') {
        match segment {
            "." => {}
            ".." => {
                segments.pop()?;
            }
            other => segments.push(other),
        }
    }
    Some(segments.join("/"))
}

fn above_root(referencing: &str, span: ByteSpan, ref_path: &str, root: &str) -> Refusal {
    Refusal::new(
        referencing,
        span,
        DiagCode::RefAboveRoot,
        format!(
            "`@{ref_path}` in `{referencing}` walks above the composition root `{root}`; a \
             ref may reach anywhere inside the root and nowhere outside it"
        ),
    )
}

/// One `@path(...)` read, and whether it reads at the referencing node's own `t`.
struct RefSite {
    path: String,
    span: ByteSpan,
    at_t: bool,
}

fn collect_refs(e: &Expr, out: &mut Vec<RefSite>) {
    if let Expr::Ref {
        path, arg, span, ..
    } = e
    {
        out.push(RefSite {
            path: path.clone(),
            span: *span,
            at_t: **arg == Expr::Var("t".to_string()),
        });
    }
    for c in children(e, Binds::Substitute) {
        collect_refs(c, out);
    }
}

/// Every path this expression names, `@ref` or bareword call alike.
fn collect_named(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Call { name, .. } => out.push(name.clone()),
        Expr::Ref { path, .. } => out.push(path.clone()),
        _ => {}
    }
    for c in children(e, Binds::Substitute) {
        collect_named(c, out);
    }
}

type Holds<'a> = &'a dyn Fn(&str) -> bool;

fn check_refs(
    (nodes, held): (&BTreeMap<String, Expr>, Holds),
    texts: &HashMap<String, String>,
    root: &str,
) -> Vec<Refusal> {
    let mut refusals = Vec::new();
    for (path, expr) in nodes {
        let mut refs = Vec::new();
        collect_refs(expr, &mut refs);
        for site in refs {
            let Some(target) = resolve_ref_path(path, &site.path) else {
                refusals.push(above_root(path, site.span, &site.path, root));
                continue;
            };
            if !held(&target) {
                let hint = texts
                    .get(path)
                    .and_then(|text| digit_segment_hint(text, site.span.end))
                    .map(|seg| {
                        format!(
                            "; a path segment cannot begin with a digit, so `{seg}` was not \
                             read as part of this reference"
                        )
                    })
                    .unwrap_or_default();
                refusals.push(Refusal::new(
                    path.clone(),
                    site.span,
                    DiagCode::DanglingRef,
                    format!(
                        "`@{}` does not resolve to any file (looked for `{target}`){hint}",
                        site.path
                    ),
                ));
            }
        }
    }
    refusals
}

/// `end` is where the lexer's digit-segment rule stopped: on a `/` whose next byte is a digit.
/// Re-inspecting the source there recovers the segment the path lost, e.g. `/1-raw`.
fn digit_segment_hint(text: &str, end: usize) -> Option<String> {
    let b = text.as_bytes();
    if b.get(end) != Some(&b'/') || !b.get(end + 1).is_some_and(u8::is_ascii_digit) {
        return None;
    }
    let mut j = end + 1;
    while j < b.len() && (b[j].is_ascii_alphanumeric() || matches!(b[j], b'_' | b'-')) {
        j += 1;
    }
    Some(text[end..j].to_string())
}

#[derive(Clone, Copy, PartialEq)]
enum Color {
    White,
    Gray,
    Black,
}

/// A node's resolved, in-graph edges (dangling refs are already refused before this runs).
fn resolved_refs<'a>(node: &str, nodes: &'a BTreeMap<String, Expr>) -> Vec<Edge<'a>> {
    let mut raw = Vec::new();
    if let Some(expr) = nodes.get(node) {
        collect_refs(expr, &mut raw);
    }
    raw.into_iter()
        .filter_map(|site| {
            let target = resolve_ref_path(node, &site.path)?;
            nodes.get_key_value(&target).map(|(k, _)| Edge {
                target: k.as_str(),
                span: site.span,
                at_t: site.at_t,
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Edge<'a> {
    target: &'a str,
    span: ByteSpan,
    at_t: bool,
}

struct Frame<'a> {
    node: &'a str,
    refs: Vec<Edge<'a>>,
    idx: usize,
}

/// An explicit-stack DFS: a hostile, arbitrarily long acyclic ref chain walks in a heap-sized
/// `Vec`, never the OS call stack. It walks ONLY the reads written at bare `t`, so a cycle in
/// it carries no delay at any sample rate; one holding a shifted read is the engine's to settle.
fn check_cycles(nodes: &BTreeMap<String, Expr>) -> Vec<Refusal> {
    let mut color: HashMap<&str, Color> =
        nodes.keys().map(|k| (k.as_str(), Color::White)).collect();
    let mut refusals = Vec::new();
    let undelayed = |node: &str| -> Vec<Edge<'_>> {
        resolved_refs(node, nodes)
            .into_iter()
            .filter(|e| e.at_t)
            .collect()
    };

    for start in nodes.keys().map(String::as_str) {
        if color[start] != Color::White {
            continue;
        }
        color.insert(start, Color::Gray);
        let mut stack = vec![Frame {
            node: start,
            refs: undelayed(start),
            idx: 0,
        }];

        while let Some(frame) = stack.last_mut() {
            if frame.idx >= frame.refs.len() {
                color.insert(frame.node, Color::Black);
                stack.pop();
                continue;
            }
            let edge = frame.refs[frame.idx];
            let from = frame.node;
            frame.idx += 1;
            match color.get(edge.target) {
                Some(Color::Gray) => refusals.push(Refusal::new(
                    from,
                    edge.span,
                    DiagCode::RefCycle,
                    format!(
                        "a delay-free ref cycle: `{from}` reads `@{}` at its own `t`, and the \
                         reads leading back to `{from}` are all at `t` too, so this value is its \
                         own argument. A loop is well-founded only when some read on it reaches \
                         strictly into the past",
                        edge.target
                    ),
                )),
                Some(Color::White) => {
                    color.insert(edge.target, Color::Gray);
                    stack.push(Frame {
                        node: edge.target,
                        refs: undelayed(edge.target),
                        idx: 0,
                    });
                }
                _ => {}
            }
        }
    }
    refusals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(files: &[(&str, &str)]) -> crate::source::Composition {
        files.iter().copied().collect()
    }

    struct Counting {
        held: crate::source::Composition,
        asked: std::cell::RefCell<Vec<String>>,
    }

    impl Source for Counting {
        fn paths(&self) -> Result<crate::source::Listing, String> {
            panic!("a pull load must never enumerate the source")
        }
        fn get(&self, path: &str) -> Result<Option<std::borrow::Cow<'_, str>>, String> {
            self.asked.borrow_mut().push(path.to_string());
            self.held.get(path)
        }
    }

    #[test]
    fn a_pull_load_never_asks_for_a_node_the_roots_do_not_reach() {
        let source = Counting {
            held: of(&[
                (
                    "song",
                    "sin(2*pi*t)*0.1 + @drums/kick*0.5 + gain(@drums/kick, k=2)\n",
                ),
                ("drums/kick", "sin(2*pi*50*t)\n"),
                ("gain", "x*k*sin(2*pi*t)\n"),
                ("zombie", "@nowhere\n"),
            ]),
            asked: std::cell::RefCell::new(Vec::new()),
        };
        let g = load_reaching(&source, &["song"]).expect("the reached set is whole");
        let mut reached: Vec<&str> = g.paths().collect();
        reached.sort();
        assert_eq!(reached, ["drums/kick", "gain", "song"]);
        let asked = source.asked.borrow().clone();
        assert!(
            !asked.contains(&"zombie".to_string()),
            "an unreferenced node was read: {asked:?}"
        );
        assert!(
            asked.contains(&"gain".to_string()),
            "a bareword invocation names a file too: {asked:?}"
        );
        let mut once = asked.clone();
        once.sort();
        once.dedup();
        assert_eq!(
            once.len(),
            asked.len(),
            "`sin` is called from two root nodes and must cost one read: {asked:?}"
        );
    }

    #[test]
    fn same_file_cycle_is_refused() {
        let files: Vec<(&str, &str)> = vec![("a", "@a + 1\n")];
        let errs = load(&of(&files)).unwrap_err();
        assert!(errs.iter().any(|r| r.code == DiagCode::RefCycle));
    }

    /// The relaxation this file owns: a loop with a shifted read on it is left for the engine,
    /// which alone knows how many samples that shift is worth.
    #[test]
    fn a_mutual_cycle_with_a_shifted_read_on_it_parses() {
        let files: Vec<(&str, &str)> =
            vec![("a", "@b(t - 0.01s)*0.5\n"), ("b", "@a*0.5 + sin(t)\n")];
        let g = load(&of(&files)).unwrap();
        assert!(g.expr("a").is_some() && g.expr("b").is_some());

        let files: Vec<(&str, &str)> = vec![("ring", "@ring(t - 0.002s)*0.9 + sin(t)\n")];
        assert!(load(&of(&files)).is_ok());
    }

    /// Reaching a node down a delayed leg first must not settle it: only the undelayed reads
    /// are a graph here, and `a -> c -> d -> a` is a loop in it whichever way `d` was found.
    #[test]
    fn a_delay_free_leg_is_found_even_when_a_delayed_one_reaches_it_first() {
        let mut files: Vec<(&str, &str)> = vec![
            ("a", "@b + @c\n"),
            ("b", "@d(t - 0.01s)\n"),
            ("c", "@d\n"),
            ("d", "@a\n"),
        ];
        let errs = load(&of(&files)).unwrap_err();
        assert!(
            errs.iter().any(|r| r.code == DiagCode::RefCycle),
            "a -> c -> d -> a is delay-free: {errs:?}"
        );

        files.push(("c", "@d(t - 0.01s)\n"));
        assert!(
            load(&of(&files)).is_ok(),
            "every leg now carries a shift, so the engine sizes it"
        );
    }

    #[test]
    fn deep_ref_chain_cycle_is_refused() {
        let files: Vec<(&str, &str)> =
            vec![("a", "@b\n"), ("b", "@c\n"), ("c", "@d\n"), ("d", "@a\n")];
        let errs = load(&of(&files)).unwrap_err();
        assert!(errs.iter().any(|r| r.code == DiagCode::RefCycle));
    }

    /// A parameter list does not make a self-reference well-founded: an invocation is still a
    /// ref, and a ref cycle is still a cycle. `self(...)` stays the one exemption.
    #[test]
    fn a_function_invoking_itself_is_refused_but_self_inside_one_is_not() {
        let files: Vec<(&str, &str)> = vec![("f", "x + @f(t, x=x*0.5)\n")];
        let errs = load(&of(&files)).unwrap_err();
        assert!(errs.iter().any(|r| r.code == DiagCode::RefCycle));

        let files: Vec<(&str, &str)> = vec![("a", "@b(t, x=x)\n"), ("b", "@a(t, x=x)\n")];
        let errs = load(&of(&files)).unwrap_err();
        assert!(errs.iter().any(|r| r.code == DiagCode::RefCycle));

        let files: Vec<(&str, &str)> = vec![("comb", "x + g*self(t - delay)\n")];
        assert!(load(&of(&files)).is_ok());
    }

    /// A bare path is exactly what it always was; `./` spells that out and `../` walks up.
    #[test]
    fn a_dot_segment_resolves_against_the_referencing_files_own_directory() {
        let at = |file: &str, path: &str| resolve_ref_path(file, path).unwrap();
        assert_eq!(at("lead", "kick"), "kick");
        assert_eq!(at("piano/bench", "partials"), "piano/partials");
        assert_eq!(at("piano/bench", "./partials"), "piano/partials");
        assert_eq!(at("piano/bench", "../variables/bpm"), "variables/bpm");
        assert_eq!(at("a/b/c/d", "../../x"), "a/x");
        assert_eq!(at("song", "./drums/kick"), "drums/kick");
    }
}
