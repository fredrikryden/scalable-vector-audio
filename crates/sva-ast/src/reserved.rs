// Concern: finds the reserved variables' values and lowers each read of them, again where one changes | Non-concern: a song's moves of a variable (song/) | IO: (&mut Graph[, beside]) -> Refusals

use crate::diag::{ByteSpan, DiagCode};
use crate::expr::{Address, Arg, BinOp, Binds, DEGREE, Expr, Literal, children, map_children_ok};
use crate::filename::{FileSpan, SpanUnit, parse_filename};
use crate::graph::{Defined, Graph, VARIABLES, resolve_ref_path};
use crate::ingest::{Data, Parsed, base_name, body_at};
use crate::refusal::Refusal;
use crate::scale::{BUILTIN_SCALES, Scale, builtin_scale};
use crate::score::{Item, Ratio};

/// Each one the node `variables/<name>`: a `b` literal or span reads `bpm` and `meter`, and a
/// `deg` call reads `key`, its tonic, and `scale`, a built-in scale's name or a ref to the
/// scale file it counts in.
pub const RESERVED_VARIABLES: [&str; 4] = ["bpm", "meter", "key", "scale"];

const BARS: [&str; 2] = ["bpm", "meter"];
const DEGREES: [&str; 2] = ["key", "scale"];

pub fn variable(name: &str) -> String {
    format!("{VARIABLES}/{name}")
}

/// The reserved variables a node's written body, defaults and file span read; a `repeat` or
/// `concat` of a file spanned in bars reads that span.
pub(crate) fn reads(
    body: &Expr,
    defaults: &[(String, Expr)],
    span: Option<FileSpan>,
) -> Vec<&'static str> {
    let mut bars = in_bars(span);
    let mut degrees = false;
    for e in std::iter::once(body).chain(defaults.iter().map(|(_, value)| value)) {
        scan(e, &mut bars, &mut degrees);
    }
    let mut out = Vec::new();
    if bars {
        out.extend(BARS);
    }
    if degrees {
        out.extend(DEGREES);
    }
    out
}

/// The same of a parsed file: a score reads bars where it counts in them or binds one, and
/// degrees where a line counts one.
pub(crate) fn parsed_reads(parsed: &Parsed, span: Option<FileSpan>) -> Vec<&'static str> {
    let Some(Data::Score(score)) = &parsed.data else {
        return reads(&parsed.expr, &parsed.defaults, span);
    };
    let mut bars = in_bars(span);
    let mut degrees = score.reads_degrees();
    for (_, item) in score.items() {
        let lasts = match item {
            Item::Event(event) => {
                for bind in &event.binds {
                    scan(&bind.value, &mut bars, &mut degrees);
                }
                event.len.is_some()
            }
            Item::Move(m) => m.ramp.is_some(),
            _ => false,
        };
        bars |= score.unit == SpanUnit::Bars && (lasts || item.at().amount != Ratio::ZERO);
    }
    let mut out = Vec::new();
    if bars {
        out.extend(BARS);
    }
    if degrees {
        out.extend(DEGREES);
    }
    out
}

fn scan(e: &Expr, bars: &mut bool, degrees: &mut bool) {
    match e {
        Expr::Lit(Literal::Bars(_)) => *bars = true,
        Expr::Call { name, .. } if name == DEGREE => *degrees = true,
        Expr::Call { name, args, .. } if name == "repeat" || name == "concat" => {
            *bars |= args.iter().any(|arg| match arg {
                Arg::Pos(Expr::Ref { path, .. }) => in_bars(spanned(path)),
                _ => false,
            });
        }
        _ => {}
    }
    for c in children(e, Binds::Substitute) {
        scan(c, bars, degrees);
    }
}

fn spanned(path: &str) -> Option<FileSpan> {
    parse_filename(base_name(path)).1
}

fn in_bars(span: Option<FileSpan>) -> bool {
    span.is_some_and(|span| span.unit == SpanUnit::Bars)
}

/// `variables/bpm` and `variables/meter` together: seconds per bar is `beats * 60 / bpm`.
#[derive(Clone, Debug, PartialEq)]
pub struct Tempo {
    pub bpm: f64,
    pub meter: String,
    beats: f64,
}

impl Tempo {
    pub fn seconds_per_bar(&self) -> f64 {
        self.beats * 60.0 / self.bpm
    }

    pub fn beats(&self) -> f64 {
        self.beats
    }

    fn seconds(&self, bars: f64) -> Expr {
        let num = |v: f64| Box::new(Expr::Lit(Literal::Num(v)));
        Expr::Bin(
            BinOp::Mul,
            num(bars),
            Box::new(Expr::Bin(BinOp::Div, num(self.beats * 60.0), num(self.bpm))),
        )
    }
}

/// What lowering reads off each reserved variable; the tonic stays a ref, so only whether it
/// is held.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Values {
    pub tempo: Option<Tempo>,
    pub scale: Option<Scale>,
    tonic: bool,
}

impl Values {
    /// Each reserved variable a node lowered under `other` reads otherwise under these.
    fn changed(&self, other: &Values) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.tempo != other.tempo {
            out.extend(BARS);
        }
        if self.scale != other.scale {
            out.push("scale");
        }
        out
    }
}

fn tempo_refusal(name: &str, reason: String) -> Refusal {
    Refusal::new(variable(name), ByteSpan::at(0), DiagCode::BadTempo, reason)
}

/// An unparsable numerator is refused, not guessed.
fn beats_per_bar(meter: &str) -> Result<f64, Refusal> {
    let numerator = &meter[..meter.find('/').unwrap_or(meter.len())];
    numerator.trim().parse::<f64>().map_err(|e| {
        tempo_refusal(
            "meter",
            format!(
                "meter `{meter}`'s beats-per-bar component `{numerator}` is not a number: {e}; \
                 a meter is `<beats>/<note>`, as `4/4`"
            ),
        )
    })
}

fn tempo(bpm: Option<&Expr>, meter: Option<&Expr>) -> Result<Option<Tempo>, Refusal> {
    match (bpm, meter) {
        (None, None) => Ok(None),
        (Some(Expr::Lit(Literal::Num(bpm))), Some(Expr::Lit(Literal::Str(meter)))) => {
            if *bpm <= 0.0 {
                return Err(tempo_refusal(
                    "bpm",
                    format!("bpm must be positive, got {bpm}"),
                ));
            }
            Ok(Some(Tempo {
                bpm: *bpm,
                meter: meter.clone(),
                beats: beats_per_bar(meter)?,
            }))
        }
        (Some(_), Some(_)) => Err(tempo_refusal(
            "bpm",
            "`variables/bpm` must be a number and `variables/meter` a meter, as `4/4`".to_string(),
        )),
        (Some(_), None) => Err(tempo_refusal(
            "meter",
            "`variables/bpm` is present but `variables/meter` is missing; both or neither"
                .to_string(),
        )),
        (None, Some(_)) => Err(tempo_refusal(
            "bpm",
            "`variables/meter` is present but `variables/bpm` is missing; both or neither"
                .to_string(),
        )),
    }
}

/// The one whole degree a `deg` call names, folded from literals.
pub(crate) fn degree(args: &[Arg]) -> Result<i64, String> {
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

/// `deg(n)`'s value: the tonic `variables/key` holds, read at `t`, times degree n's ratio.
pub(crate) fn tonic_times(key: &str, ratio: f64, span: ByteSpan) -> Expr {
    let tonic = Expr::Ref {
        path: key.to_string(),
        arg: Box::new(Expr::Var("t".to_string())),
        binds: Vec::new(),
        address: Address::Time,
        span,
    };
    Expr::Bin(
        BinOp::Mul,
        Box::new(tonic),
        Box::new(Expr::Lit(Literal::Num(ratio))),
    )
}

/// `variables/key` as a node at `path` spells a ref to it.
pub(crate) fn key_ref(path: &str) -> String {
    format!(
        "{}{}",
        "../".repeat(path.matches('/').count()),
        variable("key")
    )
}

/// A built-in scale's name as written: one word, or words joined by `-`, which an expression
/// reads as subtraction, so `harmonic-minor` is the name it spells.
fn scale_name(written: &Expr) -> Option<String> {
    match written {
        Expr::Var(word) => Some(word.clone()),
        Expr::Bin(BinOp::Sub, left, right) => {
            Some(format!("{}-{}", scale_name(left)?, scale_name(right)?))
        }
        _ => None,
    }
}

fn builtin_names() -> String {
    BUILTIN_SCALES.map(|(name, ..)| name).join(", ")
}

impl Graph {
    /// Each reserved variable's value, read off the node this graph holds, else `beside`'s.
    pub fn values(&self, beside: Option<&Graph>) -> Result<Values, Vec<Refusal>> {
        let held = |path: &str| self.defines(path) || beside.is_some_and(|b| b.defines(path));
        let node = |name: &str| {
            let path = variable(name);
            self.source(&path)
                .or_else(|| beside?.source(&path))
                .map(|d| &d.body)
        };
        let tempo = tempo(node("bpm"), node("meter"));
        let scale = match node("scale") {
            None => Ok(None),
            Some(written) => self
                .scale_at(written, &variable("scale"), beside)
                .map(Some)
                .map_err(|reason| {
                    Refusal::new(
                        variable("scale"),
                        ByteSpan::at(0),
                        DiagCode::BadScale,
                        reason,
                    )
                }),
        };
        match (tempo, scale) {
            (Ok(tempo), Ok(scale)) => Ok(Values {
                tempo,
                scale,
                tonic: held(&variable("key")),
            }),
            (tempo, scale) => Err(tempo.err().into_iter().chain(scale.err()).collect()),
        }
    }

    /// The scale `written` at `referencing` names: a built-in by its name, any other by a bare
    /// ref to its file.
    pub(crate) fn scale_at(
        &self,
        written: &Expr,
        referencing: &str,
        beside: Option<&Graph>,
    ) -> Result<Scale, String> {
        let target = match written {
            Expr::Ref {
                path: to,
                arg,
                binds,
                address: Address::Time,
                ..
            } if binds.is_empty() && **arg == Expr::Var("t".to_string()) => {
                resolve_ref_path(referencing, to)
            }
            _ => {
                return match scale_name(written) {
                    Some(name) => builtin_scale(&name).ok_or_else(|| {
                        format!(
                            "`{referencing}` names `{name}`, which is no built-in scale; the \
                             built-ins are {}, and any other scale is a file of cents named by \
                             a ref, as `@../scales/custom`",
                            builtin_names()
                        )
                    }),
                    None => Err(format!(
                        "`{referencing}` holds `{}`; a scale is a built-in named bare, as \
                         `minor`, or a scale file named by a ref, as `@../scales/custom`",
                        crate::print::render(written)
                    )),
                };
            }
        };
        let found = target
            .as_deref()
            .and_then(|target| self.scale(target).or_else(|| beside?.scale(target)));
        match (target, found) {
            (Some(_), Some(scale)) => Ok(scale.clone()),
            (target, _) => Err(format!(
                "`{referencing}` names {}, and a scale is a file of rows of cents above the \
                 tonic, the last the period, as `scales/custom` would hold them, or a built-in \
                 named bare: {}",
                match target {
                    Some(t) if self.defines(&t) || beside.is_some_and(|b| b.defines(&t)) => {
                        format!("`{t}`, no scale")
                    }
                    Some(t) => format!("`{t}`, which no file holds"),
                    None => "a path above the composition root".to_string(),
                },
                builtin_names()
            )),
        }
    }

    /// Every read of a reserved variable lowered to its value: a `b` literal or span to
    /// seconds, a `deg` call to the tonic's ref times its ratio. Beside a graph lowered before,
    /// each of its nodes that read a variable under another value than it holds now is read
    /// again from its text and lowered here.
    pub fn lower_reserved(&mut self, beside: Option<&Graph>) -> Result<(), Vec<Refusal>> {
        let values = self.values(beside)?;
        let mut refusals = Vec::new();
        if let Some(beside) = beside {
            let stale = |path: &str| match beside.lowered_under(path) {
                None => false,
                Some(was) => {
                    let changed = values.changed(was);
                    beside
                        .reserved_reads(path)
                        .iter()
                        .any(|v| changed.contains(v))
                }
            };
            let stale: Vec<String> = beside
                .paths()
                .filter(|path| !self.defines(path) && stale(path))
                .map(str::to_string)
                .collect();
            for path in stale {
                if let Err(refused) = self.reread(&path, beside) {
                    refusals.push(refused);
                }
            }
        }
        let reading: Vec<String> = self
            .paths()
            .filter(|path| !self.reserved_reads(path).is_empty())
            .map(str::to_string)
            .collect();
        for path in reading {
            let defined = self.defined(&path).expect("a node this graph holds");
            match self.lowered(&path, defined, &values) {
                Ok(lowered) => self.lower(&path, lowered, &values),
                Err(refused) => refusals.extend(refused),
            }
        }
        match refusals.is_empty() {
            true => Ok(()),
            false => Err(refusals),
        }
    }

    /// `defined` at `path` with its reads of a reserved variable lowered under `values`; a
    /// refusal in a score names its line and column.
    pub(crate) fn lowered(
        &self,
        path: &str,
        defined: &Defined,
        values: &Values,
    ) -> Result<Defined, Vec<Refusal>> {
        self.lowered_with(path, defined, values, None)
    }

    /// The same, each `deg(n)`'s ratio read off `ratio` where given: a song whose scale moves
    /// passes degree n's ratio lane there.
    pub(crate) fn lowered_with(
        &self,
        path: &str,
        defined: &Defined,
        values: &Values,
        ratio: Option<&dyn Fn(i64) -> Expr>,
    ) -> Result<Defined, Vec<Refusal>> {
        let reads = match self.defines(path) {
            true => self.reserved_reads(path).to_vec(),
            false => reads(&defined.body, &defined.defaults, None),
        };
        let missing = |name: &str| match name {
            "bpm" | "meter" => values.tempo.is_none().then(|| {
                Refusal::new(
                    path,
                    ByteSpan::at(0),
                    DiagCode::BadTempo,
                    format!(
                        "`{path}` counts in bars, `b`, and this composition declares no tempo; \
                         add `variables/bpm` and `variables/meter`"
                    ),
                )
            }),
            "key" => (!values.tonic).then(|| {
                Refusal::new(
                    path,
                    ByteSpan::at(0),
                    DiagCode::BadScale,
                    format!(
                        "`{path}` calls `deg`, which counts from the tonic `variables/key` \
                         holds, and this composition has none"
                    ),
                )
            }),
            _ => None,
        };
        let mut refusals: Vec<Refusal> = reads.iter().filter_map(|name| missing(name)).collect();
        refusals.dedup_by(|a, b| a.reason == b.reason);
        if !refusals.is_empty() {
            return Err(refusals);
        }
        let text = self.text(path);
        let score = self.score(path).is_some();
        let refuse = |span: ByteSpan, message: String| match (score, text) {
            (true, Some(text)) => crate::score::refused(
                format!("{path}{}", crate::score::EXTENSION),
                text,
                span.start,
                DiagCode::BadScore,
                &message,
            ),
            _ => Refusal::new(path, span, DiagCode::BadScale, message),
        };
        let from = match (score, text) {
            (false, Some(text)) => body_at(text),
            _ => 0,
        };
        let mut lowering = Lowering {
            values,
            key: key_ref(path),
            refuse: &refuse,
            refusals: Vec::new(),
            ratio,
        };
        let body = lowering.lower(&defined.body, from);
        let defaults = defined
            .defaults
            .iter()
            .map(|(name, value)| (name.clone(), lowering.lower(value, 0)))
            .collect();
        match lowering.refusals.is_empty() {
            true => Ok(Defined { body, defaults }),
            false => Err(lowering.refusals),
        }
    }
}

struct Lowering<'a> {
    values: &'a Values,
    key: String,
    refuse: &'a dyn Fn(ByteSpan, String) -> Refusal,
    refusals: Vec<Refusal>,
    ratio: Option<&'a dyn Fn(i64) -> Expr>,
}

impl Lowering<'_> {
    /// `e` with each read lowered; `from` is where `e`'s spans start in its file. A degree it
    /// cannot read is refused and left as written.
    fn lower(&mut self, e: &Expr, from: usize) -> Expr {
        match e {
            Expr::Lit(Literal::Bars(bars)) => self
                .values
                .tempo
                .as_ref()
                .expect("a node counting in bars is refused with no tempo")
                .seconds(*bars),
            Expr::Call { name, args, span } if name == DEGREE && self.ratio.is_some() => {
                match degree(args) {
                    Ok(n) => {
                        let ratio = (self.ratio.expect("matched"))(n);
                        let tonic = tonic_times(&self.key, 1.0, *span);
                        let Expr::Bin(_, tonic, _) = tonic else {
                            unreachable!("a product")
                        };
                        Expr::Bin(BinOp::Mul, tonic, Box::new(ratio))
                    }
                    Err(message) => {
                        let at = ByteSpan::new(span.start + from, span.end + from);
                        self.refusals.push((self.refuse)(at, message));
                        e.clone()
                    }
                }
            }
            Expr::Call { name, args, span } if name == DEGREE => {
                let ratio = degree(args).and_then(|n| match &self.values.scale {
                    None => Err(
                        "`deg` counts in the scale `variables/scale` names, and this \
                         composition has none; write a built-in's name in `variables/scale`, \
                         as `major`, or a ref to a scale file"
                            .to_string(),
                    ),
                    Some(scale) => scale.ratio(n).ok_or_else(|| {
                        format!("`deg` {n} lies so far from the tonic its ratio is no number")
                    }),
                });
                match ratio {
                    Ok(ratio) => tonic_times(&self.key, ratio, *span),
                    Err(message) => {
                        let at = ByteSpan::new(span.start + from, span.end + from);
                        self.refusals.push((self.refuse)(at, message));
                        e.clone()
                    }
                }
            }
            _ => map_children_ok(e, Binds::Substitute, |c| self.lower(c, from)),
        }
    }
}
