// Concern: parses a cost's text, or builds its tree, into leaves, loads and terms, and types it | Non-concern: evaluating it (mod.rs), reading a file (load.rs) | IO: (text or Tree) -> Cost

use sva_ast::{Token, TokenKind, tokenize};

use crate::cli_error::CliError;
use crate::query::{Asked, asked, call};

/// The readings a cost subtracts, each by the rule `residual.rs` states for it.
pub const SUBTRACTS: [&str; 7] = [
    "envelope", "loudness", "bands", "crest", "spectrum", "onsets", "pitch",
];

pub const REDUCERS: [&str; 5] = ["norm", "sum", "dot", "max", "abs"];

/// How many arguments a reducer takes: `dot` a weight vector and a vector, the rest one.
pub fn takes(reducer: &str) -> usize {
    match reducer {
        "dot" => 2,
        _ => 1,
    }
}

/// The name a leaf of a reading is written under, `repr.bands(...)`.
pub const REPR: &str = "repr";

/// Each arithmetic operator by name, as a tree names it.
pub const OPERATORS: [(&str, Op); 4] = [
    ("add", Op::Add),
    ("sub", Op::Sub),
    ("mul", Op::Mul),
    ("div", Op::Div),
];

/// A leaf that reads a JSON file instead of rendering: the cost's one file read.
pub const LOAD_JSON: &str = "load_json";

#[derive(Clone, Debug, PartialEq)]
pub struct Leaf {
    /// The render expression, as `sva-cli render` takes it.
    pub target: String,
    /// A recording handed over as a value, which `target` reads as `sample("<its key>")`.
    pub recording: Option<std::sync::Arc<sva_ast::Recording>>,
    pub asked: Asked,
    pub written: String,
}

/// One step of a jq path: `.name` or `['any key']`, or `[3]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Key(String),
    Index(usize),
}

/// Where a load's JSON comes from: a file and a path into it, as the CLI writes
/// `load_json("<file>", "<jq path>")`, or the value itself, as a page hands it over.
#[derive(Clone, Debug, PartialEq)]
pub enum From {
    File {
        file: String,
        path: Vec<Step>,
        path_text: String,
    },
    Value(serde_json::Value),
}

/// A value the cost reads instead of rendering: read and checked once, before any render.
#[derive(Clone, Debug, PartialEq)]
pub struct Load {
    pub from: From,
    pub written: String,
}

/// What a load held once read: a number, a vector, or a reading the leaf it subtracts from
/// decoded.
#[derive(Clone, Debug, PartialEq)]
pub enum Held {
    Number(f64),
    Vector(Vec<f64>),
    Reading(sva_engine::Output),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Leaf(usize),
    Loaded(usize),
    Num(f64),
    Vector(Vec<f64>),
    Neg(Box<Node>),
    Bin(Op, Box<Node>, Box<Node>),
    Reduce(&'static str, Vec<Node>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub text: String,
    pub sign: f64,
    pub node: Node,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    pub text: String,
    pub leaves: Vec<Leaf>,
    pub loads: Vec<Load>,
    /// Each top-level `+` or `-` term, which the answer reports on its own.
    pub terms: Vec<Term>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Reading(usize),
    /// A load holding a reading, which only a subtraction from a leaf reads.
    Object(usize),
    /// A load not read yet: a number, a vector or a reading.
    Unknown,
    Vector,
    Scalar,
}

fn refused(code: &'static str, message: String) -> CliError {
    CliError::usage(code, message)
}

/// Where a cost's text was written, so a refusal can point into it.
#[derive(Clone, Copy)]
enum Whence<'a> {
    Argv,
    /// A cost file's path; the text is the file's, its comments blanked, so bytes line up.
    File(&'a str),
}

impl Whence<'_> {
    fn refused(self, text: &str, code: &'static str, byte: usize, what: String) -> CliError {
        match self {
            Whence::Argv => refused(code, format!("{what}, at byte {byte} of the cost")),
            Whence::File(file) => {
                let before = &text[..byte.min(text.len())];
                let line = before.matches('\n').count() + 1;
                let column = byte - before.rfind('\n').map_or(0, |at| at + 1) + 1;
                CliError::Located {
                    code,
                    message: format!("{what}, at line {line}, byte {column} of `{file}`"),
                    file: file.to_string(),
                    byte,
                    line,
                    column,
                }
            }
        }
    }
}

pub fn cost(text: &str) -> Result<Cost, CliError> {
    parsed(text, Whence::Argv)
}

/// The cost a file holds, `file` its path in the composition: every `;` comment line and
/// trailing comment is no part of it, and it may run over many lines.
pub fn cost_file(file: &str, content: &str) -> Result<Cost, CliError> {
    let blanked: Vec<String> = content
        .split('\n')
        .map(|line| {
            let code = sva_ast::strip_line_comment(line);
            format!("{code}{}", " ".repeat(line.len() - code.len()))
        })
        .collect();
    let text = blanked.join("\n");
    if text.trim().is_empty() {
        return Err(refused(
            "solve.bad_cost",
            format!("`{file}` holds no cost, only comments; write one under its annotation line"),
        ));
    }
    let mut cost = parsed(&text, Whence::File(file))?;
    let spaced = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
    cost.text = spaced(&cost.text);
    for term in &mut cost.terms {
        term.text = spaced(&term.text);
    }
    Ok(cost)
}

fn parsed(text: &str, whence: Whence) -> Result<Cost, CliError> {
    let tokens = tokenize(text).map_err(|d| {
        whence.refused(
            text,
            "solve.bad_cost",
            d.span.start,
            format!("the cost does not read: {}", d.message),
        )
    })?;
    let mut parser = Parser {
        whence,
        text,
        tokens: &tokens,
        at: 0,
        leaves: Vec::new(),
        loads: Vec::new(),
        terms: Vec::new(),
    };
    let (node, ..) = parser.sum(true)?;
    if let Some(extra) = tokens.get(parser.at) {
        return Err(parser.unexpected(extra, "the cost ends here"));
    }
    let Parser {
        leaves,
        loads,
        terms,
        ..
    } = parser;
    assembled(text.trim().to_string(), node, terms, leaves, loads)
}

/// One cost: each top-level term reported apart where every term types to one number, else
/// the whole as one term, then typed. The text and tree forms both end here.
fn assembled(
    text: String,
    node: Node,
    terms: Vec<Term>,
    leaves: Vec<Leaf>,
    loads: Vec<Load>,
) -> Result<Cost, CliError> {
    let mut cost = Cost {
        text: text.clone(),
        leaves,
        loads,
        terms,
    };
    let typing = Typing {
        leaves: &cost.leaves,
        loads: &cost.loads,
        held: None,
    };
    let split = cost.terms.iter().all(|t| {
        typing
            .kind(&t.node)
            .is_ok_and(|k| matches!(k, Kind::Scalar | Kind::Unknown))
    });
    if !split || cost.terms.is_empty() {
        cost.terms = vec![Term {
            text,
            sign: 1.0,
            node,
        }];
    }
    cost.typed(None)?;
    Ok(cost)
}

impl Cost {
    /// Every term one number: before the loads are read, `held` is `None` and a load may be
    /// anything; once read, each is what it held.
    pub fn typed(&self, held: Option<&[Held]>) -> Result<(), CliError> {
        let typing = Typing {
            leaves: &self.leaves,
            loads: &self.loads,
            held,
        };
        for term in &self.terms {
            typing.term(term)?;
        }
        Ok(())
    }

    /// The leaf each load subtracts from or is subtracted from, where it is one.
    pub fn paired(&self) -> Vec<Option<usize>> {
        let mut out = vec![None; self.loads.len()];
        for term in &self.terms {
            pairs(&term.node, &mut out);
        }
        out
    }
}

fn pairs(node: &Node, out: &mut [Option<usize>]) {
    match node {
        Node::Bin(Op::Sub, a, b) => match (&**a, &**b) {
            (Node::Leaf(x), Node::Loaded(j)) | (Node::Loaded(j), Node::Leaf(x)) => {
                out[*j] = Some(*x);
            }
            _ => {
                pairs(a, out);
                pairs(b, out);
            }
        },
        Node::Bin(_, a, b) => {
            pairs(a, out);
            pairs(b, out);
        }
        Node::Neg(x) => pairs(x, out),
        Node::Reduce(_, args) => args.iter().for_each(|a| pairs(a, out)),
        Node::Leaf(_) | Node::Loaded(_) | Node::Num(_) | Node::Vector(_) => {}
    }
}

struct Typing<'c> {
    leaves: &'c [Leaf],
    loads: &'c [Load],
    held: Option<&'c [Held]>,
}

impl Typing<'_> {
    fn term(&self, term: &Term) -> Result<(), CliError> {
        match self.kind(&term.node)? {
            Kind::Scalar | Kind::Unknown => {}
            Kind::Vector => {
                return Err(refused(
                    "solve.vector_cost",
                    format!(
                        "`{}` is a vector, and a cost is one number; reduce it, as \
                         `norm(...)`, `sum(abs(...))`, `max(abs(...))` or `dot([w1, w2], ...)`",
                        term.text
                    ),
                ));
            }
            Kind::Reading(leaf) => return Err(unpaired(&self.leaves[leaf])),
            Kind::Object(load) => return Err(unpaired_load(&self.loads[load])),
        }
        Ok(())
    }

    fn kind(&self, node: &Node) -> Result<Kind, CliError> {
        let leaves = self.leaves;
        let loaded = |k: Kind| match k {
            Kind::Reading(at) => Err(unpaired(&leaves[at])),
            Kind::Object(at) => Err(unpaired_load(&self.loads[at])),
            other => Ok(other),
        };
        Ok(match node {
            Node::Leaf(at) => Kind::Reading(*at),
            Node::Loaded(at) => match self.held.map(|held| &held[*at]) {
                None => Kind::Unknown,
                Some(Held::Number(_)) => Kind::Scalar,
                Some(Held::Vector(_)) => Kind::Vector,
                Some(Held::Reading(_)) => Kind::Object(*at),
            },
            Node::Num(_) => Kind::Scalar,
            Node::Vector(_) => Kind::Vector,
            Node::Neg(x) => loaded(self.kind(x)?)?,
            Node::Bin(op, a, b) => match (self.kind(a)?, self.kind(b)?) {
                (Kind::Reading(x), Kind::Reading(y)) if *op == Op::Sub => {
                    let (l, r) = (&leaves[x], &leaves[y]);
                    if l.asked != r.asked {
                        return Err(refused(
                            "solve.mismatched_readings",
                            format!(
                                "`{}` and `{}` are not one reading with one set of arguments, \
                                 so they do not subtract; read both sides alike, as `repr.{}(@a) \
                                 - repr.{}(sample(\"../ref.wav\"))`",
                                l.written, r.written, l.asked.name, l.asked.name
                            ),
                        ));
                    }
                    Kind::Vector
                }
                (Kind::Reading(_), Kind::Object(_) | Kind::Unknown)
                | (Kind::Object(_) | Kind::Unknown, Kind::Reading(_))
                    if *op == Op::Sub =>
                {
                    Kind::Vector
                }
                (Kind::Reading(at), _) | (_, Kind::Reading(at)) => {
                    return Err(unpaired(&leaves[at]));
                }
                (Kind::Object(at), _) | (_, Kind::Object(at)) => {
                    return Err(unpaired_load(&self.loads[at]));
                }
                (Kind::Unknown, _) | (_, Kind::Unknown) => Kind::Unknown,
                (Kind::Scalar, Kind::Scalar) => Kind::Scalar,
                (Kind::Vector, Kind::Vector) if matches!(op, Op::Add | Op::Sub) => Kind::Vector,
                (Kind::Vector, Kind::Scalar) | (Kind::Scalar, Kind::Vector)
                    if matches!(op, Op::Mul) =>
                {
                    Kind::Vector
                }
                (Kind::Vector, Kind::Scalar) if *op == Op::Div => Kind::Vector,
                _ => {
                    return Err(refused(
                        "solve.bad_cost",
                        "a vector adds or subtracts only a vector, and scales only by a number; \
                         reduce it first, as `norm(...)`"
                            .to_string(),
                    ));
                }
            },
            Node::Reduce(name, args) => {
                let kinds = args
                    .iter()
                    .map(|a| self.kind(a).and_then(loaded))
                    .collect::<Result<Vec<_>, _>>()?;
                let vector = |k: &Kind| matches!(k, Kind::Vector | Kind::Unknown);
                match (*name, kinds.as_slice()) {
                    ("abs", [one]) => *one,
                    ("dot", [w, v]) if vector(w) && vector(v) => Kind::Scalar,
                    ("norm" | "sum" | "max", [v]) if vector(v) => Kind::Scalar,
                    _ => {
                        return Err(refused(
                            "solve.bad_cost",
                            format!(
                                "`{name}` takes {}",
                                match *name {
                                    "dot" => "two vectors, `dot([1, 0.5], v)`",
                                    "abs" => "one number or vector",
                                    _ => "one vector, as a subtraction of two readings",
                                }
                            ),
                        ));
                    }
                }
            }
        })
    }
}

fn unpaired(leaf: &Leaf) -> CliError {
    refused(
        "solve.unpaired_reading",
        format!(
            "`{}` is a reading, which a cost subtracts from another of its kind before reducing, \
             as `norm({} - repr.{}(sample(\"../ref.wav\")))`",
            leaf.written, leaf.written, leaf.asked.name
        ),
    )
}

/// A leaf's reading, `call` as `--representation` spells it, with only its own arguments.
fn own_arguments(call_text: &str, written: &str) -> Result<Asked, CliError> {
    let asked = asked(&call(call_text)?)?;
    if !SUBTRACTS.contains(&asked.name.as_str()) {
        return Err(refused(
            "solve.unsupported_reading",
            format!(
                "`{}` is no reading a cost subtracts; it subtracts {}",
                asked.name,
                SUBTRACTS.join(", ")
            ),
        ));
    }
    if asked.node.is_some() || asked.dest.is_some() || asked.brief || asked.skim {
        return Err(refused(
            "solve.bad_cost",
            format!("`{written}` takes only its reading's own arguments"),
        ));
    }
    Ok(asked)
}

fn unpaired_load(load: &Load) -> CliError {
    refused(
        "solve.unpaired_load",
        format!(
            "`{}` holds a reading, which a cost subtracts from a `repr.` leaf of that reading \
             before reducing, as `norm(repr.bands(@a) - {})`; a number or a list of numbers \
             loads as a weight",
            load.written, load.written
        ),
    )
}

struct Parser<'a> {
    whence: Whence<'a>,
    text: &'a str,
    tokens: &'a [Token],
    at: usize,
    leaves: Vec<Leaf>,
    loads: Vec<Load>,
    terms: Vec<Term>,
}

type Spanned = (Node, usize, usize);

impl Parser<'_> {
    fn peek(&self) -> Option<&TokenKind> {
        self.tokens.get(self.at).map(|t| &t.kind)
    }

    fn end(&self) -> usize {
        self.at
            .checked_sub(1)
            .map_or(0, |at| self.tokens[at].span.end)
    }

    fn unexpected(&self, token: &Token, wanted: &str) -> CliError {
        self.whence.refused(
            self.text,
            "solve.bad_cost",
            token.span.start,
            format!(
                "`{}`: {wanted}; a cost is `repr.<reading>(<target>, <args>)` and \
                 `load_json(\"<file>\", \"<jq path>\")` leaves, subtracted in pairs, reduced \
                 by {} and weighted by numbers",
                &self.text[token.span.start..token.span.end],
                REDUCERS.join(", ")
            ),
        )
    }

    fn ended(&self, wanted: &str) -> CliError {
        self.whence.refused(
            self.text,
            "solve.bad_cost",
            self.text.trim_end().len(),
            format!("the cost ends where {wanted}"),
        )
    }

    fn expect(&mut self, kind: &TokenKind, wanted: &str) -> Result<(), CliError> {
        match self.tokens.get(self.at) {
            Some(t) if &t.kind == kind => {
                self.at += 1;
                Ok(())
            }
            Some(t) => Err(self.unexpected(t, wanted)),
            None => Err(self.ended(wanted)),
        }
    }

    /// `top` records each term, for the answer to report one by one.
    fn sum(&mut self, top: bool) -> Result<Spanned, CliError> {
        let first = self.product()?;
        let start = first.1;
        let mut parts = vec![(1.0, first)];
        while let Some(sign) = self.peek().and_then(|k| match k {
            TokenKind::Plus => Some(1.0),
            TokenKind::Minus => Some(-1.0),
            _ => None,
        }) {
            self.at += 1;
            parts.push((sign, self.product()?));
        }
        let end = parts.last().map_or(start, |(_, (_, _, e))| *e);
        if top && parts.len() > 1 {
            self.terms = parts
                .iter()
                .map(|(sign, (node, s, e))| Term {
                    text: self.text[*s..*e].trim().to_string(),
                    sign: *sign,
                    node: node.clone(),
                })
                .collect();
        }
        let mut parts = parts.into_iter();
        let (_, (mut node, ..)) = parts.next().expect("a first operand");
        for (sign, (part, ..)) in parts {
            let op = if sign > 0.0 { Op::Add } else { Op::Sub };
            node = Node::Bin(op, Box::new(node), Box::new(part));
        }
        Ok((node, start, end))
    }

    fn product(&mut self) -> Result<Spanned, CliError> {
        let (mut node, start, mut end) = self.unary()?;
        while let Some(op) = self.peek().and_then(|k| match k {
            TokenKind::Star => Some(Op::Mul),
            TokenKind::Slash => Some(Op::Div),
            _ => None,
        }) {
            self.at += 1;
            let (rhs, _, e) = self.unary()?;
            end = e;
            node = Node::Bin(op, Box::new(node), Box::new(rhs));
        }
        Ok((node, start, end))
    }

    fn unary(&mut self) -> Result<Spanned, CliError> {
        let start = self.tokens.get(self.at).map_or(0, |t| t.span.start);
        if self.peek() == Some(&TokenKind::Minus) {
            self.at += 1;
            let (x, _, end) = self.unary()?;
            return Ok((Node::Neg(Box::new(x)), start, end));
        }
        let node = self.primary()?;
        Ok((node, start, self.end()))
    }

    fn primary(&mut self) -> Result<Node, CliError> {
        let Some(token) = self.tokens.get(self.at) else {
            return Err(self.ended("a value was expected"));
        };
        self.at += 1;
        match &token.kind {
            TokenKind::Num(n) => Ok(Node::Num(*n)),
            TokenKind::LParen => {
                let (inner, ..) = self.sum(false)?;
                self.expect(&TokenKind::RParen, "a `(` closes here")?;
                Ok(inner)
            }
            TokenKind::LBracket => self.vector(),
            TokenKind::Ident(name) if name == REPR => self.leaf(token.span.start),
            TokenKind::Ident(name) if name == LOAD_JSON => self.load(token.span.start),
            TokenKind::Ident(name) => match REDUCERS.iter().find(|r| **r == name) {
                Some(reducer) => {
                    self.expect(&TokenKind::LParen, "a reducer is called, as `norm(...)`")?;
                    let mut args = vec![self.sum(false)?.0];
                    while self.peek() == Some(&TokenKind::Comma) {
                        self.at += 1;
                        args.push(self.sum(false)?.0);
                    }
                    self.expect(&TokenKind::RParen, "its `)` closes here")?;
                    Ok(Node::Reduce(reducer, args))
                }
                None => Err(self.unexpected(token, "no such reducer or leaf")),
            },
            _ => Err(self.unexpected(token, "a value was expected")),
        }
    }

    fn vector(&mut self) -> Result<Node, CliError> {
        let mut out = Vec::new();
        loop {
            let sign = match self.peek() {
                Some(TokenKind::Minus) => {
                    self.at += 1;
                    -1.0
                }
                _ => 1.0,
            };
            match self.tokens.get(self.at) {
                Some(Token {
                    kind: TokenKind::Num(n),
                    ..
                }) => out.push(sign * n),
                Some(t) => return Err(self.unexpected(t, "a weight vector holds numbers")),
                None => return Err(self.ended("a weight vector's `]` was expected")),
            }
            self.at += 1;
            match self.peek() {
                Some(TokenKind::Comma) => self.at += 1,
                Some(TokenKind::RBracket) => {
                    self.at += 1;
                    return Ok(Node::Vector(out));
                }
                _ => return Err(self.ended("a weight vector's `,` or `]` was expected")),
            }
        }
    }

    fn path_text(&mut self, wanted: &str) -> Result<(String, usize), CliError> {
        match self.tokens.get(self.at) {
            Some(Token {
                kind: TokenKind::Str(text),
                span,
            }) => {
                self.at += 1;
                Ok((text.clone(), span.start))
            }
            Some(t) => Err(self.unexpected(t, wanted)),
            None => Err(self.ended(wanted)),
        }
    }

    /// `load_json("<file>", "<jq path>")`.
    fn load(&mut self, start: usize) -> Result<Node, CliError> {
        let wanted = "`load_json` takes a file and a jq path, as `load_json(\"crop.json\", \
                      \".readings['bands(prune=1)']\")`";
        self.expect(&TokenKind::LParen, wanted)?;
        let (file, _) = self.path_text(wanted)?;
        self.expect(&TokenKind::Comma, wanted)?;
        let (path_text, at) = self.path_text(wanted)?;
        self.expect(&TokenKind::RParen, wanted)?;
        let written = self.text[start..self.end()].to_string();
        let path = jq_path(&path_text).map_err(|why| {
            self.whence.refused(
                self.text,
                "solve.bad_path",
                at,
                format!(
                    "`{path_text}` is no jq path: {why}; a path is only `.name` steps, `[3]` \
                     indices and `['any key']` for a key that is not a plain name, as \
                     `.readings['bands(prune=1)']`, and none of jq's filters"
                ),
            )
        })?;
        self.loads.push(Load {
            from: From::File {
                file,
                path,
                path_text,
            },
            written,
        });
        Ok(Node::Loaded(self.loads.len() - 1))
    }

    /// `repr.<reading>(<target>, <k>=<v>...)`: the target is the text up to the first comma
    /// outside every bracket, so an interval `[0, 2s)` stays in it.
    fn leaf(&mut self, start: usize) -> Result<Node, CliError> {
        self.expect(&TokenKind::Dot, "`repr` is followed by `.<reading>`")?;
        let name = match self.tokens.get(self.at).map(|t| &t.kind) {
            Some(TokenKind::Ident(name)) => name.clone(),
            _ => return Err(self.ended("a reading's name follows `repr.`")),
        };
        self.at += 1;
        if !SUBTRACTS.contains(&name.as_str()) {
            return Err(refused(
                "solve.unsupported_reading",
                format!(
                    "`repr.{name}` is no reading a cost subtracts; it subtracts {}",
                    SUBTRACTS.join(", ")
                ),
            ));
        }
        let open = self.at;
        self.expect(&TokenKind::LParen, "a reading is called on its target")?;
        let mut depth = 1;
        let mut pieces = Vec::new();
        let mut from = self.tokens[open].span.end;
        while depth > 0 {
            let Some(token) = self.tokens.get(self.at) else {
                return Err(self.ended(&format!("`repr.{name}(`'s `)` was expected")));
            };
            match token.kind {
                TokenKind::LParen | TokenKind::LBracket => depth += 1,
                TokenKind::RParen | TokenKind::RBracket => depth -= 1,
                TokenKind::Comma if depth == 1 => {
                    pieces.push(self.text[from..token.span.start].trim().to_string());
                    from = token.span.end;
                }
                _ => {}
            }
            if depth == 0 {
                pieces.push(self.text[from..token.span.start].trim().to_string());
            }
            self.at += 1;
        }
        let written = self.text[start..self.end()].to_string();
        let mut pieces = pieces.into_iter();
        let target = pieces.next().filter(|t| !t.is_empty()).ok_or_else(|| {
            refused(
                "solve.bad_cost",
                format!("`{written}` reads no target; write one, as `repr.{name}(@master)`"),
            )
        })?;
        let args: Vec<String> = pieces.collect();
        let asked = own_arguments(&format!("{name}({})", args.join(", ")), &written)?;
        self.leaves.push(Leaf {
            target,
            recording: None,
            asked,
            written,
        });
        Ok(Node::Leaf(self.leaves.len() - 1))
    }
}

/// `.a.b`, `[3]`, `['any key']` or `.['any key']`; `.` alone is the whole document.
pub fn jq_path(text: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    let mut rest = text.trim();
    if rest == "." {
        return Ok(steps);
    }
    if !rest.starts_with(['.', '[']) {
        return Err("it does not start with `.` or `[`".to_string());
    }
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('.') {
            if after.starts_with('[') {
                rest = after;
                continue;
            }
            let end = after
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(after.len());
            let name = &after[..end];
            if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
                return Err(format!(
                    "`.{}` names no key; write one that is not a plain name in brackets, \
                     `['{}']`",
                    after.split(['.', '[']).next().unwrap_or(after),
                    after.split(['.', '[']).next().unwrap_or(after)
                ));
            }
            steps.push(Step::Key(name.to_string()));
            rest = &after[end..];
        } else if let Some(after) = rest.strip_prefix("['") {
            let Some(close) = after.find("']") else {
                return Err(format!("`['{after}` is never closed by `']`"));
            };
            steps.push(Step::Key(after[..close].to_string()));
            rest = &after[close + 2..];
        } else if let Some(after) = rest.strip_prefix('[') {
            let Some(close) = after.find(']') else {
                return Err(format!("`[{after}` is never closed by `]`"));
            };
            let index = after[..close].trim().parse::<usize>().map_err(|_| {
                format!(
                    "`[{}]` is no index from 0; a key goes in single quotes, `['{}']`",
                    &after[..close],
                    &after[..close]
                )
            })?;
            steps.push(Step::Index(index));
            rest = &after[close + 1..];
        } else {
            return Err(format!("`{rest}` is no `.name`, `[3]` or `['key']` step"));
        }
    }
    Ok(steps)
}

/// What a leaf reads: a render expression, or a recording handed over as its samples.
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    Text(String),
    Samples { rate: u32, planes: Vec<Vec<f32>> },
}

/// A cost as a tree, as a front end with no text to parse builds one: each node one form,
/// every reference a value. [`built`] makes the same [`Cost`] the text grammar does.
#[derive(Clone, Debug, PartialEq)]
pub enum Tree {
    Num(f64),
    Weights(Vec<f64>),
    /// `call` as `--representation` spells it, `bands(prune=1)`.
    Repr {
        call: String,
        target: Target,
    },
    Value(serde_json::Value),
    Neg(Box<Tree>),
    Bin(Op, Box<Tree>, Box<Tree>),
    /// One of [`REDUCERS`], with the arguments [`takes`] says.
    Reduce(&'static str, Vec<Tree>),
}

/// `node` as the text grammar writes it, an operand of a sum or product in parentheses.
fn printed(node: &Node, leaves: &[Leaf], loads: &[Load]) -> String {
    let whole = |n: &Node| printed(n, leaves, loads);
    let each = |n: &Node| match n {
        Node::Bin(..) => format!("({})", whole(n)),
        _ => whole(n),
    };
    match node {
        Node::Leaf(at) => leaves[*at].written.clone(),
        Node::Loaded(at) => loads[*at].written.clone(),
        Node::Num(n) => format!("{n}"),
        Node::Vector(v) => {
            let v: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
            format!("[{}]", v.join(", "))
        }
        Node::Neg(x) => format!("-({})", each(x)),
        Node::Bin(op, a, b) => {
            let op = match op {
                Op::Add => "+",
                Op::Sub => "-",
                Op::Mul => "*",
                Op::Div => "/",
            };
            format!("{} {op} {}", each(a), each(b))
        }
        Node::Reduce(name, args) => {
            let args: Vec<String> = args.iter().map(whole).collect();
            format!("{name}({})", args.join(", "))
        }
    }
}

struct Building {
    leaves: Vec<Leaf>,
    loads: Vec<Load>,
}

impl Building {
    fn node(&mut self, tree: Tree) -> Result<Node, CliError> {
        Ok(match tree {
            Tree::Num(n) => Node::Num(n),
            Tree::Weights(v) => Node::Vector(v),
            Tree::Neg(x) => Node::Neg(Box::new(self.node(*x)?)),
            Tree::Bin(op, a, b) => {
                Node::Bin(op, Box::new(self.node(*a)?), Box::new(self.node(*b)?))
            }
            Tree::Reduce(reducer, args) => {
                let args = args
                    .into_iter()
                    .map(|a| self.node(a))
                    .collect::<Result<Vec<_>, _>>()?;
                Node::Reduce(reducer, args)
            }
            Tree::Value(value) => {
                let shape = crate::decode::shape(&value);
                self.loads.push(Load {
                    written: format!("value {} ({shape})", self.loads.len()),
                    from: From::Value(value),
                });
                Node::Loaded(self.loads.len() - 1)
            }
            Tree::Repr { call, target } => {
                let (target, recording) = match target {
                    Target::Text(text) if text.trim().is_empty() => {
                        return Err(refused(
                            "solve.bad_cost",
                            format!("`{call}` reads no target; give it one, as `@master`"),
                        ));
                    }
                    Target::Text(text) => (text, None),
                    Target::Samples { rate, planes } => {
                        let key = format!("#recording-{}", self.leaves.len());
                        let recording = sva_ast::Recording {
                            path: key.clone(),
                            rate,
                            planes: std::sync::Arc::new(planes),
                        };
                        (
                            format!("sample(\"{key}\")"),
                            Some(std::sync::Arc::new(recording)),
                        )
                    }
                };
                let shown = match &recording {
                    Some(r) => format!("<recording at {} Hz>", r.rate),
                    None => target.clone(),
                };
                let probe = self::call(&call)?;
                asked(&probe)?;
                let args: Vec<String> =
                    probe.args.iter().map(|(k, v)| format!("{k}={v}")).collect();
                let args = args.join(", ");
                let written = match args.is_empty() {
                    true => format!("repr.{}({shown})", probe.name),
                    false => format!("repr.{}({shown}, {args})", probe.name),
                };
                let asked = own_arguments(&call, &written)?;
                self.leaves.push(Leaf {
                    target,
                    recording,
                    asked,
                    written,
                });
                Node::Leaf(self.leaves.len() - 1)
            }
        })
    }
}

/// Each top-level `+` or `-` operand, with its sign, as the text grammar splits them.
fn split(node: Node, out: &mut Vec<(f64, Node)>) {
    match node {
        Node::Bin(Op::Add, a, b) => {
            split(*a, out);
            out.push((1.0, *b));
        }
        Node::Bin(Op::Sub, a, b) => {
            split(*a, out);
            out.push((-1.0, *b));
        }
        other => out.push((1.0, other)),
    }
}

/// `tree` as the [`Cost`] its text would parse to, typed by the same rules.
pub fn built(tree: Tree) -> Result<Cost, CliError> {
    let mut building = Building {
        leaves: Vec::new(),
        loads: Vec::new(),
    };
    let node = building.node(tree)?;
    let Building { leaves, loads } = building;
    let text = printed(&node, &leaves, &loads);
    let mut parts = Vec::new();
    split(node.clone(), &mut parts);
    let terms = match parts.len() {
        1 => Vec::new(),
        _ => parts
            .into_iter()
            .map(|(sign, node)| Term {
                text: printed(&node, &leaves, &loads),
                sign,
                node,
            })
            .collect(),
    };
    assembled(text, node, terms, leaves, loads)
}
