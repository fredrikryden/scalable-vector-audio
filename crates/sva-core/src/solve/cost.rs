// Concern: parses and types a solve's cost: `repr.` leaves, subtractions, reducers, weights | Non-concern: evaluating it (residual.rs, mod.rs) | IO: (text) -> Cost or CliError

use sva_ast::{Token, TokenKind, tokenize};

use crate::cli_error::CliError;
use crate::query::{Asked, asked, call};

/// The readings a cost subtracts, each by the rule `residual.rs` states for it.
pub const SUBTRACTS: [&str; 7] = [
    "envelope", "loudness", "bands", "crest", "spectrum", "onsets", "pitch",
];

pub const REDUCERS: [&str; 5] = ["norm", "sum", "dot", "max", "abs"];

#[derive(Clone, Debug, PartialEq)]
pub struct Leaf {
    /// The render expression, as `sva-cli render` takes it.
    pub target: String,
    pub asked: Asked,
    pub written: String,
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
    /// Each top-level `+` or `-` term, which the answer reports on its own.
    pub terms: Vec<Term>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Reading(usize),
    Vector,
    Scalar,
}

fn refused(code: &'static str, message: String) -> CliError {
    CliError::usage(code, message)
}

pub fn cost(text: &str) -> Result<Cost, CliError> {
    let tokens = tokenize(text).map_err(|d| {
        refused(
            "solve.bad_cost",
            format!("`{text}` does not read as a cost: {}", d.message),
        )
    })?;
    let mut parser = Parser {
        text,
        tokens: &tokens,
        at: 0,
        leaves: Vec::new(),
        terms: Vec::new(),
    };
    let (node, ..) = parser.sum(true)?;
    if let Some(extra) = tokens.get(parser.at) {
        return Err(parser.unexpected(extra, "the cost ends here"));
    }
    let Parser { leaves, terms, .. } = parser;
    let mut cost = Cost {
        text: text.to_string(),
        leaves,
        terms,
    };
    let split = cost
        .terms
        .iter()
        .all(|t| kind(&t.node, &cost.leaves).is_ok_and(|k| k == Kind::Scalar));
    if !split || cost.terms.is_empty() {
        cost.terms = vec![Term {
            text: text.trim().to_string(),
            sign: 1.0,
            node,
        }];
    }
    for term in &cost.terms {
        match kind(&term.node, &cost.leaves)? {
            Kind::Scalar => {}
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
            Kind::Reading(leaf) => return Err(unpaired(&cost.leaves[leaf])),
        }
    }
    Ok(cost)
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

fn kind(node: &Node, leaves: &[Leaf]) -> Result<Kind, CliError> {
    Ok(match node {
        Node::Leaf(at) => Kind::Reading(*at),
        Node::Num(_) => Kind::Scalar,
        Node::Vector(_) => Kind::Vector,
        Node::Neg(x) => match kind(x, leaves)? {
            Kind::Reading(at) => return Err(unpaired(&leaves[at])),
            other => other,
        },
        Node::Bin(op, a, b) => match (kind(a, leaves)?, kind(b, leaves)?) {
            (Kind::Reading(x), Kind::Reading(y)) if *op == Op::Sub => {
                let (l, r) = (&leaves[x], &leaves[y]);
                if l.asked != r.asked {
                    return Err(refused(
                        "solve.mismatched_readings",
                        format!(
                            "`{}` and `{}` are not one reading with one set of arguments, so \
                             they do not subtract; read both sides alike, as `repr.{}(@a) - \
                             repr.{}(sample(\"../ref.wav\"))`",
                            l.written, r.written, l.asked.name, l.asked.name
                        ),
                    ));
                }
                Kind::Vector
            }
            (Kind::Reading(at), _) | (_, Kind::Reading(at)) => {
                return Err(unpaired(&leaves[at]));
            }
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
                .map(|a| kind(a, leaves))
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(Kind::Reading(at)) = kinds.iter().find(|k| matches!(k, Kind::Reading(_))) {
                return Err(unpaired(&leaves[*at]));
            }
            match (*name, kinds.as_slice()) {
                ("abs", [one]) => *one,
                ("dot", [Kind::Vector, Kind::Vector]) => Kind::Scalar,
                ("norm" | "sum" | "max", [Kind::Vector]) => Kind::Scalar,
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

struct Parser<'a> {
    text: &'a str,
    tokens: &'a [Token],
    at: usize,
    leaves: Vec<Leaf>,
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
        refused(
            "solve.bad_cost",
            format!(
                "`{}` at byte {} of the cost: {wanted}; a cost is `repr.<reading>(<target>, \
                 <args>)` leaves, subtracted in pairs, reduced by {} and weighted by numbers",
                &self.text[token.span.start..token.span.end],
                token.span.start,
                REDUCERS.join(", ")
            ),
        )
    }

    fn ended(&self, wanted: &str) -> CliError {
        refused(
            "solve.bad_cost",
            format!("the cost `{}` ends where {wanted}", self.text),
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
            TokenKind::Ident(name) if name == "repr" => self.leaf(token.span.start),
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
        let one = call(&format!("{name}({})", args.join(", ")))?;
        let asked = asked(&one)?;
        if asked.node.is_some() || asked.dest.is_some() || asked.brief || asked.skim {
            return Err(refused(
                "solve.bad_cost",
                format!("`{written}` takes only its reading's own arguments"),
            ));
        }
        self.leaves.push(Leaf {
            target,
            asked,
            written,
        });
        Ok(Node::Leaf(self.leaves.len() - 1))
    }
}
