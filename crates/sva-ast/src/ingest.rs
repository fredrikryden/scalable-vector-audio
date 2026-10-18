// Concern: classifies one file's raw content and parses it, a score's lines too | Non-concern: walking the directory, resolving refs (graph.rs) | IO: (path, content) -> Parsed or a Diag

use crate::diag::{ByteSpan, Diag, DiagCode};
use crate::expr::{Arg, Binds, Expr, Literal, SERIES, children};
use crate::parser::parse;
use crate::scale::{Scale, is_scale, parse_scale};
use crate::score::{EXTENSION, Score, is_old_score, parse_score};

/// What a file holds beside the expression every other pass reads.
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    Score(Score),
    Scale(Scale),
}

#[derive(Debug)]
pub struct Parsed {
    pub expr: Expr,
    pub data: Option<Data>,
    pub defaults: Vec<(String, Expr)>,
}

/// A leading `<name> = <expr>` line states what a caller that says nothing about `<name>` gets.
/// Everything after the last one is the body, read exactly as a file with no defaults is.
pub fn parse_file(content: &str) -> Result<Parsed, Diag> {
    if is_old_score(content) {
        return Err(Diag::new(
            DiagCode::BadScore,
            ByteSpan::at(0),
            format!(
                "this file opens with `{{`, a 0.13 score; a score is now a `{EXTENSION}` file: \
                 rename it with `{EXTENSION}`, make its leading `;` doc comment the line-1 \
                 annotation, {{\"annotation\": \"Models: ... | Neglects: ... | IO: ... | Tags: \
                 ...\"}}, and every other `;` line a timed comment, {{\"at\": \"4b\", \"comment\": \
                 \"...\"}}, or a `comment` key on the line it explains"
            ),
        ));
    }
    let rows = code_rows(content);
    let split = default_rows(&rows);
    let (heads, _) = rows.split_at(split);
    let body_at = rows.get(split).map_or(content.len(), |(at, _)| *at);
    let defaults = read_defaults(heads)?;

    let mut parsed = parse_body(&content[body_at..]).map_err(|d| shift(d, body_at))?;
    parsed.expr = crate::series::written_out(&parsed.expr).map_err(|d| shift(d, body_at))?;
    // Liveness runs back from the body: only a live default's own reads count.
    let mut live = vec![false; defaults.len()];
    for at in (0..defaults.len()).rev() {
        let name = &defaults[at].0;
        live[at] = occurs_free(&parsed.expr, name)
            || defaults[at + 1..]
                .iter()
                .enumerate()
                .any(|(k, (_, value))| live[at + 1 + k] && occurs_free(value, name));
    }
    for (at, (name, _)) in defaults.iter().enumerate() {
        if !live[at] {
            return Err(Diag::new(
                DiagCode::BadDefault,
                ByteSpan::new(0, body_at),
                format!("`{name}` has a default but the body never reads it"),
            ));
        }
    }
    parsed.defaults = defaults;
    Ok(parsed)
}

/// A `.jsonl` file at `path`: its lines, and the files they name as the body loading follows.
pub fn parse_score_file(path: &str, content: &str) -> Result<Parsed, Diag> {
    let score = parse_score(content)?;
    Ok(Parsed {
        expr: score.reads(path),
        data: Some(Data::Score(score)),
        defaults: Vec::new(),
    })
}

/// The byte the body starts at: past the comment and default lines that lead a node's text.
pub fn body_at(content: &str) -> usize {
    let rows = code_rows(content);
    rows.get(default_rows(&rows))
        .map_or(content.len(), |(at, _)| *at)
}

/// Each default line's name and the text of its value, as written.
pub(crate) fn default_texts(content: &str) -> Vec<(String, String)> {
    let rows = code_rows(content);
    let (heads, _) = rows.split_at(default_rows(&rows));
    heads
        .iter()
        .map(|(_, line)| {
            let eq = line.find('=').expect("a default line holds one");
            (
                line[..eq].trim().to_string(),
                line[eq + 1..].trim().to_string(),
            )
        })
        .collect()
}

fn default_rows(rows: &[(usize, &str)]) -> usize {
    rows.iter()
        .position(|(_, line)| !is_default_line(line))
        .unwrap_or(rows.len())
}

/// An expression never opens with one: it is a value, never an assignment.
fn is_default_line(line: &str) -> bool {
    let code = line.trim_start();
    let end = code
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(code.len());
    let (name, rest) = code.split_at(end);
    !name.is_empty()
        && !name.starts_with(|c: char| c.is_ascii_digit())
        && rest.trim_start().starts_with('=')
}

fn read_defaults(heads: &[(usize, &str)]) -> Result<Vec<(String, Expr)>, Diag> {
    let mut out: Vec<(String, Expr)> = Vec::new();
    for (at, line) in heads {
        let eq = line.find('=').expect("is_default_line found one");
        let name = line[..eq].trim().to_string();
        let value = parse(&line[eq + 1..])
            .map(|e| relocate(&e, at + eq + 1))
            .map_err(|d| shift(d, at + eq + 1))?;
        if out.iter().any(|(seen, _)| *seen == name) {
            return Err(Diag::new(
                DiagCode::BadDefault,
                ByteSpan::new(*at, at + line.len()),
                format!("`{name}` is given a default twice"),
            ));
        }
        out.push((name, value));
    }
    Ok(out)
}

/// A default is parsed out of its own line, so every span it carries has to be put back where
/// the file reads — a refusal inside one is located like any other.
pub(crate) fn relocate(e: &Expr, by: usize) -> Expr {
    let moved = |s: &ByteSpan| ByteSpan::new(s.start + by, s.end + by);
    match e {
        Expr::Lit(_) | Expr::Var(_) => e.clone(),
        Expr::Bin(op, l, r) => Expr::Bin(*op, Box::new(relocate(l, by)), Box::new(relocate(r, by))),
        Expr::SelfRef { arg, address, span } => Expr::SelfRef {
            arg: Box::new(relocate(arg, by)),
            address: *address,
            span: moved(span),
        },
        Expr::Indexed { name, arg, span } => Expr::Indexed {
            name: name.clone(),
            arg: Box::new(relocate(arg, by)),
            span: moved(span),
        },
        Expr::Ref {
            path,
            arg,
            binds,
            address,
            span,
        } => Expr::Ref {
            address: *address,
            path: path.clone(),
            arg: Box::new(relocate(arg, by)),
            binds: binds
                .iter()
                .map(|(k, v)| (k.clone(), relocate(v, by)))
                .collect(),
            span: moved(span),
        },
        Expr::Call { name, args, span } => Expr::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| match a {
                    Arg::Pos(e) => Arg::Pos(relocate(e, by)),
                    Arg::Named(k, e) => Arg::Named(k.clone(), relocate(e, by)),
                })
                .collect(),
            span: moved(span),
        },
    }
}

fn shift(d: Diag, by: usize) -> Diag {
    Diag::new(
        d.code,
        ByteSpan::new(d.span.start + by, d.span.end + by),
        d.message,
    )
}

/// Whether `name` is read: as a variable or a call, anywhere but inside a `sum` binding it.
pub fn occurs_free(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Call {
            name: called, args, ..
        } if called == SERIES => match args.as_slice() {
            [Arg::Pos(Expr::Var(index)), lo, hi, body] => {
                let [lo, hi, body] = [lo, hi, body].map(|(Arg::Pos(x) | Arg::Named(_, x))| x);
                occurs_free(lo, name)
                    || occurs_free(hi, name)
                    || (index != name && occurs_free(body, name))
            }
            _ => children(e, Binds::Substitute)
                .into_iter()
                .any(|c| occurs_free(c, name)),
        },
        Expr::Var(n) => n == name,
        Expr::Call { name: called, .. } | Expr::Indexed { name: called, .. } if called == name => {
            true
        }
        _ => children(e, Binds::Substitute)
            .into_iter()
            .any(|c| occurs_free(c, name)),
    }
}

/// One line is an expression; rows that are all numbers are a scale's cents; anything else
/// over more than one line is refused, since a body is one expression.
fn parse_body(content: &str) -> Result<Parsed, Diag> {
    if let Some(why) = not_node_marker(content) {
        return Err(not_node_diag(why));
    }
    let rows: Vec<(usize, &str)> = code_rows(content)
        .into_iter()
        .filter(|(_, line)| !line.trim().is_empty())
        .collect();
    if rows.len() > 1 {
        if !is_scale(&rows) {
            return Err(multiline_diag(rows[1].0, rows.len()));
        }
        return Ok(Parsed {
            expr: Expr::Lit(Literal::Str("scale".to_string())),
            data: Some(Data::Scale(parse_scale(&rows)?)),
            defaults: Vec::new(),
        });
    }
    let expr = parse_scalar_file(content, &rows)?;
    Ok(Parsed {
        expr,
        data: None,
        defaults: Vec::new(),
    })
}

/// Classified from the comment-stripped row, never the raw text, so a `; ...` header cannot
/// leave `4/4` looking like a division. The expression arm still parses the whole file.
fn parse_scalar_file(content: &str, rows: &[(usize, &str)]) -> Result<Expr, Diag> {
    let code = rows.first().map_or("", |(_, line)| *line).trim();
    if code.is_empty() {
        return Ok(Expr::Lit(Literal::Str(String::new())));
    }
    if let Ok(n) = code.parse::<f64>()
        && n.is_finite()
    {
        return Ok(Expr::Lit(Literal::Num(n)));
    }
    if is_bare_token(code) {
        return Ok(Expr::Lit(Literal::Str(code.to_string())));
    }
    parse(content)
}

/// The one whole-file value the grammar cannot spell is a meter like `4/4`, whose `/` would
/// otherwise divide. Everything else is math: `C4` is a note, `0.25b` a bar literal.
fn is_bare_token(s: &str) -> bool {
    s.contains('/')
        && s.chars()
            .all(|c| c.is_alphanumeric() || matches!(c, '.' | '/' | '_'))
}

/// Narrow, cheap: names bench's own generated HTML landing in a composition directory, which
/// would otherwise refuse as a body over many lines.
fn not_node_marker(content: &str) -> Option<&'static str> {
    let mut end = content.len().min(2048);
    while !content.is_char_boundary(end) {
        end -= 1;
    }
    let head = content[..end].to_ascii_lowercase();
    if head.contains("<!doctype html") {
        Some("starts with `<!DOCTYPE html>`")
    } else if head.contains("<html") {
        Some("contains an `<html>` tag")
    } else {
        None
    }
}

fn not_node_diag(why: &str) -> Diag {
    Diag::new(
        DiagCode::NotNodeContent,
        ByteSpan::at(0),
        format!("this file does not parse as a node: it {why}, which looks like HTML"),
    )
}

fn multiline_diag(at: usize, rows: usize) -> Diag {
    Diag::new(
        DiagCode::MultilineBody,
        ByteSpan::at(at),
        format!(
            "this body runs over {rows} lines, and a body is one expression on one line; timed \
             notes go in a score, one JSON object per line, and a scale's rows are numbers of cents"
        ),
    )
}

/// A blank line stays a row; a comment-only line is dropped, so `; ...` above a scale is no row.
pub fn code_rows(content: &str) -> Vec<(usize, &str)> {
    let mut lines: Vec<&str> = content.split('\n').collect();
    if lines.last().is_some_and(|l| l.trim().is_empty()) && lines.len() > 1 {
        lines.pop();
    }
    let mut rows = Vec::new();
    let mut at = 0usize;
    for line in lines {
        let code = crate::lexer::strip_line_comment(line);
        if code.len() == line.len() || !code.trim().is_empty() {
            rows.push((at, code));
        }
        at += line.len() + 1;
    }
    rows
}

pub fn base_name(rel_path: &str) -> &str {
    rel_path.rsplit('/').next().unwrap_or(rel_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_number_files_are_literals() {
        let p = parse_file("120\n").unwrap();
        assert_eq!(p.expr, Expr::Lit(Literal::Num(120.0)));
        assert!(p.data.is_none());
    }

    #[test]
    fn bare_token_files_are_string_literals_not_expressions() {
        assert_eq!(
            parse_file("4/4\n").unwrap().expr,
            Expr::Lit(Literal::Str("4/4".to_string()))
        );
    }

    #[test]
    fn a_file_holding_one_identifier_is_that_identifier_not_a_string() {
        assert_eq!(
            parse_file("C4\n").unwrap().expr,
            Expr::Var("C4".to_string())
        );
        assert_eq!(
            parse_file("pi\n").unwrap().expr,
            Expr::Var("pi".to_string())
        );
    }

    #[test]
    fn single_line_expression_files_parse_as_expressions() {
        let p = parse_file("sin(2*pi*50*t) * exp(-t*30)\n").unwrap();
        assert!(matches!(p.expr, Expr::Bin(..)));
        assert!(p.data.is_none());
    }

    #[test]
    fn a_body_over_many_lines_refuses_located_unless_every_row_is_a_number() {
        let err = parse_file("; lane\n@kick\n@snare\n").unwrap_err();
        assert_eq!(err.code, DiagCode::MultilineBody);
        assert_eq!(err.span, ByteSpan::at(13));
        let scale = parse_file("; a scale\n200\n400 ; the third\n\n1200\n").unwrap();
        let Some(Data::Scale(scale)) = scale.data else {
            panic!("rows of numbers are a scale")
        };
        assert_eq!(scale.cents, [200.0, 400.0, 1200.0]);
    }

    #[test]
    fn a_file_with_no_extension_opening_with_a_brace_refuses_with_the_rename() {
        let err = parse_file("; doc\n\n{\"at\": \"0b\", \"node\": \"@../bass\"}\n").unwrap_err();
        assert_eq!(err.code, DiagCode::BadScore);
        assert!(err.message.contains(".jsonl"), "{}", err.message);
        assert!(err.message.contains("annotation"), "{}", err.message);
        let p = parse_score_file("score/a", "{\"at\": \"0b\", \"node\": \"@../bass\"}\n").unwrap();
        assert!(matches!(p.data, Some(Data::Score(_))));
        assert!(matches!(p.expr, Expr::Ref { ref path, .. } if path == "../bass"));
    }

    #[test]
    fn html_content_refuses_as_not_a_node() {
        let html = "<!DOCTYPE html>\n<html>\n<head></head>\n<body>bench output</body>\n</html>\n";
        let err = parse_file(html).unwrap_err();
        assert_eq!(err.code, DiagCode::NotNodeContent);
        assert!(err.message.contains("HTML"));
    }

    #[test]
    fn comment_lines_count_as_no_expression_line() {
        let p = parse_file("; the body\nsin(2*pi*50*t)\n").unwrap();
        assert_eq!(p.expr, parse_file("sin(2*pi*50*t)\n").unwrap().expr);
    }

    #[test]
    fn a_header_comment_leaves_every_scalar_shape_alone() {
        let header = "; Concern: the time signature | Non-concern: the tempo | IO: none\n";
        assert_eq!(
            parse_file(&format!("{header}4/4\n")).unwrap().expr,
            Expr::Lit(Literal::Str("4/4".to_string()))
        );
        assert_eq!(
            parse_file(&format!("{header}120\n")).unwrap().expr,
            Expr::Lit(Literal::Num(120.0))
        );
        assert_eq!(
            parse_file(&format!("{header}C4\n")).unwrap().expr,
            Expr::Var("C4".to_string())
        );
        assert_eq!(
            parse_file("4/4 ; four on the floor\n").unwrap().expr,
            Expr::Lit(Literal::Str("4/4".to_string())),
            "a trailing comment is stripped too"
        );
        assert_eq!(
            parse_file("; only a comment\n").unwrap().expr,
            Expr::Lit(Literal::Str(String::new())),
            "a file with no code is empty, not malformed"
        );
    }

    #[test]
    fn blank_lines_around_an_expression_leave_it_one_expression() {
        let plain = parse_file("sin(2*pi*220*t)\n").unwrap();
        for padded in [
            "sin(2*pi*220*t)\n\n",
            "sin(2*pi*220*t)\n\n\n",
            "\nsin(2*pi*220*t)\n",
        ] {
            let p = parse_file(padded).unwrap();
            assert_eq!(p.expr, plain.expr, "{padded:?}");
        }
        let defaulted = parse_file("f = 220\n\nsin(2*pi*f*t)\n").unwrap();
        assert_eq!(
            defaulted.expr,
            parse_file("f = 220\nsin(2*pi*f*t)\n").unwrap().expr
        );
        assert_eq!(
            parse_file("120\n\n").unwrap().expr,
            Expr::Lit(Literal::Num(120.0))
        );
    }

    #[test]
    fn malformed_expression_syntax_refuses() {
        assert!(parse_file("sin(\n").is_err());
    }
}
