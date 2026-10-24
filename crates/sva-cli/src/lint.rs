// Concern: what a composition can be told about itself without rendering a sample | Non-concern: rendering, or judging how it sounds | IO: (dir[, target]) -> Vec<Finding> or CliError

use std::path::Path;

use sva_ast::parse_doc_comment;
use sva_ast::{Graph, Source, ref_spans};
use sva_core::{CliError, Job, LintCode, LintViolation, Severity, lint_diagnostic};
use sva_core::{Diagnostic, prepared, settled};

/// These ride a `success` envelope; a refusal raises [`sva_core::CliError::LintRefused`].
pub struct Finding {
    pub code: LintCode,
    pub severity: Severity,
    pub subject: String,
    pub message: String,
    /// The 1-based line the check found it on, where it knows one.
    pub line: Option<usize>,
}

impl Finding {
    pub fn diagnostic(&self) -> Diagnostic {
        lint_diagnostic(
            self.code,
            &self.subject,
            &self.message,
            self.severity,
            self.line,
        )
    }
}

pub struct LintReport {
    pub nodes: usize,
    pub findings: Vec<Finding>,
    /// What each song's files earn: warnings, each located at its line.
    pub song: Vec<sva_ast::song::Warning>,
    /// The seconds a render of the target reads.
    pub interval: Option<(f64, f64)>,
}

/// With no target, every file's own rules and every entry point's types; with one, those of
/// each file it reaches and the interval a render of it reads.
pub fn lint(dir: &Path, target: Option<&str>) -> Result<LintReport, CliError> {
    lint_bounded(dir, target, BODY_CHARS)
}

/// The same, a body's expression allowed `body_chars` characters.
pub fn lint_bounded(
    dir: &Path,
    target: Option<&str>,
    body_chars: usize,
) -> Result<LintReport, CliError> {
    let source = crate::heard::Heard::at(dir)?;
    match target {
        None => lint_files(&source, prepared(&source)?, body_chars),
        Some(target) => lint_reaching(&source, target, body_chars),
    }
}

/// Every entry point is typed, which types every file it reaches; one that no render can take
/// whole is left to the render that refuses it. A lint violation refuses before a type does.
fn lint_files(
    source: &dyn Source,
    graph: Graph,
    body_chars: usize,
) -> Result<LintReport, CliError> {
    let violations = lint_violations(source, &graph, body_chars);
    let mut song = Vec::new();
    if violations.is_empty() {
        for root in crate::trace::entry_points(&graph) {
            song.extend(sva_core::types(&Job::over(source, &format!("@{root}")))?);
        }
    }
    let report = verdict(graph.paths().count(), violations, per_file(&graph), None)?;
    Ok(LintReport { song, ..report })
}

/// The findings no file's neighbours change.
fn per_file(graph: &Graph) -> Vec<Finding> {
    let mut findings = crate::variables::key_findings(graph);
    findings.extend(crate::bare::bare_findings(graph));
    findings
}

/// The status is the verdict; `data.diagnostics[]` carries everything the scan found.
fn verdict(
    nodes: usize,
    violations: Vec<LintViolation>,
    findings: Vec<Finding>,
    interval: Option<(f64, f64)>,
) -> Result<LintReport, CliError> {
    if violations.is_empty() {
        return Ok(LintReport {
            nodes,
            findings,
            song: Vec::new(),
            interval,
        });
    }
    let mut every = violations;
    every.extend(findings.into_iter().map(|f| LintViolation {
        code: f.code,
        severity: f.severity,
        subject: f.subject,
        message: f.message,
        line: f.line,
    }));
    Err(CliError::LintRefused(every))
}

/// The files the target reaches, each under its own rules, then what a render of it decides:
/// a lint violation refuses before a structural one.
fn lint_reaching(
    source: &dyn Source,
    target: &str,
    body_chars: usize,
) -> Result<LintReport, CliError> {
    let expr = sva_core::target(target)?.expr;
    let held = sva_core::roots_of(source, &expr)?;
    let roots: Vec<&str> = held.iter().map(String::as_str).collect();
    let graph = settled(sva_ast::load_reaching(source, &roots))?;
    let violations = lint_violations(source, &graph, body_chars);
    let mut interval = None;
    let mut song = Vec::new();
    if violations.is_empty() {
        let (render, warned) = sva_core::plan(&Job::over(source, target))?;
        song = warned;
        let rate = f64::from(render.config.rate);
        interval = render
            .range
            .map(|r| (r.start_secs(render.config.rate), r.end as f64 / rate));
    }
    let report = verdict(
        graph.paths().count(),
        violations,
        per_file(&graph),
        interval,
    )?;
    Ok(LintReport { song, ..report })
}

/// Refuses with every violation found, not just the first: a tree-wide scan, unlike
/// `check_arity`/`check_feedback`'s single graph evaluation.
fn lint_violations(source: &dyn Source, graph: &Graph, body_chars: usize) -> Vec<LintViolation> {
    let mut violations = Vec::new();
    violations.extend(doc_comment_violations(source, graph));
    violations.extend(long_comment_block_violations(source, graph));
    violations.extend(long_expression_body_violations(source, graph, body_chars));
    violations.extend(crate::rates::rate_violations(graph));
    violations.extend(crate::arity::arity_violations(graph));
    violations
}

/// Even a justified multi-line block stays under this.
const LONG_COMMENT_BLOCK_CHARS: usize = 1000;

/// The line-1 doc-comment's own run is excluded from the count.
fn long_comment_block_violations(source: &dyn Source, graph: &Graph) -> Vec<LintViolation> {
    let mut violations = Vec::new();
    for path in graph.paths() {
        let Ok(Some(text)) = source.get(path) else {
            continue;
        };
        // start line, running char total, first line's own char length.
        let mut run: Option<(usize, usize, usize)> = None;
        for (idx, line) in text.lines().enumerate() {
            if line.trim_start().starts_with(';') {
                let line_chars = line.chars().count();
                run = Some(match run {
                    Some((start, chars, first)) => (start, chars + line_chars, first),
                    None => (idx + 1, line_chars, line_chars),
                });
            } else {
                flush_comment_run(run.take(), path, &mut violations);
            }
        }
        flush_comment_run(run, path, &mut violations);
    }
    violations
}

/// Drops the header line's own char count (not a flat `1`) when the run starts at line 1.
fn flush_comment_run(
    run: Option<(usize, usize, usize)>,
    path: &str,
    violations: &mut Vec<LintViolation>,
) {
    let Some((start, chars, first)) = run else {
        return;
    };
    let chars = if start == 1 {
        chars.saturating_sub(first)
    } else {
        chars
    };
    if chars > LONG_COMMENT_BLOCK_CHARS {
        violations.push(LintViolation {
            code: LintCode::LongCommentBlock,
            severity: Severity::Error,
            subject: path.to_string(),
            message: format!(
                "`{path}` has a {chars}-character comment block, over the \
                 {LONG_COMMENT_BLOCK_CHARS}-character threshold — split it or trim it"
            ),
            line: Some(start),
        });
    }
}

/// A score's doc comment is its line-1 `annotation`, the same four fields under the same codes.
fn annotation_violations(graph: &Graph) -> Vec<LintViolation> {
    let mut violations = Vec::new();
    for path in graph.paths() {
        let Some(score) = graph.score(path) else {
            continue;
        };
        let file = format!("{path}{}", sva_ast::score::EXTENSION);
        let (code, message) = match &score.annotation {
            None => (
                LintCode::MissingComment,
                format!(
                    "`{file}`'s line 1 is no annotation — every score opens with {{\"annotation\": \
                     \"Models: ... | Neglects: ... | IO: ... -> ... | Tags: ...\"}} documenting it"
                ),
            ),
            Some(text) => match parse_doc_comment(&format!("; {text}")) {
                Ok(_) => continue,
                Err(reason) => (
                    LintCode::MalformedComment,
                    format!(
                        "`{file}`'s annotation does not match `Models: ... | Neglects: ... | IO: \
                         ... -> ... | Tags: ...` ({reason})"
                    ),
                ),
            },
        };
        violations.push(LintViolation {
            code,
            severity: Severity::Error,
            subject: file,
            message,
            line: Some(1),
        });
    }
    violations
}

/// Exactly one doc-comment line at the top, or `lint` refuses it. Only the leading run is
/// judged, so a scale's inline comments stay out of scope; a score's is its annotation.
fn doc_comment_violations(source: &dyn Source, graph: &Graph) -> Vec<LintViolation> {
    let mut violations = annotation_violations(graph);
    for path in graph.paths().filter(|path| graph.score(path).is_none()) {
        let Ok(Some(text)) = source.get(path) else {
            continue;
        };
        let header: Vec<&str> = text
            .lines()
            .take_while(|line| line.trim_start().starts_with(';'))
            .collect();
        match header.len() {
            0 => violations.push(LintViolation {
                code: LintCode::MissingComment,
                severity: Severity::Error,
                subject: path.to_string(),
                message: format!(
                    "`{path}` has no `;`-comment — every composition node must carry one `; \
                     Models: ... | Neglects: ... | IO: ... -> ... | Tags: ...` line \
                     documenting it"
                ),
                line: None,
            }),
            1 => {
                if let Err(reason) = parse_doc_comment(header[0]) {
                    violations.push(LintViolation {
                        code: LintCode::MalformedComment,
                        severity: Severity::Error,
                        subject: path.to_string(),
                        message: format!(
                            "`{path}`'s comment does not match `Models: ... | Neglects: ... \
                             | IO: ... -> ... | Tags: ...` ({reason})"
                        ),
                        line: None,
                    });
                }
            }
            n => violations.push(LintViolation {
                code: LintCode::MultilineComment,
                severity: Severity::Error,
                subject: path.to_string(),
                message: format!(
                    "`{path}` has {n} `;`-comment lines — exactly one is required, written \
                     denser instead of split across lines"
                ),
                line: None,
            }),
        }
    }
    violations
}

/// Generous on purpose: a backstop against runaway generation that `--body-chars` raises, not
/// a budget on complex models. A knot table's literals are data and count nothing.
pub const BODY_CHARS: usize = 10_000;

/// A verbose local `@ref` name is a naming choice, not equation complexity.
const REF_PLACEHOLDER: &str = "@x";

/// Excludes the leading `;`-comment run, and a score, whose lines are events and no body. A
/// node over budget gets one recount with ref names collapsed, so a verbose `@ref` cannot trip it.
fn long_expression_body_violations(
    source: &dyn Source,
    graph: &Graph,
    most: usize,
) -> Vec<LintViolation> {
    let mut violations = Vec::new();
    for path in graph.paths().filter(|path| graph.score(path).is_none()) {
        let Ok(Some(text)) = source.get(path) else {
            continue;
        };
        let body: String = text
            .lines()
            .skip_while(|line| line.trim_start().starts_with(';'))
            .collect::<Vec<_>>()
            .join("\n");
        let body = without_knot_tables(&body);
        let chars = body.chars().count();
        if chars <= most {
            continue;
        }
        let (chars, has_refs) = ref_stripped_char_count(&body);
        if chars > most {
            let message = if has_refs {
                format!(
                    "`{path}` has a {chars}-character expression body even with every `@ref` \
                     name collapsed to `{REF_PLACEHOLDER}`, over the {most}-character \
                     threshold — decompose it into sub-nodes, or raise it with `--body-chars`"
                )
            } else {
                format!(
                    "`{path}` has a {chars}-character expression body, over the {most}-character \
                     threshold — decompose it into sub-nodes, or raise it with `--body-chars`"
                )
            };
            violations.push(LintViolation {
                code: LintCode::LongExpressionBody,
                severity: Severity::Error,
                subject: path.to_string(),
                message,
                line: None,
            });
        }
    }
    violations
}

/// One text, comments blanked, so a table across lines is found whole.
fn without_knot_tables(body: &str) -> String {
    let flat: String = body
        .split('\n')
        .map(|line| {
            let kept = sva_ast::strip_line_comment(line);
            kept.to_string() + &" ".repeat(line.len() - kept.len())
        })
        .collect::<Vec<_>>()
        .join(" ");
    let mut kept = String::with_capacity(body.len());
    let mut from = 0;
    for span in sva_ast::knot_table_spans(&flat) {
        kept.push_str(&body[from..span.start]);
        from = span.end;
    }
    kept.push_str(&body[from..]);
    kept
}

/// Only reached once the raw count already exceeds the cap. Collapses each `@ref`'s `@path`
/// head to a placeholder; a trailing `(...)` invocation stays, being real expression content.
/// The `bool` says whether any ref was actually collapsed, so the error message can be worded
/// accurately for a body already over budget on pure literal content.
fn ref_stripped_char_count(body: &str) -> (usize, bool) {
    let spans = ref_spans(body);
    let has_refs = !spans.is_empty();
    let mut reduced = String::with_capacity(body.len());
    let mut cursor = 0;
    for span in spans {
        reduced.push_str(&body[cursor..span.start]);
        reduced.push_str(REF_PLACEHOLDER);
        cursor = span.end;
    }
    reduced.push_str(&body[cursor..]);
    (reduced.chars().count(), has_refs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn dir_of(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sva-cli-lint-{name}-{:x}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (rel, content) in files {
            let path = dir.join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }
        dir
    }

    /// A well-formed doc comment, for a fixture node the test isn't about.
    fn doc(models: &str) -> String {
        format!(
            "; Models: {models} | Neglects: nothing, it's a fixture | IO: t -> mix | Tags: \
             fixture\n"
        )
    }

    #[test]
    fn both_lint_response_paths_answer_the_same_diagnostic_shape() {
        let dir = dir_of(
            "one-envelope",
            &[
                ("master", &(doc("the mix") + "@kick*0.5\n")),
                ("kick", &(doc("a thump") + "sin(2*pi*50*t)\n")),
                ("variables/key", &(doc("the key") + "sin(t)\n")),
            ],
        );
        let report = lint(&dir, None).expect("a documented composition lints clean");
        let found: Vec<Diagnostic> = report.findings.iter().map(Finding::diagnostic).collect();
        let json = sva_core::success_envelope(
            &crate::output::lint_data(&dir.display().to_string(), None, report.nodes, None),
            &found,
        );
        assert!(json.contains("\"diagnostics\""), "{json}");
        assert!(
            json.contains("\"code\": \"lint.key_is_not_a_pitch\""),
            "{json}"
        );
        assert!(json.contains("\"severity\": \"warning\""), "{json}");
        assert!(
            json.contains("\"location\": { \"file\": \"variables/key\", \"span\": null, \"start\": null, \"end\": null }"),
            "{json}"
        );
        assert!(json.contains("\"help\""), "{json}");

        let undocumented = dir_of(
            "one-envelope-refused",
            &[("master", "@kick*0.5\n"), ("kick", "sin(t)\n")],
        );
        let Err(err) = lint(&undocumented, None) else {
            panic!("an undocumented node must refuse");
        };
        let refused = sva_core::error_envelope(err.code(), &err.message(), &err.diagnostics());
        assert!(refused.contains("\"diagnostics\""), "{refused}");
        assert!(
            refused.contains("\"code\": \"lint.missing_comment\""),
            "{refused}"
        );
        assert!(refused.contains("\"severity\": \"error\""), "{refused}");
        assert!(
            refused
                .contains("\"location\": { \"file\": \"kick\", \"span\": null, \"start\": null, \"end\": null }"),
            "{refused}"
        );
        assert!(refused.contains("\"help\""), "{refused}");

        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&undocumented);
    }

    fn assert_refused(result: Result<LintReport, CliError>, code: LintCode, subject: &str) {
        match result {
            Ok(report) => panic!(
                "expected a `{code:?}` refusal for `{subject}`, got a clean report: {:?}",
                report.findings.iter().map(|f| &f.code).collect::<Vec<_>>()
            ),
            Err(CliError::LintRefused(violations)) => assert!(
                violations
                    .iter()
                    .any(|v| v.code == code && v.subject == subject),
                "expected a `{code:?}` refusal for `{subject}`, got: {:?}",
                violations
                    .iter()
                    .map(|v| (v.code, &v.subject))
                    .collect::<Vec<_>>()
            ),
            Err(other) => panic!("expected a `{code:?}` refusal, got a different error: {other:?}"),
        }
    }

    #[test]
    fn plain_lint_refuses_the_type_refusal_a_targeted_lint_does() {
        let body = "crop(self(t - 1sp) + sample(sin(2*pi*100*t))*1sp, 0s, 1s)\n";
        let dir = dir_of("typed", &[("master", &(doc("a discrete loop") + body))]);
        for target in [None, Some("@master")] {
            let Err(err) = lint(&dir, target) else {
                panic!("{target:?}: `self(t - d)` in a discrete loop must refuse");
            };
            let codes: Vec<String> = err.diagnostics().into_iter().map(|d| d.code).collect();
            assert!(
                codes.iter().any(|c| c == "type.discrete_self_at_time"),
                "{target:?}: {codes:?}"
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    type ThresholdCase = (&'static str, std::path::PathBuf, Box<dyn Fn()>);

    #[test]
    fn every_length_threshold_sits_at_its_boundary_and_trips_one_char_over() {
        let cases: Vec<ThresholdCase> = vec![
            (
                "long-comment-block, 1000 chars",
                dir_of(
                    "boundary-comment",
                    &[(
                        "master",
                        &(doc("a fixture signal")
                            + "sin(t)\n"
                            + &format!(";{}", "x".repeat(999))
                            + "\n"),
                    )],
                ),
                Box::new(|| {
                    let trailing = format!(";{}", "x".repeat(1000));
                    let dir = dir_of(
                        "long-comment",
                        &[
                            (
                                "long",
                                &(doc("a fixture signal") + "sin(t)\n" + &trailing + "\n"),
                            ),
                            ("master", &(doc("a fixture signal") + "@long\n")),
                        ],
                    );
                    assert_refused(lint(&dir, None), LintCode::LongCommentBlock, "long");
                    assert_refused(
                        lint(&dir, Some("@master([0, 1s])")),
                        LintCode::LongCommentBlock,
                        "long",
                    );
                }) as Box<dyn Fn()>,
            ),
            (
                "long-expression-body, 10000 chars",
                dir_of(
                    "boundary-expression-body",
                    &[(
                        "drone",
                        &(String::from(
                            "; Models: a sustained drone | Neglects: envelope, detune | IO: t -> \
                             amplitude | Tags: drone\n",
                        ) + &"0".repeat(10_000)
                            + "\n"),
                    )],
                ),
                Box::new(|| {
                    let body = "0".repeat(10_001);
                    let dir = dir_of(
                        "long-expression-body",
                        &[
                            (
                                "drone",
                                &(String::from(
                                    "; Models: a sustained drone | Neglects: envelope, detune | \
                                     IO: t -> amplitude | Tags: drone\n",
                                ) + &body
                                    + "\n"),
                            ),
                            (
                                "master",
                                &(String::from(WELL_FORMED_MASTER_COMMENT) + "@drone\n"),
                            ),
                        ],
                    );
                    assert_refused(lint(&dir, None), LintCode::LongExpressionBody, "drone");
                    assert_refused(
                        lint(&dir, Some("@master([0, 1s])")),
                        LintCode::LongExpressionBody,
                        "drone",
                    );
                }),
            ),
        ];
        for (label, at, over) in cases {
            assert!(
                lint(&at, None).is_ok(),
                "{label}: exactly at the threshold, not over it"
            );
            over();
        }
    }

    /// Well-formed, so it never adds noise to a scenario testing some other node's comment.
    const WELL_FORMED_MASTER_COMMENT: &str = "; Models: a test harness's master node | Neglects: nothing, it's a fixture | IO: t -> mix | Tags: fixture\n";

    #[test]
    fn a_node_with_no_comment_at_all_is_missing_comment_in_both_modes() {
        let dir = dir_of(
            "missing-comment",
            &[
                ("plucked", "sin(t)\n"),
                (
                    "master",
                    &(String::from(WELL_FORMED_MASTER_COMMENT) + "@plucked\n"),
                ),
            ],
        );
        assert_refused(lint(&dir, None), LintCode::MissingComment, "plucked");
        assert_refused(
            lint(&dir, Some("@master([0, 1s])")),
            LintCode::MissingComment,
            "plucked",
        );
    }

    #[test]
    fn a_single_free_text_comment_line_is_malformed_comment_in_both_modes() {
        let dir = dir_of(
            "malformed-comment",
            &[
                (
                    "plucked",
                    "; a plucked string, decays over time, no body resonance modeled\nsin(t)\n",
                ),
                (
                    "master",
                    &(String::from(WELL_FORMED_MASTER_COMMENT) + "@plucked\n"),
                ),
            ],
        );
        assert_refused(lint(&dir, None), LintCode::MalformedComment, "plucked");
        assert_refused(
            lint(&dir, Some("@master([0, 1s])")),
            LintCode::MalformedComment,
            "plucked",
        );
    }
}
