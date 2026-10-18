// Concern: proves each refusal and warning a score's lines and lanes earn fires with its code at its file, line and column | Non-concern: what a render sounds like | IO: (a composition) -> a diagnostic

use std::path::{Path, PathBuf};

use crate::helpers::{put, scratch};
use sva_core::{CliError, Diagnostic};

/// A tone, a node with a one-number argument, a node with no default for its pitch, and the
/// variables, at 120 bpm: one bar is 2 s.
fn composition(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = scratch(&format!("rows-{name}"));
    for (path, text) in [
        ("variables/bpm", "120\n"),
        ("variables/meter", "4/4\n"),
        ("variables/key", "A3\n"),
        ("variables/scale", "minor\n"),
        ("variables/energy", "0.5\n"),
        ("variables/wobble", "0.5*sin(2*pi*t)\n"),
        (
            "tone",
            "f0 = 220hz\nlen = 0.25s\nvel = 0.5\ncrop(vel*@variables/energy*sin(2*pi*f0*t), 0s, len)\n",
        ),
        (
            "grit",
            "drive = 0.2\ncrop(sat(sin(2*pi*110*t), drive=1 + drive), 0s, 0.5s)\n",
        ),
        ("bare", "len = 0.25s\ncrop(sin(2*pi*f0*t), 0s, len)\n"),
        (
            "odd",
            "base = 110hz\nf0 = 4*base\ncrop(sin(2*pi*f0*t), 0s, 1s)\n",
        ),
    ] {
        put(&dir, path, text);
    }
    for (path, text) in files {
        put(&dir, path, text);
    }
    dir
}

/// The first diagnostic a render of `target` answers: a refusal's, else a warning's.
fn first(dir: &Path, target: &str) -> Diagnostic {
    match sva_core::probe(dir, target) {
        Err(refused) => refused.diagnostics().into_iter().next().unwrap_or_else(|| {
            panic!("{target} refused with no diagnostic: {}", refused.message())
        }),
        Ok(rendered) => sva_core::warning_diagnostic(
            rendered
                .warnings
                .first()
                .unwrap_or_else(|| panic!("{target} rendered with no warning")),
        ),
    }
}

/// `(case, files, target, code, file, (line, column), what the message says)`.
type Row<'a> = (
    &'a str,
    &'a [(&'a str, &'a str)],
    &'a str,
    &'a str,
    &'a str,
    (usize, usize),
    &'a str,
);

/// Every row checked, every one that misses named at once.
fn check(rows: &[Row]) {
    let mut missed = Vec::new();
    for (case, files, target, code, file, (line, column), says) in rows {
        let dir = composition(case, files);
        let found = first(&dir, target);
        let held = found.code == *code
            && found.file.as_deref() == Some(*file)
            && found.start == Some((*line, *column))
            && found
                .message
                .starts_with(&format!("line {line}, column {column}: "))
            && found.message.contains(says);
        if !held {
            missed.push(format!(
                "{case}: wants {code} {file} {line}:{column} `{says}`, found {} {:?} {:?} {}",
                found.code, found.file, found.start, found.message
            ));
        }
    }
    assert!(missed.is_empty(), "{}", missed.join("\n"));
}

const S: &str = "@s";

#[test]
fn a_files_own_lines_refuse_where_they_are_written() {
    check(&[
        (
            "not-json",
            &[("s.jsonl", "{\"at\": \"0b\", \"node\": }\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "expected value",
        ),
        (
            "empty",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\"}\n\n{\"at\": \"1b\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 1),
            "empty",
        ),
        (
            "semicolon",
            &[(
                "s.jsonl",
                "; a comment\n{\"at\": \"0b\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 1),
            "a `;` line is no JSON",
        ),
        (
            "annotation-later",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\"}\n{\"annotation\": \"Models: x | Neglects: y | IO: t -> x | Tags: t\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 2),
            "line 1 only",
        ),
        (
            "at-alone",
            &[("s.jsonl", "{\"at\": \"0b\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 1),
            "a timed comment is exactly `at` and `comment`",
        ),
        (
            "mark-node",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"mark\": \"a\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 27),
            "a mark takes no `node`",
        ),
        (
            "event-shape",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"shape\": \"lin\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 31),
            "an event takes no `shape`",
        ),
        (
            "twice",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"at\": \"1b\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 31),
            "`at` is written twice",
        ),
        (
            "sound",
            &[("s.jsonl", "{\"at\": \"0b\", \"sound\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 14),
            "`sound` is now `node`",
        ),
        (
            "no-at",
            &[("s.jsonl", "{\"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 18),
            "`at` is missing",
        ),
        (
            "no-to",
            &[("s.jsonl", "{\"at\": \"0b\", \"lane\": \"energy\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 31),
            "a move needs `to`",
        ),
        (
            "shape-no-len",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 1, \"shape\": \"lin\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 50),
            "`lin` or `exp`",
        ),
        (
            "len-no-shape",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 1, \"len\": \"1b\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 48),
            "`lin` or `exp`",
        ),
        (
            "unknown-shape",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 1, \"len\": \"1b\", \"shape\": \"ease\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 63),
            "`shape` is `ease`; a ramp's shape is `lin` or `exp`",
        ),
        (
            "bare-number",
            &[("s.jsonl", "{\"at\": 0.5, \"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 8),
            "bare number",
        ),
        (
            "mixed-units",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\"}\n{\"at\": \"1s\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 8),
            "one file counts all its times in bars or all in seconds",
        ),
        (
            "time-times-time",
            &[("s.jsonl", "{\"at\": \"1b*1b\", \"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 11),
            "multiplies a time by a time",
        ),
        (
            "time-over-time",
            &[("s.jsonl", "{\"at\": \"1b/1b\", \"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 11),
            "divides a time by a time",
        ),
        (
            "time-plus-number",
            &[("s.jsonl", "{\"at\": \"1b + 1\", \"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 12),
            "a time needs its unit",
        ),
        (
            "len-zero",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"len\": \"0b\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 38),
            "a window has a length",
        ),
        (
            "reads-t",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": \"0.5 + t\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 45),
            "what moves is a lane",
        ),
        (
            "reads-ref",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": \"@grit\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 39),
            "what moves is a lane",
        ),
        (
            "builtin",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": \"sin(1)\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 39),
            "calls `sin`",
        ),
        (
            "unknown-name",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": \"loud\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 39),
            "names `loud`",
        ),
        (
            "json-true",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": true}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 38),
            "a JSON boolean",
        ),
        (
            "json-object",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": {}}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 38),
            "a JSON object",
        ),
        (
            "deg-half",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"deg\": 1.5}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 38),
            "deg(7)*1st",
        ),
        (
            "deg-and-f0",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"deg\": 1, \"f0\": 440}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 38),
            "one pitch per line",
        ),
        (
            "lists-differ",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"deg\": [1, 3], \"vel\": [1]}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 53),
            "`deg` lists 2 items, `vel` lists 1 item",
        ),
        (
            "empty-list",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": []}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 38),
            "lists no item",
        ),
        (
            "list-in-list",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": [[1]]}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 39),
            "a list in a list",
        ),
        (
            "comment-list",
            &[("s.jsonl", "{\"at\": \"0b\", \"comment\": [\"a\", \"b\"]}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 25),
            "never broadcast",
        ),
        (
            "backwards",
            &[(
                "s.jsonl",
                "{\"at\": \"1b\", \"node\": \"@tone\"}\n{\"at\": \"0.5b\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 8),
            "before 1b on line 1",
        ),
        (
            "items-backwards",
            &[(
                "s.jsonl",
                "{\"at\": [\"1b\", \"0.5b\"], \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 15),
            "earlier on this line",
        ),
        (
            "mark-name",
            &[("s.jsonl", "{\"at\": \"0b\", \"mark\": \"pre-drop\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "[a-z][a-z0-9_]*",
        ),
        (
            "mark-twice",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"mark\": \"a\"}\n{\"at\": \"1b\", \"mark\": \"a\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 22),
            "defined twice",
        ),
        (
            "mark-unknown",
            &[("s.jsonl", "{\"at\": \"mark(drop)\", \"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 9),
            "mark(drop) names no mark",
        ),
        (
            "mark-cycle",
            &[(
                "s.jsonl",
                "{\"at\": \"mark(b)\", \"mark\": \"a\"}\n{\"at\": \"mark(a)\", \"mark\": \"b\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 8),
            "defined in a cycle",
        ),
        (
            "mark-in-value",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"mark\": \"a\"}\n{\"at\": \"0b\", \"node\": \"@tone\", \"vel\": \"mark(a)\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 39),
            "a mark stands only in `at` and `len`",
        ),
        (
            "deg-in-time",
            &[("s.jsonl", "{\"at\": \"deg(1)\", \"node\": \"@tone\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 9),
            "in a time field",
        ),
        (
            "position-scaled",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"mark\": \"a\"}\n{\"at\": \"2*mark(a)\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 10),
            "a mark is a position",
        ),
        (
            "position-in-len",
            &[(
                "s.jsonl",
                "{\"at\": \"1b\", \"mark\": \"a\"}\n{\"at\": \"0b\", \"node\": \"@tone\", \"len\": \"mark(a)\"}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (2, 38),
            "`len` holds a position",
        ),
        (
            "jsonl-ref",
            &[("s.jsonl", "{\"at\": \"0b\", \"node\": \"@tone.jsonl\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "a ref drops `.jsonl`",
        ),
        (
            "unresolved",
            &[("s.jsonl", "{\"at\": \"0b\", \"node\": \"@nope\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "does not resolve",
        ),
        (
            "itself",
            &[
                ("s.jsonl", "{\"at\": \"0b\", \"node\": \"@t2\"}\n"),
                ("t2.jsonl", "{\"at\": \"1b\", \"node\": \"@s\"}\n"),
            ],
            S,
            "bad-score",
            "t2.jsonl",
            (1, 22),
            "a score places itself",
        ),
        (
            "mixed-len",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"node\": [\"@t2\", \"@tone\"], \"len\": \"1b\"}\n",
                ),
                ("t2.jsonl", "{\"at\": \"0b\", \"node\": \"@tone\"}\n"),
            ],
            S,
            "bad-score",
            "s.jsonl",
            (1, 47),
            "split the line",
        ),
    ]);
}

#[test]
fn a_file_no_extension_names_opening_with_a_brace_and_a_ref_two_files_match_refuse() {
    check(&[(
        "two-files",
        &[
            ("s.jsonl", "{\"at\": \"0b\", \"node\": \"@tone\"}\n"),
            ("s", "0.5\n"),
        ],
        S,
        "bad-score",
        "s.jsonl",
        (1, 1),
        "matches two files",
    )]);
    let dir = composition(
        "old",
        &[("s", "; doc\n{\"at\": \"0b\", \"node\": \"@tone\"}\n")],
    );
    let Err(CliError::Refusals(refused)) = sva_core::probe(&dir, S) else {
        panic!("a 0.13 score refuses");
    };
    assert_eq!(refused[0].at.path, "s");
    for says in [".jsonl", "annotation", "timed comment", "`comment` key"] {
        assert!(refused[0].reason.contains(says), "{}", refused[0].reason);
    }
}

#[test]
fn a_line_is_checked_against_the_node_it_plays() {
    check(&[
        (
            "unread",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"cutoff\": 900}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 31),
            "`cutoff` is no parameter `@tone` reads",
        ),
        (
            "chord",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"chord\": [1, 3]}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 31),
            "a chord is a `deg` list",
        ),
        (
            "unbound",
            &[("s.jsonl", "{\"at\": \"0b\", \"node\": \"@bare\"}\n")],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "reads `f0`, which nothing binds",
        ),
        (
            "score-param",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"node\": \"@t2\", \"vel\": 1}\n",
                ),
                ("t2.jsonl", "{\"at\": \"0b\", \"node\": \"@tone\"}\n"),
            ],
            S,
            "bad-score",
            "s.jsonl",
            (1, 29),
            "a line placing a score takes `len` alone",
        ),
        (
            "move-score",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"node\": \"@t2\", \"lane\": \"vel\", \"to\": 1}\n",
                ),
                ("t2.jsonl", "{\"at\": \"0b\", \"node\": \"@tone\"}\n"),
            ],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "is a score, which reads no parameter",
        ),
        (
            "move-unread",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"lane\": \"cutoff\", \"to\": 1}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 39),
            "`@tone` reads no `cutoff`",
        ),
        (
            "no-variable",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"lane\": \"heat\", \"to\": 1}\n",
            )],
            S,
            "bad-score",
            "s.jsonl",
            (1, 22),
            "the variables there are `bpm`, `energy`",
        ),
    ]);
}

#[test]
fn a_lane_refuses_what_it_cannot_hold() {
    let at = |s: &str| format!("{{\"at\": \"0b\", \"node\": \"@tone\", \"len\": \"4b\"}}\n{s}");
    check(&[
        (
            "meter",
            &[(
                "s.jsonl",
                &at("{\"at\": \"0b\", \"lane\": \"meter\", \"to\": \"3/4\"}\n"),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 22),
            "`meter` takes no moves",
        ),
        (
            "equation",
            &[(
                "s.jsonl",
                &at("{\"at\": \"0b\", \"lane\": \"wobble\", \"to\": 1}\n"),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 22),
            "an equation",
        ),
        (
            "tie",
            &[(
                "s.jsonl",
                &at(
                    "{\"at\": \"1b\", \"lane\": \"energy\", \"to\": 1}\n{\"at\": \"1b\", \"lane\": \"energy\", \"to\": 0.2}\n",
                ),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (3, 8),
            "this line and s.jsonl:2",
        ),
        (
            "inside-ramp",
            &[(
                "s.jsonl",
                &at(
                    "{\"at\": \"1b\", \"lane\": \"energy\", \"to\": 1, \"len\": \"2b\", \"shape\": \"lin\"}\n{\"at\": \"2b\", \"lane\": \"energy\", \"to\": 0.2}\n",
                ),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (3, 8),
            "inside the ramp from 1b to 3b",
        ),
        (
            "home-no-constant",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@odd\", \"lane\": \"f0\", \"to\": \"440hz\", \"len\": \"1b\", \"shape\": \"exp\"}\n",
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (1, 50),
            "set it first",
        ),
        (
            "kind",
            &[(
                "s.jsonl",
                &at("{\"at\": \"0b\", \"node\": \"@tone\", \"lane\": \"f0\", \"to\": 0.5}\n"),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 51),
            "`to` is a plain number",
        ),
        (
            "exp-sign",
            &[(
                "s.jsonl",
                &at(
                    "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": -1, \"len\": \"1b\", \"shape\": \"exp\"}\n",
                ),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 38),
            "from 0.5 to -1",
        ),
        (
            "scale-ramp",
            &[(
                "s.jsonl",
                &at(
                    "{\"at\": \"0b\", \"lane\": \"scale\", \"to\": \"dorian\", \"len\": \"1b\", \"shape\": \"lin\"}\n",
                ),
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 37),
            "`scale` takes sets only",
        ),
        (
            "bpm-before-zero",
            &[(
                "s.jsonl",
                "{\"at\": \"-1b\", \"lane\": \"bpm\", \"to\": 100}\n{\"at\": \"0b\", \"node\": \"@tone\"}\n",
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (1, 8),
            "before bar 0",
        ),
        (
            "crop-splits",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"node\": \"@t2\", \"len\": \"1b\"}\n",
                ),
                (
                    "t2.jsonl",
                    "{\"at\": \"0b\", \"node\": \"@tone\"}\n{\"at\": \"0.5b\", \"lane\": \"energy\", \"to\": 1, \"len\": \"1b\", \"shape\": \"lin\"}\n",
                ),
            ],
            S,
            "bad-lane",
            "t2.jsonl",
            (2, 8),
            "a `len` crop at 1b cuts this ramp",
        ),
        (
            "engine",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@grit\"}\n{\"at\": \"0.1b\", \"node\": \"@grit\", \"lane\": \"drive\", \"to\": 0.9}\n",
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 56),
            "engine.non_constant_argument",
        ),
        (
            "set-in-default",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@odd\"}\n{\"at\": \"0.1b\", \"node\": \"@odd\", \"lane\": \"f0\", \"to\": \"440hz\"}\n",
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (2, 8),
            "set the lane before the note",
        ),
    ]);
}

#[test]
fn a_move_needs_song_time_and_bars_while_the_tempo_moves() {
    check(&[
        (
            "seconds-while-moving",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"lane\": \"bpm\", \"to\": 100}\n{\"at\": \"0b\", \"node\": \"@t2\"}\n",
                ),
                (
                    "t2.jsonl",
                    "{\"at\": \"1s\", \"lane\": \"energy\", \"to\": 1}\n",
                ),
            ],
            S,
            "bad-lane",
            "t2.jsonl",
            (1, 8),
            "reached through seconds",
        ),
        (
            "bpm-through-seconds",
            &[(
                "s.jsonl",
                "{\"at\": \"0s\", \"lane\": \"bpm\", \"to\": 100}\n",
            )],
            S,
            "bad-lane",
            "s.jsonl",
            (1, 8),
            "a `bpm` move on `variables/bpm` is reached through seconds",
        ),
        (
            "no-song-time",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 1}\n",
                ),
                ("w", "@s(2*t)\n"),
            ],
            "@w",
            "bad-lane",
            "s.jsonl",
            (1, 8),
            "has no song time",
        ),
        (
            "bars-no-song-time",
            &[
                (
                    "s.jsonl",
                    "{\"at\": \"0b\", \"lane\": \"bpm\", \"to\": 100}\n{\"at\": \"0b\", \"node\": \"@w\"}\n",
                ),
                ("w", "@t2(2*t)\n"),
                ("t2.jsonl", "{\"at\": \"1b\", \"node\": \"@tone\"}\n"),
            ],
            S,
            "bad-lane",
            "t2.jsonl",
            (1, 8),
            "it has no song time",
        ),
        (
            "twice-placed",
            &[
                ("s.jsonl", "{\"at\": [\"0b\", \"0b\"], \"node\": \"@t2\"}\n"),
                (
                    "t2.jsonl",
                    "{\"at\": \"0b\", \"node\": \"@tone\"}\n{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 1}\n",
                ),
            ],
            S,
            "bad-lane",
            "t2.jsonl",
            (2, 8),
            "one line placed twice at one position",
        ),
    ]);
    let laned = composition(
        "laned-no-song-time",
        &[
            (
                "s.jsonl",
                "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 1}\n{\"at\": \"0b\", \"node\": \"@w\"}\n",
            ),
            ("w", "@tone(2*t)\n"),
        ],
    );
    let found = first(&laned, S);
    assert_eq!(found.code, "bad-lane", "{}", found.message);
    assert_eq!(found.file.as_deref(), Some("w"));
    assert!(
        found.message.contains("has no song time"),
        "{}",
        found.message
    );
}

/// A degree with no scale refuses at its line, saying what a scale's own refusal says.
#[test]
fn a_degree_with_no_scale_refuses_on_its_line() {
    let dir = composition(
        "no-scale",
        &[(
            "s.jsonl",
            "{\"at\": \"0b\", \"node\": \"@tone\", \"deg\": 3}\n",
        )],
    );
    std::fs::remove_file(dir.join("variables/scale")).expect("a scale to remove");
    let found = first(&dir, S);
    assert_eq!(found.code, "bad-score", "{}", found.message);
    assert_eq!(found.file.as_deref(), Some("s.jsonl"));
    assert_eq!(found.start, Some((1, 38)), "{}", found.message);
    assert!(
        found.message.contains("`variables/scale`"),
        "{}",
        found.message
    );
}

#[test]
fn a_line_over_a_ramp_and_a_lane_reaching_nothing_warn() {
    check(&[
        (
            "overrides",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@tone\", \"lane\": \"vel\", \"to\": 1, \"len\": \"1b\", \"shape\": \"lin\"}\n{\"at\": \"0.5b\", \"node\": \"@tone\", \"vel\": 0.2}\n",
            )],
            S,
            "line-overrides-lane",
            "s.jsonl",
            (2, 33),
            "the line wins",
        ),
        (
            "nothing",
            &[(
                "s.jsonl",
                "{\"at\": \"0b\", \"node\": \"@grit\", \"lane\": \"drive\", \"to\": 1}\n{\"at\": \"0b\", \"node\": \"@tone\"}\n",
            )],
            S,
            "lane-reaches-nothing",
            "s.jsonl",
            (1, 39),
            "no instance in the song reads it",
        ),
    ]);
}
