// Concern: states a structure edit rewrites a composition's text: a move plays `@master` bit for bit, a growth grafts a finer starter | Non-concern: whether an edit pays | IO: (composition) -> files

use sva_ast::Composition;
use sva_core::model::Model;
use sva_core::seed::{Limits, Readings, harmonic, read, starter};
use sva_core::solve::{Vary, vary};
use sva_core::structure::{Edit, Proposal, applied, assign, grow, merge, split};
use sva_core::{Job, Output, Representation, execute};
use sva_engine::Tier;

const BODY: &str = "crop(vel*exp(0 - t/dec)*sin(2*pi*f0*t), 0s, 0.3s)\n";

/// Two kin nodes at different defaults, and a score whose third line is silent.
fn composition() -> Composition {
    [
        ("master", "@score/notes\n".to_string()),
        (
            "instruments/a",
            format!("f0 = 220hz\ndec = 0.1s\n{BODY}"),
        ),
        (
            "instruments/b",
            format!("f0 = 660hz\ndec = 0.2s\n{BODY}"),
        ),
        (
            "score/notes.jsonl",
            [
                r#"{"at": "0s", "node": "@../instruments/a", "f0": "230hz", "vel": 0.5}"#,
                r#"{"at": "0.1s", "node": "@../instruments/a", "f0": "640hz", "dec": "0.15s", "vel": 0.4}"#,
                r#"{"at": "0.15s", "node": "@../instruments/a", "vel": 0}"#,
                r#"{"at": "0.2s", "node": "@../instruments/b", "vel": 0.3}"#,
                r#"{"at": "0.3s", "node": "@../instruments/b", "f0": "700hz", "vel": 0.6}"#,
                "",
            ]
            .join("\n"),
        ),
    ]
    .into_iter()
    .collect()
}

fn varies() -> Vec<Vary> {
    [
        "f0=50hz..2000hz/1.22,log",
        "dec=0.01s..2s/1.2,log",
        "vel=0..1/0.01",
    ]
    .iter()
    .map(|v| vary(v).expect("a vary"))
    .collect()
}

fn model(c: &Composition) -> Model {
    let graph = sva_ast::load_reaching(c, &["master"]).expect("the composition loads");
    Model::of(&graph, "master").expect("a model")
}

fn samples(c: &Composition) -> Vec<u64> {
    let rendered = execute(Job::over(c, "@master"), &Tier::default()).expect("a render");
    let answer = rendered.answer(&rendered.target, Representation::Samples);
    let Output::Samples(b) = answer.expect("samples").value else {
        panic!("samples answer samples");
    };
    b.planes.iter().flatten().map(|v| v.to_bits()).collect()
}

fn text(c: &Composition, path: &str) -> Option<String> {
    sva_ast::Source::get(c, path)
        .ok()
        .flatten()
        .map(|t| t.into_owned())
}

/// `p`'s files laid over the composition.
fn edited(c: &Composition, v: &[Vary], p: &Proposal) -> Composition {
    let files = applied(&model(c), &|path| text(c, path), v, p).expect("the edit applies");
    assert!(!files.is_empty(), "`{}` rewrites a file", p.label);
    let mut out = c.clone();
    for (path, text) in files {
        out.insert(path, text);
    }
    out
}

#[test]
fn every_split_merge_and_cluster_plays_the_same_samples() {
    let (c, v) = (composition(), varies());
    let m = model(&c);
    let mut proposals = split::proposals(&m, &v, assign::REASSIGN_ROUNDS);
    proposals.extend(merge::proposals(&m, &v));
    proposals.push(assign::cluster(&m, &v, assign::REASSIGN_ROUNDS));
    assert_eq!(proposals.len(), 5, "two splits, two merges, a cluster");
    let before = samples(&c);
    for p in &proposals {
        assert!(
            samples(&edited(&c, &v, p)) == before,
            "`{}` plays as before",
            p.label
        );
    }
}

#[test]
fn a_merge_leaves_its_node_used_by_no_line() {
    let (c, v) = (composition(), varies());
    for p in merge::proposals(&model(&c), &v) {
        let Some(Edit::Move { to, .. }) = p.edits.first() else {
            panic!("a merge moves lines");
        };
        let after = model(&edited(&c, &v, &p));
        assert!(after.lines.iter().all(|l| l.node == *to), "{}", p.label);
    }
}

#[test]
fn a_split_copies_its_node_as_a_kin_one_named_after_it() {
    let (c, v) = (composition(), varies());
    let m = model(&c);
    let p = split::proposals(&m, &v, assign::REASSIGN_ROUNDS)
        .into_iter()
        .find(|p| {
            p.edits
                .first()
                .is_some_and(|e| matches!(e, Edit::Copy { from, .. } if from == "instruments/a"))
        })
        .expect("a split of `a`");
    assert_eq!(
        p.edits[0],
        Edit::Copy {
            from: "instruments/a".to_string(),
            path: "instruments/a-2".to_string()
        }
    );
    let after = model(&edited(&c, &v, &p));
    assert!(after.kin("instruments/a", "instruments/a-2"));
}

const RATE: f64 = 48_000.0;
const COARSE: Limits = Limits {
    harmonics: 4,
    lone: 1,
};
const FINER: Limits = Limits {
    harmonics: 8,
    lone: 1,
};

/// Two plucks of one tone, 220 Hz at 0 s and 330 Hz at 0.3 s, each with eight harmonics.
fn plucks() -> Readings {
    let pluck = |n: usize, at: f64, hz: f64| {
        let t = n as f64 / RATE - at;
        if t < 0.0 {
            return 0.0;
        }
        let w = |k: usize| libm::sin(2.0 * std::f64::consts::PI * hz * k as f64 * t) / k as f64;
        0.1 * sva_formula::exp(-6.0 * t) * (1..=8).map(w).sum::<f64>()
    };
    let x = (0..28_800).map(|n| (pluck(n, 0.0, 220.0) + pluck(n, 0.3, 330.0)) as f32);
    read(&[x.collect()], RATE)
}

/// The starter written at `COARSE`, and what it varies.
fn coarse(r: &Readings) -> (Composition, Vec<Vary>) {
    written(r, COARSE)
}

fn written(r: &Readings, at: Limits) -> (Composition, Vec<Vary>) {
    let s = starter(r, "clip.wav", at);
    let v = s.varies.iter().map(|v| vary(v).expect("a vary")).collect();
    (s.files.into_iter().collect(), v)
}

fn grown(c: &Composition, v: &[Vary], r: &Readings, to: Limits) -> Vec<Proposal> {
    grown_from(c, v, r, COARSE, to)
}

fn grown_from(
    c: &Composition,
    v: &[Vary],
    r: &Readings,
    from: Limits,
    to: Limits,
) -> Vec<Proposal> {
    grow::proposals(&model(c), &|path| text(c, path), v, r, from, to).expect("a growth")
}

fn voice_nodes(c: &Composition) -> Vec<String> {
    let m = model(c);
    let voices = m
        .nodes
        .keys()
        .filter(|n| n.starts_with("instruments/voice-"));
    voices.cloned().collect()
}

fn names(range: std::ops::RangeInclusive<usize>) -> Vec<String> {
    range.map(|k| format!("h{k}")).collect()
}

#[test]
fn growing_to_the_limits_written_proposes_nothing() {
    let r = plucks();
    let (c, v) = coarse(&r);
    assert_eq!(grown(&c, &v, &r, COARSE), Vec::new());
}

#[test]
fn growing_harmonics_offers_each_voice_kind_its_node_and_lines_at_eight() {
    let r = plucks();
    let (c, v) = coarse(&r);
    let kinds = voice_nodes(&c);
    let proposals = grown(&c, &v, &r, FINER);
    assert_eq!(proposals.len(), kinds.len(), "one growth per voice kind");
    for p in &proposals {
        let after = model(&edited(&c, &v, p));
        for node in &kinds {
            let defaults: Vec<&String> =
                after.nodes[node].defaults.iter().map(|(n, _)| n).collect();
            for h in names(1..=8) {
                assert!(defaults.contains(&&h), "`{node}` reads `{h}`");
            }
            let lines: Vec<_> = after.lines.iter().filter(|l| l.node == *node).collect();
            assert!(!lines.is_empty(), "`{node}` places a line");
            for l in lines {
                for h in names(5..=8) {
                    let held = l.fields.iter().any(|(f, _)| *f == h);
                    assert!(held, "the line at {} s holds `{h}`", l.at);
                }
            }
        }
    }
}

#[test]
fn a_grown_node_is_the_finer_seed_node_keeping_each_default_it_wrote() {
    let r = plucks();
    let (mut c, v) = coarse(&r);
    let node = voice_nodes(&c).remove(0);
    let fitted = text(&c, &node)
        .expect("the voice node")
        .replace("tilt = 1\n", "tilt = 1.25\n")
        .replace("h3 = 0\n", "h3 = -3\n")
        .replace("damp = 0\n", "")
        .replace("*exp(0 - damp*k*f0*t/1000)", "");
    c.insert(node.clone(), fitted);
    let p = grown(&c, &v, &r, FINER).remove(0);
    let expected = harmonic(8)
        .replace("tilt = 1\n", "tilt = 1.25\n")
        .replace("h3 = 0\n", "h3 = -3\n");
    assert_eq!(text(&edited(&c, &v, &p), &node), Some(expected));
}

#[test]
fn a_line_the_coarse_score_dropped_gets_no_graft() {
    let r = plucks();
    let (full, v) = coarse(&r);
    let score = "score/clip.jsonl";
    let mut first = true;
    let kept: String = text(&full, score)
        .expect("the score")
        .lines()
        .filter(|row| {
            let voice = row.contains("\"@../instruments/voice-");
            let keep = !voice || first;
            first &= !voice;
            keep
        })
        .map(|row| format!("{row}\n"))
        .collect();
    let mut dropped = full.clone();
    dropped.insert(score.to_string(), kept);
    let voice_lines = |c: &Composition| -> Vec<(f64, Vec<(String, f64)>)> {
        let m = model(&edited(c, &v, &grown(c, &v, &r, FINER).remove(0)));
        let lines = m.lines.into_iter().filter(|l| l.node.contains("/voice-"));
        lines.map(|l| (l.at, l.fields)).collect()
    };
    let (all, left) = (voice_lines(&full), voice_lines(&dropped));
    assert_eq!(all.len(), 2, "the coarse score places two voice lines");
    assert_eq!(
        left,
        all[..1],
        "the dropped line stays dropped, the kept one grafted alike"
    );
}

/// Each lone partial line's onset, `f0` and `vel`.
fn partials(m: &Model) -> Vec<(f64, Option<f64>, Option<f64>)> {
    let lines = m.lines.iter().filter(|l| l.node == "instruments/partial");
    let field = |l: &sva_core::model::Line, name: &str| {
        l.fields.iter().find(|(f, _)| f == name).map(|f| f.1)
    };
    lines
        .map(|l| (l.at, field(l, "f0"), field(l, "vel")))
        .collect()
}

#[test]
fn growing_lone_partials_adds_sounding_ones_at_each_onset_in_order() {
    let r = plucks();
    for lone in [0, 1] {
        let from = Limits { harmonics: 4, lone };
        let to = Limits {
            harmonics: 4,
            lone: lone + 2,
        };
        let (c, v) = written(&r, from);
        let before = partials(&model(&c));
        let proposals = grown_from(&c, &v, &r, from, to);
        assert_eq!(proposals.len(), 2, "one growth per onset from {from:?}");
        for p in &proposals {
            let after = partials(&model(&edited(&c, &v, p)));
            let added: Vec<_> = after.iter().filter(|l| !before.contains(l)).collect();
            assert_eq!(added.len(), 2, "`{}` adds two partials", p.label);
            for (_, _, vel) in added {
                let vel = vel.expect("an added partial writes its `vel`");
                assert!(vel > 0.0, "`{}` enters at its reading's level", p.label);
            }
        }
    }
}
