// Concern: states a structure edit rewrites a composition's text: a move plays `@master` bit for bit, a growth grafts a finer starter | Non-concern: whether an edit pays | IO: (composition) -> files

use sva_ast::Composition;
use sva_core::model::Model;
use sva_core::seed::{Limits, Readings, read, starter};
use sva_core::solve::{Vary, vary};
use sva_core::structure::{Proposal, applied, assign, grow, merge, split};
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

const RATE: f64 = 48_000.0;

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

fn written(r: &Readings, at: Limits) -> (Composition, Vec<Vary>) {
    let s = starter(r, "clip.wav", at);
    let v = s.varies.iter().map(|v| vary(v).expect("a vary")).collect();
    (s.files.into_iter().collect(), v)
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
fn growing_lone_partials_adds_sounding_ones_at_each_hit_in_order() {
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
        let onsets: Vec<f64> = proposals
            .iter()
            .filter_map(|p| p.label.strip_prefix("lone partials at "))
            .map(|at| at.trim_end_matches('s').parse().expect("an onset"))
            .collect();
        assert_eq!(
            onsets.len(),
            proposals.len(),
            "one growth per hit from {from:?}"
        );
        for pluck in [0.0, 0.3] {
            assert!(
                onsets.iter().any(|at: &f64| (at - pluck).abs() <= 0.005),
                "a growth at the pluck at {pluck} s from {from:?}: {onsets:?}"
            );
        }
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
