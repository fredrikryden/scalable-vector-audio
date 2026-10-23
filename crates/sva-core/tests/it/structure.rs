// Concern: states a structure edit rewrites a composition's text and leaves what `@master` plays bit for bit | Non-concern: whether an edit pays | IO: (two kin nodes, a score) -> files, samples

use sva_ast::Composition;
use sva_core::model::Model;
use sva_core::solve::{Vary, vary};
use sva_core::structure::{Edit, Proposal, applied, assign, merge, split};
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

/// `p`'s files laid over the composition.
fn edited(c: &Composition, p: &Proposal) -> Composition {
    let texts = |path: &str| {
        sva_ast::Source::get(c, path)
            .ok()
            .flatten()
            .map(|t| t.into_owned())
    };
    let files = applied(&model(c), &texts, &varies(), p).expect("the edit applies");
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
            samples(&edited(&c, p)) == before,
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
        let after = model(&edited(&c, &p));
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
    let after = model(&edited(&c, &p));
    assert!(after.kin("instruments/a", "instruments/a-2"));
}
