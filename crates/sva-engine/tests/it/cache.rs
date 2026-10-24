// Concern: proves a warm render answers as a cold one did and an edit re-renders only what it reaches | Non-concern: the cap (stores.rs) | IO: (a composition, twice) -> the same bytes

use std::fs;
use std::path::{Path, PathBuf};

use crate::fixtures::{dir_of, samples};
use sva_engine::{Ask, Render, RenderConfig, Representation, Tier, render};

const SECONDS: f64 = 0.05;
const RATE: u32 = 8_000;

fn store_all() -> Tier {
    Tier::default()
}

fn rendered(dir: &Path, root: &str, cache: &Tier) -> Render {
    let graph = sva_ast::parse_composition(dir).expect("a composition that parses");
    render(&graph, root, RenderConfig::seconds(RATE, SECONDS), cache)
        .unwrap_or_else(|e| panic!("rendering `{root}`: {e}"))
}

/// Every node held as samples, which is what a reading that consumes buffers asks for.
fn all_of(dir: &Path, root: &str, nodes: &[&str], cache: &Tier) -> Render {
    let graph = sva_ast::parse_composition(dir).expect("a composition that parses");
    let asks = nodes
        .iter()
        .map(|node| Ask {
            node: (*node).to_string(),
            representation: Representation::Samples,
        })
        .collect();
    render(
        &graph,
        root,
        RenderConfig::seconds(RATE, SECONDS).asking(asks),
        cache,
    )
    .unwrap_or_else(|e| panic!("rendering `{root}`: {e}"))
}

fn write(dir: &Path, rel: &str, text: &str) {
    fs::write(dir.join(rel), text).expect("a node file");
}

/// One composition of a chord, a voicing over it and a master that sums both.
fn chain(name: &str) -> PathBuf {
    dir_of(
        name,
        &[
            ("chord", "sin(2*pi*256*t) + sin(2*pi*384*t)\n"),
            ("voiced", "@chord*0.5\n"),
            ("master", "@voiced + @chord*0.25\n"),
        ],
    )
}

#[test]
fn a_warm_render_is_byte_identical_to_a_cold_one() {
    let dir = chain("warm-cold");
    let cache = store_all();
    let cold = samples(&rendered(&dir, "master", &cache));
    assert!(cache.bytes() > 0, "the cold render filled the store");
    let warm = samples(&rendered(&dir, "master", &cache));
    assert_eq!(cold, warm);
}

#[test]
fn editing_one_file_re_renders_it_and_its_dependents_and_nothing_else() {
    let dir = chain("edited");
    let cache = store_all();
    let nodes = ["chord", "voiced", "master"];
    let before = samples(&all_of(&dir, "master", &nodes, &cache));
    write(&dir, "voiced", "@chord*0.75\n");
    let after = samples(&all_of(&dir, "master", &nodes, &cache));
    assert_eq!(
        before["chord"], after["chord"],
        "the untouched law is one value"
    );
    assert_ne!(before["voiced"], after["voiced"]);
    assert_ne!(before["master"], after["master"]);
}

/// A loop of refs closes through no `self`, so nothing substitutes it: it says so rather
/// than unrolling forever.
#[test]
fn a_loop_of_refs_refuses_rather_than_substituting_forever() {
    let dir = dir_of(
        "ref-loop",
        &[
            ("a", "sin(2*pi*220*t) + 0.5*@b(t - 0.01s)\n"),
            ("b", "@a(t - 0.01s)*0.5\n"),
            ("master", "@a\n"),
        ],
    );
    let graph = sva_ast::parse_composition(&dir).expect("a composition");
    let refused = render(
        &graph,
        "master",
        RenderConfig::seconds(RATE, SECONDS),
        &Tier::default(),
    );
    let Err(refused) = refused else {
        panic!("a loop of refs has no law");
    };
    assert_eq!(refused.code(), "engine.cyclic_substitution");
}

/// The label is part of the value a collapse produced, so it is stored with the
/// buffer and answered on a hit. Warm and cold are one command answering one way.
#[test]
fn a_warm_hit_carries_the_cold_label() {
    let dir = dir_of(
        "warm-label",
        &[("chord", "sin(2*pi*256*t) + sin(2*pi*384*t)\n")],
    );
    let label_of = |cache: &Tier| {
        let held = rendered(&dir, "chord", cache);
        held.labels
            .get(&held.root)
            .expect("the root collapsed")
            .clone()
    };

    let cache = Tier::new(1 << 20);
    let cold = label_of(&cache);
    assert_eq!(
        cold.source,
        sva_engine::Source::Exact,
        "the cold run is exact"
    );
    let sva_engine::Detail::Lines { summed, .. } = &cold.detail else {
        panic!("a line spectrum states what it summed: {cold:?}");
    };
    assert!(*summed > 0, "the lines are counted");

    let warm = label_of(&cache);
    assert_eq!(
        warm.detail, cold.detail,
        "a hit answers the row the cold run took"
    );
    assert_eq!(warm.source, cold.source);
    assert_eq!(warm.rate, cold.rate);
    assert_eq!(warm.profile, cold.profile);
}

/// A profile's floor decides how many terms a series keeps, so a render under another floor
/// computes its own value rather than answering with the first profile's.
#[test]
fn a_render_under_another_floor_is_not_answered_by_the_first() {
    let dir = dir_of(
        "floors",
        &[("node", "sum(k, 1, inf, cos(2*pi*440*t)/(k*k))\n")],
    );
    let graph = sva_ast::parse_composition(&dir).expect("a composition that parses");
    let under = |floor_db: f64, cache: &Tier| {
        let config = RenderConfig {
            profile: sva_engine::Profile {
                floor_db,
                floor_db_above_5k: floor_db,
                ..sva_engine::PSYCHOACOUSTIC_V1
            },
            ..RenderConfig::seconds(RATE, SECONDS)
        };
        samples(&render(&graph, "node", config, cache).expect("a render"))
    };
    let cold = under(-60.0, &store_all());
    assert!(
        cold != under(-20.0, &store_all()),
        "the floor moves the samples"
    );
    let cache = store_all();
    under(-20.0, &cache);
    assert!(
        under(-60.0, &cache) == cold,
        "a warm render under another floor is the cold one"
    );
}
