// Concern: proves memory never passes its cap | Non-concern: what a key stands for (cache.rs) | IO: (a composition, a tier) -> what it holds

use crate::fixtures::graph_of;
use sva_ast::Graph;
use sva_engine::{CacheStats, RenderConfig, Tier, render};

const SECONDS: f64 = 0.05;
const RATE: u32 = 8_000;

/// Two sampled voices `a` and `b` each read.
fn forked(f: u32) -> Graph {
    graph_of(
        "forked",
        &[
            ("x", &format!("sample(sin(2*pi*{f}*t))*0.5\n")),
            ("y", &format!("sample(sin(2*pi*{}*t))*0.5\n", f + 1)),
            ("a", "@x*0.5 + @y*0.25\n"),
            ("b", "@x*0.25 + @y*0.5\n"),
            ("master", "@a + @b\n"),
        ],
    )
}

fn stats(graph: &Graph, cache: &Tier) -> CacheStats {
    render(graph, "master", RenderConfig::seconds(RATE, SECONDS), cache)
        .expect("a render")
        .cache_stats
        .expect("a render handed a store reports on it")
}

fn within(cache: &Tier) {
    assert!(
        cache.bytes() <= cache.max_bytes(),
        "{} over {}",
        cache.bytes(),
        cache.max_bytes()
    );
}

#[test]
fn the_cap_holds_after_every_render_and_a_lowered_cap() {
    let cache = Tier::new(20_000);
    let mut evicted = 0;
    for f in (1..=8).map(|k| 100 * k) {
        let after = stats(&forked(f), &cache);
        within(&cache);
        assert!(after.bytes <= after.max_bytes, "{after:?}");
        assert_eq!(after.entries, cache.entries());
        evicted += after.evictions;
    }
    assert!(evicted > 0, "eight renders overflow the cap");
    assert_eq!(evicted, cache.evictions(), "every eviction is reported");
    cache.set_max_bytes(4_000);
    within(&cache);
}
