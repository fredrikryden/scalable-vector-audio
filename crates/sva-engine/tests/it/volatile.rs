// Concern: proves a volatile render sounds as a plain one and a volatile name must be bound | Non-concern: memory's cap or evictions (stores.rs) | IO: (a composition, names) -> CacheStats

use crate::fixtures::{graph_of, samples};
use sva_ast::Graph;
use sva_engine::{Render, RenderConfig, Tier, render};

const SECONDS: f64 = 0.05;
const RATE: u32 = 8_000;

/// A note through a tone knob, and a send that never reads the knob.
fn mix(cutoff: f64) -> Graph {
    graph_of(
        "volatile",
        &[
            (
                "note",
                "sample(sin(2*pi*220*t) + 0.3*sin(2*pi*660*t))*0.5\n",
            ),
            ("tone", "lowpass(x, cutoff=cutoff, q=0.7)\n"),
            ("send", "lowpass(x, cutoff=3000, q=0.5)*0.2\n"),
            (
                "master",
                &format!("@tone(t, x=@note, cutoff={cutoff}) + @send(t, x=@note)\n"),
            ),
        ],
    )
}

fn run(graph: &Graph, volatile: &[&str], cache: &Tier) -> Result<Render, sva_engine::EngineError> {
    let mut config = RenderConfig::seconds(RATE, SECONDS);
    config.volatile = volatile.iter().map(|n| (*n).to_string()).collect();
    render(graph, "master", config, cache)
}

#[test]
fn a_volatile_render_sounds_exactly_as_a_plain_one() {
    for cutoff in [300.0, 1500.0] {
        let store = Tier::default();
        let volatile = run(&mix(cutoff), &["cutoff"], &store).expect("a render");
        let plain = run(&mix(cutoff), &[], &Tier::default()).expect("a render");
        assert_eq!(samples(&volatile), samples(&plain), "at {cutoff}");
        let answered = run(&mix(cutoff), &["cutoff"], &store).expect("a render");
        assert_eq!(
            samples(&answered),
            samples(&plain),
            "and from the store at {cutoff}"
        );
    }
}

#[test]
fn a_name_the_target_binds_nowhere_is_refused() {
    let Err(refused) = run(&mix(400.0), &["cutof"], &Tier::default()) else {
        panic!("a misspelled knob renders nothing")
    };
    assert_eq!(refused.code(), "render.volatile_unbound");
    assert!(refused.to_string().contains("cutof"), "{refused}");
}
