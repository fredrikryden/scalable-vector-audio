// Concern: proves a ledger names every ref under its target, whatever each holds | Non-concern: the arithmetic of one entry (sva-samples) | IO: (a composition) -> LedgerEntry per node

use crate::fixtures::graph_of;
use sva_engine::{Ask, LedgerEntry, Output, RenderConfig, Representation, Tier, answer, render};

fn ledger(name: &str, files: &[(&str, &str)], depth: usize) -> Vec<LedgerEntry> {
    let g = graph_of(name, files);
    let asks = vec![Ask {
        node: "master".to_string(),
        representation: Representation::Ledger { depth },
    }];
    let held = render(
        &g,
        "master",
        RenderConfig::seconds(8_000, 1.0).asking(asks),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    let id = held.id("master").expect("the root");
    match answer(&held, id, Representation::Ledger { depth })
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .value
    {
        Output::Ledger(entries) => entries,
        other => panic!("expected a ledger, got {other:?}"),
    }
}

fn named<'a>(entries: &'a [LedgerEntry], node: &str) -> &'a LedgerEntry {
    entries
        .iter()
        .find(|e| e.node == node)
        .unwrap_or_else(|| panic!("`{node}` is not in the ledger: {:?}", names(entries)))
}

fn names(entries: &[LedgerEntry]) -> Vec<&str> {
    entries.iter().map(|e| e.node.as_str()).collect()
}

/// The tree a ledger walks is the ref tree, and a closed form's refs are written in its own body.
#[test]
fn a_law_master_ledgers_each_ref_share() {
    let entries = ledger(
        "law-master",
        &[
            ("quiet", "0.1*sin(2*pi*200*t)\n"),
            ("loud", "0.8*sin(2*pi*300*t)\n"),
            ("master", "@quiet(t) + @loud(t)\n"),
        ],
        3,
    );

    assert_eq!(
        named(&entries, "master").share,
        Some(1.0),
        "the target carries its own whole energy"
    );
    let quiet = named(&entries, "quiet");
    let loud = named(&entries, "loud");
    assert!(
        (quiet.rms - 0.1 / 2f64.sqrt()).abs() < 1e-3,
        "a sine at 0.1 reads its own rms: {}",
        quiet.rms
    );
    assert!(
        loud.share.expect("an audio share") > quiet.share.expect("an audio share"),
        "the louder ref carries the larger share: {quiet:?} {loud:?}"
    );
}

/// A share is a share of a sum; crediting a factor with the whole product counts it twice.
#[test]
fn a_factor_of_a_product_has_no_share() {
    let entries = ledger(
        "factor-of-a-product",
        &[
            ("carrier", "sin(2*pi*220*t)\n"),
            ("shape", "0.5 + 0.5*cos(2*pi*3*t)\n"),
            ("master", "@carrier(t)*@shape(t)\n"),
        ],
        1,
    );

    for node in ["carrier", "shape"] {
        let held = named(&entries, node);
        assert_eq!(held.share, None, "`{node}` is a factor, not an addend");
        assert!(held.rms > 0.0, "`{node}` still states what it holds");
    }
    assert_eq!(named(&entries, "master").share, Some(1.0));
}

/// A sampled node reads its refs through slots, and silencing every slot but one leaves what
/// that ref put into the buffer. Without that a sampled target answers one row: itself.
#[test]
fn ledger_walks_through_a_sampled_node_to_its_slots() {
    let entries = ledger(
        "sampled-master",
        &[
            (
                "hat",
                "crop(0.3*sin(2*pi*3000*t), 0s, 0.4s, rise=0.002s, fall=0.2s)\n",
            ),
            ("chord", "0.2*sin(2*pi*220*t)\n"),
            ("glue", "sample(@chord(t)) + 0.3*self[idx(t) - 1]\n"),
            (
                "master",
                "crop(0.6*(@glue(t) + sample(@hat(t)) + sample(@hat(t - 0.5s))), 0s, 1s)\n",
            ),
        ],
        2,
    );

    let held = names(&entries);
    for node in ["glue", "hat"] {
        assert!(
            held.contains(&node),
            "`{node}` is a buffer master reads: {held:?}"
        );
        assert!(named(&entries, node).rms > 0.0, "and it sounds");
    }
    let under_master: f64 = entries
        .iter()
        .filter(|e| e.depth == 1)
        .filter_map(|e| e.share)
        .sum();
    assert!(
        (under_master - 1.0).abs() < 1e-9,
        "the slots of a sum are the whole of it: {under_master}"
    );
    assert!(
        named(&entries, "hat").share.expect("an audio share") > 0.0,
        "both reads of one buffer stand in the one row that buffer gets"
    );
    assert!(
        held.contains(&"chord"),
        "and the walk carries on through `glue` into the closed form it sampled: {held:?}"
    );
}

/// A mono addend beside a stereo one is isolated by silencing the stereo read: the zero that
/// stands for it keeps both channels, so the run writes the width it holds and ends.
#[test]
fn a_mono_addend_beside_a_stereo_one_is_shared_in_every_channel() {
    let entries = ledger(
        "mono-beside-stereo",
        &[
            (
                "wide",
                "join(sample(sin(2*pi*440*t)), sample(sin(2*pi*550*t)))\n",
            ),
            ("mono", "sample(sin(2*pi*330*t))\n"),
            ("master", "@wide + 0.5*@mono\n"),
        ],
        2,
    );
    let rows =
        |node: &str| -> Vec<&LedgerEntry> { entries.iter().filter(|e| e.node == node).collect() };
    assert_eq!(
        rows("mono").len(),
        2,
        "one row per channel: {:?}",
        names(&entries)
    );
    for (wide, mono) in rows("wide").into_iter().zip(rows("mono")) {
        let (wide, mono) = (
            wide.share.expect("an addend"),
            mono.share.expect("an addend"),
        );
        assert!(
            (wide - 0.8).abs() < 1e-3 && (mono - 0.2).abs() < 1e-3,
            "a unit sine against a half one carries four fifths of the energy: {wide} {mono}"
        );
    }
}

fn rendered(g: &sva_ast::Graph, secs: f64, asks: &[(&str, Representation)]) -> sva_engine::Render {
    let asks = asks
        .iter()
        .map(|(node, representation)| Ask {
            node: node.to_string(),
            representation: *representation,
        })
        .collect();
    let config = RenderConfig::seconds(8_000, secs).asking(asks);
    render(g, "master", config, &Tier::default()).expect("a render")
}

/// Tallied block by block, a ledger states bit for bit the sums over whole buffers: a stateful
/// row's contribution, run beside it, is the samples that row's own run computes.
#[test]
fn a_tallied_ledger_states_the_sums_over_whole_buffers() {
    let g = graph_of(
        "tallied-is-whole",
        &[
            ("a", "sample(0.4*sin(2*pi*220*t))\n"),
            ("b", "sample(0.3*sin(2*pi*330*t) + 0.1*sin(2*pi*1250*t))\n"),
            ("f", "lowpass(@b, cutoff=900, q=0.707)\n"),
            ("master", "@a + @f\n"),
        ],
    );
    let secs = 3.0;
    let tallied = rendered(&g, secs, &[("master", Representation::Ledger { depth: 2 })]);
    let master = tallied.id("master").expect("the root");
    let Output::Ledger(entries) = answer(&tallied, master, Representation::Ledger { depth: 2 })
        .expect("a ledger")
        .value
    else {
        panic!("expected a ledger");
    };
    let whole = rendered(
        &g,
        secs,
        &[
            ("master", Representation::Samples),
            ("a", Representation::Samples),
            ("f", Representation::Samples),
        ],
    );
    let samples = |node: &str| {
        let held = whole.output(whole.id(node).expect(node)).expect(node);
        held.plane(0).to_vec()
    };
    let (master, a, f) = (samples("master"), samples("a"), samples("f"));
    let row = |x: &[f64], reader: &[f64]| {
        let energy: f64 = reader.iter().map(|v| v * v).sum();
        let product: f64 = x.iter().zip(reader).map(|(p, q)| p * q).sum();
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
        let peak = x.iter().fold(0f64, |m, v| m.max(v.abs()));
        let min = x.iter().fold(f64::INFINITY, |m, v| m.min(*v));
        let max = x.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        (rms, peak, min, max, product / energy)
    };
    let expected = vec![
        ("master", 0, row(&master, &master)),
        ("a", 1, row(&a, &master)),
        ("f", 1, row(&f, &master)),
        ("b", 2, row(&f, &f)),
    ];
    let found: Vec<_> = entries
        .iter()
        .map(|e| {
            let share = e.share.expect("an addend");
            (
                e.node.as_str(),
                e.depth,
                (e.rms, e.peak, e.min, e.max, share),
            )
        })
        .collect();
    assert_eq!(found, expected);
}
