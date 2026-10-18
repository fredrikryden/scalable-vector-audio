// Concern: proves a ledger names every ref under its target, whatever each holds | Non-concern: the arithmetic of one entry (sva-samples) | IO: (a composition) -> LedgerEntry per node

use crate::fixtures::graph_of;
use sva_engine::{
    Ask, LedgerEntry, Output, Range, RenderConfig, Representation, Tier, answer, render,
};

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

/// Forwarding one ref is not a reason to vanish: the file is a node and carries all of it.
#[test]
fn a_pass_through_node_appears_in_the_ledger() {
    let entries = ledger(
        "pass-through",
        &[
            ("tone", "0.4*sin(2*pi*200*t)\n"),
            ("voice", "@tone(t)\n"),
            ("master", "@voice(t)\n"),
        ],
        3,
    );

    let held = names(&entries);
    assert!(held.contains(&"voice"), "{held:?}");
    assert_eq!(
        named(&entries, "voice").share,
        Some(1.0),
        "a node that forwards one ref forwards all of its energy"
    );
    assert!(held.contains(&"tone"), "{held:?}");
}

/// Two chains reach one node at two depths. The reading walks level by level, so the node
/// is held at the shorter chain's depth, and a longer chain reaching it first must not
/// spend the budget that would have carried its own children.
#[test]
fn a_node_two_chains_reach_is_held_at_the_shorter_one() {
    let entries = ledger(
        "diamond",
        &[
            ("leaf", "0.2*sin(2*pi*500*t)\n"),
            ("near", "@leaf(t)\n"),
            ("far", "@near(t)\n"),
            ("master", "@far(t) + @near(t)\n"),
        ],
        2,
    );

    let held = names(&entries);
    assert!(held.contains(&"near"), "depth 1 through `near`: {held:?}");
    assert!(held.contains(&"far"), "depth 1 through `far`: {held:?}");
    assert!(
        held.contains(&"leaf"),
        "`leaf` sits two hops down the short chain: {held:?}"
    );
    assert!(
        named(&entries, "leaf").rms > 0.0,
        "a named node is a rendered one, never a zero standing in for one"
    );
}

/// An entry is what a ref contributed to the node reading it, at that node's own offset, not
/// what the ref's own buffer holds from its own zero.
#[test]
fn a_shifted_section_contributes_nothing_before_its_bar() {
    let entries = ledger(
        "shifted-section",
        &[
            ("sections/s1", "crop(0.5*sin(2*pi*220*t), 0s, 2s)\n"),
            ("sections/s2", "crop(0.5*sin(2*pi*330*t), 0s, 2s)\n"),
            (
                "master",
                "crop(@sections/s1(t) + @sections/s2(t - 2s), 0s, 4s)\n",
            ),
        ],
        1,
    );

    let early = named(&entries, "sections/s2");
    assert_eq!(early.rms, 0.0, "s2 sounds from the third second on");
    assert_eq!(early.share, Some(0.0));
    let sounding = named(&entries, "sections/s1");
    assert!(
        (sounding.share.expect("an audible share") - 1.0).abs() < 1e-9,
        "s1 is the whole of the first second: {:?}",
        sounding.share
    );
}

/// A share is of the node reading the ref, not of the target two hops up.
#[test]
fn a_ref_is_a_share_of_the_node_reading_it() {
    let entries = ledger(
        "share-of-its-reader",
        &[
            ("voice", "0.5*sin(2*pi*220*t)\n"),
            ("section", "@voice(t)\n"),
            ("master", "0.1*@section(t)\n"),
        ],
        2,
    );

    let section = named(&entries, "section");
    assert!(
        (section.share.expect("an audible share") - 1.0).abs() < 1e-9,
        "the section is the whole of master's own level: {:?}",
        section.share
    );
    let voice = named(&entries, "voice");
    assert!(
        (voice.share.expect("an audible share") - 1.0).abs() < 1e-9,
        "and the voice is the whole of the section's, not ten times master's: {:?}",
        voice.share
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

/// An inline solver holds no slot; the factor beside it is still no addend of the product.
#[test]
fn a_constant_factor_beside_an_inline_solver_has_no_share() {
    let entries = ledger(
        "factor-beside-a-solver",
        &[
            ("gain", "0.5\n"),
            ("master", "sample(@gain) * chaigne_askenfelt(261.63)\n"),
        ],
        1,
    );
    let gain = named(&entries, "gain");
    assert_eq!(gain.share, None, "`gain` is a factor, not an addend");
    assert!(
        (gain.rms - 0.5).abs() < 1e-12,
        "`gain` states what it holds, not the product: {}",
        gain.rms
    );
    assert_eq!(named(&entries, "master").share, Some(1.0));
}

/// A self-read written inline moves as a slot does, so the factor beside it is no addend of
/// the product.
#[test]
fn a_constant_factor_beside_a_self_read_has_no_share() {
    let master = "sample(@gain) * self[idx(t) - 1] + sample(0.25)\n";
    let entries = ledger("self-read", &[("gain", "0.5\n"), ("master", master)], 1);
    let gain = named(&entries, "gain");
    assert_eq!(gain.share, None, "`gain` is a factor, not an addend");
    assert!(
        (gain.rms - 0.5).abs() < 1e-12,
        "`gain` states its own 0.5, not the product: {}",
        gain.rms
    );
    assert_eq!(named(&entries, "master").share, Some(1.0));
}

/// A ref the window does not reach contributed nothing to it, whatever its own buffer holds.
#[test]
fn a_ref_outside_the_window_has_zero_share() {
    let entries = ledger(
        "outside-the-window",
        &[
            ("early", "crop(0.5*sin(2*pi*220*t), 0s, 0.5s)\n"),
            ("late", "crop(0.5*sin(2*pi*330*t), 0s, 0.5s)\n"),
            ("master", "@early(t) + @late(t - 2s)\n"),
        ],
        1,
    );

    assert_eq!(named(&entries, "late").share, Some(0.0));
    assert_eq!(named(&entries, "late").rms, 0.0);
    assert!(
        (named(&entries, "early").share.expect("an audio share") - 1.0).abs() < 1e-12,
        "the one ref the window reaches is the whole of it"
    );
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

/// A ledger over a range that starts late sums its energies over that range alone: a section
/// silent there accounts for none of it.
#[test]
fn a_ledger_over_a_late_range_shares_that_range_alone() {
    let g = graph_of(
        "windowed-ledger",
        &[
            ("early", "crop(0.8*sin(2*pi*200*t), 0s, 0.5s)\n"),
            ("late", "crop(0.3*sin(2*pi*300*t), 0.5s, 1s)\n"),
            ("master", "@early(t) + @late(t)\n"),
        ],
    );
    let depth = 1;
    let asks = vec![Ask {
        node: "master".to_string(),
        representation: Representation::Ledger { depth },
    }];
    let config = RenderConfig {
        range: Range {
            start: Some(4_000),
            end: Some(8_000),
        },
        ..RenderConfig::at(8_000).asking(asks)
    };
    let held = render(&g, "master", config, &Tier::default()).expect("a render");
    let id = held.id("master").expect("the root");
    let Output::Ledger(second_half) = answer(&held, id, Representation::Ledger { depth })
        .expect("a ledger")
        .value
    else {
        panic!("expected a ledger");
    };
    assert_eq!(named(&second_half, "early").share, Some(0.0));
    assert_eq!(named(&second_half, "early").rms, 0.0);
    let late = named(&second_half, "late");
    assert!(
        (late.share.expect("an audible share") - 1.0).abs() < 1e-9,
        "late is the whole of the second half: {late:?}"
    );
}

/// Depth counts refs from file to file: a factor two files down states what it holds over
/// the whole range, past where its reader's crop stops reading it.
#[test]
fn a_factor_two_files_down_states_what_it_holds_over_the_range() {
    let entries = ledger(
        "nested-factor",
        &[
            ("noise", "2*sample(rand(t, seed=3)) - 1\n"),
            ("shape", "crop(exp(-t/0.1), 0s, 0.25s)\n"),
            (
                "voice",
                "lowpass(@noise, cutoff=2000, q=0.7)*sample(@shape)\n",
            ),
            ("master", "@voice\n"),
        ],
        2,
    );
    let noise = named(&entries, "noise");
    assert_eq!(noise.share, None, "`noise` is a factor, not an addend");
    assert!(
        (noise.rms - (1.0f64 / 3.0).sqrt()).abs() < 0.05,
        "`noise` states its own level over the whole second: {}",
        noise.rms
    );
}

/// A row read one step late is computed to the range's end, factors under it included.
#[test]
fn a_row_read_late_states_what_it_holds_to_the_end() {
    let entries = ledger(
        "late-row",
        &[
            ("noise", "2*sample(rand(t, seed=3)) - 1\n"),
            ("shape", "crop(exp(-t/0.1), 0s, 2s)\n"),
            (
                "voice",
                "lowpass(@noise, cutoff=2000, q=0.7)*sample(@shape)\n",
            ),
            ("master", "@voice(t - 18sp)\n"),
        ],
        3,
    );
    for node in ["noise", "shape"] {
        let held = named(&entries, node);
        assert_eq!(held.share, None, "`{node}` is a factor, not an addend");
        assert!(held.rms > 0.0, "`{node}` states what it holds");
    }
    assert_eq!(named(&entries, "voice").share, Some(1.0));
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

/// A ledger holds each row a block at a time, never over the whole range: a longer range of
/// sixteen rows, chirps so no period folds them, grows what it holds by what it grows the
/// render of their sum alone.
#[test]
fn a_ledger_holds_its_rows_a_block_at_a_time() {
    let mut files: Vec<(String, String)> = (0..16)
        .map(|i| {
            let hz = 100 + 37 * i;
            (
                format!("v{i}"),
                format!("sample(0.05*sin(2*pi*{hz}*t + 3*t*t))\n"),
            )
        })
        .collect();
    let sum: Vec<String> = (0..16).map(|i| format!("@v{i}")).collect();
    files.push(("master".to_string(), format!("{}\n", sum.join(" + "))));
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let g = graph_of("ledger-memory", &files);
    let held = |secs: f64, representation: Representation| {
        let mut config = RenderConfig::seconds(8_000, secs).asking(vec![Ask {
            node: "master".to_string(),
            representation,
        }]);
        config.threads = std::num::NonZeroUsize::MIN;
        render(&g, "master", config, &Tier::default()).expect("a render")
    };
    let ledger = Representation::Ledger { depth: 1 };
    let grown = |representation| {
        held(120.0, representation).held_bytes - held(30.0, representation).held_bytes
    };
    let (plain, tallied) = (grown(Representation::Samples), grown(ledger));
    let row = 90 * 8_000 * std::mem::size_of::<f64>();
    assert!(
        tallied <= plain + row / 8,
        "ninety more seconds grew a ledger of 16 rows by {tallied} bytes, the plain render by \
         {plain}; one row over them is {row}"
    );
    let long = held(120.0, ledger);
    let id = long.id("master").expect("the root");
    let Output::Ledger(entries) = answer(&long, id, ledger).expect("a ledger").value else {
        panic!("expected a ledger");
    };
    assert_eq!(entries.len(), 17, "{:?}", names(&entries));
    let shares: f64 = entries.iter().skip(1).filter_map(|e| e.share).sum();
    assert!(
        (shares - 1.0).abs() < 1e-9,
        "the parts are the whole: {shares}"
    );
}
