// Concern: proves a render that skips exactly-zero work writes the bits of the work done whole | Non-concern: what any row computes (sva-samples) | IO: (a composition, a range) -> samples

use crate::fixtures::graph_of;
use sva_ast::Graph;
use sva_engine::{Extent, Range, Render, RenderConfig, Tier, render};

const RATE: u32 = 8_000;

fn notes() -> Graph {
    graph_of(
        "skipping-exact-zero-work",
        &[
            ("note", "crop(sin(2*pi*440*t)*exp(-t/0.2), 0s, 0.5s)\n"),
            ("song", "@note(t) + @note(t - 2s) + @note(t - 4s)\n"),
        ],
    )
}

fn sampled_notes() -> Graph {
    graph_of(
        "skipping-exact-zero-work-sampled",
        &[
            (
                "note",
                "crop(lowpass(sample(0.5*sin(2*pi*440*t)), cutoff=2000, q=0.7), 0s, 0.5s)\n",
            ),
            ("song", "@note(t) + @note(t - 2s) + @note(t - 4s)\n"),
        ],
    )
}

fn over(g: &Graph, target: &str, secs: f64, cache: &Tier) -> Render {
    render(g, target, config(secs), cache).unwrap_or_else(|e| panic!("{target}: {e}"))
}

fn config(secs: f64) -> RenderConfig {
    RenderConfig {
        range: Range {
            start: Some(0),
            end: Some((secs * f64::from(RATE)) as i64),
        },
        ..RenderConfig::at(RATE)
    }
}

/// A song reading one note three times computes the note once, over its own half second:
/// each sample is the note's own samples added from +0 in the order written, and a render
/// reaching further finds the note whole in the store.
#[test]
fn a_note_read_three_times_is_computed_once_and_added() {
    let g = notes();
    let cache = Tier::default();
    let held = over(&g, "song", 6.0, &cache);
    let note = held.id("note").expect("the note");
    assert_eq!(
        held.evaluated(note),
        vec![Extent::new(0, i64::from(RATE / 2))]
    );
    let alone = over(&g, "note", 6.0, &Tier::default());
    let own = alone
        .output(alone.root)
        .expect("the note")
        .plane(0)
        .to_vec();
    let at = |n: i64| {
        usize::try_from(n)
            .ok()
            .and_then(|n| own.get(n))
            .copied()
            .unwrap_or(0.0)
    };
    let samples = held.output(held.root).expect("the song").plane(0).to_vec();
    let gap = 2 * i64::from(RATE);
    for (n, sample) in samples.iter().enumerate() {
        let n = n as i64;
        let added = 0.0 + at(n) + at(n - gap) + at(n - 2 * gap);
        assert_eq!(sample.to_bits(), added.to_bits(), "sample {n}");
    }

    let further = over(&g, "song", 8.0, &cache);
    let stats = further.cache_stats.as_ref().expect("stats");
    assert_eq!(stats.hits(), stats.lookups.len(), "{stats:?}");
    let reread = further.output(further.root).expect("the song").plane(0)[..samples.len()].to_vec();
    assert_eq!(reread, samples);
}

/// A term scaled by constants is as zero as its read where that read is, so a sum of scaled
/// notes, as a grid's rows are, writes the same bits.
#[test]
fn a_scaled_sampled_term_is_read_only_where_it_is_nonzero() {
    let g = sampled_notes();
    let scaled = graph_of(
        "skipping-exact-zero-work-scaled",
        &[
            (
                "note",
                "crop(lowpass(sample(0.5*sin(2*pi*440*t)), cutoff=2000, q=0.7), 0s, 0.5s)\n",
            ),
            (
                "song",
                "@note(t)*0.5 + @note(t - 2s)*-0.75 + @note(t - 4s)*0.25\n",
            ),
        ],
    );
    let held = over(&scaled, "song", 4.5, &Tier::default());
    let song = held.output(held.root).expect("the song").plane(0).to_vec();
    let alone = over(&g, "note", 4.5, &Tier::default());
    let note = alone
        .output(alone.root)
        .expect("the note")
        .plane(0)
        .to_vec();
    let at = |n: i64| {
        usize::try_from(n)
            .ok()
            .and_then(|n| note.get(n))
            .copied()
            .unwrap_or(0.0)
    };
    let gap = i64::from(RATE) * 2;
    for (n, sample) in song.iter().enumerate() {
        let n = n as i64;
        let added = 0.0 + at(n) * 0.5 + at(n - gap) * -0.75 + at(n - 2 * gap) * 0.25;
        assert_eq!(sample.to_bits(), added.to_bits(), "sample {n}");
    }
}

/// A raised-cosine edge clamped shut holds its factor at exactly zero once its line passes
/// the clamp: the tone it shapes is computed only up to there, and renders the bits it does
/// with nothing cut.
#[test]
fn a_clamped_edge_ends_what_it_shapes_where_it_holds_zero() {
    let shut = "(1 - (0.5 - 0.5*cos(pi*min(1, max(0, (t - 0.1)/0.03)))))";
    let g = graph_of(
        "skipping-exact-zero-work-clamp",
        &[
            ("tone", &format!("crop({shut}*sin(2*pi*200*t), 0s, 2s)\n")),
            ("top", "sample(@tone(t))\n"),
        ],
    );
    let config = |silence_threshold_dbfs| RenderConfig {
        profile: sva_engine::Profile {
            silence_threshold_dbfs,
            ..sva_engine::PSYCHOACOUSTIC_V1
        },
        ..RenderConfig::seconds(RATE, 2.0)
    };
    let held = render(&g, "top", config(-120.0), &Tier::default()).expect("a render");
    let tone = held.id("tone").expect("the tone");
    let shut_at = (0.13 * f64::from(RATE)) as i64;
    let computed = held.evaluated(tone);
    assert!(!computed.is_empty(), "the tone is a value of its own");
    assert!(
        computed.iter().all(|e| e.end <= shut_at + 2),
        "{computed:?} past {shut_at}"
    );
    let exact = render(&g, "top", config(f64::NEG_INFINITY), &Tier::default()).expect("a render");
    let bits = |r: &Render| -> Vec<u64> {
        let out = r.output(r.root).expect("the root");
        out.plane(0).iter().map(|v| v.to_bits()).collect()
    };
    assert_eq!(bits(&held), bits(&exact));
}
