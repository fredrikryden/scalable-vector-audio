// Concern: proves an open render, or a stream's term, ends where its bound at the root is under the silence threshold | Non-concern: deriving a bound | IO: (a composition, a profile) -> samples, the cut

use std::cell::RefCell;

use crate::fixtures::{Now, added, graph_of, next};
use sva_ast::Graph;
use sva_engine::{PSYCHOACOUSTIC_V1, Profile, Render, RenderConfig, Stream, StreamConfig, Tier};

const RATE: u32 = 8_000;

fn fades() -> Graph {
    graph_of(
        "faded",
        &[
            ("fade", "sin(2*pi*200*t)*exp(-t/0.15)\n"),
            ("tone", "0.5*sin(2*pi*300*t)\n"),
            ("mix", "@fade(t) + @tone(t)\n"),
            ("echo", "sample(x) + 0.5*self[idx(t - 0.25s)]\n"),
            ("echoed", "@echo(t, x=crop(sin(2*pi*200*t), 0s, 0.1s))\n"),
        ],
    )
}

fn rendered(g: &Graph, target: &str, profile: Profile) -> Render {
    let config = RenderConfig {
        profile,
        ..RenderConfig::at(RATE)
    };
    sva_engine::render(g, target, config, &Tier::default())
        .unwrap_or_else(|e| panic!("{target}: {e}"))
}

fn plane(r: &Render) -> Vec<f64> {
    r.output(r.root).expect("the root").plane(0).to_vec()
}

/// Where the render's own label says `node` was cut.
fn cut(r: &Render, node: &str) -> Option<i64> {
    let cutting = r.labels[&r.root]
        .cutting_below_silence_threshold
        .clone()
        .expect("a render states its silence threshold");
    cutting
        .treated_as_silent_from_sample
        .iter()
        .find(|(n, _)| n == node)
        .map(|(_, at)| *at)
}

fn at(profile_db: f64) -> Profile {
    Profile {
        silence_threshold_dbfs: profile_db,
        ..PSYCHOACOUSTIC_V1
    }
}

/// `exp(-t/0.15)` falls under -120 dBFS a little past 2.07 s: an open render of the fade ends
/// at the sample its bound stays under, and says so.
#[test]
fn a_decayed_sound_ends_where_its_bound_stays_under_the_silence_threshold() {
    let g = fades();
    let fade = rendered(&g, "fade", PSYCHOACOUSTIC_V1);
    let cutting = fade.labels[&fade.root]
        .cutting_below_silence_threshold
        .clone()
        .expect("a stated silence threshold");
    assert_eq!(cutting.silence_threshold_dbfs, -120.0);
    let from = cut(&fade, "fade").expect("the fade is cut");
    let seconds = from as f64 / f64::from(RATE);
    assert!((2.07..2.2).contains(&seconds), "cut at {seconds} s");
    let samples = plane(&fade);
    assert_eq!(samples.len() as i64, from, "it ends at its cut");
    assert!(samples.iter().rev().take(40).any(|v| *v != 0.0));
}

/// A closed interval renders its whole length: no node is cut under it, however quiet.
#[test]
fn a_closed_interval_cuts_nothing() {
    let g = fades();
    let closed = RenderConfig {
        profile: PSYCHOACOUSTIC_V1,
        ..RenderConfig::seconds(RATE, 4.0)
    };
    let mix = sva_engine::render(&g, "mix", closed, &Tier::default()).expect("a render");
    let cutting = mix.labels[&mix.root]
        .cutting_below_silence_threshold
        .clone()
        .expect("a stated silence threshold");
    assert!(
        cutting.treated_as_silent_from_sample.is_empty(),
        "{:?}",
        cutting.treated_as_silent_from_sample
    );
    let exact = RenderConfig {
        profile: at(f64::NEG_INFINITY),
        ..RenderConfig::seconds(RATE, 4.0)
    };
    let uncut = sva_engine::render(&g, "mix", exact, &Tier::default()).expect("a render");
    let (a, b) = (plane(&mix), plane(&uncut));
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// Notes each falling under the silence threshold leave the sum early only while all that left early,
/// summed at the root, stay under it: decaying notes sounding as one, every 32nd of a second,
/// differ from the stream with cuts off by less than the silence threshold, and few sound at once.
#[test]
fn notes_let_go_early_stay_under_the_level_together() {
    let g = graph_of("faded-together", &[("pad", "sin(2*pi*f0*t)\n")]);
    let played = |silence_threshold_dbfs: f64| {
        let config = StreamConfig {
            block: 50,
            channels: None,
            render: RenderConfig {
                profile: at(silence_threshold_dbfs),
                ..RenderConfig::at(RATE)
            },
        };
        let expr =
            |text: &str| sva_ast::parse_expr(text).unwrap_or_else(|e| panic!("{}", e.message));
        let stream = Stream::open(&g, &expr("@notes"), config, &Tier::default()).now();
        let stream = RefCell::new(stream.expect("opens"));
        let (mut samples, mut most) = (Vec::new(), 0);
        for _ in 0..160 {
            let at = stream.borrow().position();
            let note = expr(&format!("0.2*exp(-(t - {at}sp)/0.05)"));
            added(&stream, &g, &note, &Tier::default())
                .now()
                .unwrap_or_else(|e| panic!("{e}"));
            for _ in 0..5 {
                let block = next(&mut stream.borrow_mut()).unwrap_or_else(|e| panic!("{e}"));
                samples.extend_from_slice(block.expect("plays on").plane(0));
            }
            most = most.max(stream.borrow().counts().terms);
        }
        (samples, most, stream.borrow().counts().terms)
    };
    let (cutting, most, _) = played(-120.0);
    let (exact, _, kept) = played(f64::NEG_INFINITY);
    assert_eq!(kept, 160, "with cuts off every note stays");
    assert!(most < 40, "{most} sound at once");
    let moved = cutting.iter().zip(&exact).map(|(a, b)| (a - b).abs());
    let moved = moved.fold(0.0f64, f64::max);
    assert!(moved > 0.0 && moved < 1e-6, "the root moved {moved}");
}

/// A felt strike rings the bank as an impulse does, from the contact on: an open render of it
/// ends where the ring falls under the threshold, not when the felt lets go.
#[test]
fn a_struck_string_rings_on_past_its_contact() {
    let g = graph_of(
        "struck",
        &[
            ("rung", "string(220)\n"),
            ("struck", "string(220, vel=3)\n"),
        ],
    );
    let ends = |target: &str| {
        let held = rendered(&g, target, PSYCHOACOUSTIC_V1);
        let len = plane(&held).len();
        assert_eq!(
            cut(&held, target),
            Some(len as i64),
            "{target} ends at its cut"
        );
        len
    };
    let (rung, struck) = (ends("rung"), ends("struck"));
    assert!(
        struck > RATE as usize,
        "the strike rings past a second, as the impulse's {rung} samples do: {struck}"
    );
}
