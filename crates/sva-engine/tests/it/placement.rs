// Concern: proves a placed read lands within half a sample, and a periodic value holds across its period | Non-concern: what a value holds | IO: (a composition) -> evaluated segments, onsets

use crate::fixtures::{Now, graph_of, next};
use sva_ast::Graph;
use sva_engine::{Range, RenderConfig, Stream, StreamConfig, Tier, render};

const RATE: u32 = 44_100;

/// `files` at `bpm`, `beats` to the bar, every bar lowered to seconds.
fn at_tempo(name: &str, files: &[(&str, &str)], (bpm, beats): (f64, u32)) -> Graph {
    let (bpm, meter) = (format!("{bpm}\n"), format!("{beats}/4\n"));
    let mut files = files.to_vec();
    files.extend([
        ("variables/bpm", bpm.as_str()),
        ("variables/meter", meter.as_str()),
    ]);
    let mut g = graph_of(name, &files);
    g.lower_reserved(None).expect("a tempo");
    g
}

fn streamed(
    g: &Graph,
    target: &str,
    rate: u32,
    seconds: i64,
    each: &mut dyn FnMut(i64, &[f64]),
) -> Stream {
    let config = StreamConfig {
        block: 1 << 15,
        channels: None,
        render: RenderConfig {
            range: Range {
                start: Some(0),
                end: Some(seconds * i64::from(rate)),
            },
            ..RenderConfig::at(rate)
        },
    };
    let at = sva_ast::parse_expr(target).expect("a target");
    let mut stream = Stream::open(g, &at, config, &Tier::default())
        .now()
        .expect("it streams");
    while let Some(block) = next(&mut stream).expect("a block") {
        each(block.start, block.plane(0));
    }
    stream
}

/// 384 beats at 128 bpm, 22,050 Hz: the last as close as the first.
#[test]
fn every_onset_lands_within_half_a_sample_and_the_tempo_never_drifts() {
    let rate = 22_050;
    let beat = 60.0 / 128.0;
    let bars: Vec<(String, String)> = (0..24)
        .map(|bar| {
            let clicks: Vec<String> = (0..16)
                .map(|k| format!("@click(t - {}b)", 16 * bar + k))
                .collect();
            (format!("bar{bar}"), format!("{}\n", clicks.join(" + ")))
        })
        .collect();
    let song: Vec<String> = bars.iter().map(|(name, _)| format!("@{name}")).collect();
    let song = format!("{}\n", song.join(" + "));
    let mut files: Vec<(&str, &str)> = vec![("click", "crop(1, 0s, 1sp)\n"), ("song", &song)];
    files.extend(
        bars.iter()
            .map(|(name, body)| (name.as_str(), body.as_str())),
    );
    let g = at_tempo("snapped", &files, (128.0, 1));
    let mut onsets = Vec::new();
    streamed(&g, "@song", rate, 180, &mut |start, plane| {
        onsets.extend(
            plane
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.0)
                .map(|(i, v)| (start + i as i64, *v)),
        );
    });
    assert_eq!(onsets.len(), 384);
    for (k, (at, v)) in onsets.iter().enumerate() {
        let exact = k as f64 * beat * f64::from(rate);
        assert_eq!(*v, 1.0, "beat {k}");
        assert!(
            (*at as f64 - exact).abs() <= 0.5,
            "beat {k}: {at} for {exact}"
        );
        let even = exact.round_ties_even() as i64;
        assert_eq!(*at, even, "beat {k}: ties round to even");
    }
}

/// 1 kHz repeats every 441 samples; each range starting past 0 wraps or folds into it.
#[test]
fn a_periodic_value_read_across_its_period_from_past_zero_holds_every_sample() {
    let g = graph_of("periodic-offset", &[("s", "sin(2*pi*1000*t)\n")]);
    for (start, end) in [(300, 600), (441, 445), (11_025, 11_029), (440, 444)] {
        let mut config = RenderConfig::at(RATE);
        config.range = Range {
            start: Some(start),
            end: Some(end),
        };
        let rendered = render(&g, "s", config, &Tier::default()).unwrap_or_else(|e| panic!("{e}"));
        let out = rendered
            .output(rendered.id("s").expect("s"))
            .expect("its samples");
        assert_eq!(
            (out.start, out.plane(0).len()),
            (start, (end - start) as usize)
        );
        for (i, x) in out.plane(0).iter().enumerate() {
            let n = start + i as i64;
            let exact = libm::sin(std::f64::consts::TAU * 1000.0 * n as f64 / f64::from(RATE));
            assert!(
                (x - exact).abs() < 1e-9,
                "[{start}, {end}) sample {n}: {x} for {exact}"
            );
        }
    }
}
