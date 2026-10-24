// Concern: proves an edit plays the whole render of the edited expression where nothing before now moved, and carries changed state | Non-concern: the blocks before it | IO: (a stream, exprs) -> blocks

use std::cell::RefCell;

use crate::fixtures::{Now, added, edited, graph_of, next, removed};
use sva_ast::Graph;
use sva_engine::{Change, Changed, Placed, change};
use sva_engine::{Range, RenderConfig, Stream, StreamConfig, Tier, render};

const RATE: u32 = 8_000;
const BLOCK: usize = 256;

/// piano3 with its damper wired to `release`: lifted until then, ramped in over 0.03 s.
const PIANO: &str = "release = inf\n0.0014822 * chaigne_askenfelt(f0, vel=vel, \
    damper_r=0.1*pow(262/f0, 2)*crop(min(1, (t - release)/0.03), release, inf), \
    b=max(1.4e-4, 4.1e-4*pow(f0/262, 1.9)), strike_pos=0.12, hammer_mass=0.009, \
    hammer_k=2e10*max(1, pow(f0/523, 0.6)), hammer_p=3, damp_dc=1.327*exp(0.394*log(f0/262) \
    - 0.12*log(f0/262)*log(f0/262)), damp_freq=0.00044109413838472024, unison_count=3, \
    detune=1, bridge_coupling=97.2*pow(f0/262, 0.677), bridge_mass=1.04*exp(0.466*log(f0/262) \
    - 0.506*log(f0/262)*log(f0/262)), string1_cents=-0.2, string2_cents=0, string3_cents=0.25, \
    string1_hammer_k_ratio=1, string2_hammer_k_ratio=0.8, string3_hammer_k_ratio=0.6)\n";

const PLUCK: &str = "release = inf\nchaigne_askenfelt(f0, damper_r=0.1*pow(262/f0, 2)\
    *crop(min(1, (t - release)/0.03), release, inf))\n";

const ECHO: &str = "feedback = 0.35\nsample(x) + feedback*self[idx(t - 0.25s)]\n";

fn composition(released: f64) -> Graph {
    graph_of(
        "edit",
        &[
            ("piano", PIANO),
            ("echo", ECHO),
            ("pluck", PLUCK),
            (
                "key",
                "release = inf\n@echo(t, x=@piano(t, f0=261.63, vel=4.5, release=release))\n",
            ),
            ("released", &format!("@key(t, release={released})\n")),
            (
                "note",
                "release = inf\n@piano(t, f0=261.63, vel=4.5, release=release)\n",
            ),
            ("played", &format!("@note(t, release={released})\n")),
            (
                "toned",
                "release = inf\nlowpass(highpass(@note(t, release=release), cutoff=180, q=0.7), cutoff=2400, \
                 q=1.3)\n",
            ),
            ("toned_up", &format!("@toned(t, release={released})\n")),
            (
                "string",
                "release = inf\n@pluck(t, f0=523.25, release=release)\n",
            ),
            ("struck", &format!("@string(t, release={released})\n")),
            (
                "doubled",
                "release = inf\n@string(t, release=release) + 0.5*@string(t - 0.3s, release=release)\n",
            ),
            ("doubled_up", &format!("@doubled(t, release={released})\n")),
            ("burst", "crop(sin(2*pi*440*t), 0s, 0.05s)\n"),
            (
                "blip",
                "crop(lowpass(sample(sin(2*pi*f0*t)), cutoff=2000, q=0.7), 0s, 0.4s)\n",
            ),
        ],
    )
}

fn config_at(rate: u32, end: Option<i64>) -> StreamConfig {
    StreamConfig {
        block: BLOCK,
        channels: None,
        render: RenderConfig {
            range: Range {
                start: Some(0),
                end,
            },
            ..RenderConfig::at(rate)
        },
    }
}

fn expr(text: &str) -> sva_ast::Expr {
    sva_ast::parse_expr(text).unwrap_or_else(|e| panic!("`{text}`: {}", e.message))
}

fn opened(g: &Graph, text: &str, cache: &Tier) -> RefCell<Stream> {
    opened_at(RATE, g, text, cache)
}

fn opened_at(rate: u32, g: &Graph, text: &str, cache: &Tier) -> RefCell<Stream> {
    let end = Some(4 * i64::from(rate));
    let opened = Stream::open(g, &expr(text), config_at(rate, end), cache).now();
    RefCell::new(opened.unwrap_or_else(|e| panic!("{e}")))
}

fn edit(stream: &RefCell<Stream>, g: &Graph, text: &str) {
    edited(stream, g, &expr(text), &Tier::default())
        .now()
        .unwrap_or_else(|e| panic!("`{text}`: {e}"));
}

fn blocks(stream: &RefCell<Stream>, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count * BLOCK);
    for _ in 0..count {
        let block = next(&mut stream.borrow_mut()).unwrap_or_else(|e| panic!("{e}"));
        out.extend_from_slice(block.expect("a stream with no end").plane(0));
    }
    out
}

fn whole(g: &Graph, target: &str, samples: usize) -> Vec<f64> {
    whole_at(RATE, g, target, samples)
}

fn whole_at(rate: u32, g: &Graph, target: &str, samples: usize) -> Vec<f64> {
    let secs = samples as f64 / f64::from(rate);
    let held = render(
        g,
        target,
        RenderConfig::seconds(rate, secs),
        &Tier::default(),
    )
    .unwrap_or_else(|e| panic!("{target}: {e}"));
    let id = held.id(target).expect("the root");
    held.output(id).expect("a buffer").plane(0).to_vec()
}

/// Key-up is an edit binding `release` at sample `k`, on the block grid: the held blocks and
/// the released ones after are one whole render with `release = k/rate`.
fn released_at(target: &str, whole_target: &str, k: usize) {
    released_at_rate(RATE, target, whole_target, k);
}

fn released_at_rate(rate: u32, target: &str, whole_target: &str, k: usize) {
    let release = k as f64 / f64::from(rate);
    let g = composition(release);
    let held = opened_at(rate, &g, &format!("@{target}(t)"), &Tier::default());
    let mut heard = blocks(&held, k / BLOCK);
    let released = opened_at(rate, &g, &format!("@{target}(t)"), &Tier::default());
    blocks(&released, k / BLOCK);
    edit(&released, &g, &format!("@{target}(t, release={release})"));
    heard.extend(blocks(&released, 8));

    let want = whole_at(rate, &g, whole_target, heard.len());
    assert_ne!(
        want[k + 2 * BLOCK..],
        blocks(&held, 8)[2 * BLOCK..],
        "{target}: the damper did nothing"
    );
    assert_eq!(heard, want, "{target} at {rate}");
}

#[test]
fn key_up_over_an_echo_is_the_whole_render_released_there() {
    released_at("key", "released", 12 * BLOCK);
}

/// A note pressed at `t0` is a term shifted there, and its key-up at `t1` binds that term's
/// `release`: the stream is the whole render of the final expression, echo tail and all. The
/// first note pressed again at `t2` is its own node read a second time, further back than the
/// stream still holds, so it is stepped again from its start.
#[test]
fn a_note_pressed_and_released_mid_stream_is_the_whole_render_of_the_last_edit() {
    let (t0, t1, t2) = (5 * BLOCK, 9 * BLOCK, 12 * BLOCK);
    let release = (t1 - t0) as f64 / f64::from(RATE);
    let first = "@pluck(t, f0=440)";
    let pressed = format!("{first} + @pluck(t - {t0}sp, f0=660)");
    let released = format!("{first} + @pluck(t - {t0}sp, f0=660, release={release})");
    let last = format!("{released} + @pluck(t - {t2}sp, f0=440)");
    let g = composition(1.0);
    let echoed = |x: &str| format!("@echo(t, x={x})");
    let stream = opened(&g, &echoed(first), &Tier::default());
    let mut heard = blocks(&stream, t0 / BLOCK);
    edit(&stream, &g, &echoed(&pressed));
    heard.extend(blocks(&stream, (t1 - t0) / BLOCK));
    edit(&stream, &g, &echoed(&released));
    heard.extend(blocks(&stream, (t2 - t1) / BLOCK));
    edit(&stream, &g, &echoed(&last));
    heard.extend(blocks(&stream, 16));

    let mut whole_g = g.clone();
    assert!(whole_g.define("final", expr(&echoed(&last))));
    let want = whole(&whole_g, "final", heard.len());
    assert_ne!(
        want,
        whole(&g, "string", heard.len()),
        "the press was heard"
    );
    assert_eq!(heard, want);
}

/// A term removed while it sounds is cut where the stream stands and leaves the sum there:
/// what it played stays, echo and all, and the stream is the whole render of it cut there.
#[test]
fn a_term_removed_while_it_sounds_is_cut_where_the_stream_stands() {
    let g = composition(1.0);
    let stream = opened(&g, "@echo(t, x=@notes)", &Tier::default());
    let add = |term: &str| {
        added(&stream, &g, &expr(term), &Tier::default())
            .now()
            .unwrap_or_else(|e| panic!("`{term}`: {e}"))
    };
    let (a, b) = (
        format!("@blip(t - {BLOCK}sp, f0=200)"),
        format!("@blip(t - {}sp, f0=300)", 4 * BLOCK),
    );
    let first = add(&a);
    add(&b);
    let mut heard = blocks(&stream, 6);
    assert_eq!(
        removed(&stream, first, &Tier::default()).now().ok(),
        Some(true)
    );
    assert_eq!(
        removed(&stream, first, &Tier::default()).now().ok(),
        Some(false),
        "a removed handle"
    );
    assert_eq!(
        stream.borrow().counts().terms,
        1,
        "the cut term left the sum"
    );
    heard.extend(blocks(&stream, 30));

    let cut = (6 * BLOCK) as f64 / f64::from(RATE);
    let mut whole_g = g.clone();
    let last = format!("@echo(t, x=crop({a}, -inf, {cut}s) + {b})");
    assert!(whole_g.define("final", expr(&last)));
    assert_eq!(heard, whole(&whole_g, "final", heard.len()), "{last}");
}

/// Key-up on a note the store answered finds no state at the edit's instant, only one marked
/// before it: an exact stream computes the string on from there, a live one computes none of
/// its past and starts it silent, named dropped.
#[test]
fn a_live_edit_starts_a_node_with_no_state_silent_and_names_it() {
    let g = composition(1.0);
    let cache = Tier::default();
    cache.set_mark_every(BLOCK);
    render(&g, "string", RenderConfig::seconds(RATE, 0.5), &cache).expect("a render");
    let held = "@echo(t, x=@pluck(t, f0=523.25))";
    let released = "@echo(t, x=@pluck(t, f0=523.25, release=0.1))";
    let (exact, live) = (opened(&g, held, &cache), opened(&g, held, &cache));
    live.borrow_mut().go_live();
    let before = blocks(&live, 8);
    assert_eq!(before, blocks(&exact, 8));
    edit(&exact, &g, released);
    edit(&live, &g, released);
    assert!(exact.borrow().dropped().is_empty());
    assert_eq!(
        live.borrow().dropped().len(),
        1,
        "{:?}",
        live.borrow().dropped()
    );
    assert!(
        live.borrow().dropped()[0].starts_with("pluck("),
        "{:?}",
        live.borrow().dropped()
    );
    assert_ne!(blocks(&live, 4), blocks(&exact, 4));
}

/// The same with the loop written as the target itself: the stream's root is the stateful
/// value, and it takes the past of the root it replaces.
#[test]
fn a_stateful_target_edited_in_place_keeps_its_tail_ringing() {
    let g = composition(1.0);
    let loop_at = |feedback: f64| format!("sample(@burst) + {feedback}*self[idx(t - 0.25s)]");
    let stream = opened(&g, &loop_at(0.35), &Tier::default());
    let k = 13 * BLOCK;
    let mut heard = blocks(&stream, k / BLOCK);
    edit(&stream, &g, &loop_at(0.5));
    heard.extend(blocks(&stream, 12));
    let delay = RATE as usize / 4;
    assert!(heard[k..].iter().any(|v| *v != 0.0), "the tail rings on");
    for n in k..heard.len() {
        assert_eq!(heard[n], 0.5 * heard[n - delay], "sample {n}");
    }
}

/// A pad held until `release`, then fading, its support running on to the hour.
const PAD: &str = "release = inf\ncrop(sin(2*pi*f0*t)*exp(-t/0.25), 0s, release) + \
    crop(exp(-release/0.25)*sin(2*pi*f0*t)*exp(-(t - release)/0.05), release, 3600s)\n";

/// A term placed at its landing has its sample 0 at the sample the add lands at, and a key-up
/// placed there too is written from that same sample 0: the stream is the whole render of the
/// note shifted to where it landed.
#[test]
fn a_term_placed_at_its_landing_plays_from_its_sample_zero_there() {
    let g = graph_of("landing", &[("pad", PAD), ("note", "@pad(t, f0=300)\n")]);
    let stream = opened(&g, "@notes", &Tier::default());
    let mut heard = blocks(&stream, 5);
    let held = expr("@pad(t, f0=300)");
    let build = |_: &Stream| {
        Ok::<_, sva_engine::EngineError>(Change::Add(g.clone(), held.clone(), Placed::Landing))
    };
    let Ok(Changed::Added(note)) = change(&stream, build, &Tier::default()).now() else {
        panic!("an add answers its handle");
    };
    let landed = 5 * BLOCK as i64;
    assert_eq!(stream.borrow().landed(note), Some(landed));
    heard.extend(blocks(&stream, 2));
    let up = expr(&format!("@pad(t, f0=300, release={}sp)", 2 * BLOCK));
    let build = |_: &Stream| {
        Ok::<_, sva_engine::EngineError>(Change::Replace(
            note,
            g.clone(),
            up.clone(),
            Placed::Landing,
        ))
    };
    assert_eq!(
        change(&stream, build, &Tier::default()).now().ok(),
        Some(Changed::Held(true))
    );
    heard.extend(blocks(&stream, 8));

    let alone = whole(&g, "note", 2 * BLOCK);
    assert!(alone.iter().any(|v| *v != 0.0));
    assert_eq!(
        heard[5 * BLOCK..7 * BLOCK],
        alone[..],
        "the attack is whole"
    );
    let mut whole_g = g.clone();
    let last = format!("@pad(t - {landed}sp, f0=300, release={}sp)", 2 * BLOCK);
    assert!(whole_g.define("final", expr(&last)));
    assert_eq!(heard, whole(&whole_g, "final", heard.len()), "{last}");
}

fn widths() -> Graph {
    graph_of(
        "widths",
        &[
            ("mono", "crop(sin(2*pi*200*t), 0s, 0.1s)\n"),
            (
                "pair",
                "crop(join(sin(2*pi*300*t), sin(2*pi*301*t)), 0s, 0.1s)\n",
            ),
            (
                "three",
                "crop(join(sin(2*pi*300*t), sin(2*pi*301*t), sin(2*pi*302*t)), 0s, 0.1s)\n",
            ),
        ],
    )
}

/// A live stream over `text` playing `channels`, the target's own where `None`.
fn opened_with(
    g: &Graph,
    text: &str,
    channels: Option<usize>,
) -> Result<RefCell<Stream>, sva_engine::EngineError> {
    let config = StreamConfig {
        channels,
        ..config_at(RATE, Some(4 * i64::from(RATE)))
    };
    let mut stream = Stream::open(g, &expr(text), config, &Tier::default()).now()?;
    stream.go_live();
    Ok(RefCell::new(stream))
}

/// An add or an edit wider than the stream plays is refused, coded, and the stream plays on
/// as it was; so is a stream opened narrower than its target.
#[test]
fn what_is_wider_than_the_stream_is_refused() {
    let g = widths();
    let add = |stream: &RefCell<Stream>, term: &str| {
        added(stream, &g, &expr(term), &Tier::default()).now()
    };
    let mono = opened_with(&g, "@notes", None).unwrap_or_else(|e| panic!("{e}"));
    add(&mono, "@mono(t - 512sp)").unwrap_or_else(|e| panic!("{e}"));
    let widened = add(&mono, "@pair(t - 1024sp)").expect_err("a stereo term widens it");
    assert_eq!(widened.code(), "engine.stream_width", "{widened}");
    let edit = edited(&mono, &g, &expr("join(@notes, @notes)"), &Tier::default()).now();
    let widened = edit.expect_err("a stereo target widens it");
    assert_eq!(widened.code(), "engine.stream_width", "{widened}");
    assert_eq!(
        (mono.borrow().width(), mono.borrow().counts().terms),
        (1, 1)
    );
    let heard = blocks(&mono, 8);
    let mut whole_g = g.clone();
    assert!(whole_g.define("final", expr("@mono(t - 512sp)")));
    assert_eq!(heard, whole(&whole_g, "final", heard.len()), "plays on");

    let stereo = opened_with(&g, "@notes", Some(2)).unwrap_or_else(|e| panic!("{e}"));
    let widened = add(&stereo, "@three(t - 512sp)").expect_err("three channels widen it");
    assert_eq!(widened.code(), "engine.stream_width", "{widened}");
    let narrow = opened_with(&g, "@pair", Some(1))
        .err()
        .expect("a stereo target in mono");
    assert_eq!(narrow.code(), "engine.stream_width", "{narrow}");
}

/// An edit whose graph holds a node the stream plays otherwise plays that node as the edit's
/// graph holds it, from the edit on.
#[test]
fn an_edit_plays_a_node_as_its_own_graph_holds_it() {
    let graph = |level: &str| {
        graph_of(
            "renewed",
            &[
                ("bed", &format!("{level}*sin(2*pi*110*t)\n")),
                ("tone", "0.1*sin(2*pi*440*t)\n"),
            ],
        )
    };
    let config = StreamConfig {
        block: BLOCK,
        channels: None,
        render: RenderConfig::at(RATE),
    };
    let expr = |text: &str| sva_ast::parse_expr(text).expect("an expression");
    let (loud, soft) = (graph("0.5"), graph("0.25"));
    let target = expr("@bed + @tone");
    let stream = Stream::open(&loud, &target, config, &Tier::default()).now();
    let stream = RefCell::new(stream.expect("a stream"));
    next(&mut stream.borrow_mut()).expect("a block");
    edited(&stream, &soft, &expr("@bed(t) + @tone"), &Tier::default())
        .now()
        .expect("edited");
    let heard = next(&mut stream.borrow_mut())
        .expect("a block")
        .expect("a block");
    let whole = render(
        &soft,
        "bed",
        RenderConfig::seconds(RATE, 0.1),
        &Tier::default(),
    )
    .expect("a render");
    let bed = whole.output(whole.root).expect("the bed").plane(0).to_vec();
    let tone =
        |n: usize| 0.1 * libm::sin(std::f64::consts::TAU * 440.0 * n as f64 / f64::from(RATE));
    for (k, v) in heard.plane(0).iter().enumerate() {
        let n = BLOCK + k;
        assert!((v - (bed[n] + tone(n))).abs() < 1e-9, "sample {n}: {v}");
    }
}
