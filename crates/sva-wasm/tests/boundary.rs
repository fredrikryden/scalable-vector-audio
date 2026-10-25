// Concern: drives every JS export through real wasm | Non-concern: what a sample is worth (sva-engine), the pipeline (sva-cli) | IO: (a built composition) -> assertions
#![cfg(target_arch = "wasm32")]

//! The gap a native test cannot see: `wasm32-unknown-unknown` has no clock and no threads, and
//! reaching for either compiles clean and traps only at RUNTIME. Hence one session for the
//! whole surface. Run it with `wasm-pack test --node crates/sva-wasm`.

use std::task::{Context, Poll, Waker};
use sva_wasm::{Composition, Rendering, Stream};

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::wasm_bindgen_test;

/// Through js-sys, not a local `extern "C"`: an import naming `JSON.parse` a second time inside
/// this package collides with the crate's own shim at link time.
fn as_text(value: &JsValue) -> String {
    js_sys::JSON::stringify(value)
        .map(String::from)
        .unwrap_or_else(|_| unreachable!("the value is JSON"))
}

/// A collection's own list: every array this tool answers sits under `items`.
fn items(value: &JsValue, name: &str) -> js_sys::Array {
    js_sys::Array::from(&field(&field(value, name), "items"))
}

fn field(value: &JsValue, name: &str) -> JsValue {
    js_sys::Reflect::get(value, &JsValue::from_str(name))
        .unwrap_or_else(|_| unreachable!("{name} is readable"))
}

/// `{ rate: 8000 }` and each of `more`, as a page writes its options.
fn options(more: &[(&str, JsValue)]) -> JsValue {
    let held = js_sys::Object::new();
    for (key, value) in [("rate", JsValue::from(8000))].iter().chain(more) {
        js_sys::Reflect::set(&held, &(*key).into(), value)
            .unwrap_or_else(|_| unreachable!("an object takes a key"));
    }
    held.into()
}

/// A render with no persistent store never waits, so one poll finishes it.
trait Now {
    fn rendered(
        &self,
        target: &str,
        representations: Option<Vec<String>>,
        options: JsValue,
    ) -> Result<Rendering, JsValue>;
}

impl Now for Composition {
    fn rendered(
        &self,
        target: &str,
        representations: Option<Vec<String>>,
        options: JsValue,
    ) -> Result<Rendering, JsValue> {
        let mut render = std::pin::pin!(self.render(target, representations, options));
        let mut context = Context::from_waker(Waker::noop());
        match render.as_mut().poll(&mut context) {
            Poll::Ready(done) => done,
            Poll::Pending => unreachable!("a render in memory alone never waits"),
        }
    }
}

/// A stream in memory alone never waits either.
fn now<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(done) => done,
        Poll::Pending => unreachable!("a stream in memory alone never waits"),
    }
}

fn page() -> Composition {
    let mut held = Composition::new(Some("a-registry".to_string()));
    held.insert("master", "@partials/one*0.5\n");
    held.insert("partials/one", "sin(2*pi*100*t)\n");
    held.insert("wide", "join(sin(2*pi*100*t), sin(2*pi*200*t))\n");
    held.insert("unreached", "this is not an expression (\n");
    held
}

/// `@<node>` over its first second, read for `representations`.
fn asking(of: &Composition, node: &str, representations: &[&str]) -> Rendering {
    let asked = representations.iter().map(|r| (*r).to_string()).collect();
    of.rendered(&format!("@{node}([0, 1s])"), Some(asked), options(&[]))
        .unwrap_or_else(|_| unreachable!("`{node}` renders"))
}

fn render(of: &Composition, node: &str) -> Rendering {
    asking(of, node, &["samples"])
}

fn plane(of: &Rendering) -> Vec<f32> {
    of.samples(0)
        .unwrap_or_else(|_| unreachable!("one component"))
}

fn readings(of: &Rendering) -> JsValue {
    of.representations(JsValue::UNDEFINED).unwrap_or_else(|e| {
        unreachable!("representations answer: {}", as_text(&field(&e, "refusal")))
    })
}

#[wasm_bindgen_test]
fn every_export_survives_the_boundary() {
    let held = page();

    let one = render(&held, "partials/one");
    assert_eq!(one.sample_rate(), 8000);
    assert_eq!(one.channels(), 1);
    assert_eq!(one.duration_secs(), 1.0);

    let drawn = plane(&one);
    assert_eq!(drawn.len(), 8000, "a typed array of every sample");
    let want = (2.0 * std::f64::consts::PI * 100.0 * (2.0 / 8000.0)).sin() as f32;
    assert!((drawn[2] - want).abs() < 1e-4, "{} vs {want}", drawn[2]);

    let part = held
        .rendered("@partials/one([0.25s, 0.5s])", None, options(&[]))
        .unwrap_or_else(|_| unreachable!("an interval renders"));
    assert_eq!(plane(&part).len(), 2000, "an interval narrows the array");
    let read = field(&readings(&part), "representations");
    let component = items(&field(&field(&read, "samples"), "value"), "components").get(0);
    let paging = field(&field(&component, "values"), "pagination");
    assert_eq!(
        field(&paging, "count").as_f64(),
        Some(2000.0),
        "and the reading the CLI prints"
    );
    assert_eq!(
        plane(&part)[..],
        drawn[2000..4000],
        "and trims the output alone"
    );

    let wide = render(&held, "wide");
    assert_eq!(wide.channels(), 2);
    assert!(
        wide.samples(2).is_err(),
        "a component that is not there refuses, and does not trap"
    );

    assert!(held.cache_max_bytes() > 0.0, "a default budget is held");
}

#[wasm_bindgen_test]
fn a_representation_crosses_as_the_object_the_cli_puts_under_data() {
    let one = asking(&page(), "partials/one", &["envelope", "samples"]);

    let asked = readings(&one);
    let spelled = as_text(&asked);
    assert!(
        spelled.contains("\"target\":\"@partials/one([0, 1s])\"")
            && spelled.contains("\"envelope\":"),
        "the CLI's own keys: {spelled}"
    );
    assert_eq!(field(&asked, "sample_rate").as_f64(), Some(8000.0));

    let read = field(&asked, "representations");
    let component = items(&field(&field(&read, "samples"), "value"), "components").get(0);
    let values = field(&component, "values");
    assert_eq!(
        js_sys::Array::from(&field(&values, "items")).length(),
        4096,
        "capped exactly as stdout caps it, where the typed array is not"
    );
    let paging = field(&values, "pagination");
    assert_eq!(
        field(&paging, "count").as_f64(),
        Some(8000.0),
        "the count is the whole reading's, not the page's"
    );
    assert_eq!(
        field(&paging, "has_more").as_bool(),
        Some(true),
        "and it says so rather than reading as the whole"
    );

    let whole = one
        .representations(every_value())
        .unwrap_or_else(|e| unreachable!("`limit: null` answers: {}", as_text(&e)));
    let values = field(
        &items(
            &field(
                &field(&field(&whole, "representations"), "samples"),
                "value",
            ),
            "components",
        )
        .get(0),
        "values",
    );
    assert_eq!(
        js_sys::Array::from(&field(&values, "items")).length(),
        8000,
        "`limit: null` holds every value"
    );
    assert_eq!(
        field(&field(&values, "pagination"), "has_more").as_bool(),
        Some(false)
    );
}

/// `{ limit: null }`: every value of every array, as a reading written to a file holds.
fn every_value() -> JsValue {
    let held = js_sys::Object::new();
    js_sys::Reflect::set(&held, &"limit".into(), &JsValue::NULL)
        .unwrap_or_else(|_| unreachable!("an object takes a key"));
    held.into()
}

/// PCM a page holds is read as `sva-cli analyze` reads a file: the same `data` object, the
/// same refusal of a reading that needs a graph, and channels of one length.
#[wasm_bindgen_test]
fn pcm_a_page_holds_is_analyzed_as_the_cli_analyzes_a_file() {
    let tone: Vec<f32> = (0..8000)
        .map(|i| (std::f64::consts::TAU * 220.0 * f64::from(i) / 8000.0).sin() as f32 * 0.5)
        .collect();
    let channels = |planes: &[&[f32]]| -> js_sys::Array {
        planes
            .iter()
            .map(|p| JsValue::from(js_sys::Float32Array::from(*p)))
            .collect()
    };
    let asked = |names: &[&str]| names.iter().map(|n| (*n).to_string()).collect::<Vec<_>>();
    let read = sva_wasm::analyze(
        "crop.wav",
        8000,
        channels(&[&tone, &tone]),
        asked(&["envelope", "stereo"]),
        JsValue::UNDEFINED,
    )
    .unwrap_or_else(|e| unreachable!("a tone analyzes: {}", as_text(&field(&e, "refusal"))));
    let spelled = as_text(&read);
    assert!(
        spelled.contains("\"target\":\"crop.wav\"") && spelled.contains("\"stereo\":"),
        "the CLI's own keys: {spelled}"
    );
    assert_eq!(field(&read, "sample_rate").as_f64(), Some(8000.0));

    // Three seconds: more band values than stdout's cap.
    let long: Vec<f32> = tone.iter().copied().cycle().take(24000).collect();
    let bands = |options: JsValue| {
        let read = sva_wasm::analyze(
            "crop.wav",
            8000,
            channels(&[&long]),
            asked(&["bands"]),
            options,
        )
        .unwrap_or_else(|e| unreachable!("a tone's bands: {}", as_text(&field(&e, "refusal"))));
        let value = field(&field(&field(&read, "representations"), "bands"), "value");
        let rms = field(&items(&value, "bands").get(0), "rms");
        let paging = field(&rms, "pagination");
        let held = js_sys::Array::from(&field(&rms, "items")).length();
        (
            field(&paging, "has_more").as_bool(),
            f64::from(held),
            field(&paging, "count").as_f64(),
        )
    };
    let (more, held, count) = bands(JsValue::UNDEFINED);
    assert_eq!(
        (more, held),
        (Some(true), 4096.0),
        "capped as stdout caps, of {count:?}"
    );
    let (more, held, count) = bands(every_value());
    assert_eq!(
        (more, Some(held)),
        (Some(false), count),
        "`limit: null` reads every value"
    );

    let zero = js_sys::Object::new();
    js_sys::Reflect::set(&zero, &"limit".into(), &JsValue::from(0))
        .unwrap_or_else(|_| unreachable!("an object takes a key"));
    let none = sva_wasm::analyze(
        "crop.wav",
        8000,
        channels(&[&tone]),
        asked(&["bands"]),
        zero.into(),
    );
    refused_as(none.err(), "wasm.bad_argument");

    let graph = sva_wasm::analyze(
        "crop.wav",
        8000,
        channels(&[&tone]),
        asked(&["ledger"]),
        JsValue::UNDEFINED,
    );
    refused_as(graph.err(), "query.not_analyzable");
    let uneven = sva_wasm::analyze(
        "crop.wav",
        8000,
        channels(&[&tone, &tone[..10]]),
        asked(&["envelope"]),
        JsValue::UNDEFINED,
    );
    refused_as(uneven.err(), "wasm.bad_argument");
}

/// A stem playing a riff a bar late, read once dry beside an echo in `master`.
fn riffed() -> Composition {
    let mut held = Composition::new(None);
    for (path, text) in [
        ("variables/bpm", "60\n"),
        ("variables/meter", "4/4\n"),
        ("variables/key", "A3\n"),
        ("variables/scale", "major\n"),
        (
            "tone",
            "f0 = 220\nlen = 0.25s\nsin(2*pi*f0*t)*crop(1, 0s, len)\n",
        ),
        (
            "riff.jsonl",
            "{\"at\": \"0b\", \"node\": \"@tone\", \"deg\": 3, \"len\": \"0.5b\"}\n\
             {\"at\": \"1b\", \"node\": \"@tone\", \"f0\": \"A2\"}\n",
        ),
        ("stem", "0.5*@riff(t - 1b)\n"),
        ("master", "@riff(t) + @stem(t)\n"),
    ] {
        held.insert(path, text);
    }
    held
}

/// Each event `song` answers as (at, sounds, len) in seconds, hz and node.
fn notes(song: &JsValue) -> Vec<(f64, f64, Option<f64>, Option<f64>, String)> {
    let seconds = |v: JsValue| match v.is_null() {
        true => None,
        false => field(&v, "seconds").as_f64(),
    };
    items(song, "events")
        .iter()
        .map(|e| {
            (
                seconds(field(&e, "at")).unwrap_or(f64::NAN),
                seconds(field(&e, "sounds")).unwrap_or(f64::NAN),
                seconds(field(&e, "len")),
                field(&e, "hz").as_f64(),
                field(&e, "node").as_string().unwrap_or_default(),
            )
        })
        .collect()
}

/// `song` derives the events `sva-cli outline` lists, over the `set` a render takes, and
/// renders nothing: a note read dry and a bar late is one event, sounding at its song time.
#[wasm_bindgen_test]
fn a_song_answers_its_placed_notes_without_rendering() {
    let held = riffed();
    let object = |key: &str, value: JsValue| {
        let held = js_sys::Object::new();
        js_sys::Reflect::set(&held, &key.into(), &value)
            .unwrap_or_else(|_| unreachable!("an object takes a key"));
        JsValue::from(held)
    };
    let set = Some(object("set", object("bpm", "120".into())));
    let derive = |target: &str, with: Option<JsValue>| {
        held.song(target, with.unwrap_or(JsValue::UNDEFINED))
            .unwrap_or_else(|e| {
                unreachable!("{target} derives: {}", as_text(&field(&e, "refusal")))
            })
    };
    let third = 220.0 * 2f64.powf(4.0 / 12.0);
    let rounded = |mut held: Vec<(f64, f64, Option<f64>, Option<f64>, String)>| {
        for n in &mut held {
            n.3 = n.3.map(|hz| (hz * 1e6).round() / 1e6);
        }
        held
    };
    let want = |late: f64, bar: f64| {
        rounded(vec![
            (0.0, late, Some(0.5 * bar), Some(third), "tone".to_string()),
            (bar, bar + late, None, Some(110.0), "tone".to_string()),
        ])
    };
    assert_eq!(rounded(notes(&derive("@master", None))), want(0.0, 4.0));
    assert_eq!(rounded(notes(&derive("@stem", None))), want(4.0, 4.0));
    assert_eq!(rounded(notes(&derive("@stem", set))), want(2.0, 2.0));
    let refused = held.song("@stem", options(&[]));
    assert!(refused.is_err(), "`song` takes `set` alone");
}

/// `threads` is a ceiling: a build that starts none computes on one thread whatever a page
/// asks, to the same samples, and refuses only a thread count of nothing.
#[wasm_bindgen_test]
fn a_render_asks_at_most_the_threads_the_module_started() {
    let threads = |n: u32| options(&[("threads", JsValue::from(n))]);
    let held = page();
    let alone = plane(&render(&held, "master"));
    for n in [1, 4] {
        let asked = held
            .rendered("@master([0, 1s])", None, threads(n))
            .unwrap_or_else(|_| unreachable!("`threads: {n}` renders"));
        assert_eq!(plane(&asked), alone, "on {n}");
    }
    let refused = held.rendered("@master([0, 1s])", None, threads(0));
    assert!(refused.is_err(), "no thread at all refuses");
    assert!(
        now(sva_wasm::start_threads(2)).is_err(),
        "a build without wasm threads starts none"
    );
    let only = js_sys::Object::new();
    js_sys::Reflect::set(&only, &"threads".into(), &JsValue::from(3))
        .unwrap_or_else(|_| unreachable!("an object takes a key"));
    let opened = now(Composition::open(None, None, only.into()))
        .unwrap_or_else(|_| unreachable!("`open` takes `threads`"));
    assert!(
        now(opened.stream("1", 64, threads(2))).is_ok(),
        "and so does a stream"
    );
}

/// A located refusal is the contract everywhere else in this engine, so it has to cross as one:
/// data on a thrown `Error`, never a trap that takes the module down with it.
#[wasm_bindgen_test]
fn a_refusal_crosses_as_data_and_the_module_keeps_working() {
    let mut held = page();
    held.insert("master", "@nowhere*2\n");

    let refused = held
        .rendered("@master([0, 1s])", None, options(&[]))
        .err()
        .unwrap_or_else(|| unreachable!("a dangling ref refuses"));

    assert_eq!(
        field(&refused, "name").as_string().as_deref(),
        Some("validation_error"),
        "the CLI's own error code"
    );
    let envelope = field(&refused, "refusal");
    let spelled = as_text(&envelope);
    assert!(
        spelled.contains("\"status\":\"error\"") && spelled.contains("\"diagnostics\""),
        "the envelope the CLI would have printed: {spelled}"
    );
    let found = field(&field(&envelope, "data"), "diagnostics");
    assert_eq!(
        field(&field(&found, "pagination"), "count").as_f64(),
        Some(1.0),
        "a collection carries its own count: {spelled}"
    );
    let first = js_sys::Array::from(&field(&found, "items")).get(0);
    assert!(
        field(&first, "location").is_object(),
        "and it is located: {spelled}"
    );
    let details = field(&field(&envelope, "error"), "details");
    assert_eq!(
        field(&details, "count").as_f64(),
        Some(1.0),
        "`error.details`, the third key the standard mandates: {spelled}"
    );
    assert_eq!(
        items(&details, "codes").get(0).as_string(),
        field(&first, "code").as_string(),
        "and it names the code a page branches on: {spelled}"
    );

    held.insert("master", "@partials/one*0.5\n");
    assert_eq!(
        render(&held, "master").channels(),
        1,
        "one node replaced, and the session renders again"
    );
}

const BLOCK: usize = 256;

/// `@<node>` over four seconds, which no test reaches the end of.
fn opened(held: &Composition, node: &str) -> Stream {
    now(held.stream(&format!("@{node}([0, 4s])"), BLOCK, options(&[])))
        .unwrap_or_else(|e| unreachable!("`{node}` streams: {}", as_text(&field(&e, "refusal"))))
}

fn blocks(stream: &Stream, count: usize) -> Vec<f32> {
    let mut heard = Vec::with_capacity(count * BLOCK);
    for _ in 0..count {
        heard.extend(read(stream));
    }
    heard
}

/// The block from where the stream stands, its frames interleaved.
fn read(stream: &Stream) -> Vec<f32> {
    read_at(stream, stream.position() as usize)
}

/// The block from sample `at`, its frames interleaved.
fn read_at(stream: &Stream, at: usize) -> Vec<f32> {
    let width = stream.channels();
    let out = js_sys::Float32Array::new_with_length((width * BLOCK) as u32);
    let took = stream
        .read(at as f64, &out)
        .unwrap_or_else(|_| unreachable!("a block"));
    out.subarray(0, (took * width) as u32).to_vec()
}

fn floats(n: usize) -> js_sys::Float32Array {
    js_sys::Float32Array::new_with_length(n as u32)
}

fn refused_as(refused: Option<JsValue>, code: &str) {
    let refused = refused.unwrap_or_else(|| unreachable!("`{code}` refuses"));
    assert!(
        as_text(&field(&refused, "refusal")).contains(code),
        "{}",
        as_text(&refused)
    );
}

/// A sampled filter over a closed form: the stream's blocks are the render's samples. A term
/// added to `@notes` crosses as a handle, and replacing it with one binding `release` at the
/// stream's position closes the envelope there.
#[wasm_bindgen_test]
fn a_stream_crosses_block_by_block_and_a_replaced_term_releases_it() {
    let mut held = page();
    held.insert(
        "filtered",
        "lowpass(sample(@partials/one), cutoff=300, q=0.7)\n",
    );
    held.insert("gated", "release = inf\ncrop(@filtered, 0s, release)\n");
    let stream = opened(&held, "filtered");
    assert_eq!((stream.channels(), stream.sample_rate()), (1, 8000));
    let heard = blocks(&stream, 8000 / BLOCK);
    let whole = plane(&render(&held, "filtered"));
    assert_eq!(heard[..], whole[..heard.len()]);

    let refusal = |e: JsValue| unreachable!("{}", as_text(&field(&e, "refusal")));
    let gated = opened(&held, "notes");
    let key = now(gated.add("@gated", None)).unwrap_or_else(refusal);
    assert!(blocks(&gated, 3).iter().any(|v| *v != 0.0));
    let at = gated.position() / 8000.0;
    let released = now(gated.replace(key, &format!("@gated(t, release={at})"), None));
    assert_eq!(released.ok(), Some(true));
    assert!(blocks(&gated, 2).iter().all(|v| *v == 0.0));
    assert_eq!(
        now(gated.remove(key)).ok(),
        Some(false),
        "a released term leaves the sum once the stream passes it"
    );
    let key = now(gated.add("@gated", None)).unwrap_or_else(refusal);
    assert_eq!(now(gated.remove(key)).ok(), Some(true));
    assert_eq!(
        now(gated.remove(key)).ok(),
        Some(false),
        "a handle removed is held no more"
    );
    refused_as(
        now(gated.add("@gated([0, 1s])", None)).err(),
        "validation_error",
    );
    refused_as(now(gated.edit("@notes([0, 1s])")).err(), "validation_error");
}

/// A node the page inserts anew while a stream plays it is a new version: the stream's next
/// edit plays the inserted text, from where it lands bit for bit the render of the edited page.
#[wasm_bindgen_test]
fn a_stream_edit_after_an_insert_plays_the_inserted_text() {
    let mut held = page();
    let stream = opened(&held, "master");
    blocks(&stream, 4);
    held.insert("partials/one", "sin(2*pi*200*t)\n");
    now(stream.edit("@master"))
        .unwrap_or_else(|e| unreachable!("{}", as_text(&field(&e, "refusal"))));
    let at = stream.position() as usize;
    let heard = blocks(&stream, 8);
    let whole = plane(&render(&held, "master"));
    assert!(heard.iter().any(|v| *v != 0.0), "silence tests nothing");
    assert_eq!(heard[..], whole[at..at + heard.len()]);
}

/// A read fills a view onto a playout ring's SharedArrayBuffer (a plain ArrayBuffer where the
/// host has none) from its offset, and touches nothing else of the ring.
#[wasm_bindgen_test]
fn a_read_fills_a_view_onto_a_shared_ring() {
    let held = page();
    let stream = opened(&held, "wide");
    let (frame, ring) = (2 * 4, 4 * BLOCK);
    let shared =
        js_sys::Reflect::has(&js_sys::global(), &"SharedArrayBuffer".into()).unwrap_or(false);
    let buffer: JsValue = if shared {
        js_sys::SharedArrayBuffer::new((ring * frame) as u32).into()
    } else {
        js_sys::ArrayBuffer::new((ring * frame) as u32).into()
    };
    js_sys::Float32Array::new(&buffer).fill(7.0, 0, (ring * 2) as u32);
    let view = js_sys::Float32Array::new_with_byte_offset_and_length(
        &buffer,
        (BLOCK * frame) as u32,
        (2 * BLOCK) as u32,
    );
    let took = stream
        .read(0.0, &view)
        .unwrap_or_else(|e| unreachable!("a block: {}", as_text(&e)));
    assert_eq!(took, BLOCK);
    let whole = js_sys::Float32Array::new(&buffer).to_vec();
    assert!(
        whole[..2 * BLOCK].iter().all(|&v| v == 7.0),
        "before the view"
    );
    assert!(
        whole[4 * BLOCK..].iter().all(|&v| v == 7.0),
        "after the view"
    );
    let wide = render(&held, "wide");
    for c in 0..2 {
        let rendered = wide
            .samples(c)
            .unwrap_or_else(|_| unreachable!("two components"));
        let heard: Vec<f32> = whole[2 * BLOCK..4 * BLOCK]
            .iter()
            .skip(c)
            .step_by(2)
            .copied()
            .collect();
        assert_eq!(heard[..], rendered[..BLOCK], "component {c}");
    }
}

#[wasm_bindgen_test]
fn a_stream_refuses_what_it_cannot_take_at_the_boundary() {
    let held = page();
    let stream = opened(&held, "partials/one");
    refused_as(stream.read(0.5, &floats(BLOCK)).err(), "wasm.bad_argument");
    refused_as(stream.read(0.0, &floats(0)).err(), "engine.empty_read");
    read(&stream);
    refused_as(
        stream.read(0.0, &floats(BLOCK)).err(),
        "engine.stream_behind",
    );
    let wide = opened(&held, "wide");
    refused_as(wide.read(0.0, &floats(3)).err(), "wasm.bad_argument");
    let open = |options: JsValue| now(held.stream("@master([0, 1s])", BLOCK, options)).err();
    refused_as(open(JsValue::from(3)), "wasm.bad_argument");
    refused_as(
        open(self::options(&[("until", JsValue::from_str("2"))])),
        "validation_error",
    );
    refused_as(
        open(self::options(&[("release", JsValue::from_str("soon"))])),
        "wasm.bad_argument",
    );
    let empty = now(held.stream("@master([0, 1s])", 0, options(&[])));
    refused_as(empty.err(), "engine.no_stream");
}

/// An in-memory directory handle, counting under `read` each file read. A missing name rejects
/// as `NotFoundError`; a write lands on close; a file moves whole. A file is open from
/// `createWritable` to `close` and while it moves, and removing or moving it then rejects, as
/// OPFS does. Each call installs fresh exclusive, queued `navigator.locks`.
fn fake_directory() -> JsValue {
    js_sys::Function::new_no_args(
        r#"
        const held = new Map();
        const locks = {
            async request(name, ...rest) {
                const callback = rest[rest.length - 1];
                const options = rest.length > 1 ? rest[0] : {};
                if (options.ifAvailable && held.has(name)) return callback(null);
                while (held.has(name)) await held.get(name);
                const lock = { name, mode: "exclusive" };
                let release;
                held.set(name, new Promise((r) => { release = r; }));
                try { return await callback(lock); }
                finally { held.delete(name); release(); }
            },
            async query() {
                return { held: [...held.keys()].map((name) => ({ name, mode: "exclusive" })), pending: [] };
            },
        };
        if (typeof globalThis.navigator !== "object" || globalThis.navigator === null) {
            Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true, writable: true });
        }
        Object.defineProperty(globalThis.navigator, "locks", { value: locks, configurable: true });
        const missing = () => Object.assign(new Error("missing"), { name: "NotFoundError" });
        const busy = () => Object.assign(new Error("open elsewhere"), { name: "NoModificationAllowedError" });
        const read = { count: 0 };
        const directory = (name) => {
            const files = new Map();
            const dirs = new Map();
            const open = new Map();
            const opened = (name, by) => {
                const count = (open.get(name) || 0) + by;
                if (count === 0) open.delete(name); else open.set(name, count);
            };
            const held = {
                name,
                read,
                files,
                dirs,
                open,
                async getFileHandle(name, options) {
                    if (!files.has(name)) {
                        if (!(options && options.create)) throw missing();
                        files.set(name, new Uint8Array(0));
                    }
                    return {
                        async getFile() {
                            const bytes = files.get(name);
                            if (!bytes) throw missing();
                            read.count += 1;
                            const blob = (held) => ({
                                size: held.length,
                                async arrayBuffer() { return held.slice().buffer; },
                                slice(start, end) { return blob(held.slice(start, end)); },
                            });
                            return blob(bytes);
                        },
                        async createWritable() {
                            let pending = new Uint8Array(0);
                            opened(name, 1);
                            return {
                                async write(data) { pending = new Uint8Array(data); },
                                async close() { files.set(name, pending); opened(name, -1); },
                            };
                        },
                        async move(into, as) {
                            if (!files.has(name)) throw missing();
                            if (open.has(name) || into.open.has(as)) throw busy();
                            opened(name, 1);
                            into.open.set(as, (into.open.get(as) || 0) + 1);
                            await null;
                            const bytes = files.get(name);
                            files.delete(name);
                            into.files.set(as, bytes);
                            opened(name, -1);
                            const left = into.open.get(as) - 1;
                            if (left === 0) into.open.delete(as); else into.open.set(as, left);
                        },
                    };
                },
                async getDirectoryHandle(name, options) {
                    if (!dirs.has(name)) {
                        if (!(options && options.create)) throw missing();
                        dirs.set(name, directory(name));
                    }
                    return dirs.get(name);
                },
                async removeEntry(name) {
                    if (open.has(name)) throw busy();
                    if (!files.delete(name) && !dirs.delete(name)) throw missing();
                },
                keys() {
                    const names = [...files.keys(), ...dirs.keys()];
                    let at = 0;
                    return { async next() { return at < names.length ? { done: false, value: names[at++] } : { done: true }; } };
                },
            };
            return held;
        };
        return directory("sva");
        "#,
    )
    .call0(&JsValue::NULL)
    .unwrap_or_else(|_| unreachable!("the fake builds"))
}

/// Every value's file, beside the store's own index.
fn values_in(dir: &JsValue) -> usize {
    js_sys::Array::from(&field(dir, "files"))
        .iter()
        .filter_map(|pair| js_sys::Array::from(&pair).get(0).as_string())
        .filter(|name| name != "index")
        .count()
}

/// `Composition.open` over `dir`, its store's budget `store_max_bytes` where given.
async fn open_over(dir: &JsValue, store_max_bytes: Option<f64>) -> Composition {
    let budget = store_max_bytes.map_or(JsValue::UNDEFINED, |bytes| {
        let held = js_sys::Object::new();
        js_sys::Reflect::set(&held, &"store_max_bytes".into(), &bytes.into())
            .unwrap_or_else(|_| unreachable!("an object takes a key"));
        held.into()
    });
    Composition::open(None, Some(dir.clone().into()), budget)
        .await
        .unwrap_or_else(|e| unreachable!("it opens: {}", as_text(&e)))
}

async fn over_store(dir: &JsValue) -> Composition {
    let mut held = open_over(dir, None).await;
    held.insert("master", "sample(sin(2*pi*100*t))*0.5\n");
    held
}

async fn stored_render(held: &Composition) -> JsValue {
    held.render("@master([0, 0.1s])", None, options(&[]))
        .await
        .unwrap_or_else(|e| unreachable!("it renders: {}", as_text(&e)))
        .stats()
        .unwrap_or_else(|_| unreachable!("stats answer"))
}

#[wasm_bindgen_test]
async fn a_render_writes_the_directory_nothing_until_the_page_persists() {
    let dir = fake_directory();
    let held = over_store(&dir).await;
    stored_render(&held).await;
    assert_eq!(values_in(&dir), 0, "a render writes nothing");
    let written = held
        .persist()
        .await
        .unwrap_or_else(|_| unreachable!("persisted"));
    assert!(written > 0);
    assert_eq!(values_in(&dir), written);

    let warm = stored_render(&over_store(&dir).await).await;
    assert_eq!(
        field(&warm, "computed").as_f64(),
        Some(0.0),
        "{}",
        as_text(&warm)
    );
    assert_eq!(Composition::new(None).persist().await.ok(), Some(0));
}

async fn persisted(held: &Composition) {
    held.persist()
        .await
        .unwrap_or_else(|e| unreachable!("persisted: {}", as_text(&e)));
}

/// A budget that is no whole number is refused by name, as every option is.
#[wasm_bindgen_test]
async fn a_store_budget_that_is_no_whole_number_is_refused() {
    let held = js_sys::Object::new();
    js_sys::Reflect::set(&held, &"store_max_bytes".into(), &1.5.into())
        .unwrap_or_else(|_| unreachable!("an object takes a key"));
    let refused = Composition::open(None, Some(fake_directory().into()), held.into()).await;
    let e = refused.err().unwrap_or_else(|| unreachable!("refused"));
    assert_eq!(
        field(&e, "name").as_string().as_deref(),
        Some("validation_error")
    );
}

/// `@master` rendered with `out: null`, asking `readings`, or none.
async fn prepared(held: &Composition, readings: &[&str]) -> Rendering {
    let readings = (!readings.is_empty()).then(|| readings.iter().map(|r| r.to_string()).collect());
    held.render(
        "@master([0, 0.1s])",
        readings,
        options(&[("out", JsValue::NULL)]),
    )
    .await
    .unwrap_or_else(|e| unreachable!("it renders: {}", as_text(&e)))
}

/// A render with `out: null` hands back no samples and reads only what it asks. Persisted, the
/// target renders from the store alone; prepared again, nothing is computed.
#[wasm_bindgen_test]
async fn a_render_with_out_null_hands_back_no_samples() {
    let dir = fake_directory();
    let held = over_store(&dir).await;
    let bare = prepared(&held, &[]).await;
    refused_as(bare.samples(0).err(), "wasm.bad_argument");
    let asked = field(&readings(&bare), "representations");
    let keys = js_sys::Object::keys(asked.unchecked_ref::<js_sys::Object>());
    assert_eq!(keys.length(), 0, "{}", as_text(&asked));
    let stats = bare.stats().unwrap_or_else(|_| unreachable!("stats"));
    assert!(field(&stats, "computed").as_f64() > Some(0.0));
    assert_eq!(values_in(&dir), 0, "a render writes nothing");
    held.persist()
        .await
        .unwrap_or_else(|_| unreachable!("persisted"));

    let reader = over_store(&dir).await;
    let hit = stored_render(&reader).await;
    assert_eq!(
        field(&hit, "computed").as_f64(),
        Some(0.0),
        "{}",
        as_text(&hit)
    );
    let again = prepared(&over_store(&dir).await, &[]).await;
    let stats = again.stats().unwrap_or_else(|_| unreachable!("stats"));
    assert_eq!(
        field(&stats, "computed").as_f64(),
        Some(0.0),
        "{}",
        as_text(&stats)
    );
    let work = again.work().unwrap_or_else(|_| unreachable!("work"));
    assert_eq!(field(&work, "computed_samples").as_f64(), Some(0.0));

    let read = prepared(&reader, &["envelope"]).await;
    let asked = field(&readings(&read), "representations");
    assert!(
        !field(&asked, "envelope").is_undefined(),
        "{}",
        as_text(&asked)
    );
    refused_as(read.samples(0).err(), "wasm.bad_argument");
    let samples = Some(vec!["samples".to_string()]);
    let refused = reader
        .render(
            "@master([0, 0.1s])",
            samples,
            options(&[("out", JsValue::NULL)]),
        )
        .await;
    refused_as(refused.err(), "wasm.bad_argument");
    let named = reader
        .render(
            "@master([0, 0.1s])",
            None,
            options(&[("out", JsValue::TRUE)]),
        )
        .await;
    refused_as(named.err(), "wasm.bad_argument");
}

/// `index` set to what an older format wrote, over the values a newer one left under it.
fn aged(dir: &JsValue) {
    let files: js_sys::Map = field(dir, "files").into();
    let text = js_sys::Uint8Array::from(&b"sva store format 1\n"[..]);
    files.set(&"index".into(), &text);
}

/// What `work` resolves to, run as its own task beside the others.
fn task(work: impl Future<Output = Result<JsValue, JsValue>> + 'static) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(work)
}

/// A directory holding the values of `over_store`'s render, under an older format's index.
async fn aged_directory() -> JsValue {
    let dir = fake_directory();
    let first = over_store(&dir).await;
    stored_render(&first).await;
    first
        .persist()
        .await
        .unwrap_or_else(|e| unreachable!("persisted: {}", as_text(&e)));
    aged(&dir);
    dir
}

/// Four workers render the same sound with `out: null` and one renders it, each over its own
/// store on `dir`, all at once, and each persists.
async fn workers(dir: &JsValue) {
    let workers: js_sys::Array = (0..5)
        .map(|n| {
            let dir = dir.clone();
            task(async move {
                let mut held = open_over(&dir, None).await;
                held.insert("master", "sample(sin(2*pi*100*t))*0.5\n");
                let out = match n {
                    0 => options(&[]),
                    _ => options(&[("out", JsValue::NULL)]),
                };
                drop(held.render("@master([0, 0.1s])", None, out).await?);
                Ok(JsValue::from(held.persist().await?))
            })
        })
        .collect();
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::all(&workers))
        .await
        .unwrap_or_else(|e| unreachable!("every worker succeeds: {}", as_text(&e)));
}

/// An entry another holder keeps open is left by every commit over it, and no worker fails;
/// once it closes, the next persist writes it.
#[wasm_bindgen_test]
async fn an_entry_open_elsewhere_fails_no_worker_and_the_next_persist_writes_it() {
    let dir = aged_directory().await;
    let files: js_sys::Map = field(&dir, "files").into();
    let name = files
        .keys()
        .into_iter()
        .filter_map(|name| name.ok()?.as_string())
        .find(|name| name != "index")
        .unwrap_or_else(|| unreachable!("a value's file"));
    let handle = js_call(&dir, "getFileHandle", &[name.clone().into()]).await;
    let writable = js_call(&handle, "createWritable", &[]).await;
    workers(&dir).await;
    assert!(files.has(&name.clone().into()), "open, so left");
    js_call(&writable, "close", &[]).await;

    let held = over_store(&dir).await;
    let cold = stored_render(&held).await;
    let missed = items(&cold, "lookups")
        .iter()
        .any(|l| field(&l, "outcome").as_string().as_deref() != Some("hit"));
    assert!(missed, "the entry left open was missed: {}", as_text(&cold));
    held.persist()
        .await
        .unwrap_or_else(|e| unreachable!("persisted: {}", as_text(&e)));
    let warm = stored_render(&over_store(&dir).await).await;
    assert_eq!(
        field(&warm, "computed").as_f64(),
        Some(0.0),
        "{}",
        as_text(&warm)
    );
}

/// Every call on `dir` and the directories in it rejects from now on, as on a revoked handle.
fn revoked(dir: &JsValue) {
    js_sys::Function::new_with_args(
        "dir",
        r#"
        const refuse = async () => { throw Object.assign(new Error("revoked"), { name: "SecurityError" }); };
        const revoke = (dir) => {
            for (const name of ["getFileHandle", "getDirectoryHandle", "removeEntry"]) dir[name] = refuse;
            dir.keys = () => ({ next: refuse });
            for (const inner of dir.dirs.values()) revoke(inner);
        };
        revoke(dir);
        "#,
    )
    .call1(&JsValue::NULL, dir)
    .unwrap_or_else(|_| unreachable!("the fake revokes"));
}

async fn channel(held: &Composition) -> Vec<f32> {
    held.render("@master([0, 0.1s])", None, options(&[]))
        .await
        .unwrap_or_else(|e| unreachable!("it renders: {}", as_text(&e)))
        .samples(0)
        .unwrap_or_else(|e| unreachable!("its samples: {}", as_text(&e)))
}

/// A directory that rejects every call fails no render, its out kept or not, whether it failed
/// after the store opened or before: each renders a storeless composition's samples.
#[wasm_bindgen_test]
async fn a_directory_that_rejects_every_call_fails_no_render() {
    let dir = fake_directory();
    let before = over_store(&dir).await;
    revoked(&dir);
    let after = over_store(&dir).await;
    let mut memory = Composition::new(None);
    memory.insert("master", "sample(sin(2*pi*100*t))*0.5\n");
    let bare = channel(&memory).await;
    for held in [&before, &after] {
        assert_eq!(channel(held).await, bare);
        let read = prepared(held, &["envelope"]).await;
        let asked = field(&readings(&read), "representations");
        assert!(!field(&asked, "envelope").is_undefined());
    }
    assert_eq!(after.persist().await.ok(), Some(0));
}

/// `of.method(...args)`, awaited.
async fn js_call(of: &JsValue, method: &str, args: &[JsValue]) -> JsValue {
    let method: js_sys::Function = field(of, method).into();
    let args: js_sys::Array = args.iter().collect();
    let promise = method
        .apply(of, &args)
        .unwrap_or_else(|_| unreachable!("`{method:?}` answers"));
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise))
        .await
        .unwrap_or_else(|e| unreachable!("it settles: {}", as_text(&e)))
}

/// A tap's add awaits the store while the stream plays on: placed `at: "landing"`, the note's
/// sample 0 is the first sample it lands at, so the attack is whole however late it lands.
#[wasm_bindgen_test]
async fn an_add_placed_at_landing_plays_its_note_from_sample_zero() {
    let dir = fake_directory();
    let held = over_store(&dir).await;
    let stream = held
        .stream(
            "@notes([0, 1s])",
            BLOCK,
            options(&[("live", JsValue::TRUE)]),
        )
        .await
        .unwrap_or_else(|e| unreachable!("it streams: {}", as_text(&e)));
    let placed = |at: &str| {
        let held = js_sys::Object::new();
        js_sys::Reflect::set(&held, &"at".into(), &at.into())
            .unwrap_or_else(|_| unreachable!("an object takes a key"));
        JsValue::from(held)
    };
    let landing = placed("landing");
    let mut add = std::pin::pin!(stream.add("@master", Some(landing)));
    let polled = add.as_mut().poll(&mut Context::from_waker(Waker::noop()));
    assert!(polled.is_pending(), "the add awaits the directory");
    blocks(&stream, 1);
    let handle = add
        .await
        .unwrap_or_else(|e| unreachable!("added: {}", as_text(&e)));
    let landed = stream.landed(handle);
    assert_eq!(
        landed,
        Some(stream.position()),
        "it lands at a block's start"
    );
    assert!(landed > Some(0.0), "the stream played on while it waited");

    let mut alone = Composition::new(None);
    alone.insert("master", "sample(sin(2*pi*100*t))*0.5\n");
    let note = plane(&render(&alone, "master"));
    assert_eq!(blocks(&stream, 2)[..], note[..2 * BLOCK]);
    refused_as(
        now(stream.add("@master", Some(placed("rather")))).err(),
        "validation_error",
    );
}

/// `of.method(...args)` as a page calls it, what it returns or throws.
fn called(of: &JsValue, method: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    let method: js_sys::Function = field(of, method).into();
    method.apply(of, &args.iter().collect())
}

/// A page reopening its stream frees the old one with an edit in flight: the free succeeds,
/// the edit rejects as a conflict, and no handle of the old stream names a term of the new.
#[wasm_bindgen_test]
async fn a_stream_freed_with_an_edit_in_flight_rejects_it_and_its_handles_name_nothing_new() {
    let held = page();
    let open = || now(held.stream("@notes([0, 1s])", BLOCK, options(&[])));
    let old = open().unwrap_or_else(|e| unreachable!("it streams: {}", as_text(&e)));
    let kept = now(old.add("@master", None)).unwrap_or_else(|e| unreachable!("{}", as_text(&e)));
    let old = JsValue::from(old);
    let pending = called(&old, "add", &["@master(t - 0.1s)".into()])
        .unwrap_or_else(|e| unreachable!("add answers a promise: {}", as_text(&e)));
    let freed = called(&old, "free", &[]);
    assert!(
        freed.is_ok(),
        "{}",
        as_text(&freed.err().unwrap_or_default())
    );
    let settled = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(pending)).await;
    let refused = settled
        .err()
        .unwrap_or_else(|| unreachable!("the edit rejects"));
    assert_eq!(
        field(&refused, "name").as_string().as_deref(),
        Some("conflict")
    );

    let new = open().unwrap_or_else(|e| unreachable!("it streams: {}", as_text(&e)));
    let added = now(new.add("@master", None)).unwrap_or_else(|e| unreachable!("{}", as_text(&e)));
    assert_ne!(added, kept, "a handle names one term of one stream");
    let replaced = now(new.replace(kept, "@master(t - 0.2s)", None));
    assert_eq!(
        replaced.ok(),
        Some(false),
        "the old handle names nothing here"
    );
    assert_eq!(now(new.remove(kept)).ok(), Some(false));
}

/// A thousand refs, each reading the next as a gain, a max or a filter does, render on the
/// stack a page's wasm runs on.
#[wasm_bindgen_test]
fn a_chain_a_thousand_refs_deep_renders() {
    let mut held = Composition::new(None);
    held.insert("c0", "sample(crop(sin(2*pi*220*t), 0s, 0.1s))\n");
    for k in 1..=1000 {
        let read = format!("@c{}(t)", k - 1);
        let body = match k % 3 {
            0 => format!("0.999*{read}"),
            1 => format!("max({read}, -1)"),
            _ => format!("crop(lowpass({read}, 3000), 0s, 1s)"),
        };
        held.insert(&format!("c{k}"), &format!("{body}\n"));
    }
    let drawn = plane(&render(&held, "c1000"));
    assert!(drawn.iter().any(|v| *v != 0.0), "the chain sounds");
}

/// A saw computed whole before the first block, dear enough for a store to keep, and a filter
/// over it block by block.
fn filtered(held: &mut Composition) {
    held.insert(
        "tone",
        "sample(crop(sum(k, 1, 60, sin(2*pi*k*55*t)/k), 0s, 30s))\n",
    );
    held.insert("wet", "lowpass(@tone, cutoff=1000)\n");
}

const WET: &str = "@wet([0, 30s])";

async fn work_of(held: &Composition, options: JsValue) -> Result<f64, JsValue> {
    let rendered = held.render(WET, None, options).await?;
    let work = rendered.work().unwrap_or_else(|_| unreachable!("work"));
    Ok(field(&work, "computed_samples")
        .as_f64()
        .unwrap_or(f64::NAN))
}

/// An `AbortController` the page aborts on its own turn stops the render it handed.
#[wasm_bindgen_test]
async fn an_aborted_controller_stops_a_render() {
    let controller = js_sys::Reflect::get(&js_sys::global(), &"AbortController".into())
        .unwrap_or_else(|_| unreachable!("a global AbortController"));
    let controller = js_sys::Reflect::construct(&controller.into(), &js_sys::Array::new())
        .unwrap_or_else(|_| unreachable!("a controller"));
    let signal = field(&controller, "signal");
    js_sys::Function::new_with_args("c", "setTimeout(() => c.abort(), 0);")
        .call1(&JsValue::NULL, &controller)
        .unwrap_or_else(|_| unreachable!("the abort is queued"));
    let mut held = Composition::new(None);
    filtered(&mut held);
    let stopped = work_of(&held, options(&[("signal", signal)])).await;
    let error = stopped.err().unwrap_or_else(|| unreachable!("it stops"));
    assert_eq!(
        field(&error, "name").as_string().as_deref(),
        Some("AbortError")
    );
}

/// A `signal` that is no `AbortSignal` refuses by name.
#[wasm_bindgen_test]
fn a_signal_that_is_no_abort_signal_refuses() {
    let refused = page().rendered("@master([0, 0.1s])", None, options(&[("signal", 3.into())]));
    let error = refused.err().unwrap_or_else(|| unreachable!("it refuses"));
    assert_eq!(
        field(&error, "name").as_string().as_deref(),
        Some("validation_error")
    );
}

/// Every report a subscription to `held` hears from here on.
fn subscribed(held: &Composition) -> js_sys::Array {
    let heard = js_sys::Array::new();
    let push = js_sys::Function::new_with_args("stats", "this.push(stats)");
    held.subscribe(push.bind0(&heard));
    heard
}

fn count(of: &JsValue, path: &[&str]) -> f64 {
    let at = path.iter().fold(of.clone(), |at, name| field(&at, name));
    at.as_f64()
        .unwrap_or_else(|| unreachable!("{path:?} is a number in {}", as_text(of)))
}

/// Two seconds of `@master` heard through `heard`: the reports of its one call, held to what a
/// subscriber may draw from them and to what the finished render states of itself. Its last.
async fn reported(held: &Composition, heard: &js_sys::Array) -> JsValue {
    let from = heard.length();
    let render = held
        .render("@master([0, 2s])", None, options(&[]))
        .await
        .unwrap_or_else(|e| unreachable!("it renders: {}", as_text(&e)));
    let reports: Vec<JsValue> = heard.iter().skip(from as usize).collect();
    let (first, last) = (&reports[0], &reports[reports.len() - 1]);
    let all = as_text(heard);
    assert_eq!(
        field(first, "state").as_string().as_deref(),
        Some("working")
    );
    assert_eq!(field(last, "state").as_string().as_deref(), Some("idle"));
    assert_eq!(count(last, &["working"]), 0.0);
    let call = |report: &JsValue| field(report, "call");
    let text = |of: &JsValue, name: &str| field(of, name).as_string().unwrap_or_default();
    assert_eq!(text(&call(first), "phase"), "parsing", "{all}");
    assert_eq!(text(&call(last), "phase"), "done", "{all}");
    assert_eq!(text(&call(last), "kind"), "render");
    assert_eq!(text(&call(last), "target"), "@master([0, 2s])");
    let mut at = 0.0;
    let mut delivering = 0;
    let (mut worked, mut typed) = (0.0, 0.0);
    for report in &reports {
        assert!(field(report, "boot").is_null(), "{all}");
        let call = call(report);
        assert_eq!(
            count(&call, &["id"]),
            count(first, &["call", "id"]),
            "{all}"
        );
        let samples = count(&call, &["samples"]);
        assert!(samples >= at, "samples fell: {all}");
        at = samples;
        let work = count(&call, &["cached_samples"]) + count(&call, &["computed_samples"]);
        assert!(work >= worked, "work done fell: {all}");
        worked = work;
        let typing = field(&call, "typing");
        if !typing.is_null() {
            let (now, nodes) = (count(&typing, &["typed"]), count(&typing, &["nodes"]));
            assert!(
                now >= typed && now <= nodes,
                "typing fell or overran: {all}"
            );
            typed = now;
        }
        match text(&call, "phase").as_str() {
            "delivering" => {
                delivering += 1;
                assert_eq!(count(&call, &["total_samples"]), 16000.0, "{all}");
                assert!(!field(&call, "planned_samples").is_null(), "{all}");
            }
            "parsing" => {
                assert!(typing.is_null(), "{all}");
                assert!(field(&call, "planned_samples").is_null(), "{all}");
            }
            "typing" | "looking_up" => {
                assert!(field(&call, "total_samples").is_null(), "{all}");
                assert!(field(&call, "planned_samples").is_null(), "{all}");
            }
            _ => {}
        }
    }
    assert!(delivering > 2, "progress arrives while it delivers: {all}");
    let done = call(last);
    assert_eq!(count(&done, &["samples"]), 16000.0, "{all}");
    assert_eq!(count(&done, &["total_samples"]), 16000.0, "{all}");
    assert_eq!(worked, count(&done, &["planned_samples"]), "{all}");
    assert_eq!(typed, count(&done, &["typing", "nodes"]), "{all}");

    let stats = render.stats().unwrap_or_else(|_| unreachable!("stats"));
    let work = render.work().unwrap_or_else(|_| unreachable!("work"));
    let state = format!("{} against {}", as_text(&done), as_text(&stats));
    let lookups = |name: &str| count(&done, &["lookups", name]);
    let ended = |name: &str| count(&stats, &[name]);
    assert_eq!(
        count(&done, &["computed_samples"]),
        count(&work, &["computed_samples"])
    );
    assert_eq!(lookups("hit"), ended("hits"), "{state}");
    assert_eq!(
        lookups("from_memory") + lookups("from_store"),
        ended("hits")
    );
    assert_eq!(lookups("miss") + lookups("prefix"), ended("computed"));
    let new = ended("stored") + ended("replaced") + ended("extended");
    assert_eq!(lookups("new"), new, "{state}");
    assert_eq!(lookups("reused"), ended("reused"), "{state}");
    for name in [
        "disk_lookups",
        "disk_reads",
        "disk_read_bytes",
        "promotions",
    ] {
        let told = count(&done, &["tier", name]);
        assert_eq!(told, count(&stats, &["tier", name]), "{name}: {state}");
    }
    last.clone()
}

#[wasm_bindgen_test]
async fn a_subscriber_hears_each_render_work_then_rest_with_the_counts_it_ends_at() {
    let dir = fake_directory();
    let cold = over_store(&dir).await;
    let heard = subscribed(&cold);
    let opening = heard.get(0);
    assert_eq!(heard.length(), 1, "a subscription is answered at once");
    assert_eq!(
        field(&opening, "state").as_string().as_deref(),
        Some("idle")
    );
    assert!(field(&opening, "call").is_null());

    let computed = reported(&cold, &heard).await;
    let of = |report: &JsValue, path: &[&str]| count(report, path);
    assert!(of(&computed, &["call", "lookups", "miss"]) > 0.0);
    assert_eq!(of(&computed, &["call", "lookups", "hit"]), 0.0);
    assert!(of(&computed, &["call", "computed_samples"]) >= 16000.0);
    assert_eq!(of(&computed, &["call", "cached_samples"]), 0.0);
    assert!(of(&computed, &["call", "typing", "nodes"]) > 0.0);

    let before = heard.length();
    persisted(&cold).await;
    let storing: Vec<JsValue> = heard.iter().skip(before as usize).collect();
    let phases: Vec<String> = storing
        .iter()
        .filter_map(|report| field(&field(report, "call"), "phase").as_string())
        .collect();
    assert_eq!(phases, ["storing", "done"], "{}", as_text(&heard));

    let warm = over_store(&dir).await;
    let heard = subscribed(&warm);
    let stored = reported(&warm, &heard).await;
    let state = as_text(&stored);
    assert!(
        of(&stored, &["call", "lookups", "from_store"]) > 0.0,
        "{state}"
    );
    assert_eq!(
        of(&stored, &["call", "lookups", "from_memory"]),
        0.0,
        "{state}"
    );
    assert_eq!(of(&stored, &["call", "computed_samples"]), 0.0, "{state}");
    let read_back = of(&stored, &["call", "cached_samples"]);
    assert_eq!(read_back, 16000.0, "{state}");
    assert_eq!(
        of(&stored, &["call", "cached_from_store_samples"]),
        read_back,
        "{state}"
    );
    assert!(
        of(&stored, &["call", "tier", "disk_read_bytes"]) > 0.0,
        "{state}"
    );

    let resident = reported(&warm, &heard).await;
    let state = as_text(&resident);
    assert!(
        of(&resident, &["call", "lookups", "from_memory"]) > 0.0,
        "{state}"
    );
    assert_eq!(
        of(&resident, &["call", "lookups", "from_store"]),
        0.0,
        "{state}"
    );
    assert_eq!(of(&resident, &["call", "computed_samples"]), 0.0, "{state}");
    assert_eq!(
        of(&resident, &["call", "cached_samples"]),
        16000.0,
        "{state}"
    );
    assert_eq!(
        of(&resident, &["call", "cached_from_store_samples"]),
        0.0,
        "{state}"
    );
    assert_eq!(of(&resident, &["call", "typing", "nodes"]), 0.0, "{state}");
    assert_eq!(
        of(&resident, &["totals", "cached_samples"]),
        32000.0,
        "{state}"
    );
    assert_eq!(
        of(&resident, &["call", "tier", "disk_reads"]),
        0.0,
        "{state}"
    );
    assert_eq!(of(&resident, &["totals", "samples"]), 32000.0, "{state}");
    let both =
        of(&stored, &["call", "lookups", "hit"]) + of(&resident, &["call", "lookups", "hit"]);
    assert_eq!(
        of(&resident, &["totals", "lookups", "hit"]),
        both,
        "{state}"
    );

    warm.unsubscribe();
    let before = heard.length();
    stored_render(&warm).await;
    assert_eq!(heard.length(), before, "unsubscribed, it hears no more");
}

/// Each report's boot phase, or `call` for one of a call and `rest` for one of neither.
fn steps(heard: &js_sys::Array, from: u32) -> Vec<String> {
    let reports = heard.iter().skip(from as usize);
    let step = |report: JsValue| {
        let (boot, call) = (field(&report, "boot"), field(&report, "call"));
        match (boot.is_null(), call.is_null()) {
            (false, _) => field(&boot, "phase").as_string().unwrap_or_default(),
            (true, false) => "call".to_string(),
            (true, true) => "rest".to_string(),
        }
    };
    reports.map(step).collect()
}

#[wasm_bindgen_test]
async fn the_modules_subscriber_hears_the_boot_then_every_composition() {
    let heard = js_sys::Array::new();
    let push = js_sys::Function::new_with_args("stats", "this.push(stats)");
    sva_wasm::subscribe(push.bind0(&heard));
    assert_eq!(heard.length(), 0, "nothing is open yet");

    let dir = fake_directory();
    let cold = over_store(&dir).await;
    let opening = [
        "opening_store",
        "reading_index",
        "index_read",
        "opened",
        "rest",
    ];
    assert_eq!(steps(&heard, 0), opening, "{}", as_text(&heard));
    let states: Vec<String> = heard
        .iter()
        .filter_map(|report| field(&report, "state").as_string())
        .collect();
    let all = as_text(&heard);
    assert_eq!(
        states,
        ["booting", "booting", "booting", "idle", "idle"],
        "{all}"
    );
    let store = |report: u32| field(&field(&heard.get(report), "boot"), "store");
    assert!(field(&store(0), "entries").is_null(), "{all}");
    assert!(count(&store(0), &["max_bytes"]) > 0.0, "{all}");
    assert_eq!(count(&store(2), &["entries"]), 0.0, "{all}");
    assert_eq!(count(&store(3), &["bytes"]), 0.0, "{all}");

    stored_render(&cold).await;
    let calls = steps(&heard, opening.len() as u32);
    assert!(calls.len() > 2 && calls.iter().all(|step| step == "call"));
    persisted(&cold).await;

    let before = heard.length();
    let warm = over_store(&dir).await;
    assert_eq!(steps(&heard, before), opening, "{}", as_text(&heard));
    let read = field(&field(&heard.get(before + 2), "boot"), "store");
    let state = as_text(&read);
    assert!(count(&read, &["entries"]) > 0.0, "{state}");
    assert!(count(&read, &["bytes"]) > 0.0, "{state}");
    assert!(count(&read, &["index_bytes"]) > 0.0, "{state}");

    let before = heard.length();
    let own = subscribed(&warm);
    stored_render(&warm).await;
    assert_eq!(heard.length(), before, "its own subscriber hears it");
    assert!(own.length() > 1);
    warm.unsubscribe();
    assert_eq!(steps(&heard, before), ["rest"], "the module's again");
    let unstored = page();
    assert_eq!(steps(&heard, before + 1), ["rest"], "one made later too");

    sva_wasm::unsubscribe();
    let before = heard.length();
    render(&unstored, "master");
    stored_render(&warm).await;
    over_store(&fake_directory()).await;
    assert_eq!(heard.length(), before, "unsubscribed, it hears no more");
}

#[wasm_bindgen_test]
fn a_nodes_outline_is_its_bodys_parse_tree() {
    let text = "; Models: a voice | Neglects: all | IO: (t) -> amplitude\nf0 = C4\nsin(f0*t)\n";
    let tree = sva_wasm::outline(text).unwrap_or_else(|_| unreachable!("a body parses"));
    assert_eq!(
        field(&tree, "text").as_string().as_deref(),
        Some("sin(f0*t)\n")
    );
    let root = field(&tree, "outline");
    assert_eq!(field(&root, "kind").as_string().as_deref(), Some("call"));
    assert!(
        sva_wasm::outline("sin(").is_err(),
        "a parse refusal crosses"
    );
}

#[wasm_bindgen_test]
fn every_reading_a_render_answers_is_listed() {
    let listed = sva_wasm::readings().unwrap_or_else(|_| unreachable!("the table crosses"));
    let names: Vec<String> = listed
        .iter()
        .filter_map(|row| field(&row, "name").as_string())
        .collect();
    assert_eq!(names.len() as u32, listed.length());
    for name in ["lines", "samples", "ledger", "bindings"] {
        assert!(names.iter().any(|n| n == name), "{name} in {names:?}");
    }
}

/// A one-sine clip at 48 kHz, as `seed` and a cost's reference take its channels.
fn sine_clip() -> (u32, js_sys::Array) {
    let rate = 48_000u32;
    let clip: Vec<f32> = (0..rate * 3 / 10)
        .map(|n| {
            let t = f64::from(n) / f64::from(rate);
            (0.1 * js_sys::Math::exp(-6.0 * t)
                * js_sys::Math::sin(std::f64::consts::TAU * 220.0 * t)) as f32
        })
        .collect();
    let channels = js_sys::Array::of1(&js_sys::Float32Array::from(clip.as_slice()));
    (rate, channels)
}

/// `{ key: value, ... }` as a page writes it.
fn object(pairs: &[(&str, JsValue)]) -> js_sys::Object {
    let object = js_sys::Object::new();
    for (key, value) in pairs {
        js_sys::Reflect::set(&object, &(*key).into(), value)
            .unwrap_or_else(|_| unreachable!("an object takes a key"));
    }
    object
}

/// While the solver is rebuilt, `solve` and `seed` each throw their stub code as the CLI
/// refuses it; an option no solve takes still refuses by name first.
#[wasm_bindgen_test]
fn solve_and_seed_throw_unbuilt() {
    let code = |refused: JsValue| {
        let envelope = as_text(&field(&refused, "refusal"));
        assert_eq!(
            field(&refused, "name").as_string().as_deref(),
            Some("validation_error"),
            "{envelope}"
        );
        envelope
    };
    let comp = page();
    let vary = js_sys::Array::of1(&JsValue::from("gain=0..1/0.25"));
    let asked = object(&[
        ("method", JsValue::from("prox")),
        ("steps", JsValue::from(3)),
    ]);
    let refused = now(comp.solve(JsValue::from("1"), vary.clone(), asked.into()))
        .err()
        .unwrap_or_else(|| unreachable!("solve refuses"));
    assert!(code(refused).contains("solve.unbuilt"));
    let unknown = object(&[("spiral", JsValue::from(1))]);
    let refused = now(comp.solve(JsValue::from("1"), vary, unknown.into()))
        .err()
        .unwrap_or_else(|| unreachable!("an unknown option refuses"));
    assert!(code(refused).contains("wasm.bad_argument"));
    let (rate, channels) = sine_clip();
    let refused = sva_wasm::seed(rate, channels, "clip.wav", None)
        .err()
        .unwrap_or_else(|| unreachable!("seed refuses"));
    assert!(code(refused).contains("seed.unbuilt"));
}

/// A composition holding the starter `seeded` hands over, the cost text `seed` hands with it,
/// the starter's varies, and the clip as the recording that text's `sample("clip.wav")` reads.
fn seeded_fit(
    rate: u32,
    channels: &js_sys::Array,
    seeded: &JsValue,
) -> (Composition, JsValue, js_sys::Array, JsValue) {
    let mut comp = Composition::new(Some("a-fit".to_string()));
    for file in items(seeded, "files").iter() {
        let (path, text) = (field(&file, "path"), field(&file, "text"));
        comp.insert(
            &path.as_string().unwrap_or_default(),
            &text.as_string().unwrap_or_default(),
        );
    }
    let varies = js_sys::Array::from(&field(&field(seeded, "varies"), "items"));
    let clip = object(&[("rate", JsValue::from(rate)), ("channels", channels.into())]);
    let recordings = object(&[("clip.wav", clip.into())]);
    (comp, field(seeded, "cost"), varies, recordings.into())
}

/// A starter seeded at `{ harmonics: 1, lone: 1 }` solves through `grow` up to two harmonics:
/// the composition as given solved, then each level grown, the chosen stage written. `progress`
/// hears each listed stage as it finishes with its terms, its files rendering over the starter,
/// the last least one's the answer's `files`; activities are timed off wasm's clock.
#[wasm_bindgen_test]
#[ignore = "spec for the solve rewrite"]
fn a_coarse_starter_grows_through_its_levels() {
    let (rate, channels) = sine_clip();
    let limits = object(&[("harmonics", JsValue::from(1)), ("lone", JsValue::from(1))]);
    let seeded = sva_wasm::seed(rate, channels.clone(), "clip.wav", Some(limits))
        .unwrap_or_else(|e| unreachable!("a clip seeds: {}", as_text(&e)));
    let (comp, cost, varies, recordings) = seeded_fit(rate, &channels, &seeded);
    let heard = js_sys::Array::new();
    let push = js_sys::Function::new_with_args("line", "this.push(line)");
    let continuation = object(&[
        ("scheme", JsValue::from("grow")),
        ("harmonics", JsValue::from(2)),
        ("lone", JsValue::from(1)),
    ]);
    let asked = object(&[
        ("method", JsValue::from("prox")),
        ("continuation", continuation.into()),
        ("steps", JsValue::from(3)),
        ("heartbeat", JsValue::from(0)),
        ("write", JsValue::from(true)),
        ("recordings", recordings),
        ("progress", push.bind0(&heard).into()),
    ]);
    let answered = now(comp.solve(cost, varies, asked.into())).unwrap_or_else(|e| {
        unreachable!(
            "the priced solve answers: {}",
            as_text(&field(&e, "refusal"))
        )
    });
    let data = field(&answered, "data");
    assert_eq!(
        field(&field(&data, "continuation"), "scheme")
            .as_string()
            .as_deref(),
        Some("grow")
    );
    let labels: Vec<String> = items(&data, "stages")
        .iter()
        .filter_map(|s| field(&s, "label").as_string())
        .collect();
    assert_eq!(labels[..2], ["start", "settle"], "{labels:?}");
    assert!(
        labels[2..].iter().all(|l| l == "grow h=2 lone=1"),
        "{labels:?}"
    );
    assert!(field(&data, "chosen").is_object());
    assert!(
        items(&data, "written").length() > 0,
        "the chosen stage is written"
    );
    let finished: Vec<JsValue> = heard
        .iter()
        .filter(|l| field(l, "finished").as_bool() == Some(true))
        .collect();
    assert!(
        heard.length() as usize > finished.len(),
        "the steps are heard too"
    );
    let heard_labels: Vec<String> = finished
        .iter()
        .filter_map(|l| field(l, "stage").as_string())
        .collect();
    assert_eq!(heard_labels, labels, "a finished line per listed stage");
    assert!(
        heard
            .iter()
            .any(|l| field(&l, "action").as_string().as_deref() == Some("descending")),
        "activities are heard"
    );
    for line in &finished {
        for key in ["residual_bits", "bits", "numbers"] {
            assert!(
                field(line, key).as_f64().is_some(),
                "{key}: {}",
                as_text(line)
            );
        }
        let (mut fresh, ..) = seeded_fit(rate, &channels, &seeded);
        for file in items(line, "files").iter() {
            fresh.insert(
                &field(&file, "path").as_string().unwrap_or_default(),
                &field(&file, "text").as_string().unwrap_or_default(),
            );
        }
        assert!(
            fresh
                .rendered("@master([0, 0.1s])", None, options(&[]))
                .is_ok(),
            "a stage's files render: {}",
            as_text(line)
        );
    }
    let least = finished
        .iter()
        .rfind(|l| field(l, "least").as_bool() == Some(true))
        .unwrap_or_else(|| unreachable!("a stage is least"));
    assert_eq!(
        field(least, "length_bits").as_f64(),
        field(&field(&data, "chosen"), "length_bits").as_f64(),
        "the last least stage is the chosen one"
    );
    assert_eq!(
        as_text(&field(least, "files")),
        as_text(&field(&data, "files")),
        "its files are the answer's"
    );
}

/// `seed`'s cost text, its `sample("clip.wav")` read from `recordings`, solves to the same
/// stages, choice and files as the tree it stands for, the clip passed as a value; a text the
/// grammar refuses refuses as `sva-cli solve` does, and a recording no page passed is named.
#[wasm_bindgen_test]
#[ignore = "spec for the solve rewrite"]
fn a_seeded_cost_text_solves_as_its_tree() {
    let (rate, channels) = sine_clip();
    let seeded = sva_wasm::seed(rate, channels.clone(), "clip.wav", None)
        .unwrap_or_else(|e| unreachable!("a clip seeds: {}", as_text(&e)));
    let (comp, text, varies, recordings) = seeded_fit(rate, &channels, &seeded);
    let leaf = |target: JsValue| {
        JsValue::from(object(&[
            ("repr", JsValue::from("internal")),
            ("target", target),
        ]))
    };
    let clip = object(&[(
        "samples",
        object(&[("rate", JsValue::from(rate)), ("channels", channels.into())]).into(),
    )]);
    let residual = js_sys::Array::of2(&leaf(JsValue::from("@master")), &leaf(clip.into()));
    let numbers = object(&[
        ("repr", JsValue::from("numbers")),
        ("target", JsValue::from("@master")),
    ]);
    let norm = object(&[("norm", object(&[("sub", residual.into())]).into())]);
    let tree = object(&[("add", js_sys::Array::of2(&norm, &numbers).into())]);
    let asked = |recordings: Option<&JsValue>| {
        let mut pairs = vec![
            ("method", JsValue::from("prox")),
            ("steps", JsValue::from(2)),
        ];
        pairs.extend(recordings.map(|r| ("recordings", r.clone())));
        JsValue::from(object(&pairs))
    };
    let solved = |cost: JsValue| {
        let answered = now(comp.solve(cost, varies.clone(), asked(Some(&recordings))))
            .unwrap_or_else(|e| {
                unreachable!(
                    "the priced solve answers: {}",
                    as_text(&field(&e, "refusal"))
                )
            });
        let data = field(&answered, "data");
        ["stages", "chosen", "files"].map(|key| as_text(&field(&data, key)))
    };
    assert_eq!(solved(text.clone()), solved(tree.into()));
    let refusal = |cost: &str, recordings: Option<&JsValue>| {
        let refused = now(comp.solve(cost.into(), varies.clone(), asked(recordings)))
            .err()
            .unwrap_or_else(|| unreachable!("`{cost}` refuses"));
        as_text(&field(&refused, "refusal"))
    };
    let broken = refusal("norm(repr.internal(@master)", Some(&recordings));
    assert!(broken.contains("solve.bad_cost"), "{broken}");
    let unheld = refusal(&text.as_string().unwrap_or_default(), None);
    assert!(
        unheld.contains("`sample(\\\"clip.wav\\\")` names no recording"),
        "{unheld}"
    );
}
