// Concern: proves memory over a disk answers renders and streams from the root down, writes until persist, keeps a version and budget | Non-concern: a real medium | IO: (composition, fake disk) -> hits

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll, Waker};

use crate::disk::{Held, Memory, disk, now, opened};
use crate::fixtures::{graph_of, next, samples};
use sva_ast::Graph;
use sva_engine::{
    Backend, Buffer, CacheStats, Change, Changed, EngineError, Handle, Hash, Hear, INDEX_NAME,
    Never, Out, Outcome, Placed, Range, Render, RenderConfig, STORE_FORMAT, Session, Store, Stream,
    StreamConfig, Tier, change, render, render_in, render_over,
};
use sva_formula::FOURIER_DUAL_RULES_VERSION;

const SECONDS: f64 = 0.05;
const RATE: u32 = 8_000;

/// A render's node lookups alone, each of which memory may pass to the disk: the walk looks up
/// each node before any value under it is asked.
fn nodes(stats: &CacheStats) -> Vec<&sva_engine::Lookup> {
    let mut named = std::collections::BTreeSet::new();
    let looked = stats
        .lookups
        .iter()
        .filter(|l| l.outcome != Outcome::Reused);
    looked.filter(|l| named.insert(l.node.as_str())).collect()
}

fn rendered(graph: &Graph, store: &Tier<Memory>) -> Render {
    now(render_over(
        graph,
        "master",
        RenderConfig::seconds(RATE, SECONDS),
        store,
    ))
    .expect("a render")
}

fn stats(render: &Render) -> &CacheStats {
    render
        .cache_stats
        .as_ref()
        .expect("a render through a store reports on it")
}

fn two_voices(name: &str, y: u32) -> Graph {
    graph_of(
        name,
        &[
            (
                "x",
                "lowpass(sample(sin(2*pi*220*t)), cutoff=2000, q=0.7)*0.5\n",
            ),
            (
                "y",
                &format!("lowpass(sample(sin(2*pi*{y}*t)), cutoff=2000, q=0.7)*0.5\n"),
            ),
            ("master", "@x*0.5 + @y*0.25\n"),
        ],
    )
}

fn outcomes(stats: &CacheStats, node: &str) -> Vec<Outcome> {
    let found: Vec<Outcome> = stats
        .lookups
        .iter()
        .filter(|l| l.node == node)
        .map(|l| l.outcome)
        .collect();
    assert!(!found.is_empty(), "`{node}` was looked up: {stats:?}");
    found
}

#[test]
fn a_warm_store_answers_every_value_and_writes_nothing_until_persist() {
    let memory = Memory::default();
    let graph = two_voices("warm", 330);
    let store = opened(&memory, u64::MAX);
    let cold = rendered(&graph, &store);
    assert!(memory.entries().is_empty(), "a render writes nothing");
    assert!(now(store.persist()).expect("persisted").written > 0);
    assert!(!memory.entries().is_empty());

    let warm = rendered(&graph, &opened(&memory, u64::MAX));
    let stats = stats(&warm);
    assert_eq!(stats.computed(), 0, "every value is a hit: {stats:?}");
    assert!(stats.lookups.iter().any(|l| l.outcome == Outcome::Hit));
    assert_eq!(samples(&cold), samples(&warm), "a hit is the bits computed");
    let (cold, warm) = (
        cold.labels[&cold.root].clone(),
        warm.labels[&warm.root].clone(),
    );
    assert_eq!(cold, warm, "a hit carries its label");
}

#[test]
fn an_edit_recomputes_only_the_edited_node_and_its_readers() {
    let memory = Memory::default();
    let store = opened(&memory, u64::MAX);
    rendered(&two_voices("edit-before", 330), &store);
    now(store.persist()).expect("persisted");

    let edited = rendered(&two_voices("edit-after", 440), &opened(&memory, u64::MAX));
    let stats = stats(&edited);
    assert!(outcomes(stats, "x").iter().all(|o| *o == Outcome::Hit));
    for node in ["y", "master"] {
        assert!(outcomes(stats, node).iter().any(|o| *o != Outcome::Hit));
    }
}

/// `memory`'s index as format 1 wrote it, beside `n` of its entries.
fn left_by_format_1(memory: &Memory, n: usize) -> Vec<String> {
    let mut index = String::from("sva store format 1\n");
    let mut names = Vec::new();
    for i in 0..n {
        let name = format!("{:032x}", 0xbeef_0000 + i);
        memory.set(&name, vec![0; 64]);
        index += &format!("{name} 64\n");
        names.push(name);
    }
    memory.set(INDEX_NAME, index.into_bytes());
    names
}

/// The calls one open makes over what format 1 left of `n` entries.
fn opening_over_format_1(n: usize) -> usize {
    let memory = Memory::default();
    left_by_format_1(&memory, n);
    memory.calls.store(0, Ordering::Relaxed);
    opened(&memory, u64::MAX);
    memory.calls.load(Ordering::Relaxed)
}

/// Opening over a store another format wrote costs the same however much it holds: a fresh
/// index, and none of its files listed, read or removed.
#[test]
fn opening_over_another_formats_store_never_touches_its_files() {
    assert_eq!(opening_over_format_1(10), opening_over_format_1(800));
    let memory = Memory::default();
    let mut names = left_by_format_1(&memory, 800);
    memory.reads.lock().unwrap().clear();
    let store = opened(&memory, u64::MAX);
    let reads = memory.reads.lock().unwrap().clone();
    assert_eq!(reads.len(), 1, "{reads:?}");
    assert_eq!(reads[0].0, INDEX_NAME);
    assert_eq!(memory.lists.load(Ordering::Relaxed), 0);
    names.sort();
    assert_eq!(memory.entries(), names, "none removed");
    assert_eq!(memory.bytes(INDEX_NAME), Some(format_only()));
    assert_eq!(disk(&store).bytes(), 0);
}

/// The index as every build at this format and Fourier dual rules version writes it for an empty store, whatever its
/// engine version or sources: the bytes a store in a user's browser keeps across releases.
fn format_only() -> Vec<u8> {
    format!("sva store format {STORE_FORMAT} table {FOURIER_DUAL_RULES_VERSION}\n").into_bytes()
}

/// A store a build with other Fourier dual rules wrote opens empty: its values were computed under
/// duals this build no longer has, though their keys and bytes read alike.
#[test]
fn a_store_other_fourier_dual_rules_wrote_opens_empty() {
    let memory = Memory::default();
    let graph = two_voices("fourier-dual-rules", 330);
    let store = opened(&memory, u64::MAX);
    rendered(&graph, &store);
    now(store.persist()).expect("persisted");
    let index = String::from_utf8(memory.bytes(INDEX_NAME).expect("an index")).expect("text");
    let ours = format!("sva store format {STORE_FORMAT} table {FOURIER_DUAL_RULES_VERSION}\n");
    let theirs = format!(
        "sva store format {STORE_FORMAT} table {}\n",
        FOURIER_DUAL_RULES_VERSION - 1
    );
    let rest = index
        .strip_prefix(&ours)
        .expect("the index names its Fourier dual rules version");
    memory.set(INDEX_NAME, (theirs + rest).into_bytes());
    for name in memory.entries() {
        if name != INDEX_NAME {
            let entry = memory.bytes(&name).expect("an entry");
            memory.set(
                &name,
                under_fourier_dual_rules(entry, FOURIER_DUAL_RULES_VERSION - 1),
            );
        }
    }

    let reopened = opened(&memory, u64::MAX);
    assert_eq!(disk(&reopened).bytes(), 0);
    let warm = rendered(&graph, &reopened);
    assert!(stats(&warm).computed() > 0, "{:?}", stats(&warm));
}

/// An entry's bytes as a build with Fourier dual rules `version` writes them: its header's tag
/// names the version, and the header's checksum covers the tag.
fn under_fourier_dual_rules(mut entry: Vec<u8>, version: u64) -> Vec<u8> {
    let head = 8 + u64::from_le_bytes(entry[..8].try_into().expect("a length")) as usize;
    let tag = 8 + 4 + 4;
    assert_eq!(&entry[8..12], b"SVAh");
    entry[tag..tag + 8].copy_from_slice(&version.to_le_bytes());
    let sum = entry[8..head - 8]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        });
    entry[head - 8..head].copy_from_slice(&sum.to_le_bytes());
    entry
}

/// The format a digest of these values' stored bytes was pinned under, and the digest.
const PINNED: (u32, u64) = (53, 11238238686593349323);

/// A change to how a value is encoded, or to what the engine computes for any construct here,
/// fails this until `STORE_FORMAT` is bumped and the digest pinned again: rows, a filter, a
/// shifted read, a warped read, a loop, a solver under a min-and-max ramp, a pointwise form,
/// noise and a filter's ringing.
#[test]
fn what_a_store_writes_changes_only_with_its_format() {
    let damped = "chaigne_askenfelt(261.63, damper_r=0.1*crop(max(0, min(1, (t - 0.02s)/0.03s)), \
        0.02s, inf))\n";
    let graph = graph_of(
        "pinned",
        &[
            ("x", "sample(sin(2*pi*220*t))*0.5\n"),
            (
                "f",
                "lowpass(sample(sin(2*pi*220*t)) + sample(sin(2*pi*3000*t))*0.3, cutoff=800)\n",
            ),
            ("echo", "@x + 0.5*self[idx(t - 0.01s)]\n"),
            ("tone", "sin(2*pi*330*t)\n"),
            ("s", damped),
            (
                "chirp",
                "sample(sin(2*pi*(200 + 900*t)*t))*step(t - 0.01s)\n",
            ),
            ("hiss", "lowpass(noise(7, period=0.25), cutoff=1000)*0.1\n"),
            (
                "rung",
                "lowpass(sample(crop(sin(2*pi*220*t), 0s, 0.02s)), cutoff=300)\n",
            ),
            (
                "master",
                "@echo*0.5 + @f(t - 0.01s) + @s*0.25 + @chirp*0.2 + @hiss + @rung \
                 + 0.1*@tone(t + 0.001*sin(2*pi*5*t))\n",
            ),
        ],
    );
    let memory = Memory::default();
    let store = opened(&memory, u64::MAX);
    rendered(&graph, &store);
    now(store.persist()).expect("persisted");
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    for name in memory.entries() {
        let bytes = memory.bytes(&name).expect("an entry");
        for b in name.bytes().chain(bytes) {
            digest = (digest ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    assert_eq!(
        (STORE_FORMAT, digest),
        PINNED,
        "stored bytes changed: bump STORE_FORMAT and pin the new digest"
    );
}

#[test]
fn a_truncated_entry_is_computed_and_written_again() {
    let memory = Memory::default();
    let graph = two_voices("truncated", 330);
    let store = opened(&memory, u64::MAX);
    let cold = rendered(&graph, &store);
    now(store.persist()).expect("persisted");

    let root = nodes(stats(&cold))
        .into_iter()
        .find(|l| l.node == "master")
        .expect("the root was looked up")
        .key;
    let name = format!("{:016x}{:016x}", root.0, root.1);
    let whole = memory.bytes(&name).expect("an entry");
    memory.set(&name, whole[..whole.len() / 2].to_vec());
    let store = opened(&memory, u64::MAX);
    let warm = rendered(&graph, &store);
    assert!(
        stats(&warm).planned.iter().any(|n| n == "master"),
        "the root computed what the entry could not answer: {:?}",
        stats(&warm)
    );
    assert_eq!(samples(&cold), samples(&warm));
    now(store.persist()).expect("persisted");
    assert_eq!(memory.bytes(&name), Some(whole), "the entry is whole again");
}

fn tone(name: &str, hz: u32) -> Graph {
    graph_of(name, &[("master", &format!("sample(sin(2*pi*{hz}*t))\n"))])
}

/// Each tone's own keys, and the bytes its render persisted.
fn persisted_tone(memory: &Memory, max_bytes: u64, hz: u32) -> (Vec<Hash>, u64) {
    let store = opened(memory, max_bytes);
    let before = disk(&store).bytes();
    let render = rendered(&tone(&format!("tone-{hz}"), hz), &store);
    now(store.persist()).expect("persisted");
    assert!(
        disk(&store).bytes() <= max_bytes,
        "the store keeps its budget"
    );
    let keys = nodes(stats(&render)).iter().map(|l| l.key).collect();
    (keys, disk(&store).bytes().saturating_sub(before))
}

fn all_held(memory: &Memory, keys: &[Hash]) -> bool {
    let store = opened(memory, u64::MAX);
    keys.iter().all(|k| disk(&store).holds(*k))
}

/// A store's reads reach the index with its next commit, ahead of what that commit writes.
#[test]
fn past_its_budget_the_store_evicts_the_least_recently_used_first() {
    let (_, one) = persisted_tone(&Memory::default(), u64::MAX, 100);
    let budget = one * 5 / 2;
    let memory = Memory::default();
    let (a, _) = persisted_tone(&memory, budget, 100);
    let (b, _) = persisted_tone(&memory, budget, 200);
    let store = opened(&memory, budget);
    rendered(&tone("tone-100", 100), &store);
    let render = rendered(&tone("tone-300", 300), &store);
    now(store.persist()).expect("persisted");
    let c: Vec<Hash> = nodes(stats(&render)).iter().map(|l| l.key).collect();
    assert!(all_held(&memory, &a), "read since, so kept");
    assert!(!all_held(&memory, &b), "least recently used, so gone");
    assert!(all_held(&memory, &c), "just written, so kept");
}

fn bits(render: &Render) -> Vec<Vec<u64>> {
    let root = render.output(render.root).expect("the root's samples");
    root.planes
        .iter()
        .map(|plane| plane.iter().map(|v| v.to_bits()).collect())
        .collect()
}

/// `master` reads `e` through `p`; `s` and `q` are the siblings on the way down.
fn nested(name: &str, hz: u32) -> Graph {
    graph_of(
        name,
        &[
            (
                "e",
                &format!("lowpass(sample(sin(2*pi*{hz}*t)), cutoff=2000, q=0.7)*0.5\n"),
            ),
            (
                "q",
                "lowpass(sample(sin(2*pi*550*t)), cutoff=2000, q=0.7)*0.25\n",
            ),
            (
                "s",
                "lowpass(sample(sin(2*pi*330*t)), cutoff=2000, q=0.7)*0.5\n",
            ),
            ("p", "@e*0.5 + @q\n"),
            ("master", "@p + @s*0.5\n"),
        ],
    )
}

#[test]
fn a_cold_render_through_a_store_writes_the_bits_a_render_without_one_does() {
    let graph = graph_of(
        "cold-bits",
        &[
            (
                "x",
                "lowpass(sample(sin(2*pi*220*t)) + sample(sin(2*pi*3000*t))*0.3, cutoff=800)\n",
            ),
            ("y", "@x(t - 0.01s)*0.5\n"),
            ("z", "sample(sin(2*pi*440*t))*0.5\n"),
            ("master", "@x*0.5 + @y + @z*0.25\n"),
        ],
    );
    let config = RenderConfig::seconds(RATE, 1.2);
    let fresh = render(&graph, "master", config.clone(), &Tier::default()).expect("a render");
    let store = opened(&Memory::default(), u64::MAX);
    let cold = now(render_over(&graph, "master", config, &store)).expect("a render");
    assert!(
        bits(&fresh)[0].iter().any(|b| *b != 0),
        "silence tests nothing"
    );
    assert_eq!(bits(&cold), bits(&fresh));
}

/// A medium that refuses every write fails no render, its out dropped or not: the samples are a
/// render's over memory alone, and each one's stats say why nothing was written; once the medium
/// takes writes, a persist writes what memory still holds.
#[test]
fn a_store_refusing_every_write_fails_no_render() {
    let memory = Memory::default();
    let store = opened(&memory, u64::MAX);
    store.set_max_bytes(1);
    memory.refusing.store(true, Ordering::Relaxed);
    let graph = two_voices("refusing", 330);
    let through = rendered(&graph, &store);
    let config = RenderConfig::seconds(RATE, SECONDS);
    let fresh = render(&graph, "master", config.clone(), &Tier::default()).expect("a render");
    assert_eq!(samples(&through), samples(&fresh));
    assert!(stats(&through).unstaged.is_some(), "{:?}", stats(&through));
    let prepared = dropped(&graph, config, &store);
    assert!(
        stats(&prepared).unstaged.is_some(),
        "{:?}",
        stats(&prepared)
    );
    assert!(now(store.persist()).is_err());
    memory.refusing.store(false, Ordering::Relaxed);
    assert!(now(store.persist()).expect("persisted").written > 0);
}

/// A run stored from an instant off the grid's blocks is read back off the disk by a stream
/// in blocks of any size, crossing chunk edges anywhere: the bits a cold render writes, none
/// of it computed.
#[test]
fn a_stored_run_streams_off_the_disk_in_blocks_of_any_size() {
    let graph = graph_of(
        "stored-blocks",
        &[
            (
                "tone",
                "lowpass(sample(crop(sin(2*pi*220*t), 0.1234s, 1.9s)), cutoff=900)\n",
            ),
            ("warm", "@tone\n"),
        ],
    );
    let config = RenderConfig {
        range: Range {
            start: Some(0),
            end: Some(16_000),
        },
        ..RenderConfig::at(RATE)
    };
    let memory = Memory::default();
    let store = opened(&memory, u64::MAX);
    now(render_over(&graph, "warm", config.clone(), &store)).expect("a render");
    now(store.persist()).expect("persisted");
    let cold = render(&graph, "warm", config.clone(), &Tier::default()).expect("a render");
    let cold = cold.output(cold.root).expect("the root");
    let cold: Vec<u64> = cold.plane(0).iter().map(|v| v.to_bits()).collect();
    assert!(cold.iter().any(|b| *b != 0), "silence tests nothing");
    for block in [1, 333, 4_095, 4_096, 4_097, 9_000] {
        let player = opened(&memory, u64::MAX);
        let stream = StreamConfig {
            block,
            channels: None,
            render: config.clone(),
        };
        let at = sva_ast::parse_expr("@warm").expect("an expression");
        let mut stream = now(Stream::open(&graph, &at, stream, &player)).expect("a stream");
        let mut heard = Vec::new();
        while let Some(held) = stream.read(stream.position(), block).expect("a block") {
            heard.extend(held.plane(0).iter().map(|v| v.to_bits()));
        }
        assert!(heard == cold, "in blocks of {block}");
        assert!(stream.stats().tier.disk_reads > 0, "in blocks of {block}");
        assert_eq!(stream.evaluated("tone"), Vec::new(), "in blocks of {block}");
    }
}

/// A string held until `release`, never released here.
const HELD: &str = "release = inf\nchaigne_askenfelt(f0, damper_r=0.1*crop(min(1, \
    (t - release)/0.03s), release, inf))\n";

fn term(text: &str) -> sva_ast::Expr {
    sva_ast::parse_expr(text).expect("an expression")
}

fn store_hits(stream: &RefCell<Stream>) -> Vec<String> {
    let hits = stream.borrow().stats().lookups.into_iter();
    hits.filter(|l| matches!(l.outcome, Outcome::Hit | Outcome::Extended))
        .map(|l| l.node)
        .collect()
}

/// A composition whose `string` a worker warmed at each of `hz` over the whole stream and
/// persisted, and each warm's new entries.
fn warmed_notes(name: &str, hz: &[&str]) -> (Graph, Memory, Vec<BTreeSet<String>>) {
    let mut nodes = vec![("string".to_string(), HELD.to_string())];
    nodes.extend(
        hz.iter()
            .map(|f| (format!("warm{f}"), format!("@string(t, f0={f})\n"))),
    );
    let nodes: Vec<(&str, &str)> = nodes
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let graph = graph_of(name, &nodes);
    let memory = Memory::default();
    let mut entries = Vec::new();
    for f in hz {
        let store = opened(&memory, u64::MAX);
        let config = RenderConfig {
            range: Range {
                start: Some(0),
                end: Some(8_000),
            },
            ..RenderConfig::at(RATE)
        };
        let before: BTreeSet<String> = memory.entries().into_iter().collect();
        now(render_over(&graph, &format!("warm{f}"), config, &store)).expect("a render");
        now(store.persist()).expect("persisted");
        let after = memory.entries().into_iter();
        entries.push(after.filter(|e| !before.contains(e)).collect());
    }
    (graph, memory, entries)
}

/// `Memory` whose every read answers only on its `delay`th poll, waking itself between.
#[derive(Clone)]
struct Slow {
    memory: Memory,
    delay: usize,
}

async fn later(polls: usize) {
    let mut left = polls;
    std::future::poll_fn(|cx| match left {
        0 => Poll::Ready(()),
        _ => {
            left -= 1;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await
}

impl Backend for Slow {
    type Lock = Held;

    async fn lock(&self) -> Result<Held, String> {
        self.memory.lock().await
    }

    async fn get(&self, name: &str) -> Result<Option<Vec<u8>>, String> {
        later(self.delay).await;
        self.memory.get(name).await
    }

    async fn get_range(&self, name: &str, from: u64, len: u64) -> Result<Option<Vec<u8>>, String> {
        later(self.delay).await;
        self.memory.get_range(name, from, len).await
    }

    async fn put(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        self.memory.put(name, bytes).await
    }

    async fn delete(&self, name: &str) -> Result<bool, String> {
        self.memory.delete(name).await
    }

    async fn list(&self) -> Result<Vec<(String, u64)>, String> {
        self.memory.list().await
    }

    async fn staging(&self) -> Result<Slow, String> {
        let memory = self.memory.staging().await?;
        Ok(Slow { memory, ..*self })
    }

    async fn rename(&self, name: &str, to: &Slow) -> Result<bool, String> {
        self.memory.rename(name, &to.memory).await
    }
}

/// Polled until it is done: every wait here wakes itself.
fn settled<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    loop {
        let polled = future
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop()));
        if let Poll::Ready(out) = polled {
            return out;
        }
    }
}

type Pending<'s> = std::pin::Pin<Box<dyn Future<Output = Result<Changed, EngineError>> + 's>>;

/// Edit `k`: at an onset, a note, and the handle it replaces, if any.
type Spec = (i64, &'static str, Option<Handle>);

fn changed_by(graph: &Graph, (at, f0, replaced): Spec) -> Change {
    match replaced {
        Some(handle) => {
            let fade = term(&format!("0.5*@string(t - {at}sp, f0={f0})"));
            Change::Replace(handle, graph.clone(), fade, Placed::Written)
        }
        None => Change::Add(
            graph.clone(),
            term(&format!("@string(t - {at}sp, f0={f0})")),
            Placed::Written,
        ),
    }
}

fn notes_over<B: Backend>(graph: &Graph, store: &Tier<B>, live: bool) -> RefCell<Stream> {
    let config = StreamConfig {
        block: 256,
        channels: None,
        render: RenderConfig {
            range: Range {
                start: Some(0),
                end: Some(8_000),
            },
            ..RenderConfig::at(RATE)
        },
    };
    let stream = settled(Stream::open(graph, &term("@notes"), config, store));
    let mut stream = stream.expect("a stream");
    if live {
        stream.go_live();
    }
    RefCell::new(stream)
}

/// A live stream edited once a block, adds of stored notes and replaces of sounding ones, over
/// a store whose every read takes several polls: every block comes when asked, each edit lands
/// while the stream plays on, and what it plays is, bit for bit, an exact stream taking each
/// edit where it landed.
#[test]
fn a_live_stream_plays_on_while_its_edits_await_a_slow_store() {
    let hz = ["261.63", "293.66", "329.63", "349.23", "392"];
    let (graph, memory, _) = warmed_notes("slow", &hz);
    let slow = Slow {
        memory: memory.clone(),
        delay: 3,
    };
    let player = settled(Store::open(slow, u64::MAX)).expect("the store opens");
    let player = Tier::over(player, u64::MAX);
    let shared = notes_over(&graph, &player, true);
    let (mut specs, mut landed, mut handles) = (Vec::<Spec>::new(), Vec::new(), Vec::new());
    let (mut pending, mut heard, mut waited) = (Vec::<(usize, Pending)>::new(), Vec::new(), 0);
    let mut added = BTreeMap::new();
    let mut k = 0;
    while k < 20 || !pending.is_empty() {
        if k < 20 {
            let at = shared.borrow().position() + 64;
            let replaced = handles.pop().filter(|_| k % 2 == 1);
            specs.push((at, hz[k % hz.len()], replaced));
            let (spec, graph) = (specs[k], &graph);
            let build = move |_: &Stream| Ok::<_, EngineError>(changed_by(graph, spec));
            pending.push((k, Box::pin(change(&shared, build, &player))));
        }
        let mut left = Vec::new();
        for (edit, mut future) in pending.drain(..) {
            match future
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
            {
                Poll::Ready(done) => {
                    if let Changed::Added(handle) = done.expect("an edit") {
                        handles.push(handle);
                        added.insert(handle, edit);
                    }
                    landed.push((heard.len(), edit));
                }
                Poll::Pending => left.push((edit, future)),
            }
        }
        waited += left.len();
        pending = left;
        let block = next(&mut shared.borrow_mut()).expect("a block");
        heard.push(block.expect("a block when asked").plane(0).to_vec());
        k += 1;
    }
    assert!(waited > 0, "some edit awaited the store across a block");

    let at_once = opened(&memory, u64::MAX);
    let reference = notes_over(&graph, &at_once, false);
    let (mut expected, mut ours) = (Vec::new(), BTreeMap::new());
    for n in 0..heard.len() {
        for (_, edit) in landed.iter().filter(|(at, _)| *at == n) {
            let (at, f0, replaced) = specs[*edit];
            let replaced = replaced.map(|live| ours[&added[&live]]);
            let build = |_: &Stream| Ok::<_, EngineError>(changed_by(&graph, (at, f0, replaced)));
            if let Changed::Added(handle) =
                now(change(&reference, build, &at_once)).expect("an edit")
            {
                ours.insert(*edit, handle);
            }
        }
        let block = next(&mut reference.borrow_mut()).expect("a block");
        let block = block.expect("a block");
        expected.push(block.plane(0).to_vec());
    }
    assert!(
        heard.iter().flatten().any(|v| *v != 0.0),
        "silence tests nothing"
    );
    assert!(
        heard == expected,
        "the slow store's live stream plays what the exact one does"
    );
    assert!(!store_hits(&shared).is_empty());
}

/// `graph`'s master rendered with its out dropped.
fn dropped(graph: &Graph, config: RenderConfig, store: &Tier<Memory>) -> Render {
    let config = RenderConfig {
        out: Out::Dropped,
        ..config
    };
    now(render_over(graph, "master", config, store)).expect("a render")
}

fn prepared(graph: &Graph, store: &Tier<Memory>) -> Render {
    dropped(graph, RenderConfig::seconds(RATE, SECONDS), store)
}

/// A target rendered with its out dropped hands back no samples; persisted, it renders from the
/// store alone, bit for bit the fresh render.
#[test]
fn a_target_prepared_and_persisted_renders_as_a_store_hit() {
    let memory = Memory::default();
    let graph = nested("prepared", 220);
    let store = opened(&memory, u64::MAX);
    let prepared = prepared(&graph, &store);
    assert!(stats(&prepared).computed() > 0, "{:?}", stats(&prepared));
    assert!(prepared.output(prepared.root).is_err(), "no samples");
    assert!(prepared.buffers.is_empty());
    assert!(memory.entries().is_empty(), "a render persists nothing");
    now(store.persist()).expect("persisted");

    let hit = rendered(&graph, &opened(&memory, u64::MAX));
    assert_eq!(stats(&hit).lookups.len(), 1, "{:?}", stats(&hit));
    assert_eq!(stats(&hit).lookups[0].outcome, Outcome::Hit);
    let fresh = render(
        &graph,
        "master",
        RenderConfig::seconds(RATE, SECONDS),
        &Tier::default(),
    );
    assert_eq!(bits(&hit), bits(&fresh.expect("a render")));
}

/// A render with its out dropped holds as much of its root however long it runs, where one
/// handing back its samples holds them all.
#[test]
fn a_render_with_its_out_dropped_holds_its_root_a_block_at_a_time() {
    let master = "lowpass(sample(0.1*saw(110*t)), cutoff=900, q=0.7)\n";
    let graph = graph_of("out-dropped", &[("master", master)]);
    let held = |secs: f64, out: Out| {
        let config = RenderConfig {
            out,
            threads: NonZeroUsize::new(2).expect("two"),
            ..RenderConfig::seconds(RATE, secs)
        };
        let done = render(&graph, "master", config, &Tier::default()).expect("a render");
        done.held_bytes
    };
    assert_eq!(held(4.0, Out::Dropped), held(8.0, Out::Dropped));
    assert!(held(8.0, Out::Kept) > held(4.0, Out::Kept));
}

/// The fade `master` reads falls under the silence threshold, so an open render of it ends there: one
/// answered by the store ends where the one that computed it did, and says so.
#[test]
fn a_hit_ends_an_open_render_where_the_render_that_stored_it_did() {
    let graph = graph_of(
        "cut-carried",
        &[
            ("fade", "sin(2*pi*200*t)*exp(-t/0.02)\n"),
            ("master", "sample(@fade)\n"),
        ],
    );
    let memory = Memory::default();
    let store = opened(&memory, u64::MAX);
    let config = RenderConfig::at(RATE);
    let cold = now(render_over(&graph, "master", config.clone(), &store)).expect("a render");
    now(store.persist()).expect("persisted");
    let cut = cold.labels[&cold.root]
        .cutting_below_silence_threshold
        .clone()
        .expect("a silence threshold");
    assert!(!cut.treated_as_silent_from_sample.is_empty(), "{cut:?}");

    let warm = now(render_over(
        &graph,
        "master",
        config,
        &opened(&memory, u64::MAX),
    ));
    let warm = warm.expect("a render");
    assert_eq!(stats(&warm).computed(), 0, "{:?}", stats(&warm));
    assert_eq!(warm.labels[&warm.root], cold.labels[&cold.root]);
    assert_eq!(bits(&warm), bits(&cold));
}

/// A filtered voice `x` under a master, `late` beside a second one.
fn voices(late: bool) -> Graph {
    let voice =
        |f: u32| format!("lowpass(sample(crop(sin(2*pi*{f}*t), 0s, 4s)), cutoff=2000, q=0.7)\n");
    let (x, y) = (voice(220), voice(330));
    match late {
        false => graph_of("voices", &[("x", &x), ("master", "@x*0.5\n")]),
        true => graph_of(
            "voices-late",
            &[("x", &x), ("y", &y), ("master", "@x*0.5 + @y*0.25\n")],
        ),
    }
}

/// From second 1, so `y`'s filter has a history to run before the first block.
fn from_second_one() -> RenderConfig {
    let mut config = RenderConfig::at(RATE);
    config.range = Range {
        start: Some(i64::from(RATE)),
        end: Some(4 * i64::from(RATE)),
    };
    config.threads = NonZeroUsize::MIN;
    config
}

/// Memory lets go of all it holds once the first block is heard.
struct LetsGo<'t, B: Backend>(&'t Tier<B>);

impl<B: Backend> Hear for LetsGo<'_, B> {
    fn heard(&self, _: &Buffer) -> bool {
        self.0.set_max_bytes(0);
        true
    }
}

fn read_to_its_end<B: Backend>(tier: &Tier<B>) {
    let rendered = |late: bool, config: RenderConfig, hear: Option<&dyn Hear>| {
        let (mut session, graph) = (Session::default(), voices(late));
        let render = render_in(&mut session, &graph, "master", config, tier, (&Never, hear));
        now(render).expect("a render")
    };
    let whole = RenderConfig {
        threads: NonZeroUsize::MIN,
        ..RenderConfig::seconds(RATE, 4.0)
    };
    rendered(false, whole, None);
    let over = rendered(true, from_second_one(), Some(&LetsGo(tier)));
    let stats = over.cache_stats.as_ref().expect("a render over memory");
    let x = stats.lookups.iter().find(|l| l.node == "x");
    assert_eq!(x.map(|l| l.outcome), Some(Outcome::Hit), "{stats:?}");
    assert!(tier.evictions() > 0, "memory let go of what it held");
    let alone =
        render(&voices(true), "master", from_second_one(), &Tier::default()).expect("a render");
    assert_eq!(samples(&over)["master"], samples(&alone)["master"]);
}

/// A render reads a node memory answered to its range's end, though it runs a history first
/// and memory lets the node go before its later blocks: over memory alone and over a disk.
#[test]
fn a_node_memory_answered_is_read_to_the_ranges_end_though_memory_lets_it_go_meanwhile() {
    read_to_its_end(&Tier::default());
    read_to_its_end(&opened(&Memory::default(), u64::MAX));
}
