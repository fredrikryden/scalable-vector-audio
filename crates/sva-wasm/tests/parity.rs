// Concern: pins the sample bits a filter fixture renders to, the same digests native and in wasm | Non-concern: the JS surface (boundary.rs), why any value is right | IO: (a composition) -> digests

//! `sva-cli` and the page render one composition to the same bits: `cargo test` pins these
//! digests native, `wasm-pack test --node crates/sva-wasm` the same ones in wasm.

use std::num::NonZeroUsize;

use sva_core::{Job, Output, Representation, asked, call, execute};
use sva_engine::Tier;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

const NODES: [(&str, &str); 8] = [
    ("src", "sample(0.2*noise(1, period=1s))\n"),
    (
        "narrow",
        "bandpass(@src(t - 0.2361s), 777.4297, 29.15361)\n",
    ),
    ("high", "bandpass(@src(t - 0.6869s), 14871.31, 557.674)\n"),
    (
        "low",
        "lowpass(lowpass(@src(t - 0.923s), 10.6047, 0.7071), 10.6047, 0.7071)\n",
    ),
    (
        "wide",
        "highpass(@src, 50, 5) + peaking(@src, 1500hz, 0.9, 6db)\n",
    ),
    (
        "sweep",
        "lowpass(@src, cutoff=1000 + 800*sin(2*pi*2*t), q=2)\n",
    ),
    (
        "tone",
        "tanh(3*sin(2*pi*110*t))*exp(-3*t) + pow(2, sin(2*pi*t))*cos(2*pi*C4*t)\n",
    ),
    (
        "master",
        "@narrow + @high + @low + @wide + @sweep + 0.1*@tone\n",
    ),
];

/// `<node> <FNV-1a of its f64 sample bits> <of its f32 ones>`.
const PINNED: &str = "
src 0dd11846683765de 495eea335ff9d34b
narrow f20a08d8b1366486 b2ab02a84621d739
high 0118a0a4c72b636d 6499c72be971da8d
low 89264ead9dee7bc7 2c1275450e06bdab
wide c7a9bc56b9bdb57a f009ce5e9f017713
sweep 94eda0be726cd095 65f1c8dbcfdada21
tone b39c6b676180771f 3ecd2048cbd265c2
master 38393608f384a255 ba2ab554083ddd6b
";

fn fnv(words: impl Iterator<Item = u64>) -> u64 {
    words.fold(0xcbf2_9ce4_8422_2325, |held, w| {
        w.to_le_bytes().iter().fold(held, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
    })
}

fn table(threads: usize) -> String {
    let held: sva_ast::Composition = NODES.into_iter().collect();
    let samples = [asked(&call("samples").expect("a call")).expect("a reading")];
    let mut lines = String::new();
    for (node, _) in NODES {
        let target = format!("@{node}([0, 0.1s])");
        let job = Job {
            rate: Some(48_000),
            asked: &samples,
            threads: NonZeroUsize::new(threads),
            ..Job::over(&held, &target)
        };
        let rendered = execute(job, &Tier::default()).unwrap_or_else(|e| panic!("{target}: {e}"));
        let answer = rendered
            .answer(&rendered.target, Representation::Samples)
            .unwrap_or_else(|e| panic!("{target}: {e}"));
        let Output::Samples(buffer) = answer.value else {
            panic!("{target}: samples answer samples");
        };
        let plane = buffer.plane(0);
        let narrow = sva_core::encode::float32(node, plane).expect("a 32-bit float holds it");
        let wide = fnv(plane.iter().map(|v| v.to_bits()));
        let f32s = fnv(narrow.iter().map(|v| u64::from(v.to_bits())));
        lines.push_str(&format!("{node} {wide:016x} {f32s:016x}\n"));
    }
    lines
}

fn holds(threads: usize) {
    let table = table(threads);
    assert_eq!(
        table.trim(),
        PINNED.trim(),
        "on {threads} thread(s), this target rendered:\n{table}"
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn filtered_noise_renders_the_pinned_bits_on_one_thread() {
    holds(1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn filtered_noise_renders_the_pinned_bits_on_four_threads() {
    holds(4);
}
