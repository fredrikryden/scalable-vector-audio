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
src c7558063189a4793 495eea335ff9d34b
narrow b4a2c3742a439c53 b2ab02a84621d739
high 016a982bca75dfaa 1e0204de279281e0
low 8a37493aa0bc518a 2c1275450e06bdab
wide 3f78c9949f00e4d1 375891442ebd5894
sweep bcb0ba7039101cbe 65f1c8dbcfdada21
tone 440e7c4848e5480a 3ecd2048cbd265c2
master 489036ff27546ff7 53d876dd19b06e0c
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

/// A priced fit of a clip rendered here, seeded and solved as `new --from` and `solve
/// --method prox` do: `<FNV-1a of each point's distance bits> <of the chosen files' bytes>`.
const PRICED: &str = "837ed46ba97e9c3c 4033d3b9d24c9c9f";

fn priced(threads: usize) -> String {
    let clip_nodes: sva_ast::Composition = [(
        "clip",
        "0.1*exp(-6*t)*(sin(2*pi*220*t) + 0.4*sin(2*pi*660*t))\n",
    )]
    .into_iter()
    .collect();
    let samples = [asked(&call("samples").expect("a call")).expect("a reading")];
    let job = Job {
        rate: Some(48_000),
        asked: &samples,
        threads: NonZeroUsize::new(threads),
        ..Job::over(&clip_nodes, "@clip([0, 0.25s])")
    };
    let rendered = execute(job, &Tier::default()).expect("the clip renders");
    let answer = rendered
        .answer(&rendered.target, Representation::Samples)
        .expect("samples");
    let Output::Samples(buffer) = answer.value else {
        panic!("samples answer samples");
    };
    let x: Vec<f64> = buffer.plane(0).to_vec();
    let channels = vec![x.iter().map(|v| *v as f32).collect::<Vec<f32>>()];
    let seeded = sva_core::seed::seed(&channels, 48_000.0, "clip.wav", sva_core::seed::LIMITS);
    let held: sva_ast::Composition = seeded
        .files
        .iter()
        .filter(|(p, _)| !p.starts_with("fits/"))
        .map(|(p, t)| (p.as_str(), t.as_str()))
        .collect();
    use sva_core::solve::{Op, Target, Tree};
    let internal = |target| Tree::Repr {
        call: "internal".to_string(),
        target,
    };
    let planes = channels.clone();
    let tree = Tree::Bin(
        Op::Add,
        Box::new(Tree::Reduce(
            "norm",
            vec![Tree::Bin(
                Op::Sub,
                Box::new(internal(Target::Text("@master".to_string()))),
                Box::new(internal(Target::Samples {
                    rate: 48_000,
                    planes,
                })),
            )],
        )),
        Box::new(Tree::Bin(
            Op::Mul,
            Box::new(Tree::Num(0.012)),
            Box::new(Tree::Repr {
                call: "numbers".to_string(),
                target: Target::Text("@master".to_string()),
            }),
        )),
    );
    let problem = sva_core::solve::Problem {
        source: &held,
        texts: &sva_core::solve::NoFiles,
        cost: sva_core::solve::built(tree).expect("the priced cost types"),
        vary: seeded
            .varies
            .iter()
            .map(|v| sva_core::solve::vary(v).expect("a vary"))
            .collect(),
        sets: Vec::new(),
        rate: Some(48_000),
        threads: NonZeroUsize::new(threads),
        stops: sva_core::solve::Stops {
            max_renders: 1,
            step_tol: 0.0,
            cost_tol: 0.0,
        },
    };
    let priced = sva_core::solve::prox::run::Priced {
        path: Some(vec![0.004, 0.04]),
        steps: 12,
    };
    let tier = Tier::default();
    let mut quiet = |_: &sva_core::solve::prox::Progress| {};
    let fit =
        sva_core::solve::prox::run::fitted(&problem, &priced, &tier, &mut quiet, &sva_core::Never);
    let mut fit = std::pin::pin!(fit);
    let waker = std::task::Waker::noop();
    let fitted = match fit
        .as_mut()
        .poll(&mut std::task::Context::from_waker(waker))
    {
        std::task::Poll::Ready(done) => done.expect("the priced solve answers"),
        std::task::Poll::Pending => unreachable!("a fit in memory alone never waits"),
    };
    let distances = fnv(fitted.points.iter().map(|p| p.predicted.to_bits()));
    let files = fnv(fitted.files.iter().flat_map(|(p, t)| {
        p.bytes()
            .chain(t.bytes())
            .map(u64::from)
            .collect::<Vec<_>>()
    }));
    format!("{distances:016x} {files:016x}")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn a_priced_fit_answers_the_pinned_bits_on_one_thread() {
    assert_eq!(priced(1), PRICED, "on one thread, this target fitted");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_priced_fit_answers_the_pinned_bits_on_four_threads() {
    assert_eq!(priced(4), PRICED, "on four threads, this target fitted");
}
