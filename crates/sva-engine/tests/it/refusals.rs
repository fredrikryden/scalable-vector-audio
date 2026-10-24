// Concern: proves each crossing FORMAT 16 names refuses under its own code and offers the cast to write | Non-concern: what types when nothing crosses (typing.rs) | IO: (a composition) -> a refusal

use crate::fixtures::graph_of;
use sva_engine::{EngineError, RenderConfig, Tier, render, types};

fn refusal(name: &str, files: &[(&str, &str)], root: &str) -> EngineError {
    types(&graph_of(name, files), root).expect_err("a refusal")
}

fn parts(e: &EngineError) -> (String, String, String) {
    match e {
        EngineError::Refused(d) => (d.code.clone(), d.message.clone(), d.help.clone()),
        other => panic!("expected a written refusal, got {other:?}"),
    }
}

#[test]
fn mixing_t_and_f_names_fourier() {
    let e = refusal(
        "t-and-f",
        &[
            ("carrier", "tanh(sin(2*pi*100*t)*3)\n"),
            ("mask", "1/(1 + pow(f/300, 8))\n"),
            ("master", "@carrier*@mask\n"),
        ],
        "master",
    );
    let (code, _, help) = parts(&e);
    assert_eq!(code, "type.domain_mismatch");
    assert!(
        help.contains("fourier"),
        "the repair names both casts: {help}"
    );
    assert!(help.contains("ifourier"), "{help}");
}

#[test]
fn fourier_on_a_nonlinearity_names_the_subterm_and_the_cast() {
    let e = refusal(
        "fourier-on-a-nonlinearity",
        &[
            ("drive", "sin(2*pi*55*t)\n"),
            ("bell", "fourier(tanh(@drive*4))\n"),
        ],
        "bell",
    );
    let (code, message, help) = parts(&e);
    assert_eq!(code, "cast.left_algebra");
    assert!(
        message.contains("left A"),
        "the blocking subterm is named: {message}"
    );
    assert!(
        help.contains("sample("),
        "the cast to write instead: {help}"
    );
}

/// The registry is what `sva-cli builtins` prints as the whole refusal vocabulary, so a code
/// written in any crate a refusal crosses and left out of it is a hole in that listing.
#[test]
fn every_code_the_engine_can_raise_is_in_the_registry() {
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut written: Vec<String> = Vec::new();
    for name in ["sva-engine", "sva-samples", "sva-formula"] {
        for path in sources(&crates.join(name).join("src")) {
            let text = std::fs::read_to_string(&path).expect("a source file");
            written.extend(codes_in(&text));
        }
    }
    written.sort();
    written.dedup();
    assert!(written.len() > 20, "the scan found {} codes", written.len());
    for code in &written {
        assert!(
            sva_engine::REGISTRY.iter().any(|(name, _)| name == code),
            "`{code}` is written in a crate a refusal crosses, and the registry never names it"
        );
    }
    let mut named: Vec<&str> = sva_engine::REGISTRY.iter().map(|(name, _)| *name).collect();
    let before = named.len();
    named.sort_unstable();
    named.dedup();
    assert_eq!(before, named.len(), "the registry names a code twice");
}

fn sources(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    for held in std::fs::read_dir(dir)
        .expect("a source directory")
        .flatten()
    {
        let path = held.path();
        match path.is_dir() {
            true => out.extend(sources(&path)),
            false if path.extension().is_some_and(|e| e == "rs") => out.push(path),
            false => {}
        }
    }
    out
}

/// A quoted `<judgment>.<name>`, which is the one shape every code in this crate is written in.
fn codes_in(text: &str) -> Vec<String> {
    const PREFIXES: [&str; 8] = [
        "type.",
        "cast.",
        "collapse.",
        "samples.",
        "grammar.",
        "engine.",
        "read.",
        "ref.",
    ];
    let mut out = Vec::new();
    for quoted in text.split('"').skip(1).step_by(2) {
        let shaped = PREFIXES.iter().any(|p| quoted.starts_with(p))
            && quoted
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c == '.')
            && quoted.matches('.').count() == 1;
        if shaped {
            out.push(quoted.to_string());
        }
    }
    out
}

/// A shoulder of no length is a hard edge, so `fall=0s` is a legal window and must render as
/// one. A shoulder outside the window it belongs to is not, and
/// names itself rather than aborting the process on a non-finite amplitude.
#[test]
fn a_zero_fall_shoulder_is_a_hard_edge_not_a_panic() {
    let rendered = |body: &str| {
        let g = graph_of("crop-edges", &[("node", body)]);
        render(
            &g,
            "node",
            RenderConfig::seconds(8_000, 4.0),
            &Tier::default(),
        )
    };
    let rms = |body: &str| {
        let held = rendered(body).unwrap_or_else(|e| panic!("{body}: {e}"));
        let plane = held
            .output(held.id("node").expect("the root"))
            .expect("a buffer")
            .plane(0)
            .to_vec();
        (plane.iter().map(|s| s * s).sum::<f64>() / plane.len() as f64).sqrt()
    };

    let cropped = rms("crop(sin(2*pi*440*t), 0s, 4s)\n");
    assert!(cropped > 0.0);
    assert!(
        (rms("crop(sin(2*pi*440*t), 0s, 4s, rise=0s, fall=0s)\n") - cropped).abs() < 1e-12,
        "two hard edges are the crop itself"
    );
    let one_sided = rms("crop(sin(2*pi*440*t), 0s, 4s, rise=1s, fall=0s)\n");
    assert!(
        one_sided > 0.0 && one_sided < cropped,
        "a rise with no fall takes energy off the front alone: {one_sided}"
    );

    for body in [
        "crop(sin(2*pi*440*t), 0s, 4s, rise=-1s, fall=1s)\n",
        "crop(sin(2*pi*440*t), 0s, 4s, rise=3s, fall=3s)\n",
    ] {
        let Err(refused) = rendered(body) else {
            panic!("{body}: a shoulder outside its own window should refuse");
        };
        assert_eq!(refused.code(), "engine.bad_crop_shoulder", "{body}");
    }
}

/// A solver's grid is sized from its arguments, so an out-of-range value must not reach one.
#[test]
fn a_finite_difference_argument_outside_its_range_refuses_before_a_grid_is_built() {
    let e = refusal(
        "physics-out-of-range",
        &[("node", "botteldooren(0)\n")],
        "node",
    );
    assert_eq!(e.code(), "engine.physics_out_of_range");
    assert!(e.to_string().contains("botteldooren"), "{e}");
}

/// A node count only the rate decides, so the grid checks it and a render carries it out.
#[test]
fn a_room_too_large_for_one_grid_refuses_with_the_ceiling_it_passed() {
    let graph = graph_of("room-too-large", &[("node", "botteldooren(40)\n")]);
    let config = sva_engine::RenderConfig::seconds(16_000, 0.01);
    let Err(refused) = sva_engine::render(&graph, "node", config, &Tier::default()) else {
        panic!("a 40 Hz room is millions of nodes");
    };
    let (code, message, _) = parts(&refused);
    assert_eq!(code, "samples.grid_too_large");
    assert!(message.contains("botteldooren"), "{message}");
    assert!(
        sva_engine::REGISTRY.iter().any(|(name, _)| *name == code),
        "`{code}` is in the vocabulary `builtins` prints"
    );
}
