// Concern: what a tempo decides about an interval's length and bar-resolved timing | Non-concern: the readings a render answers (the parent suite) | IO: (a composition) -> a duration or a refusal

use crate::helpers::{plane, put, scratch, secs};
use sva_ast::Dir;
use sva_cli::CliError;
use sva_core::{Job, PROBE, execute, probe};
use sva_engine::Tier;

/// `bpm` reaches the samples only by resolving a score's bar times, so a tempo change must
/// re-render exactly the bar-timed nodes and leave the rest byte-identical.
#[test]
fn a_bpm_change_re_renders_bar_resolved_timing_and_nothing_else() {
    let dir = scratch("bpm-cache");
    put(&dir, "variables/meter", "4/4\n");
    put(
        &dir,
        "kick",
        "gain = 1\ngain*sin(2*pi*55*t)*exp(0 - 30*t)\n",
    );
    put(&dir, "tone", "sin(2*pi*440*t)*0.1\n");
    put(
        &dir,
        "pattern-1b.jsonl",
        "{\"at\": \"0b\", \"node\": \"@kick\"}\n\
         {\"at\": \"0.25b\", \"node\": \"@kick\", \"gain\": 0.5}\n\
         {\"at\": \"0.75b\", \"node\": \"@kick\", \"gain\": 0.7}\n",
    );
    put(&dir, "master", "crop(@pattern-1b(t), 0s, 3s) + @tone*0.2\n");

    let cache = Tier::default();
    let source = Dir::at(&dir);
    let at = |target: &'static str| {
        execute(
            Job {
                ..Job::over(&source, target)
            },
            &cache,
        )
        .expect("the target renders")
    };
    let job = || at("@master([0, 3s])");
    put(&dir, "variables/bpm", "120\n");
    let fast = job();
    let fast_tone = at("@tone([0, 3s])");
    let fast_pattern = at("@pattern-1b([0, 3s])");
    put(&dir, "variables/bpm", "100\n");
    let slow = job();
    let slow_tone = at("@tone([0, 3s])");
    let slow_pattern = at("@pattern-1b([0, 3s])");

    assert_eq!(secs(&fast), 3.0);
    assert_eq!(secs(&slow), 3.0, "same window, slower grid");
    assert_eq!(
        plane(&slow_tone, PROBE),
        plane(&fast_tone, PROBE),
        "a node no bar reaches is the same signal"
    );
    assert_ne!(
        plane(&slow_pattern, PROBE),
        plane(&fast_pattern, PROBE),
        "and the hits really did move"
    );
}

/// A `b` literal is grid units: without a tempo it refuses rather than defaulting.
#[test]
fn a_bar_literal_needs_a_tempo_and_reaches_a_probe_expression_too() {
    let dir = scratch("bar-literal");
    put(&dir, "master", "crop(@tone(t - 0.5b), 0s, 2b)\n");
    put(&dir, "tone", "sin(2*pi*A4*t)\n");
    assert!(
        bad_tempo(probe(&dir, "@master")),
        "a bar literal with no bpm/meter must refuse"
    );

    put(&dir, "variables/bpm", "120\n");
    put(&dir, "variables/meter", "4/4\n");
    assert_eq!(
        secs(&probe(&dir, "@master([0, 2b])").unwrap()),
        4.0,
        "2 bars at 120bpm 4/4"
    );

    let bars = probe(&dir, "crop(1, 0s, 1b)").unwrap();
    let seconds = probe(&dir, "crop(1, 0s, 2s)").unwrap();
    assert_eq!(secs(&bars), 2.0, "the proof ends each where its crop does");
    assert_eq!(
        plane(&bars, PROBE),
        plane(&seconds, PROBE),
        "1b IS 2s at this tempo"
    );
}

fn bad_tempo<T>(answered: Result<T, CliError>) -> bool {
    matches!(answered, Err(CliError::Refusals(refused))
        if refused.iter().all(|r| r.code == sva_ast::DiagCode::BadTempo))
}

/// `--set` replaces a reserved variable for one render, and the composition's files are
/// left as they were.
#[test]
fn a_render_sets_the_tempo_without_editing_a_file() {
    let dir = scratch("set-tempo");
    put(&dir, "master", "crop(sin(2*pi*220*t), 0s, 2b)\n");
    put(&dir, "variables/bpm", "120\n");
    put(&dir, "variables/meter", "4/4\n");
    let end = |sets: &[&str]| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"));
        command.current_dir(&dir).args([
            "render",
            "@master",
            "--representation",
            "loudness",
            "--cache",
            "none",
        ]);
        for set in sets {
            command.args(["--set", set]);
        }
        let out = command.output().expect("the binary runs");
        let printed = String::from_utf8_lossy(&out.stdout).into_owned();
        let at = printed.find("\"end_secs\": ").expect("an interval") + 12;
        let digits = printed[at..].split([' ', '}']).next().expect("a number");
        digits.parse::<f64>().expect("seconds")
    };
    assert_eq!(end(&[]), 4.0, "2 bars at 120 bpm");
    assert_eq!(end(&["bpm=60"]), 8.0, "2 bars at 60 bpm");
    assert_eq!(end(&["meter=3/4"]), 3.0, "2 bars of 3/4 at 120 bpm");
    assert_eq!(
        std::fs::read_to_string(dir.join("variables/bpm")).unwrap(),
        "120\n"
    );
}
