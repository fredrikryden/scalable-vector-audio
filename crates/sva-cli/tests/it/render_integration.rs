// Concern: exercises sva-cli's probe/answer/write end to end over fixtures | Non-concern: unit-level parsing or collapse math | IO: (fixtures/*) -> asserted Rendered, JSON or CliError

mod tempo;
mod written;

use crate::helpers::{entry, fixture, json_of, ledgered, master, plane, put, scratch, secs};
use sva_cli::{CliError, error_envelope};
use sva_core::{PROBE, probe};
use sva_engine::{Output, Representation, Source};

/// `expr` over its first second, read through a node of its own.
fn second(expr: &str) -> sva_core::Rendered {
    let dir = scratch("second");
    put(&dir, "x", &format!("{expr}\n"));
    probe(&dir, "@x([0, 1s])").unwrap_or_else(|e| panic!("`{expr}`: {e}"))
}

const LOUDNESS: Representation = Representation::Loudness {
    window_secs: sva_engine::DEFAULT_LOUDNESS_WINDOW_SECS,
};

#[test]
fn basic_fixture_renders_with_bpm_meter_and_a_repeat_desugared() {
    let rendered = master(&fixture("basic")).expect("basic fixture should render");
    assert_eq!(secs(&rendered), 1.0, "the interval's own end");
    assert!(!plane(&rendered, PROBE).is_empty());

    let root = entry(&rendered, PROBE);
    assert_eq!(root.share, Some(1.0), "a node's share of itself is 1.0");
    assert_eq!(root.depth, 0);
    assert_eq!(root.clipped, Some(false));
    assert!(root.peak > 0.0 && root.peak < 1.0);
}

/// An open interval ends where the root's support does: an arrangement's desugared `concat`
/// crops its last section, so it ends where that section does.
#[test]
fn an_arranged_root_ends_where_its_last_section_does() {
    let rendered = probe(&fixture("arranged"), "@master").expect("arranged fixture should render");
    assert_eq!(secs(&rendered), 6.0, "1 bar + 2 bars at 120bpm 4/4");
    let samples = plane(&rendered, PROBE);
    assert_eq!(samples.len(), 6 * 44100);
    assert!(
        samples[5 * 44100..].iter().any(|s| *s != 0.0),
        "the last section sounds"
    );
}

#[test]
fn a_ledger_envelope_names_its_target_window_and_every_node_under_it() {
    let rendered = master(&fixture("basic")).unwrap();
    let json = json_of(&rendered, "ledger(depth=8)");

    assert!(json.contains("\"status\": \"success\""));
    assert!(json.contains("\"target\": \"@master([0, 1s])\""));
    assert!(json.contains("\"sample_rate\": 44100"));
    assert!(json.contains("\"start_secs\": 0"));
    assert!(json.contains("\"end_secs\": 1"));
    assert!(json.contains("\"written\": { \"items\": [], \"pagination\""));
    assert!(json.contains("\"node\": \"probe\""));
    for field in [
        "\"unit\"",
        "\"control\"",
        "\"rms\"",
        "\"peak\"",
        "\"share\"",
    ] {
        assert!(json.contains(field), "missing {field}");
    }
    assert!(json.contains("\"meta\"") && json.contains("\"timestamp\""));
    assert!(!json.contains("\"spectrum\""), "one query, one answer");
}

/// A reading off a closed form is exact and rate-free, one off a buffer is measured at
/// the rate it ran at, and both name the profile they were taken under.
#[test]
fn every_answer_names_its_source_profile_and_rate() {
    let rendered = second("sin(2*pi*440*t)");

    let law = rendered.answer(PROBE, Representation::Lines).unwrap();
    assert_eq!(law.source, Source::Exact);
    assert_eq!(law.profile, "psychoacoustic-v1");
    assert_eq!(law.rate, None, "a law answers without a grid");
    let Output::Lines(lines) = &law.value else {
        panic!("a pair answers lines")
    };
    assert!(lines.iter().any(|l| (l.hz - 440.0).abs() < 1e-9));

    let measured = rendered.answer(PROBE, LOUDNESS).unwrap();
    assert_eq!(measured.source, Source::Measured);
    assert_eq!(measured.rate, Some(44_100));

    let json = json_of(&rendered, "lines");
    assert!(json.contains("\"source\": \"exact\""));
    assert!(json.contains("\"profile\": \"psychoacoustic-v1\""));
    assert!(json.contains("\"rate\": null"));
}

/// The point of probing: an effect is measured against the real graph without touching disk.
#[test]
fn probe_evaluates_an_argv_expression_in_the_compositions_namespace() {
    let dir = fixture("basic");
    let before = master(&dir).unwrap();
    let probed = ledgered(&dir, "crop(@drums*0.5, 0s, 1s)").expect("probe should render");

    assert_eq!(secs(&probed), secs(&before));
    let dry = entry(&before, PROBE).rms;
    let wet = entry(&probed, PROBE).rms;
    assert!(
        wet < dry,
        "half the level of the graph it read: {dry} -> {wet}"
    );
    assert!(wet > 0.0, "and not all of it");
    assert!(
        !dir.join(PROBE).exists(),
        "a probe writes nothing to the composition"
    );
}

#[test]
fn a_probe_naming_a_missing_node_or_a_broken_expression_refuses() {
    let dir = fixture("basic");
    let Err(missing) = probe(&dir, "@nope*2") else {
        panic!("no such node");
    };
    assert_eq!(missing.code(), "not_found");
    assert!(matches!(
        probe(&dir, "lp(@drums, cutoff="),
        Err(CliError::BadProbe(_))
    ));
}

/// The `cli` standard's envelope names three keys under `error`, and a missing `details`
/// leaves an agent branching on the code with nowhere structured to read the particulars.
#[test]
fn an_error_envelope_carries_code_message_and_details() {
    let dir = scratch("envelope-details");
    put(&dir, "master", "@nope\n");

    let err = master(&dir).err().expect("a dangling ref refuses");
    let json = error_envelope(err.code(), &err.message(), &err.diagnostics());
    let (error, data) = json
        .split_once("\"data\"")
        .expect("an error envelope holds both objects");

    for key in ["\"code\"", "\"message\"", "\"details\""] {
        assert!(error.contains(key), "`error` names {key}: {json}");
    }
    assert!(error.contains(err.code()), "the code branched on: {json}");
    for found in err.diagnostics() {
        let code = sva_core::json::escape(&found.code);
        assert!(error.contains(&code), "`details` names `{code}`: {json}");
        assert!(data.contains(&code), "and `data` locates it: {json}");
    }
    assert!(data.contains("\"location\""), "located: {json}");
}

/// One meaning of "missing" per tool, per the `cli` standard: every key an object declares is
/// written and answers `null` where there is no value, rather than being left out.
#[test]
fn every_absent_reading_is_null_never_a_missing_key() {
    let dir = fixture("basic");
    let rendered = master(&dir).expect("basic renders");
    let json = json_of(&rendered, "ledger(depth=8)");
    assert!(
        json.contains("\"channel\": null"),
        "a mono row still names its channel: {json}"
    );
    assert!(
        json.contains("\"tail_db\": null"),
        "an answer that dropped nothing still names its tail: {json}"
    );

    let traced = sva_cli::trace_data(&sva_cli::trace(&dir, "drums").expect("drums traces"));
    for absent in ["\"discrete\": null", "\"file\": null", "\"loop\": null"] {
        assert!(
            traced.contains(absent),
            "a trace writes {absent} rather than dropping the key: {traced}"
        );
    }
}

/// The cache log goes to stderr alone: stdout stays the envelope it is without it.
#[test]
fn a_debug_log_writes_to_stderr_and_leaves_stdout_the_envelope() {
    let dir = scratch("cache-log");
    put(&dir, "x", "crop(sin(2*pi*220*t), 0s, 0.05s)\n");
    let run = |log: Option<&str>| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_sva-cli"));
        command
            .current_dir(&dir)
            .args([
                "render",
                "@x",
                "--representation",
                "loudness",
                "--cache",
                "none",
            ])
            .env_remove("SVA_LOG");
        if let Some(level) = log {
            command.env("SVA_LOG", level);
        }
        command.output().expect("the binary runs")
    };
    let data = |out: &std::process::Output| {
        let printed = String::from_utf8_lossy(&out.stdout).into_owned();
        printed[..printed.find("\"meta\"").expect("an envelope")].to_string()
    };
    let (plain, logged) = (run(None), run(Some("debug")));
    assert!(!logged.stderr.is_empty(), "a debug log writes to stderr");
    assert!(data(&plain).contains("\"status\": \"success\""));
    assert_eq!(data(&plain), data(&logged));
}
