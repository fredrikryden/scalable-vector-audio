// Concern: scaffolds a starter composition and the order to read it in | Non-concern: rendering it, or the JSON shape (output.rs) | IO: (dir, name) -> a written tree or CliError

use std::path::{Path, PathBuf};

use sva_core::CliError;

/// `(path under the new directory, contents)`, written in this order so a directory always
/// exists before a file lands in it. Every body stays a closed form until its own layer's
/// written `sample()`, and each equation's comment says why it is written that way rather
/// than the way that costs orders of magnitude more; a score's annotation line is its comment.
pub const FILES: [(&str, &str); 12] = [
    (
        "variables/bpm",
        r"; Models: the pulse every `b` literal is measured against | Neglects: the meter, which variables/meter states | IO: () -> beats per minute | Tags: tempo
120
",
    ),
    (
        "variables/meter",
        r"; Models: how many beats a bar holds, so `1b` resolves to seconds | Neglects: the pulse itself, which variables/bpm states | IO: () -> beats per bar | Tags: meter
4/4
",
    ),
    (
        "variables/key",
        r"; Models: the tonic every score degree and `st` offset is measured from, so transposing is editing this note | Neglects: the scale a degree counts in, which variables/scale names | IO: () -> hertz | Tags: key
C3
",
    ),
    (
        "variables/scale",
        r"; Models: the scale every `deg` counts in, one the language names, so changing mode is editing this name | Neglects: the tonic, which variables/key holds; any scale `sva-cli builtins` does not list is a file of cents named by a ref | IO: () -> a scale | Tags: scale
major
",
    ),
    (
        "voice/tone",
        r"; Models: one voice's timbre, three harmonics rolled off by a lowpass at the third, held for len | Neglects: the envelope, which lib/env owns; every parameter carries a default and the crop ends it, so a bare `@voice/tone` plays | IO: (t, f0, vel, len) -> amplitude | Tags: voice, harmonics
f0 = @../variables/key
vel = 0.2
len = 0.5s
crop(lowpass(vel*(sin(2*pi*f0*t) + 0.5*sin(2*pi*2*f0*t) + 0.3333*sin(2*pi*3*f0*t)), 3*f0), 0s, len)
",
    ),
    (
        "lib/env",
        r"; Models: one exponential decay under a shouldered crop, the shape every note wears | Neglects: the timbre it multiplies, which voice/tone owns; written as one window and never as a `min`/`max` pair, which would leave no dual and no exact reading | IO: (t, attack, decay, fade, len) -> a gain | Tags: envelope
attack = 0.005s
decay = 0.25s
fade = 0.05s
len = 0.5s
crop(exp(-t/decay), 0s, len, rise=attack, fall=fade)
",
    ),
    (
        "voice/note",
        r"; Models: one played note, a timbre under an envelope | Neglects: which pitch and when, which the scores under score/ state | IO: (t, f0, vel, len) -> amplitude | Tags: voice, note
f0 = @../variables/key
vel = 0.2
len = 0.5s
@../voice/tone(t, f0=f0, vel=vel, len=len) * @../lib/env(t, len=len)
",
    ),
    (
        "score/chord.jsonl",
        r#"{"annotation": "Models: the tonic triad, one event per voice of a `deg` list on one `at` | Neglects: rhythm, which score/phrase has; every voice is the same tone | IO: (t) -> amplitude | Tags: score, chord"}
{"at": "0b", "node": "@../voice/tone", "deg": [1, 3, 5]}
"#,
    ),
    (
        "score/phrase.jsonl",
        r#"{"annotation": "Models: two bars of eighth notes down the scale and back, each a note with its len | Neglects: dynamics; every note's `vel` is voice/note's default | IO: (t) -> amplitude | Tags: score, melody"}
{"at": "0b", "node": "@../voice/note", "deg": 8, "len": "0.25b"}
{"at": "0.25b", "node": "@../voice/note", "deg": 5, "len": "0.25b"}
{"at": "0.5b", "node": "@../voice/note", "deg": 3, "len": "0.25b"}
{"at": "0.75b", "node": "@../voice/note", "deg": 5, "len": "0.25b"}
{"at": "1b", "node": "@../voice/note", "deg": 6, "len": "0.25b"}
{"at": "1.25b", "node": "@../voice/note", "deg": 5, "len": "0.25b"}
{"at": "1.5b", "node": "@../voice/note", "deg": 3, "len": "0.25b"}
{"at": "1.75b", "node": "@../voice/note", "deg": 1, "len": "0.25b"}
"#,
    ),
    (
        "perc/hat",
        r"; Models: a closed hat -- one noise band under a shouldered crop, its 0.5 s period commensurate with this 2 s bar, its gain small because a flat noise holds half a unit of power a hertz, whatever its period, an rms of 100 under a 20 kHz ceiling | Neglects: pitch, which no hat has; a `noise(...)*exp(...)` tail would cost orders of magnitude more | IO: (t, gain) -> amplitude | Tags: percussion, noise
gain = 0.0008
crop(gain*noise(1, period=0.5, color=1db), 0s, 0.4s, rise=0.002s, fall=0.2s)
",
    ),
    (
        "fx/glue",
        r"; Models: the pitched layers glued by one short feedback -- the pitched chain's one crossing into samples, and the last thing that chain does, because `self` may read only what `sample` has already written | Neglects: the hats, which master sums in beside this under a `sample` of their own; a loop rings on forever, so it is cropped to the two bars it plays | IO: (t) -> amplitude | Tags: fx, feedback
crop(sample(0.5*@../score/chord(t) + @../score/phrase(t)) + 0.3*self[idx(t) - 1], 0s, 2b)
",
    ),
    (
        "master",
        r"; Models: two bars of the whole piece, each layer sampled on its own | Neglects: nothing it does not name; one closed form over the noise and the scores together costs far more than this, and a mono master broadcasts to any width, so no `join` of a value with itself is written | IO: (t) -> amplitude | Tags: master, arrangement
crop(0.6*(@fx/glue(t) + sample(@perc/hat(t)) + sample(@perc/hat(t - 1b))), 0s, 2b)
",
    ),
];

/// `(command, why)`, in the order a stranger should run them: what the composition IS before
/// what it sounds like. Answered beside the
/// tree `new` wrote, so the first thing an agent reads is the next thing it should do.
pub const NEXT: [(&str, &str); 9] = [
    (
        "sva-cli lint",
        "structure, comments, scores, key, whether each node plays bare; refuses before any audio",
    ),
    (
        "sva-cli builtins",
        "the whole vocabulary, and the scales variables/scale may name; nothing outside it parses",
    ),
    (
        "sva-cli trace master",
        "what master reads, and why it is samples",
    ),
    (
        "sva-cli trace score/phrase",
        "a score: one line per note, its sound shifted to `at`, each `deg` counted in variables/scale from variables/key",
    ),
    (
        "sva-cli render 'sample(@voice/tone)' --representation spectrum(peaks=3)",
        "the three harmonics the lowpass leaves, measured off its samples",
    ),
    (
        "sva-cli render 'sample(@score/chord)' --representation spectrum(peaks=9) --set scale=minor",
        "the triad in minor without editing a file; `--set key=D3` transposes it",
    ),
    (
        "sva-cli render '@perc/hat' --representation atoms",
        "a noise is a line series, one atom per line, 2 Hz apart at period=0.5",
    ),
    (
        "sva-cli render '@master' --representation ledger(skim=1)",
        "per-node rms, peak, clipped",
    ),
    (
        "sva-cli render '@master' --representation samples=/tmp/song.wav --rate 48000",
        "the audio itself, at whatever rate you name",
    ),
];

#[derive(Debug)]
pub struct Scaffolded {
    pub root: PathBuf,
    pub files: Vec<String>,
    /// `(command, why)` to run next, in order.
    pub next: Vec<(String, String)>,
    /// What the starter's limits left out of the recording, each naming its flag.
    pub dropped: Vec<String>,
}

/// What a scaffold writes, `(path, bytes)` in order, and what to run next.
struct Tree {
    files: Vec<(String, Vec<u8>)>,
    next: Vec<(String, String)>,
    dropped: Vec<String>,
}

fn starter() -> Tree {
    Tree {
        files: FILES
            .iter()
            .map(|(p, c)| (p.to_string(), c.as_bytes().to_vec()))
            .collect(),
        next: NEXT
            .iter()
            .map(|(c, w)| (c.to_string(), w.to_string()))
            .collect(),
        dropped: Vec::new(),
    }
}

/// A priced fit's starter of the recording at `from`, the recording beside it.
fn seeded(from: &Path, limits: sva_core::seed::Limits) -> Result<Tree, CliError> {
    let (planes, rate) = crate::recording::read_recording(from)?;
    let clip = from
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| {
            CliError::usage(
                "new.bad_from",
                format!("`{}` names no file", from.display()),
            )
        })?
        .to_string();
    let seeded = sva_core::seed::seed(&planes, f64::from(rate), &clip, limits);
    let bytes = std::fs::read(from)
        .map_err(|e| CliError::Io(format!("could not read {}: {e}", from.display())))?;
    let mut files: Vec<(String, Vec<u8>)> = seeded
        .files
        .into_iter()
        .map(|(p, t)| (p, t.into_bytes()))
        .collect();
    files.push((clip, bytes));
    let varies: Vec<String> = seeded
        .varies
        .iter()
        .map(|v| format!("--vary '{v}'"))
        .collect();
    let (continuation, why) = match seeded.dropped.is_empty() {
        true => (
            format!(
                "path:{}",
                sva_core::continuation::path::DEFAULT_PRICES
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            "shrinks the starter to the stage of least description length its price path \
             holds, each price per number a stage, and writes it in place",
        ),
        false => (
            "grow".to_string(),
            "fits the starter, then grows back what its limits dropped a level at a time, \
             each growth kept where it lowers the description length, and writes the stage of \
             least length in place",
        ),
    };
    let solve = format!(
        "sva-cli solve @{} --method prox --continuation {continuation} {} --write --confirm",
        sva_core::seed::COST,
        varies.join(" ")
    );
    Ok(Tree {
        files,
        next: vec![(solve, why.to_string())],
        dropped: seeded.dropped,
    })
}

/// Refuses if `<dir>/<name>` already exists, so `new` never overwrites a caller's own work.
/// An `idempotency_key` is recorded beside the composition and answers the same `Scaffolded`
/// again when the same key is retried over a tree still holding exactly this scaffold, so a
/// network retry or an agent restart converges rather than erroring.
pub fn scaffold(
    dir: &Path,
    name: &str,
    idempotency_key: Option<&str>,
    from: Option<(&Path, sva_core::seed::Limits)>,
) -> Result<Scaffolded, CliError> {
    if name.is_empty() || name.contains('/') || name.contains(std::path::MAIN_SEPARATOR) {
        return Err(CliError::usage(
            "new.bad_name",
            format!("`{name}` must be a single directory name, not a path"),
        ));
    }
    let tree = match from {
        Some((from, limits)) => seeded(from, limits)?,
        None => starter(),
    };
    let root = dir.join(name);
    if root.exists() {
        let key = idempotency_key.ok_or_else(|| CliError::Conflict {
            by: "new",
            message: format!(
                "{} already exists; `new` never overwrites a composition. State \
                 `--idempotency-key <key>` to have a retry of your own create succeed instead",
                root.display()
            ),
        })?;
        let taken = || CliError::Conflict {
            by: "new",
            message: format!(
                "{} already exists and is not what `--idempotency-key {key}` scaffolded; \
                 `new` never overwrites a composition",
                root.display()
            ),
        };
        if std::fs::read_to_string(key_path(dir, name)).ok().as_deref() != Some(key) {
            return Err(taken());
        }
        if !holds(&root, &tree) {
            return Err(taken());
        }
        return Ok(Scaffolded {
            root,
            files: tree.files.iter().map(|(path, _)| path.clone()).collect(),
            next: tree.next,
            dropped: tree.dropped,
        });
    }
    // Written aside, renamed in, the key last: a create that stopped leaves nothing behind.
    let staging = dir.join(format!(".{name}.sva-new-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    let files = staged(&staging, &tree).inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&staging);
    })?;
    std::fs::rename(&staging, &root).map_err(|e| {
        let _ = std::fs::remove_dir_all(&staging);
        CliError::Io(format!("could not put {} in place: {e}", root.display()))
    })?;
    if let Some(key) = idempotency_key {
        std::fs::write(key_path(dir, name), key).map_err(|e| {
            let _ = std::fs::remove_dir_all(&root);
            let _ = std::fs::remove_file(key_path(dir, name));
            CliError::Io(format!(
                "could not record the idempotency key beside {}: {e}",
                root.display()
            ))
        })?;
    }
    Ok(Scaffolded {
        root,
        files,
        next: tree.next,
        dropped: tree.dropped,
    })
}

fn staged(staging: &Path, tree: &Tree) -> Result<Vec<String>, CliError> {
    let mut files = Vec::with_capacity(tree.files.len());
    for (path, contents) in &tree.files {
        let full = staging.join(path);
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::Io(format!("could not create {}: {e}", parent.display())))?;
        }
        std::fs::write(&full, contents)
            .map_err(|e| CliError::Io(format!("could not write {}: {e}", full.display())))?;
        files.push(path.to_string());
    }
    Ok(files)
}

/// Beside the composition, never inside it: every file under a composition root is a node.
fn key_path(dir: &Path, name: &str) -> std::path::PathBuf {
    dir.join(format!(".{name}.sva-idempotency-key"))
}

/// Byte-for-byte, so a retry only succeeds over a tree nothing has edited since.
fn holds(root: &Path, tree: &Tree) -> bool {
    tree.files
        .iter()
        .all(|(path, contents)| std::fs::read(root.join(path)).is_ok_and(|held| held == *contents))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sva-cli-new-{name}-{:x}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn scaffold_refuses_a_name_that_already_exists() {
        let dir = tmp("exists");
        std::fs::create_dir_all(dir.join("song1")).unwrap();
        let err = scaffold(&dir, "song1", None, None).unwrap_err();
        assert!(matches!(err, CliError::Conflict { .. }));
        assert_eq!(
            err.exit_code(),
            4,
            "a taken name is a conflict, not a fault"
        );
        assert!(err.message().contains("already exists"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `cli` standard's own requirement for a create verb: a retry succeeds identically
    /// rather than erroring, so an agent restart converges instead of dead-ending.
    #[test]
    fn a_retry_under_the_same_idempotency_key_answers_the_first_call_again() {
        let dir = tmp("idempotent");
        let first = scaffold(&dir, "song1", Some("k-1"), None).expect("a fresh scaffold");
        let again = scaffold(&dir, "song1", Some("k-1"), None).expect("a retry over the same tree");
        assert_eq!(first.root, again.root);
        assert_eq!(first.files, again.files);

        std::fs::write(first.root.join("master"), "0.0\n").unwrap();
        let other = scaffold(&dir, "song1", Some("k-2"), None).unwrap_err();
        assert!(
            matches!(other, CliError::Conflict { .. }),
            "someone else's key is not a retry of this create"
        );

        std::fs::write(first.root.join("master"), "0.0\n").unwrap();
        let edited = scaffold(&dir, "song1", Some("k-1"), None).unwrap_err();
        assert!(
            matches!(edited, CliError::Conflict { .. }),
            "an edited tree is not a retry"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A caller's very first `sva-cli lint` after `new` must not immediately refuse them —
    /// the scaffold is the canonical example of the doc-comment convention, not an exception.
    /// `missing-comment`/`multiline-comment`/`malformed-comment` are hard errors now, so a
    /// violation here would surface as `Err`, not as a `Finding` to filter out.
    #[test]
    fn the_scaffolded_composition_lints_clean_of_every_doc_comment_check() {
        let dir = tmp("lints-clean");
        let scaffolded = scaffold(&dir, "song1", None, None).unwrap();
        let report = crate::lint::lint(&scaffolded.root, None);
        assert!(
            report.is_ok(),
            "a freshly scaffolded composition must carry a well-formed doc comment on every \
             node: {}",
            report.err().map(|e| e.message()).unwrap_or_default()
        );
        assert!(
            report.is_ok_and(|held| held.findings.is_empty()),
            "and no warning either: the quickstart's very first command must come back silent"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `next` line an argv change broke would send a stranger's first move into a refusal.
    #[test]
    fn every_next_command_this_build_still_parses() {
        for (command, why) in NEXT {
            let argv: Vec<String> = command
                .split_whitespace()
                .skip(1)
                .map(str::to_string)
                .collect();
            assert!(
                crate::parse_args(&argv).is_ok(),
                "`new` answers `{command}`, which this build refuses"
            );
            assert!(!why.is_empty(), "`{command}` is answered with no reason");
        }
    }
}
