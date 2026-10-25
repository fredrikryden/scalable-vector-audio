// Concern: parses argv into the subcommands this CLI answers and their options | Non-concern: what a target names (sva-core) | IO: (argv) -> Command or CliError

use std::path::PathBuf;

use sva_core::{Asked, CliError};

mod reading;
mod solve;

use reading::{analyze_args, render_args};
use solve::solve_args;

#[derive(Debug, PartialEq)]
pub struct RenderArgs {
    /// One expression; its refs name nodes from the current directory or an absolute path.
    pub target: String,
    pub until: Option<String>,
    pub rate: Option<u32>,
    /// The precision every sample is written to; the profile's own where `None`.
    pub bits: Option<i32>,
    /// The most threads it computes on; every core where `None`.
    pub threads: Option<std::num::NonZeroUsize>,
    pub asked: Vec<Asked>,
    /// The caller said a destination that already holds a file may be replaced.
    pub confirm: bool,
    pub cache: CacheAt,
    /// Reserved variables read as these texts instead of their files'.
    pub sets: Vec<(String, String)>,
}

/// Where a render's persistent store lives.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CacheAt {
    #[default]
    Platform,
    Path(PathBuf),
    Off,
}

#[derive(Debug, PartialEq)]
pub struct AnalyzeArgs {
    pub path: PathBuf,
    pub asked: Vec<Asked>,
    /// The caller said a destination that already holds a file may be replaced.
    pub confirm: bool,
}

#[derive(Debug, PartialEq)]
pub enum Command {
    Version,
    Help,
    Render(Box<RenderArgs>),
    Analyze(Box<AnalyzeArgs>),
    /// Its argv checked; the solver is being rebuilt.
    Solve,
    /// One expression, as render's target; `None` lints every file under its own rules.
    Lint {
        target: Option<String>,
        format: Format,
        body_chars: usize,
    },
    Trace {
        target: String,
    },
    Builtins,
    /// An expression's own parse tree; it reads no composition.
    Outline {
        text: String,
    },
    New {
        name: String,
        /// Present where the caller says this is a retry, per the `cli` standard's own
        /// requirement that a create verb be made idempotent.
        idempotency_key: Option<String>,
        /// A recording to start the composition from, which this build refuses.
        from: Option<PathBuf>,
    },
}

/// How the diagnostics an evaluator answers are rendered. One set of objects, two renderings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Format {
    #[default]
    Json,
    Text,
}

const STANDARD_VERBS: [(&str, &str); 4] = [
    ("validate", "lint"),
    ("list", "builtins"),
    ("create", "new"),
    ("show", "trace"),
];

fn canonical(subcommand: &str) -> &str {
    STANDARD_VERBS
        .iter()
        .find(|(alias, _)| *alias == subcommand)
        .map_or(subcommand, |(_, name)| *name)
}

/// A free function rather than a closure per parser: a closure would hold its loop's
/// iterator borrowed.
pub(crate) fn value<'a>(
    it: &mut impl Iterator<Item = &'a String>,
    name: &str,
) -> Result<String, CliError> {
    it.next()
        .cloned()
        .ok_or_else(|| CliError::usage("cli.missing_value", format!("{name} needs a value")))
}

/// `--version`/`-V` and `--help` are global flags, recognized anywhere in argv (cli standard).
pub fn parse_args(argv: &[String]) -> Result<Command, CliError> {
    if argv.iter().any(|a| a == "--version" || a == "-V") {
        return Ok(Command::Version);
    }
    if argv.iter().any(|a| a == "--help") {
        return Ok(Command::Help);
    }

    let mut it = argv.iter();
    let subcommand = canonical(
        it.next()
            .ok_or_else(|| CliError::usage("cli.missing_subcommand", "no subcommand given"))?,
    );
    let rest = it.as_slice();
    match subcommand {
        "render" => render_args(rest),
        "analyze" => analyze_args(rest),
        "solve" => solve_args(rest),
        "lint" => lint_args(rest),
        "trace" => trace_args(rest),
        "builtins" => builtins_args(rest),
        "outline" => outline_args(rest),
        "new" => new_args(rest),
        other => Err(CliError::usage(
            "cli.unknown_subcommand",
            format!("unknown subcommand `{other}`"),
        )),
    }
}

fn builtins_args(rest: &[String]) -> Result<Command, CliError> {
    match rest.first() {
        None => Ok(Command::Builtins),
        Some(extra) => Err(CliError::usage(
            "cli.unknown_argument",
            format!("builtins takes no arguments, not `{extra}`"),
        )),
    }
}

fn outline_args(rest: &[String]) -> Result<Command, CliError> {
    match rest {
        [text] => Ok(Command::Outline { text: text.clone() }),
        _ => Err(CliError::usage(
            "cli.missing_argument",
            "outline takes one <expression>, quoted as one argument",
        )),
    }
}

fn new_args(rest: &[String]) -> Result<Command, CliError> {
    let mut it = rest.iter();
    let name = match it.next() {
        Some(name) if !name.starts_with("--") => name.clone(),
        Some(_) | None => return Err(CliError::usage("cli.missing_argument", "missing <name>")),
    };
    let (mut idempotency_key, mut from) = (None, None);
    let mut sized = None;
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--idempotency-key" => idempotency_key = Some(value(&mut it, "--idempotency-key")?),
            "--from" => from = Some(PathBuf::from(value(&mut it, "--from")?)),
            "--harmonics" | "--lone" => {
                let least = usize::from(flag == "--harmonics");
                value(&mut it, flag)?
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n >= least)
                    .ok_or_else(|| {
                        CliError::usage(
                            "cli.bad_value",
                            format!("`{flag}` takes a whole number from {least}"),
                        )
                    })?;
                sized = Some(flag.clone());
            }
            _ => {
                return Err(CliError::usage(
                    "cli.unknown_argument",
                    "new takes one <name>, an optional `--from <recording>` with `--harmonics <n>` \
                     and `--lone <n>`, and an optional `--idempotency-key <key>`",
                ));
            }
        }
    }
    if let (Some(flag), None) = (sized, &from) {
        return Err(CliError::usage(
            "cli.needs_from",
            format!("`{flag}` sizes the starter `--from <recording>` writes"),
        ));
    }
    Ok(Command::New {
        name,
        idempotency_key,
        from,
    })
}

fn lint_args(rest: &[String]) -> Result<Command, CliError> {
    let mut it = rest.iter().peekable();
    let target = match it.peek() {
        Some(a) if !a.starts_with("--") => it.next().cloned(),
        _ => None,
    };
    let mut format = Format::default();
    let mut body_chars = crate::lint::BODY_CHARS;
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--body-chars" => {
                let text = it.next().map(String::as_str).unwrap_or("nothing");
                body_chars = text.parse().map_err(|_| {
                    CliError::usage(
                        "cli.bad_value",
                        format!("`--body-chars` takes a whole number of characters, not `{text}`"),
                    )
                })?;
            }
            "--format" => {
                format = match it.next().map(String::as_str) {
                    Some("json") => Format::Json,
                    Some("text") => Format::Text,
                    other => {
                        return Err(CliError::usage(
                            "cli.bad_value",
                            format!(
                                "`--format` takes json or text, not `{}`",
                                other.unwrap_or("nothing")
                            ),
                        ));
                    }
                };
            }
            extra => {
                return Err(CliError::usage(
                    "cli.unknown_argument",
                    format!(
                        "lint takes only ['<expression>'], `--format <json|text>` and \
                         `--body-chars <n>`, not `{extra}`"
                    ),
                ));
            }
        }
    }
    Ok(Command::Lint {
        target,
        format,
        body_chars,
    })
}

/// One positional and nothing else: a trace answers structure, which no option narrows.
fn trace_args(rest: &[String]) -> Result<Command, CliError> {
    match rest {
        [target] if !target.starts_with("--") => Ok(Command::Trace {
            target: target.clone(),
        }),
        [] => Err(CliError::usage(
            "cli.missing_argument",
            "missing <node|expression>",
        )),
        _ => Err(CliError::usage(
            "cli.unknown_argument",
            "trace takes one <node|expression> and nothing else",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    fn rendered(parts: &[&str]) -> RenderArgs {
        match parse_args(&argv(parts)) {
            Ok(Command::Render(args)) => *args,
            other => panic!("expected a render, got {other:?}"),
        }
    }

    /// Each mistake answers a code a caller branches on, its message quoting the text it
    /// refuses, and no banner: `--help` is where the whole surface is stated.
    #[test]
    fn each_argument_mistake_answers_its_own_code_naming_the_text() {
        let render = |tail: &[&'static str]| [&["render", "@a"][..], tail].concat();
        let cases: [(Vec<&str>, &str, &str); 14] = [
            (
                render(&["--representation", "spectre"]),
                "query.unknown_representation",
                "`spectre`",
            ),
            (
                render(&["--representation", "lines(depth=2)"]),
                "query.unknown_argument",
                "`depth`",
            ),
            (
                render(&["--representation", "ledger(depth=many)"]),
                "query.bad_value",
                "`many`",
            ),
            (
                render(&["--representation", "lines,lines"]),
                "query.duplicate",
                "`lines`",
            ),
            (
                render(&["--representation", "bindings"]),
                "query.missing_argument",
                "`bindings`",
            ),
            (
                render(&["--representation", "spectrum(peaks=8"]),
                "query.malformed",
                "`spectrum(peaks=8`",
            ),
            (
                render(&["--representation", "exact-envelope"]),
                "query.retired",
                "`exact-envelope`",
            ),
            (
                render(&["--representation", "a=x,b=x"]),
                "query.unknown_representation",
                "`a`",
            ),
            (
                render(&["--representation", "lines=x.json,atoms=./x.json"]),
                "query.shared_destination",
                "x.json",
            ),
            (
                render(&["--representation", "lines", "--bogus"]),
                "cli.unknown_argument",
                "`--bogus`",
            ),
            (
                render(&["--representation"]),
                "cli.missing_value",
                "--representation",
            ),
            (
                render(&["--representation", "lines", "--rate", "0"]),
                "cli.bad_value",
                "`0`",
            ),
            (vec!["bench"], "cli.unknown_subcommand", "`bench`"),
            (
                vec!["analyze", "/tmp/a.wav", "--representation", "ledger"],
                "query.not_analyzable",
                "`ledger`",
            ),
        ];
        for (parts, code, text) in cases {
            let Err(CliError::Usage {
                code: held,
                message,
            }) = parse_args(&argv(&parts))
            else {
                panic!("{parts:?} refuses as an argument mistake");
            };
            assert_eq!(held, code, "{parts:?}: {message}");
            assert!(message.contains(text), "{parts:?}: {message}");
            assert!(!message.contains("usage:"), "{parts:?}: {message}");
        }
    }

    #[test]
    fn version_and_help_are_recognized_anywhere_in_argv() {
        assert_eq!(parse_args(&argv(&["--version"])).unwrap(), Command::Version);
        assert_eq!(parse_args(&argv(&["-V"])).unwrap(), Command::Version);
        assert_eq!(
            parse_args(&argv(&["lint", "--help"])).unwrap(),
            Command::Help
        );
    }

    #[test]
    fn each_standard_verb_alias_reaches_the_subcommand_it_names() {
        assert_eq!(
            parse_args(&argv(&["validate"])).unwrap(),
            Command::Lint {
                target: None,
                format: Format::Json,
                body_chars: crate::lint::BODY_CHARS,
            }
        );
        assert_eq!(parse_args(&argv(&["list"])).unwrap(), Command::Builtins);
        assert_eq!(
            parse_args(&argv(&["show", "kick"])).unwrap(),
            Command::Trace {
                target: "kick".to_string(),
            }
        );
        assert_eq!(
            parse_args(&argv(&["create", "song1"])).unwrap(),
            Command::New {
                name: "song1".to_string(),
                idempotency_key: None,
                from: None,
            }
        );
    }

    /// A render names its target, its readings as calls in one comma list or several, each
    /// reading's options as its own arguments, and its flags; nothing else.
    #[test]
    fn a_render_takes_a_target_readings_with_their_arguments_and_flags() {
        let args = rendered(&[
            "render",
            "@piano([0, 2b], f0=C4)",
            "--representation",
            "samples=/tmp/out.wav, spectrum(peaks=8, frame=50ms)",
            "--representation",
            "ledger(depth=2, brief=1)=/tmp/ledger.json,bindings(node=@voice)",
            "--rate",
            "48000",
            "--bits",
            "16",
            "--until",
            "t > 1s",
            "--threads",
            "3",
        ]);
        assert_eq!(args.target, "@piano([0, 2b], f0=C4)");
        assert_eq!(args.until.as_deref(), Some("t > 1s"));
        assert_eq!(args.rate, Some(48_000));
        assert_eq!(args.bits, Some(16));
        assert_eq!(args.threads.map(std::num::NonZeroUsize::get), Some(3));
        let names: Vec<&str> = args.asked.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["samples", "spectrum", "ledger", "bindings"]);
        assert_eq!(args.asked[0].path(), Some(Path::new("/tmp/out.wav")));
        assert_eq!(
            args.asked[1]
                .representation(sva_engine::DEFAULT_SAMPLE_RATE, None)
                .unwrap(),
            sva_engine::Representation::Spectrum {
                max_peaks: 8,
                frame_secs: Some(0.05)
            }
        );
        assert_eq!(
            args.asked[2]
                .representation(sva_engine::DEFAULT_SAMPLE_RATE, None)
                .unwrap(),
            sva_engine::Representation::Ledger { depth: 2 }
        );
        assert!(args.asked[2].brief);
        assert_eq!(args.asked[2].path(), Some(Path::new("/tmp/ledger.json")));
        assert_eq!(args.asked[3].node.as_deref(), Some("voice"));
    }

    /// `alsa:` after `=` names a live device; its buffer is the reading's own argument, at
    /// the table's default until written, and the other flags and readings stand beside it.
    #[test]
    fn a_device_destination_takes_samples_and_a_buffer_beside_every_other_flag() {
        use sva_core::Destination;
        let row = sva_engine::Representation::Samples.reading();
        assert_eq!(row.takes, [sva_engine::Arg::Buffer]);
        assert_eq!(sva_engine::Arg::Buffer.key(), "buffer");
        let args = rendered(&["render", "@a", "--representation", "samples=alsa:default"]);
        assert_eq!(
            args.asked[0].dest,
            Some(Destination::Alsa {
                device: "default".to_string(),
                buffer_secs: sva_engine::DEFAULT_BUFFER_SECS,
            })
        );
        assert_eq!(args.asked[0].path(), None);
        let args = rendered(&[
            "render",
            "@a",
            "--representation",
            "samples(buffer=20ms)=alsa:hw:1,0,loudness",
            "--until",
            "t > 1s",
            "--rate",
            "44100",
        ]);
        let names: Vec<&str> = args.asked.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["samples", "loudness"]);
        assert_eq!(
            args.asked[0].dest,
            Some(Destination::Alsa {
                device: "hw:1,0".to_string(),
                buffer_secs: 0.02,
            }),
            "a comma inside a device name stays in it where no reading follows"
        );
        assert_eq!(args.asked[1].dest, None);
        assert_eq!(args.until.as_deref(), Some("t > 1s"));
        assert_eq!(args.rate, Some(44_100));
        let args = rendered(&[
            "render",
            "@a",
            "--representation",
            "lines,samples=alsa:plughw:CARD=PCH,DEV=0",
        ]);
        assert_eq!(
            args.asked[1].dest,
            Some(Destination::Alsa {
                device: "plughw:CARD=PCH,DEV=0".to_string(),
                buffer_secs: sva_engine::DEFAULT_BUFFER_SECS,
            })
        );
    }

    #[test]
    fn a_device_refuses_what_it_cannot_play_naming_the_text() {
        let render = |r: &'static str| vec!["render", "@a", "--representation", r];
        let cases: [(Vec<&str>, &str, &str); 9] = [
            (
                render("samples=alsa:hw:0,samples=alsa:hw:1"),
                "query.duplicate",
                "`samples`",
            ),
            (
                render("samples=alsa:"),
                "destination.no_device",
                "`samples=alsa:`",
            ),
            (
                render("loudness=alsa:default"),
                "destination.not_audio",
                "`loudness`",
            ),
            (
                render("samples(buffer=50ms)"),
                "query.buffer_without_device",
                "`samples`",
            ),
            (
                render("samples(buffer=50ms)=/tmp/a.wav"),
                "query.buffer_without_device",
                "`buffer`",
            ),
            (
                render("samples(buffer=0)=alsa:default"),
                "query.bad_value",
                "`0`",
            ),
            (
                render("samples(buffer=5000s)=alsa:default"),
                "query.bad_value",
                "`5000s`",
            ),
            (
                render("lines(buffer=1)"),
                "query.unknown_argument",
                "`buffer`",
            ),
            (
                vec![
                    "analyze",
                    "/tmp/a.wav",
                    "--representation",
                    "samples=alsa:hw:1",
                ],
                "destination.needs_render",
                "`alsa:hw:1`",
            ),
        ];
        for (parts, code, text) in cases {
            let Err(CliError::Usage {
                code: held,
                message,
            }) = parse_args(&argv(&parts))
            else {
                panic!("{parts:?} refuses as an argument mistake");
            };
            assert_eq!(held, code, "{parts:?}: {message}");
            assert!(message.contains(text), "{parts:?}: {message}");
        }
    }
}
