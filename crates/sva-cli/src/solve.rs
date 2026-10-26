// Concern: runs a solve on a recording: progress on stderr, each level's files and wav written as it ends, N0 kept | Non-concern: argv, the pyramid (sva-core) | IO: (SolveArgs) -> JSON or CliError

use std::cell::Cell;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use sva_core::solve::atoms::Clip;
use sva_core::solve::calibrate::rendered;
use sva_core::solve::pyramid::{Asked, Event, Stage, mono, pyramid, scales};
use sva_core::solve::report::{answer, progress};
use sva_core::{Abandon, CliError, success_envelope};

use crate::args::SolveArgs;
use crate::calibrations;
use crate::recording::read_recording;
use crate::store::wait;
use crate::wav::{SampleEncoding, write_wav};

fn io(path: &Path, e: std::io::Error) -> CliError {
    CliError::Io(format!("{}: {e}", path.display()))
}

/// Level `stage`'s files under `out/level-<k>/` and its render beside them as `level-<k>.wav`.
fn write(out: &Path, k: usize, stage: &Stage, clip: &Clip) -> Result<(String, String), CliError> {
    let dir = out.join(format!("level-{k}"));
    for (path, text) in &stage.files {
        let file = dir.join(path);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        }
        std::fs::write(&file, text).map_err(|e| io(&file, e))?;
    }
    let wav = out.join(format!("level-{k}.wav"));
    let samples: Vec<f32> = rendered(&stage.atoms, clip)
        .iter()
        .map(|s| *s as f32)
        .collect();
    write_wav(&samples, clip.rate, &wav, SampleEncoding::Float)?;
    Ok((dir.display().to_string(), wav.display().to_string()))
}

pub fn solve(args: &SolveArgs) -> Result<String, CliError> {
    let (planes, rate) = read_recording(&args.path)?;
    let held = std::fs::read_dir(&args.out).is_ok_and(|mut d| d.next().is_some());
    if held && !args.confirm {
        return Err(CliError::Conflict {
            by: "destination",
            message: format!(
                "{} already holds files; pass `--confirm` to write the levels over them",
                args.out.display()
            ),
        });
    }
    std::fs::create_dir_all(&args.out).map_err(|e| io(&args.out, e))?;
    let samples = mono(&planes);
    let clip = Clip {
        rate,
        len: samples.len(),
    };
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, stop.clone())
            .map_err(|e| CliError::Io(format!("a signal handler: {e}")))?;
    }
    let started = Instant::now();
    let mut wrote = Vec::new();
    let mut failed = None;
    let mut warnings = Vec::new();
    let levels = scales(args.levels);
    let key = (rate, levels.as_slice(), args.spl_db);
    let kept = calibrations::held(&args.cache, key, &mut warnings);
    let broke = Cell::new(false);
    let stages = wait(pyramid(
        &samples,
        rate,
        &Asked {
            levels: args.levels,
            spl_db: args.spl_db,
            threads: sva_engine::default_threads(),
        },
        &mut |event| {
            eprintln!("{}", progress(&event));
            if let Event::Finished(stage) = event {
                match write(&args.out, wrote.len() + 1, stage, &clip) {
                    Ok(paths) => wrote.push(paths),
                    Err(e) => {
                        failed = failed.take().or(Some(e));
                        broke.set(true);
                    }
                }
            }
        },
        &Stop {
            broke: &broke,
            signal: &stop,
        },
        &|| started.elapsed().as_secs_f64(),
    ));
    if let Some(kept) = &kept {
        kept.keep_all(key, &mut warnings);
    }
    if let Some(e) = failed {
        return Err(e);
    }
    Ok(success_envelope(
        &answer(&stages, args.levels, &wrote),
        &warnings,
    ))
}

/// A solve's stop: a level that could not be written, or a signal.
struct Stop<'a> {
    broke: &'a Cell<bool>,
    signal: &'a AtomicBool,
}

impl Abandon for Stop<'_> {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(std::future::ready(
            self.broke.get() || self.signal.load(Ordering::Relaxed),
        ))
    }
}
