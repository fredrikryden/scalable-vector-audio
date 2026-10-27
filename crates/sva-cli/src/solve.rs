// Concern: runs a solve on a recording: progress on stderr, each level's files and wav as it ends, N0 kept; or a first fit's files | Non-concern: argv, the fits | IO: (SolveArgs) -> JSON or CliError

use std::cell::Cell;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use sva_core::json::{num, strings};
use sva_core::solve::atoms::Clip;
use sva_core::solve::atoms::{Atom, Sound, files};
use sva_core::solve::calibrate::rendered;
use sva_core::solve::first::fit;
use sva_core::solve::grid::compared;
use sva_core::solve::pyramid::{Asked, Event, Stage, mono, pyramid, read_at};
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

/// Level `stage`'s files under `out/level-<k>/`, its render beside them as `level-<k>.wav`, and
/// its vocoded clip and atoms in that directory as `clip-coarse.wav` and `fit-coarse.wav`.
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
    std::fs::create_dir_all(&dir).map_err(|e| io(&dir, e))?;
    for (name, heard) in ["clip-coarse.wav", "fit-coarse.wav"]
        .iter()
        .zip(&stage.vocoded)
    {
        let samples: Vec<f32> = heard.iter().map(|s| *s as f32).collect();
        write_wav(&samples, stage.rate, &dir.join(name), SampleEncoding::Float)?;
    }
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
    if args.first {
        return first(args, &samples, rate);
    }
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
    let levels = read_at(rate, args.levels, args.coarsest);
    let key = (levels.as_slice(), args.spl_db);
    let kept = calibrations::held(&args.cache, key, &mut warnings);
    let broke = Cell::new(false);
    let stages = wait(pyramid(
        &samples,
        rate,
        &Asked {
            levels: args.levels,
            coarsest: args.coarsest,
            coarse: true,
            spl_db: args.spl_db,
            threads: sva_engine::default_threads(),
            every_level: args.every_level,
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

/// One sound's first fit of `samples` under `args.out`: its composition, `first.wav` (its
/// render), `clip-grid.wav` and `fit-grid.wav` (both grids sounded alike) and `grid.json`.
fn first(args: &SolveArgs, samples: &[f64], rate: u32) -> Result<String, CliError> {
    let started = Instant::now();
    let first = fit(samples, rate, args.spl_db);
    let (grid_json, mean) = compared(&first.clip_grid, &first.fit_grid, args.spl_db);
    let mut wrote = Vec::new();
    let mut text = |name: &str, body: &str| -> Result<(), CliError> {
        let file = args.out.join(name);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        }
        std::fs::write(&file, body).map_err(|e| io(&file, e))?;
        wrote.push(file.display().to_string());
        Ok(())
    };
    for (path, body) in files(&first.atoms, &first.clip) {
        text(&path, &body)?;
    }
    text("grid.json", &grid_json)?;
    let len = first.clip.len;
    for (name, heard) in [
        ("first.wav", rendered(&first.atoms, &first.clip)),
        ("clip-grid.wav", first.clip_grid.sounded(len)),
        ("fit-grid.wav", first.fit_grid.sounded(len)),
    ] {
        let wav = args.out.join(name);
        let samples: Vec<f32> = heard.iter().map(|s| *s as f32).collect();
        write_wav(&samples, first.clip.rate, &wav, SampleEncoding::Float)?;
        wrote.push(wav.display().to_string());
    }
    let count = |kind: fn(&Atom) -> bool| first.atoms.iter().filter(|a| kind(a)).count();
    let data = format!(
        "{{ \"first\": {{ \"rate_hz\": {}, \"partials\": {}, \"harmonics\": {}, \
         \"noises\": {}, \"mean_abs_db\": {}, \"secs\": {}, \"wrote\": {} }} }}",
        first.clip.rate,
        count(|a| matches!(a.sound, Sound::Partial { .. })),
        count(|a| matches!(a.sound, Sound::Harmonic { .. })),
        count(|a| matches!(a.sound, Sound::Noise { .. })),
        num(mean),
        num(started.elapsed().as_secs_f64()),
        strings(&wrote)
    );
    Ok(success_envelope(&data, &[]))
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
