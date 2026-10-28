// Concern: runs a fit on a recording, fresh or resumed: a line and render per pass, then its composition, its render, both grids | Non-concern: argv, the fit | IO: (SolveArgs) -> JSON or CliError

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use sva_core::{Abandon, CliError, success_envelope};
use sva_solve::TOLERANCE;
use sva_solve::fit::{Options, Resume, fit, mono};
use sva_solve::report::{answer, progress};
use sva_solve::sources::{DIRECTORY, SCORE, files_at, sounded};

use crate::args::SolveArgs;
use crate::recording::read_recording;
use crate::store::wait;
use crate::wav::{SampleEncoding, write_wav};

fn io(path: &Path, e: std::io::Error) -> CliError {
    CliError::Io(format!("{}: {e}", path.display()))
}

/// The fit of `args.path` under `args.out`: `pass-001.wav` and on (the sources each pass leaves,
/// rendered as it ends), its composition, `final.wav` (its render), and `clip-grid.wav` and
/// `fit-grid.wav` (both coarse grids sounded alike); resumed from `args.from` where named.
pub fn solve(args: &SolveArgs) -> Result<String, CliError> {
    let (planes, rate) = read_recording(&args.path)?;
    let from = args.from.as_deref().map(resume).transpose()?;
    let options = Options {
        tolerance: args.tolerance.unwrap_or(match from {
            Some(_) => 0.0,
            None => TOLERANCE,
        }),
        from,
        threads: sva_engine::default_threads(),
    };
    let after = options.from.as_ref().map_or(0, |r| r.passes);
    let held = std::fs::read_dir(&args.out).is_ok_and(|mut d| d.next().is_some());
    if held && !args.confirm {
        return Err(CliError::Conflict {
            by: "destination",
            message: format!(
                "{} already holds files; pass `--confirm` to write the fit over them",
                args.out.display()
            ),
        });
    }
    std::fs::create_dir_all(&args.out).map_err(|e| io(&args.out, e))?;
    stale(&args.out, after)?;
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, stop.clone())
            .map_err(|e| CliError::Io(format!("a signal handler: {e}")))?;
    }
    let started = Instant::now();
    let mut wrote = Vec::new();
    let mut failed = None;
    let mut gain_db = 0.0;
    let fitted = wait(fit(
        &mono(&planes),
        rate,
        args.spl_db,
        &options,
        &mut |line| {
            eprintln!("{}", progress(line));
            if let Some(g) = &line.geometry {
                gain_db = g.gain_db;
            }
            if line.geometry.is_some() || failed.is_some() {
                return;
            }
            let files = files_at(&line.sources, &line.shared, &line.clip, gain_db);
            let wav = args.out.join(format!("pass-{:03}.wav", line.pass));
            let written = sounded(files, &line.clip).and_then(|heard| {
                let samples: Vec<f32> = heard.iter().map(|s| *s as f32).collect();
                write_wav(&samples, line.clip.rate, &wav, SampleEncoding::Float)
            });
            match written {
                Ok(()) => wrote.push(wav.display().to_string()),
                Err(e) => {
                    failed = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
            }
        },
        &Signalled(&stop),
        &|| started.elapsed().as_secs_f64(),
    ))?;
    if let Some(e) = failed {
        return Err(e);
    }
    let composition = files_at(
        &fitted.sources,
        &fitted.shared,
        &fitted.clip,
        fitted.gain_db,
    );
    for (path, body) in &composition {
        let file = args.out.join(path);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io(parent, e))?;
        }
        std::fs::write(&file, body).map_err(|e| io(&file, e))?;
        wrote.push(file.display().to_string());
    }
    let len = fitted.clip.len;
    let back = libm::pow(10.0, fitted.gain_db / 20.0);
    for (name, heard, gain) in [
        ("final.wav", sounded(composition, &fitted.clip)?, 1.0),
        ("clip-grid.wav", fitted.clip_grid.sounded(len), back),
        ("fit-grid.wav", fitted.fit_grid.sounded(len), back),
    ] {
        let wav = args.out.join(name);
        let samples: Vec<f32> = heard.iter().map(|s| (*s * gain) as f32).collect();
        write_wav(&samples, fitted.clip.rate, &wav, SampleEncoding::Float)?;
        wrote.push(wav.display().to_string());
    }
    let secs = started.elapsed().as_secs_f64();
    Ok(success_envelope(&answer(&fitted, secs, &wrote, None), &[]))
}

/// The pass a render named `name` holds, where it is one.
fn pass_of(name: &str) -> Option<usize> {
    name.strip_prefix("pass-")?
        .strip_suffix(".wav")?
        .parse()
        .ok()
}

/// `out` without the pass renders past `after` an earlier fit left, so each one there past
/// the pass this fit resumes from is this fit's.
fn stale(out: &Path, after: usize) -> Result<(), CliError> {
    for entry in std::fs::read_dir(out).map_err(|e| io(out, e))? {
        let path = entry.map_err(|e| io(out, e))?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if pass_of(name).is_some_and(|k| k > after) {
            std::fs::remove_file(&path).map_err(|e| io(&path, e))?;
        }
    }
    Ok(())
}

/// The fit an earlier solve wrote under `dir`: its composition (`master`, its score, the
/// nodes under `atoms/`) and its passes, the last pass render there.
fn resume(dir: &Path) -> Result<Resume, CliError> {
    let read = |file: &Path| std::fs::read_to_string(file).map_err(|e| io(file, e));
    let listed = |at: &Path| -> Result<Vec<PathBuf>, CliError> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(at).map_err(|e| io(at, e))? {
            out.push(entry.map_err(|e| io(at, e))?.path());
        }
        Ok(out)
    };
    let mut files = Vec::new();
    for path in ["master", SCORE] {
        let file = dir.join(path);
        if file.is_file() {
            files.push((path.to_string(), read(&file)?));
        }
    }
    let nodes = dir.join(DIRECTORY);
    if nodes.is_dir() {
        for path in listed(&nodes)? {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_file() {
                files.push((format!("{DIRECTORY}/{name}"), read(&path)?));
            }
        }
    }
    let passes = listed(dir)?
        .iter()
        .filter_map(|p| pass_of(p.file_name().and_then(|n| n.to_str()).unwrap_or("")))
        .max()
        .unwrap_or(0);
    Resume::of(&files, passes)
}

/// A fit's stop: a signal, or a pass's render that failed to write.
struct Signalled<'a>(&'a AtomicBool);

impl Abandon for Signalled<'_> {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(std::future::ready(self.0.load(Ordering::Relaxed)))
    }
}
