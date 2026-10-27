// Concern: runs a fit on a recording: a progress line and a render per pass, then its composition, its render, both grids | Non-concern: argv, the fit | IO: (SolveArgs) -> JSON or CliError

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use sva_core::solve::atoms::{files_at, sounded};
use sva_core::solve::fit::{fit, mono};
use sva_core::solve::report::{answer, progress};
use sva_core::{Abandon, CliError, success_envelope};

use crate::args::SolveArgs;
use crate::recording::read_recording;
use crate::store::wait;
use crate::wav::{SampleEncoding, write_wav};

fn io(path: &Path, e: std::io::Error) -> CliError {
    CliError::Io(format!("{}: {e}", path.display()))
}

/// The fit of `args.path` under `args.out`: `pass-001.wav` and on (the atoms each pass leaves,
/// rendered as it ends), its composition, `final.wav` (its render), and `clip-grid.wav` and
/// `fit-grid.wav` (both coarse grids sounded alike).
pub fn solve(args: &SolveArgs) -> Result<String, CliError> {
    let (planes, rate) = read_recording(&args.path)?;
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
    stale(&args.out)?;
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
        &mut |line| {
            eprintln!("{}", progress(line));
            if let Some(g) = &line.geometry {
                gain_db = g.gain_db;
            }
            if line.pass == 0 || failed.is_some() {
                return;
            }
            let files = files_at(&line.atoms, &line.clip, gain_db);
            let wav = args.out.join(format!("pass-{:03}.wav", line.pass));
            let samples: Vec<f32> = sounded(files, &line.clip)
                .iter()
                .map(|s| *s as f32)
                .collect();
            match write_wav(&samples, line.clip.rate, &wav, SampleEncoding::Float) {
                Ok(()) => wrote.push(wav.display().to_string()),
                Err(e) => {
                    failed = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
            }
        },
        &Signalled(&stop),
        &|| started.elapsed().as_secs_f64(),
    ));
    if let Some(e) = failed {
        return Err(e);
    }
    let composition = files_at(&fitted.atoms, &fitted.clip, fitted.gain_db);
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
        ("final.wav", sounded(composition, &fitted.clip), 1.0),
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

/// `out` without the pass renders an earlier fit left, so each one there is this fit's.
fn stale(out: &Path) -> Result<(), CliError> {
    for entry in std::fs::read_dir(out).map_err(|e| io(out, e))? {
        let path = entry.map_err(|e| io(out, e))?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("pass-") && name.ends_with(".wav") {
            std::fs::remove_file(&path).map_err(|e| io(&path, e))?;
        }
    }
    Ok(())
}

/// A fit's stop: a signal, or a pass's render that failed to write.
struct Signalled<'a>(&'a AtomicBool);

impl Abandon for Signalled<'_> {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(std::future::ready(self.0.load(Ordering::Relaxed)))
    }
}
