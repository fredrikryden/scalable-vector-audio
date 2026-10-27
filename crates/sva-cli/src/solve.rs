// Concern: runs a fit on a recording: a progress line per step on stderr, then its composition, render and both grids sounded | Non-concern: argv, the fit | IO: (SolveArgs) -> JSON or CliError

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

/// The fit of `args.path` under `args.out`: its composition, `first.wav` (its render), and
/// `clip-grid.wav` and `fit-grid.wav` (both coarse grids sounded alike).
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
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, stop.clone())
            .map_err(|e| CliError::Io(format!("a signal handler: {e}")))?;
    }
    let started = Instant::now();
    let fitted = wait(fit(
        &mono(&planes),
        rate,
        args.spl_db,
        &mut |line| eprintln!("{}", progress(line)),
        &Signalled(&stop),
        &|| started.elapsed().as_secs_f64(),
    ));
    let mut wrote = Vec::new();
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
        ("first.wav", sounded(composition, &fitted.clip), 1.0),
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

/// A fit's stop: a signal.
struct Signalled<'a>(&'a AtomicBool);

impl Abandon for Signalled<'_> {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(std::future::ready(self.0.load(Ordering::Relaxed)))
    }
}
