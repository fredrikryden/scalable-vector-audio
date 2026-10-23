// Concern: runs one solve here: its store, a signal stopping it, progress lines, values written back | Non-concern: the search and the cost (sva-core) | IO: (SolveArgs) -> JSON, an exit, stderr, files

use std::io::Write;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sva_ast::variable;
use sva_core::continuation::Scheme;
use sva_core::fit::{fitted, fitted_data};
use sva_core::solve::prox::Settings;
use sva_core::solve::{
    Point, Problem, Solved, Stopped, Vary, cost_of, findings, point_json, solve as solved,
    solved_data,
};
use sva_core::{CliError, error_envelope_with, success_envelope};
use sva_engine::Abandon;

use crate::args::{Method, SolveArgs};
use crate::heard::Heard;
use crate::store::{On, opened, persisted, wait};

/// The exit a solve stopped by a signal answers, as a shell reports one.
pub const CANCELLED: u8 = 130;

struct Flagged(Arc<AtomicBool>);

impl Abandon for Flagged {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(async { self.0.load(Ordering::Relaxed) })
    }
}

/// The envelope and the exit it answers with.
pub fn solve(args: &SolveArgs) -> Result<(String, u8), CliError> {
    let dir = crate::composition::composition()?;
    let heard = Heard::at(&dir)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut warnings = Vec::new();
    let tier = opened(&args.cache, &mut warnings, On::Flag(&stop))?;
    let problem = Problem {
        source: &heard,
        texts: &heard,
        cost: cost_of(&heard, &args.cost)?,
        vary: args.vary.clone(),
        sets: args.sets.clone(),
        rate: args.rate,
        threads: args.threads,
        stops: args.stops,
    };
    if let Method::Prox(settings, scheme) = &args.method {
        let method = (settings, scheme.as_ref());
        return priced_solve(args, method, &dir, &problem, tier, stop, warnings);
    }
    let mut heard_point = |point: &Point| {
        // A reader that closed stderr loses the progress, never the answer.
        let _ = writeln!(std::io::stderr(), "{}", point_json(point));
    };
    let outcome = wait(solved(&problem, tier, &mut heard_point, &Flagged(stop)));
    persisted(tier, &mut warnings);
    let solved = outcome?;
    warnings.extend(findings(&solved, &args.vary));
    let stopped = solved.stopped_by == Stopped::Cancelled;
    if stopped {
        let data = solved_data(&solved, &args.vary, "");
        let message = format!(
            "the solve was stopped by a signal after {} render(s); `data` holds the best so far, \
             and no file was written",
            solved.renders
        );
        let json = error_envelope_with("cancelled", &message, &data, &warnings);
        return Ok((json, CANCELLED));
    }
    if !args.write {
        let data = solved_data(&solved, &args.vary, "");
        return Ok((success_envelope(&data, &warnings), 0));
    }
    let changes = changed(&dir, &solved, &args.vary)?;
    let listed = sva_core::json::list(&changes, |c| {
        format!(
            "{{ \"path\": \"{}\", \"from\": \"{}\", \"to\": \"{}\" }}",
            sva_core::json::escape(&c.path),
            sva_core::json::escape(&c.from),
            sva_core::json::escape(&c.to)
        )
    });
    if !changes.is_empty() && !args.confirm {
        let data = solved_data(&solved, &args.vary, &format!(", \"would_write\": {listed}"));
        let message = format!(
            "`--write` would change {} variable file(s), and replacing a file needs \
             `--confirm`; nothing was written",
            changes.len()
        );
        let json = error_envelope_with("conflict", &message, &data, &warnings);
        return Ok((json, 4));
    }
    for change in &changes {
        std::fs::write(dir.join(&change.path), &change.text)
            .map_err(|e| CliError::Io(format!("could not write `{}`: {e}", change.path)))?;
    }
    let data = solved_data(&solved, &args.vary, &format!(", \"written\": {listed}"));
    Ok((success_envelope(&data, &warnings), 0))
}

struct Change {
    path: String,
    from: String,
    to: String,
    text: String,
}

/// Each variable whose file would hold another value, its comments kept.
fn changed(dir: &std::path::Path, solved: &Solved, vary: &[Vary]) -> Result<Vec<Change>, CliError> {
    let mut out = Vec::new();
    for ((name, _, to), _) in solved.best.values.iter().zip(vary) {
        let path = variable(name);
        let text = std::fs::read_to_string(dir.join(&path))
            .map_err(|e| CliError::Io(format!("could not read `{path}`: {e}")))?;
        let at = sva_ast::body_at(&text);
        let body = &text[at..];
        let from = sva_ast::strip_line_comment(body).trim().to_string();
        if from == *to {
            continue;
        }
        let comment = body.find(';').map(|c| format!(" {}", body[c..].trim_end()));
        out.push(Change {
            text: format!("{}{to}{}\n", &text[..at], comment.unwrap_or_default()),
            path,
            from,
            to: to.clone(),
        });
    }
    Ok(out)
}

fn priced_solve(
    args: &SolveArgs,
    (settings, scheme): (&Settings, Option<&Scheme>),
    dir: &std::path::Path,
    problem: &Problem,
    tier: &sva_engine::Tier<crate::directory::Directory>,
    stop: Arc<AtomicBool>,
    mut warnings: Vec<sva_core::Diagnostic>,
) -> Result<(String, u8), CliError> {
    let mut heard = |line: &str| {
        let _ = writeln!(std::io::stderr(), "{line}");
    };
    let outcome = wait(fitted(
        problem,
        settings,
        scheme,
        tier,
        &mut heard,
        &Flagged(stop),
    ));
    persisted(tier, &mut warnings);
    let fitted = outcome?;
    let data = |extra: &str| fitted_data(&fitted, settings, &problem.cost.text, extra);
    if fitted.ran.cancelled {
        let message = "the priced solve was stopped by a signal; `data.stages` holds the stages \
                       it finished and the one it stopped in, `data.chosen` the best of them by \
                       prediction, and no file was written";
        let json = error_envelope_with("cancelled", message, &data(""), &warnings);
        return Ok((json, CANCELLED));
    }
    if !args.write {
        return Ok((success_envelope(&data(""), &warnings), 0));
    }
    let listed = sva_core::json::list(&fitted.files, |(p, _)| {
        format!("\"{}\"", sva_core::json::escape(p))
    });
    if !fitted.files.is_empty() && !args.confirm {
        let message = format!(
            "`--write` would rewrite {} file(s) with the chosen stage, and replacing a file needs \
             `--confirm`; nothing was written",
            fitted.files.len()
        );
        let data = data(&format!(", \"would_write\": {listed}"));
        let json = error_envelope_with("conflict", &message, &data, &warnings);
        return Ok((json, 4));
    }
    for (path, text) in &fitted.files {
        std::fs::write(dir.join(path), text)
            .map_err(|e| CliError::Io(format!("could not write `{path}`: {e}")))?;
    }
    let data = data(&format!(", \"written\": {listed}"));
    Ok((success_envelope(&data, &warnings), 0))
}
