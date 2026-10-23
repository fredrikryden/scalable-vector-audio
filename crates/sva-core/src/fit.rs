// Concern: one priced fit as both front ends ask it: the solver or a scheme's stages, the choice measured, its JSON | Non-concern: minimising, stage order | IO: (Problem, Scheme?) -> Fitted

use sva_engine::{Abandon, Backend, Tier};

use crate::cli_error::CliError;
use crate::continuation::{self, Cx, FEWEST, Ran, Reached, Scheme, Stage};
use crate::json::{escape, list, num};
use crate::output::{Diagnostic, Severity};
use crate::solve::Problem;
use crate::solve::prox::eval::Residual;
use crate::solve::prox::{self, BITS_PER_JND2, Ended, Progress, Settings};

pub struct Fitted {
    pub ran: Ran,
    /// The chosen stage's files over the composition.
    pub files: Vec<(String, String)>,
    pub model: String,
    pub reference: String,
    pub scheme: Option<Scheme>,
}

/// One progress line, its stage's label beside the step.
fn progress_json(stage: &str, p: &Progress) -> String {
    format!(
        "{{\"stage\": \"{}\", \"phase\": \"{}\", \"price\": {}, \"step\": {}, \"draws\": {}, \
         \"residual_bits\": {}, \"numbers\": {}, \"bits\": {}, \"ended\": {}}}",
        escape(stage),
        escape(p.phase),
        num(p.lambda),
        p.step,
        p.draws,
        num(p.residual_bits),
        p.numbers,
        num(p.bits),
        p.ended
            .map_or("null".to_string(), |e| format!("\"{}\"", e.name()))
    )
}

/// Without a scheme, one stage at the cost's own price; with one, its run; either through the
/// draws' schedule up to `settings.draws` ([`continuation`]). The stage of least predicted
/// length is chosen and it alone measured. `report` hears each progress line as JSON.
pub async fn fitted<B: Backend>(
    problem: &Problem<'_>,
    settings: &Settings,
    scheme: Option<&Scheme>,
    tier: &Tier<B>,
    report: &mut dyn FnMut(&str),
    abandon: &dyn Abandon,
) -> Result<Fitted, CliError> {
    let shape = prox::checked(problem)?;
    if settings.draws < FEWEST {
        return Err(CliError::usage(
            "solve.prox_draws",
            format!(
                "a priced fit's draws are {FEWEST} or more, the fewest with a standard error, \
                 not {}",
                settings.draws
            ),
        ));
    }
    let cx = Cx {
        problem,
        shape: &shape,
        settings,
        tier,
        abandon,
    };
    let ran = match scheme {
        None => {
            let label = format!("price {}", shape.price);
            let mut heard = |p: &Progress| report(&progress_json(&label, p));
            let solution = cx
                .scheduled(&[], &problem.cost, settings.steps, FEWEST, &mut heard)
                .await?;
            let cancelled = solution.cancelled() || abandon.abandoned().await;
            let stage = Stage {
                label,
                files: Vec::new(),
                cost: problem.cost.clone(),
                steps: settings.steps,
                edits: Vec::new(),
            };
            let reached = Reached {
                stage,
                solution,
                measured: None,
            };
            Ran::of(vec![reached], cancelled)
        }
        Some(scheme) => {
            let mut heard = |stage: &str, p: &Progress| report(&progress_json(stage, p));
            continuation::run(scheme, &cx, &mut heard).await?
        }
    };
    let ran = ran.measured(&cx).await?;
    let files = match (scheme, ran.chosen) {
        (None, _) => ran.stages[0].solution.files.clone(),
        (Some(_), Some(c)) => continuation::written(&cx, &ran.stages[c])?,
        (Some(_), None) => Vec::new(),
    };
    Ok(Fitted {
        ran,
        files,
        model: shape.model,
        reference: shape.reference,
        scheme: scheme.cloned(),
    })
}

fn residual_json(r: &Residual) -> String {
    format!(
        "{{ \"residual_bits\": {}, \"standard_error\": {}, \"d_prime\": {}, \"per_cell\": {} }}",
        num(r.bits),
        num(r.error),
        num(r.d_prime),
        num(r.per_cell)
    )
}

fn stage_json(r: &Reached) -> String {
    let s = &r.solution;
    format!(
        "{{ \"label\": \"{}\", \"edits\": {}, \"descents\": {}, \"price\": {}, \"numbers\": {}, \
         \"bits\": {}, \"residual_bits\": {}, \"length_bits\": {}, \"cells\": {}, \"objective\": {}, \
         \"measured\": {}, \"predicted\": {}, \"error_bits\": {}, \"stopped\": {} }}",
        escape(&r.stage.label),
        list(&r.stage.edits, |e| format!("\"{}\"", escape(e))),
        list(&s.descents, |d| format!(
            "{{ \"phase\": \"{}\", \"steps\": {}, \"cap\": {}, \"draws\": {}, \"ended\": \"{}\" }}",
            d.phase,
            d.steps,
            d.cap,
            d.draws,
            d.ended.name()
        )),
        num(s.price),
        s.numbers,
        num(s.bits),
        num(s.predicted.bits),
        num(r.length()),
        num(s.cells),
        num(s.objective()),
        r.measured
            .as_ref()
            .map_or("null".to_string(), residual_json),
        residual_json(&s.predicted),
        r.measured
            .map_or("null".to_string(), |m| num(m.bits - s.predicted.bits)),
        s.cancelled(),
    )
}

/// A warning naming `steps`, the front end's spelling of [`Settings::steps`], where a stage's
/// descent or debias took its cap's steps with its objective still falling; and one naming
/// `draws`, its spelling of [`Settings::draws`], where a stage's last descent stopped at the
/// noise of the cap's draws.
pub fn capped(f: &Fitted, settings: &Settings, (steps, draws): (&str, &str)) -> Vec<Diagnostic> {
    let mut out = stepped(f, settings, steps);
    out.extend(drawn(f, settings, draws));
    out
}

fn drawn(f: &Fitted, settings: &Settings, flag: &str) -> Vec<Diagnostic> {
    let ended: Vec<String> = f
        .ran
        .stages
        .iter()
        .filter(|r| {
            r.solution
                .descents
                .iter()
                .rfind(|d| d.phase == "descend")
                .is_some_and(|d| d.ended == Ended::Converged && d.draws == settings.draws)
        })
        .map(|r| format!("`{}`", r.stage.label))
        .collect();
    if ended.is_empty() {
        return Vec::new();
    }
    vec![
        Diagnostic::new(
            "solve.draws_cap",
            format!(
                "{} stopped at the noise of the cap's {} draws, so may stop short of where more \
                 would take it",
                ended.join(", "),
                settings.draws
            ),
        )
        .with_severity(Severity::Warning)
        .helped(format!(
            "raise {flag} (now {}), the most draws a stage's schedule doubles to",
            settings.draws
        )),
    ]
}

fn stepped(f: &Fitted, settings: &Settings, flag: &str) -> Vec<Diagnostic> {
    let ended: Vec<String> = f
        .ran
        .stages
        .iter()
        .flat_map(|r| {
            r.solution
                .descents
                .iter()
                .filter(|d| d.ended == Ended::Cap && d.cap > 0)
                .map(|d| format!("`{}` ({}, {} steps)", r.stage.label, d.phase, d.cap))
        })
        .collect();
    if ended.is_empty() {
        return Vec::new();
    }
    vec![
        Diagnostic::new(
            "solve.steps_cap",
            format!(
                "{} took the cap's steps before the objective stopped falling, so may stop \
                 short of where it would",
                ended.join(", ")
            ),
        )
        .with_severity(Severity::Warning)
        .helped(format!(
            "raise {flag} (now {}), the cap each stage's own scales with",
            settings.steps
        )),
    ]
}

/// The answer's `data`, as both front ends print it.
pub fn fitted_data(f: &Fitted, settings: &Settings, cost: &str, extra: &str) -> String {
    let ran = &f.ran;
    let chosen = ran
        .chosen
        .map_or("null".to_string(), |c| stage_json(&ran.stages[c]));
    format!(
        "{{\"method\": \"prox\", \"cost_expression\": \"{}\", \"model\": \"{}\", \"reference\": \
         \"{}\", \"bits_per_jnd2\": {}, \"steps\": {}, \"draws\": {}, \"continuation\": {}, \
         \"stopped_by\": \"{}\", \"chosen\": {chosen}, \"files\": {}{extra}, \"stages\": {}}}",
        escape(cost),
        escape(&f.model),
        escape(&f.reference),
        num(BITS_PER_JND2),
        settings.steps,
        settings.draws,
        f.scheme.as_ref().map_or("null".to_string(), Scheme::json),
        match ran.cancelled {
            true => "cancelled",
            false => "converged",
        },
        list(&f.files, |(p, t)| format!(
            "{{ \"path\": \"{}\", \"text\": \"{}\" }}",
            escape(p),
            escape(t)
        )),
        list(&ran.stages, stage_json)
    )
}
