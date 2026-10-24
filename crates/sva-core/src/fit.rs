// Concern: one priced fit as both front ends ask it: the solver or a scheme's stages, the choice measured, its JSON | Non-concern: minimising, stage order | IO: (Problem, Scheme?) -> Fitted

use sva_engine::{Abandon, Backend, Tier};

use crate::cli_error::CliError;
use crate::continuation::{self, Cx, FEWEST, Ran, Reached, Scheme, Stage};
use crate::json::{escape, list, num};
use crate::output::{Diagnostic, Severity};
use crate::solve::Problem;
use crate::solve::prox::eval::Residual;
use crate::solve::prox::heard::Hearing;
pub use crate::solve::prox::heard::{Action, Clock, Doing, Heard, Report};
use crate::solve::prox::{self, BITS_PER_JND2, Ended, Progress, Settings};

pub struct Fitted {
    pub ran: Ran,
    /// The chosen stage's files over the composition.
    pub files: Vec<(String, String)>,
    pub model: String,
    pub reference: String,
    pub scheme: Option<Scheme>,
}

/// One progress line: a step, its stage's label beside it; an activity; a stage listed, its
/// files' texts where `texts`, their paths alone otherwise; or the activity a refusal stopped.
pub fn heard_json(heard: &Heard, texts: bool) -> String {
    match heard {
        Heard::Step(stage, p) => progress_json(stage, p),
        Heard::Doing(d) => format!("{{{}}}", doing_json(d)),
        Heard::Failed { doing, code } => format!(
            "{{\"failed\": true, \"code\": \"{code}\", {}}}",
            match doing {
                Some(d) => doing_json(d),
                None => "\"action\": null".to_string(),
            }
        ),
        Heard::Listed {
            label,
            solution,
            at_cap,
            length,
            files,
            least,
        } => format!(
            "{{\"stage\": \"{}\", \"finished\": true, \"stopped\": {}, \"length_bits\": {}, \
             \"residual_bits\": {}, \"bits\": {}, \"numbers\": {}, \"predicted_at_cap\": {}, \
             \"least\": {least}, \"files\": {}}}",
            escape(label),
            solution.cancelled(),
            length.map_or("null".to_string(), num),
            num(solution.predicted.bits),
            num(solution.bits),
            solution.numbers,
            at_cap.map_or("null".to_string(), residual_json),
            list(files, |(p, t)| match texts {
                true => format!(
                    "{{ \"path\": \"{}\", \"text\": \"{}\" }}",
                    escape(p),
                    escape(t)
                ),
                false => format!("{{ \"path\": \"{}\" }}", escape(p)),
            }),
        ),
    }
}

fn doing_json(d: &Doing) -> String {
    let text = |t: &Option<String>| match t {
        Some(t) => format!("\"{}\"", escape(t)),
        None => "null".to_string(),
    };
    format!(
        "\"action\": \"{}\", \"stage\": {}, \"edit\": {}, \"count\": {}, \"part\": {}, \
         \"work\": {{ \"comparisons\": {}, \"per_second\": {} }}, \"seconds\": {}",
        d.action.name(),
        text(&d.stage),
        text(&d.edit),
        d.count.map_or("null".to_string(), |(done, of)| format!(
            "{{ \"done\": {done}, \"of\": {of}, \"unit\": \"{}\" }}",
            d.action.unit()
        )),
        d.part.map_or("null".to_string(), |(at, of)| format!(
            "{{ \"at\": {at}, \"of\": {of} }}"
        )),
        d.comparisons,
        d.per_second.map_or("null".to_string(), num),
        num(d.seconds)
    )
}

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

/// Zero steps evaluate the composition as it is and change no number, so a write of them, spelt
/// `(write, steps)` by the front end, has nothing to write.
pub fn writable(settings: &Settings, (write, steps): (&str, &str)) -> Result<(), CliError> {
    match settings.steps {
        0 => Err(CliError::usage(
            "solve.prox_write",
            format!(
                "{write} with {steps} 0 has nothing to write: 0 steps evaluate the composition \
                 as it is and change no number; drop {write}, or take {steps} above 0"
            ),
        )),
        _ => Ok(()),
    }
}

/// Without a scheme, one stage at the cost's own price; with one, its run; either through the
/// draws' schedule up to `settings.draws` ([`continuation`]). The stage of least predicted
/// length at that cap's draws is chosen and it alone measured. `report` hears each line
/// [`heard_json`] writes, activities `settings.heartbeat` apart.
pub async fn fitted<B: Backend>(
    problem: &Problem<'_>,
    settings: &Settings,
    scheme: Option<&Scheme>,
    tier: &Tier<B>,
    (report, clock): (Report<'_>, Clock<'_>),
    abandon: &dyn Abandon,
) -> Result<Fitted, CliError> {
    let hearing = Hearing::new(report, clock, settings.heartbeat);
    let out = heard(problem, settings, scheme, tier, &hearing, abandon).await;
    if let Err(e) = &out {
        hearing.failed(e);
    }
    out
}

async fn heard<B: Backend>(
    problem: &Problem<'_>,
    settings: &Settings,
    scheme: Option<&Scheme>,
    tier: &Tier<B>,
    hearing: &Hearing<'_>,
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
    if settings.steps == 0 && scheme.is_some() {
        return Err(CliError::usage(
            "solve.prox_steps",
            "a continuation's stages are descents, so 0 steps leave it nothing to do; take \
             steps above 0, or drop the continuation to evaluate the composition as it is",
        ));
    }
    let cx = Cx {
        problem,
        shape: &shape,
        settings,
        tier,
        hearing,
        abandon,
    };
    let ran = match scheme {
        None => {
            let label = format!("price {}", shape.price);
            hearing.staged(&label, (1, 1));
            let solution = cx
                .scheduled(&[], &problem.cost, settings.steps, FEWEST)
                .await?;
            let stage = Stage {
                label,
                files: Vec::new(),
                cost: problem.cost.clone(),
                steps: settings.steps,
                edits: Vec::new(),
            };
            let reached = cx.reached(stage, solution).await?;
            let least = continuation::below(&reached, &mut None);
            reached.listed(&reached.solution.files, least, hearing);
            let cancelled = reached.at_cap.is_none() || abandon.abandoned().await;
            Ran::of(vec![reached], cancelled)
        }
        Some(scheme) => continuation::run(scheme, &cx).await?,
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
         \"bits\": {}, \"draws\": {}, \"residual_bits\": {}, \"solved_length_bits\": {}, \
         \"length_bits\": {}, \"cells\": {}, \"objective\": {}, \"measured\": {}, \"predicted\": {}, \
         \"predicted_at_cap\": {}, \"error_bits\": {}, \"stopped\": {} }}",
        escape(&r.stage.label),
        list(&r.stage.edits, |e| format!("\"{}\"", escape(e))),
        list(&s.descents, |d| format!(
            "{{ \"phase\": \"{}\", \"steps\": {}, \"cap\": {}, \"draws\": {}, \"ended\": \"{}\", \
             \"noise_limited\": {} }}",
            d.phase,
            d.steps,
            d.cap,
            d.draws,
            d.ended.name(),
            d.noise_limited
        )),
        num(s.price),
        s.numbers,
        num(s.bits),
        s.draws(),
        num(s.predicted.bits),
        num(r.solved_length()),
        r.length().map_or("null".to_string(), num),
        num(s.cells),
        num(s.objective()),
        r.measured
            .as_ref()
            .map_or("null".to_string(), residual_json),
        residual_json(&s.predicted),
        r.at_cap.as_ref().map_or("null".to_string(), residual_json),
        match (r.measured, r.at_cap) {
            (Some(m), Some(p)) => num(m.bits - p.bits),
            _ => "null".to_string(),
        },
        s.cancelled(),
    )
}

/// A warning naming `steps`, the front end's spelling of [`Settings::steps`], where a stage's
/// descent or debias took its cap's steps with its objective still falling; and one naming
/// `draws`, its spelling of [`Settings::draws`], where a stage's last descent was limited by
/// the noise of the cap's draws.
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
                .is_some_and(|d| d.noise_limited && d.draws == settings.draws)
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
                "{} gained no more over its last descent's later half than the cap's {} draws \
                 resolve, so may stop short of where more would take it",
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
