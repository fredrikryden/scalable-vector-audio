// Concern: one priced fit as both front ends ask it: the solver alone or a scheme's stages, each measured, the choice, its JSON | Non-concern: minimising, stage order | IO: (Problem, Scheme?) -> Fitted

use sva_engine::{Abandon, Backend, Tier};

use crate::cli_error::CliError;
use crate::continuation::{self, Cx, Ran, Reached, Scheme, Stage};
use crate::json::{escape, list, num};
use crate::solve::Problem;
use crate::solve::prox::eval::Residual;
use crate::solve::prox::{self, BITS_PER_JND2, Progress, Settings};

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
        "{{\"stage\": \"{}\", \"phase\": \"{}\", \"price\": {}, \"step\": {}, \"residual_bits\": {}, \
         \"numbers\": {}, \"bits\": {}}}",
        escape(stage),
        escape(p.phase),
        num(p.lambda),
        p.step,
        num(p.residual_bits),
        p.numbers,
        num(p.bits)
    )
}

/// Without a scheme, the solver once at the cost's own price, then measured; with one, its run.
/// `report` hears each progress line as JSON.
pub async fn fitted<B: Backend>(
    problem: &Problem<'_>,
    settings: &Settings,
    scheme: Option<&Scheme>,
    tier: &Tier<B>,
    report: &mut dyn FnMut(&str),
    abandon: &dyn Abandon,
) -> Result<Fitted, CliError> {
    let shape = prox::checked(problem)?;
    let (ran, files) = match scheme {
        None => {
            let label = format!("price {}", shape.price);
            let mut heard = |p: &Progress| report(&progress_json(&label, p));
            let solution = prox::solve(problem, settings, tier, &mut heard, abandon).await?;
            let cancelled = solution.cancelled || abandon.abandoned().await;
            let measured = match cancelled {
                true => None,
                false => Some(prox::measure(problem, &solution.files, tier, abandon).await?),
            };
            let files = solution.files.clone();
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
                measured,
            };
            (Ran::of(vec![reached], cancelled), files)
        }
        Some(scheme) => {
            let cx = Cx {
                problem,
                shape: &shape,
                settings,
                tier,
                abandon,
            };
            let mut heard = |stage: &str, p: &Progress| report(&progress_json(stage, p));
            let ran = continuation::run(scheme, &cx, &mut heard).await?;
            let files = match ran.chosen {
                Some(c) => continuation::written(&cx, &ran.stages[c])?,
                None => Vec::new(),
            };
            (ran, files)
        }
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
        "{{ \"label\": \"{}\", \"edits\": {}, \"steps\": {}, \"price\": {}, \"numbers\": {}, \
         \"bits\": {}, \"residual_bits\": {}, \"length_bits\": {}, \"cells\": {}, \"objective\": {}, \
         \"measured\": {}, \"predicted\": {}, \"error_bits\": {}, \"stopped\": {} }}",
        escape(&r.stage.label),
        list(&r.stage.edits, |e| format!("\"{}\"", escape(e))),
        r.stage.steps,
        num(s.price),
        s.numbers,
        num(s.bits),
        num(r.residual_bits()),
        num(r.length()),
        num(s.cells),
        num(s.objective()),
        r.measured
            .as_ref()
            .map_or("null".to_string(), residual_json),
        residual_json(&s.predicted),
        r.measured
            .map_or("null".to_string(), |m| num(m.bits - s.predicted.bits)),
        s.cancelled,
    )
}

/// The answer's `data`, as both front ends print it.
pub fn fitted_data(f: &Fitted, settings: &Settings, cost: &str, extra: &str) -> String {
    let ran = &f.ran;
    let chosen = ran
        .chosen
        .map_or("null".to_string(), |c| stage_json(&ran.stages[c]));
    format!(
        "{{\"method\": \"prox\", \"cost_expression\": \"{}\", \"model\": \"{}\", \"reference\": \
         \"{}\", \"bits_per_jnd2\": {}, \"steps\": {}, \"continuation\": {}, \"stopped_by\": \
         \"{}\", \"chosen\": {chosen}, \"files\": {}{extra}, \"stages\": {}}}",
        escape(cost),
        escape(&f.model),
        escape(&f.reference),
        num(BITS_PER_JND2),
        settings.steps,
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
