// Concern: a pyramid's progress lines and its answer as JSON, the shapes FIT-UX's page reads | Non-concern: running it (pyramid.rs), where a line goes | IO: (Event | stages) -> JSON

use super::pyramid::{Event, LEVELS, RATIO, Stage, chosen};
use crate::json::{NONE, escape, list, num};

fn text(s: &str) -> String {
    format!("\"{}\"", escape(s))
}

fn files(stage: &Stage) -> String {
    list(&stage.files, |(path, body)| {
        format!("{{ \"path\": {}, \"text\": {} }}", text(path), text(body))
    })
}

/// One progress line: an activity, a descent's last step, or a stage as it finishes.
pub fn progress(event: &Event) -> String {
    match event {
        Event::Activity {
            action,
            stage,
            edit,
            count,
            part,
            comparisons,
            seconds,
        } => format!(
            "{{ \"action\": \"{}\", \"stage\": {}, \"edit\": {}, \"count\": {}, \"part\": {{ \
             \"at\": {}, \"of\": {} }}, \"work\": {{ \"comparisons\": {comparisons}, \
             \"per_second\": {NONE} }}, \"seconds\": {} }}",
            action.name(),
            text(stage),
            edit.map_or_else(|| NONE.to_string(), text),
            count.map_or_else(
                || NONE.to_string(),
                |(done, unit)| format!(
                    "{{ \"done\": {done}, \"of\": {NONE}, \"unit\": \"{unit}\" }}"
                )
            ),
            part.0,
            part.1,
            num(*seconds)
        ),
        Event::Step {
            stage,
            steps,
            residual_bits,
            numbers,
            bits,
            ended,
        } => format!(
            "{{ \"stage\": {}, \"phase\": \"solve\", \"price\": 1, \"step\": {steps}, \"draws\": \
             1, \"residual_bits\": {}, \"numbers\": {numbers}, \"bits\": {}, \"ended\": \"{}\" }}",
            text(stage),
            num(*residual_bits),
            num(*bits),
            ended.name()
        ),
        Event::Finished(stage) => format!(
            "{{ \"stage\": {}, \"finished\": true, \"stopped\": {}, \"length_bits\": {}, \
             \"residual_bits\": {}, \"bits\": {}, \"numbers\": {}, \"predicted_at_cap\": {NONE}, \
             \"least\": {}, \"seconds\": {}, \"files\": {} }}",
            text(&stage.label),
            stage.stopped,
            num(stage.length_bits),
            num(stage.residual_bits),
            num(stage.bits),
            stage.numbers(),
            stage.least,
            num(stage.seconds),
            files(stage)
        ),
    }
}

/// A stage as `data.stages` lists it; `wrote` the paths the CLI wrote it to, if any.
pub fn stage(stage: &Stage, wrote: Option<&(String, String)>) -> String {
    let wrote = wrote.map_or_else(String::new, |(dir, wav)| {
        format!(", \"directory\": {}, \"wav\": {}", text(dir), text(wav))
    });
    format!(
        "{{ \"label\": {}, \"scale\": {}, \"stopped\": {}, \"atoms\": {}, \"numbers\": {}, \
         \"bits\": {}, \"residual_bits\": {}, \"length_bits\": {}, \"least\": {}, \"seconds\": \
         {}, \"edits\": {}, \"descents\": {}, \"files\": {}{wrote} }}",
        text(&stage.label),
        num(stage.scale),
        stage.stopped,
        stage.atoms.len(),
        stage.numbers(),
        num(stage.bits),
        num(stage.residual_bits),
        num(stage.length_bits),
        stage.least,
        num(stage.seconds),
        list(&stage.edits, |e| text(e)),
        list(&stage.descents, |(steps, ended)| format!(
            "{{ \"steps\": {steps}, \"ended\": \"{}\" }}",
            ended.name()
        )),
        files(stage)
    )
}

/// The answer's `data`: the scheme, every stage, the chosen one and its files; `wrote` one
/// entry per stage where the CLI wrote them.
pub fn answer(stages: &[Stage], levels: usize, wrote: &[(String, String)]) -> String {
    let listed: Vec<(usize, &Stage)> = stages.iter().enumerate().collect();
    let chosen = chosen(stages);
    format!(
        "{{ \"continuation\": {{ \"scheme\": \"pyramid\", \"levels\": {levels}, \"ratio\": {}, \
         \"default_levels\": {LEVELS} }}, \"stages\": {}, \"chosen\": {}, \"files\": {}, \
         \"diagnostics\": {} }}",
        num(RATIO),
        list(&listed, |(k, s)| stage(s, wrote.get(*k))),
        chosen.map_or_else(|| NONE.to_string(), |s| stage(s, None)),
        chosen.map_or_else(|| list::<String>(&[], String::clone), files),
        list::<String>(&[], String::clone)
    )
}
