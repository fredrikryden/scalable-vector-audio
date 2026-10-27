// Concern: a fit's progress lines and its answer as JSON, the shapes sva-portal's page reads | Non-concern: running it (fit.rs), where a line goes | IO: (Line | Fit) -> JSON

use super::atoms::{Atom, Sound};
use super::fit::{Fit, Line};
use crate::json::{NONE, escape, list, num, strings};

fn text(s: &str) -> String {
    format!("\"{}\"", escape(s))
}

/// Numbers as a plain JSON array, `null` where one is not finite.
fn numbers(values: &[f64]) -> String {
    let each: Vec<String> = values.iter().map(|v| num(*v)).collect();
    format!("[{}]", each.join(", "))
}

/// What an atom is: its kind, key, Hz, stretch, harmonics and band, `null` where its kind
/// has none.
fn atom(atom: &Atom) -> String {
    let none = || NONE.to_string();
    let (hz, stretch, lo, hi) = match &atom.sound {
        Sound::Partial { hz } => (num(hz.base), none(), none(), none()),
        Sound::Harmonic { hz, stretch, .. } => (num(hz.base), num(*stretch), none(), none()),
        Sound::Noise { lo, hi } => (none(), none(), num(*lo), num(*hi)),
    };
    format!(
        "{{ \"key\": {}, \"kind\": \"{}\", \"hz\": {hz}, \"stretch\": {stretch}, \"harmonics\": \
         {}, \"lo_hz\": {lo}, \"hi_hz\": {hi} }}",
        atom.key,
        atom.kind(),
        atom.harmonics()
    )
}

/// One progress line per step of either pass, its pass beside it, the first with the grid's
/// shape.
pub fn progress(line: &Line) -> String {
    let rows: Vec<String> = line
        .residual_db
        .iter()
        .map(|row| {
            let each: Vec<String> = row
                .iter()
                .map(|v| v.map_or_else(|| NONE.to_string(), num))
                .collect();
            format!("[{}]", each.join(", "))
        })
        .collect();
    let geometry = line.geometry.as_ref().map_or_else(String::new, |g| {
        let edges: Vec<String> = g
            .band_edges_hz
            .iter()
            .map(|(lo, hi)| format!("[{}, {}]", num(*lo), num(*hi)))
            .collect();
        format!(
            ", \"rate_hz\": {}, \"frame_secs\": {}, \"first_centre_secs\": {}, \
             \"band_edges_hz\": [{}], \"band_floor_db\": {}, \"gain_db\": {}",
            g.rate_hz,
            num(g.frame_secs),
            num(g.first_centre_secs),
            edges.join(", "),
            numbers(&g.band_floor_db),
            num(g.gain_db)
        )
    });
    format!(
        "{{ \"pass\": {}, \"step\": {}, \"action\": \"{}\", \"atom\": {}, \"delta_dl_bits\": \
         {}, \"dl_bits\": {}, \"atom_bits\": {}, \"residual_bits\": {}, \"residual_db\": [{}], \
         \"envelopes\": {}, \"seconds\": {}{geometry} }}",
        line.pass,
        line.step,
        line.action.name(),
        line.atom.as_ref().map_or_else(|| NONE.to_string(), atom),
        num(line.delta_bits),
        num(line.atom_bits + line.residual_bits),
        num(line.atom_bits),
        num(line.residual_bits),
        rows.join(", "),
        list(&line.envelopes, |(key, db)| {
            let rows: Vec<String> = db.iter().map(|row| numbers(row)).collect();
            format!("{{ \"key\": {key}, \"db\": [{}] }}", rows.join(", "))
        }),
        num(line.seconds)
    )
}

/// The answer's `data`: the fit's counts, DL, time and atoms, its second pass's, the paths the CLI wrote, and its
/// composition's files where asked.
pub fn answer(
    fit: &Fit,
    secs: f64,
    wrote: &[String],
    files: Option<&[(String, String)]>,
) -> String {
    let count = |kind: &str| fit.atoms.iter().filter(|a| a.kind() == kind).count();
    let files = files.map_or_else(String::new, |files| {
        format!(
            ", \"files\": {}",
            list(files, |(path, body)| format!(
                "{{ \"path\": {}, \"text\": {} }}",
                text(path),
                text(body)
            ))
        )
    });
    let second = &fit.second;
    format!(
        "{{ \"fit\": {{ \"rate_hz\": {}, \"steps\": {}, \"partials\": {}, \"harmonics\": {}, \
         \"noises\": {}, \"dl_bits\": {}, \"atom_bits\": {}, \"residual_bits\": {}, \"gain_db\": \
         {}, \"stopped\": {}, \"secs\": {}, \"atoms\": {}, \"second\": {{ \"steps\": {}, \
         \"start_dl_bits\": {}, \"dl_bits\": {}, \"atom_bits\": {}, \"residual_bits\": {}, \
         \"stopped\": {}, \"atoms\": {} }}, \"wrote\": {} }}{files} }}",
        fit.clip.rate,
        fit.steps,
        count("partial"),
        count("harmonic"),
        count("noise"),
        num(fit.atom_bits + fit.residual_bits),
        num(fit.atom_bits),
        num(fit.residual_bits),
        num(fit.gain_db),
        fit.stopped,
        num(secs),
        list(&fit.atoms, atom),
        second.steps,
        num(second.start_bits),
        num(second.atom_bits + second.residual_bits),
        num(second.atom_bits),
        num(second.residual_bits),
        second.stopped,
        list(&second.atoms, atom),
        strings(wrote)
    )
}
