// Concern: a fit's progress lines and its answer as JSON, the shapes sva-portal's page reads | Non-concern: running it (fit.rs), where a line goes | IO: (Line | Fit) -> JSON

use super::atoms::{Atom, Sound};
use super::fit::{Fit, Line, Tweak};
use crate::json::{NONE, escape, list, num, strings};

fn text(s: &str) -> String {
    format!("\"{}\"", escape(s))
}

/// Numbers as a plain JSON array, `null` where one is not finite.
fn numbers(values: &[f64]) -> String {
    let each: Vec<String> = values.iter().map(|v| num(*v)).collect();
    format!("[{}]", each.join(", "))
}

/// What an atom is: its kind, key, Hz, glide, stretch, harmonics, band (a shaped noise's lowest
/// edge to its highest) and a shaped noise's bands, `null` where its kind has none.
fn atom(atom: &Atom) -> String {
    let none = || NONE.to_string();
    let (hz, stretch, lo, hi, bands) = match &atom.sound {
        Sound::Partial { hz, .. } => (num(hz.base), none(), none(), none(), none()),
        Sound::Harmonic { hz, stretch, .. } => {
            (num(hz.base), num(*stretch), none(), none(), none())
        }
        Sound::Noise { lo, hi } => (none(), none(), num(*lo), num(*hi), none()),
        Sound::Shaped { bands } => {
            let each: Vec<String> = bands
                .iter()
                .map(|(lo, hi, g)| {
                    format!(
                        "{{ \"lo_hz\": {}, \"hi_hz\": {}, \"gain_db\": {} }}",
                        num(*lo),
                        num(*hi),
                        num(*g)
                    )
                })
                .collect();
            let lo = bands.iter().map(|b| b.0).fold(f64::INFINITY, f64::min);
            let hi = bands.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
            (
                none(),
                none(),
                num(lo),
                num(hi),
                format!("[{}]", each.join(", ")),
            )
        }
    };
    format!(
        "{{ \"key\": {}, \"kind\": \"{}\", \"hz\": {hz}, \"glide\": {}, \"stretch\": \
         {stretch}, \"harmonics\": {}, \"lo_hz\": {lo}, \"hi_hz\": {hi}, \"bands\": {bands} }}",
        atom.key,
        atom.kind(),
        num(atom.glide()),
        atom.harmonics()
    )
}

/// One progress line per pass, its move beside it and the moves it skipped, their nodes
/// refused, the first (pass 0) with the grid's shape.
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
        "{{ \"pass\": {}, \"action\": \"{}\", \"adjust\": {}, \"atom\": {}, \"from_db\": {}, \"to_db\": {}, \
         \"delta_dl_bits\": {}, \"dl_bits\": {}, \"atom_bits\": {}, \"residual_bits\": {}, \
         \"residual_db\": [{}], \"envelopes\": {}, \"skipped\": {}, \"seconds\": {}{geometry} }}",
        line.pass,
        line.action.name(),
        line.tweak
            .map_or_else(|| NONE.to_string(), |t| text(t.name())),
        line.atom.as_ref().map_or_else(|| NONE.to_string(), atom),
        line.from_db.map_or_else(|| NONE.to_string(), num),
        line.to_db.map_or_else(|| NONE.to_string(), num),
        num(line.delta_bits),
        num(line.atom_bits + line.residual_bits),
        num(line.atom_bits),
        num(line.residual_bits),
        rows.join(", "),
        list(&line.envelopes, |(key, db)| {
            let rows: Vec<String> = db.iter().map(|row| numbers(row)).collect();
            format!("{{ \"key\": {key}, \"db\": [{}] }}", rows.join(", "))
        }),
        list(&line.skipped, atom),
        num(line.seconds)
    )
}

/// The answer's `data`: the fit's counts, DL, time and atoms, the paths the CLI wrote, and its
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
    format!(
        "{{ \"fit\": {{ \"rate_hz\": {}, \"passes\": {}, \"adds\": {}, \"adjusts\": {}, \
         \"adjusts_by\": {{ {} }}, \"partials\": {}, \"harmonics\": {}, \"noises\": {}, \"shaped\": {}, \"dl_bits\": {}, \
         \"atom_bits\": {}, \"residual_bits\": {}, \"gain_db\": {}, \"stopped\": {}, \"secs\": {}, \
         \"atoms\": {}, \"wrote\": {} }}{files} }}",
        fit.clip.rate,
        fit.passes,
        fit.adds,
        fit.adjusts,
        Tweak::ALL
            .iter()
            .zip(fit.tweaks)
            .map(|(t, n)| format!("{}: {n}", text(t.name())))
            .collect::<Vec<_>>()
            .join(", "),
        count("partial"),
        count("harmonic"),
        count("noise"),
        count("shaped"),
        num(fit.atom_bits + fit.residual_bits),
        num(fit.atom_bits),
        num(fit.residual_bits),
        num(fit.gain_db),
        fit.stopped,
        num(secs),
        list(&fit.atoms, atom),
        strings(wrote)
    )
}
