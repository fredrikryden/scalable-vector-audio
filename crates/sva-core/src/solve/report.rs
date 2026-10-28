// Concern: a fit's progress lines and its answer as JSON, the shapes sva-portal's page reads | Non-concern: running it (fit.rs), where a line goes | IO: (Line | Fit) -> JSON

use super::atoms::{Atom, Sound};
use super::fit::{Action, Fit, Line, Tweak};
use super::optimize::End;
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

/// A pass's move in words: `start`, `add <kind>`, or `<adjust> <kind>`.
fn moved(line: &Line) -> String {
    let kind = line.atom.as_ref().map_or("", |a| a.kind());
    match (line.action, line.tweak) {
        (Action::Start, _) => "start".to_string(),
        (_, Some(tweak)) => format!("{} {kind}", tweak.name()),
        (_, None) => format!("{} {kind}", line.action.name()),
    }
}

/// One progress line per pass, its move beside it and the moves it skipped, their nodes
/// refused, the first (the start, or the pass a fit resumes from) with the grid's shape; its
/// drop, the drop from the empty fit's DL so far, and the share of that this pass's is.
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
    let dl = line.atom_bits + line.residual_bits;
    let total = line.start_bits - dl;
    let share = match total > 0.0 {
        true => num((0.0 - line.delta_bits) / total),
        false => NONE.to_string(),
    };
    format!(
        "{{ \"pass\": {}, \"action\": \"{}\", \"adjust\": {}, \"move\": {}, \"atom\": {}, \"from_db\": {}, \"to_db\": {}, \
         \"delta_dl_bits\": {}, \"drop_bits\": {}, \"total_drop_bits\": {}, \"share\": {share}, \
         \"dl_bits\": {}, \"atom_bits\": {}, \"residual_bits\": {}, \
         \"residual_db\": [{}], \"envelopes\": {}, \"skipped\": {}, \"seconds\": {}, \"pass_secs\": {}{geometry} }}",
        line.pass,
        line.action.name(),
        line.tweak
            .map_or_else(|| NONE.to_string(), |t| text(t.name())),
        text(&moved(line)),
        line.atom.as_ref().map_or_else(|| NONE.to_string(), atom),
        line.from_db.map_or_else(|| NONE.to_string(), num),
        line.to_db.map_or_else(|| NONE.to_string(), num),
        num(line.delta_bits),
        num(0.0 - line.delta_bits),
        num(total),
        num(dl),
        num(line.atom_bits),
        num(line.residual_bits),
        rows.join(", "),
        list(&line.envelopes, |(key, db)| {
            let rows: Vec<String> = db.iter().map(|row| numbers(row)).collect();
            format!("{{ \"key\": {key}, \"db\": [{}] }}", rows.join(", "))
        }),
        list(&line.skipped, atom),
        num(line.seconds),
        num(line.pass_secs)
    )
}

/// A composition's files as a list of `{ path, text }`.
pub fn composition(files: &[(String, String)]) -> String {
    list(files, |(path, body)| {
        format!("{{ \"path\": {}, \"text\": {} }}", text(path), text(body))
    })
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
        format!(", \"files\": {}", composition(files))
    });
    format!(
        "{{ \"fit\": {{ \"rate_hz\": {}, \"passes\": {}, \"adds\": {}, \"adjusts\": {}, \
         \"adjusts_by\": {{ {} }}, \"partials\": {}, \"harmonics\": {}, \"noises\": {}, \"shaped\": {}, \"dl_bits\": {}, \
         \"atom_bits\": {}, \"residual_bits\": {}, \"gain_db\": {}, \"from_pass\": {}, \"tolerance\": {}, \
         \"ended\": \"{}\", \"stopped\": {}, \"secs\": {}, \
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
        fit.from_pass,
        num(fit.tolerance),
        fit.ended.name(),
        fit.ended == End::Stopped,
        num(secs),
        list(&fit.atoms, atom),
        strings(wrote)
    )
}
