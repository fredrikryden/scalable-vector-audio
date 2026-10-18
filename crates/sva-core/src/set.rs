// Concern: lays the variables a caller sets over a composition | Non-concern: what a variable's text means (sva-ast) | IO: (source, sets) -> an Overlay or CliError

use std::collections::BTreeMap;

use sva_ast::{Overlay, RESERVED_VARIABLES, Source, VARIABLES, variable};

use crate::cli_error::CliError;

pub fn set(written: &str) -> Result<(String, String), CliError> {
    match written.split_once('=') {
        Some((name, text)) => Ok((name.trim().to_string(), text.to_string())),
        None => Err(CliError::usage(
            "cli.bad_value",
            format!(
                "`--set {written}` has no `=`; it takes a variable's name and its text, as \
                 `--set key=G3`"
            ),
        )),
    }
}

/// `source` with each variable `sets` names, bare as `key` or `energy`, holding its text
/// instead of its file's.
pub fn overlaid<'a>(
    source: &'a dyn Source,
    sets: &[(String, String)],
) -> Result<Overlay<'a>, CliError> {
    let mut over = BTreeMap::new();
    for (name, text) in sets {
        if let Some(bare) = name.strip_prefix(&format!("{VARIABLES}/")) {
            return Err(CliError::usage(
                "render.set_path",
                format!(
                    "`{name}` is the variable's file; a set names the variable alone, as \
                     `{bare}={text}`"
                ),
            ));
        }
        let held = source.get(&variable(name)).map_err(CliError::Io)?.is_some();
        if !RESERVED_VARIABLES.contains(&name.as_str()) && !held {
            let listed = source.paths().map_err(CliError::Io)?;
            let mut there: Vec<&str> = listed
                .found
                .iter()
                .filter_map(|p| p.strip_prefix(&format!("{VARIABLES}/")))
                .collect();
            there.sort_unstable();
            return Err(CliError::usage(
                "render.set_unreserved",
                format!(
                    "`{name}` names no file under `{VARIABLES}/`; a render sets a variable a file \
                     there holds, or a reserved one ({}); the variables there are {}",
                    RESERVED_VARIABLES.join(", "),
                    match there.is_empty() {
                        true => "none".to_string(),
                        false => format!("`{}`", there.join("`, `")),
                    }
                ),
            ));
        }
        if over.insert(variable(name), format!("{text}\n")).is_some() {
            return Err(CliError::usage(
                "render.set_twice",
                format!("`{name}` is set twice; set each variable once"),
            ));
        }
    }
    Ok(Overlay::new(source, over))
}
