// Concern: lays the reserved variables a caller sets over a composition | Non-concern: what a variable's text means (sva-ast) | IO: (source, sets) -> an Overlay or CliError

use std::collections::BTreeMap;

use sva_ast::{Overlay, RESERVED_VARIABLES, Source, variable};

use crate::cli_error::CliError;

pub fn set(written: &str) -> Result<(String, String), CliError> {
    match written.split_once('=') {
        Some((path, text)) => Ok((path.trim().to_string(), text.to_string())),
        None => Err(CliError::usage(
            "cli.bad_value",
            format!(
                "`--set {written}` has no `=`; it takes a reserved variable and its text, as \
                 `--set variables/key=G3`"
            ),
        )),
    }
}

/// `source` with each reserved variable `sets` names holding its text instead of its file's.
pub fn overlaid<'a>(
    source: &'a dyn Source,
    sets: &[(String, String)],
) -> Result<Overlay<'a>, CliError> {
    let reserved = RESERVED_VARIABLES.map(variable);
    let mut over = BTreeMap::new();
    for (path, text) in sets {
        if !reserved.contains(path) {
            return Err(CliError::usage(
                "render.set_unreserved",
                format!(
                    "`{path}` is no reserved variable; a render sets only {}",
                    reserved.join(", ")
                ),
            ));
        }
        if over.insert(path.clone(), format!("{text}\n")).is_some() {
            return Err(CliError::usage(
                "render.set_twice",
                format!("`{path}` is set twice; set each variable once"),
            ));
        }
    }
    Ok(Overlay::new(source, over))
}
