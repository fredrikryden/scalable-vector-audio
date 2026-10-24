// Concern: the grammar of a node's `Models | Neglects | IO | Tags` line | Non-concern: finding that line, how loudly to object (sva-cli) | IO: (&str) -> DocComment or why not

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocComment {
    pub models: String,
    pub neglects: String,
    pub io_in: String,
    pub io_out: String,
    pub tags: Vec<String>,
}

/// A leading `;` is tolerated, so a caller may pass the node's source line as written.
pub fn parse(line: &str) -> Result<DocComment, String> {
    let content = line.trim_start().trim_start_matches(';').trim();
    let fields: Vec<&str> = content.split(" | ").collect();
    let [models, neglects, io, tags] = fields.as_slice() else {
        return Err("expected exactly four ` | `-delimited fields".to_string());
    };

    let models = models
        .strip_prefix("Models:")
        .ok_or("first field must start with `Models:`")?
        .trim();
    if models.is_empty() {
        return Err("`Models:` field is empty".to_string());
    }

    let neglects = neglects
        .strip_prefix("Neglects:")
        .ok_or("second field must start with `Neglects:`")?
        .trim();
    if neglects.is_empty() {
        return Err("`Neglects:` field is empty".to_string());
    }

    let io = io
        .strip_prefix("IO:")
        .ok_or("third field must start with `IO:`")?
        .trim();
    let Some((input, output)) = io.split_once("->") else {
        return Err("`IO:` field must be `<input> -> <output>`".to_string());
    };
    let (input, output) = (input.trim(), output.trim());
    if input.is_empty() || output.is_empty() {
        return Err("`IO:` field's input and output must both be non-empty".to_string());
    }

    let tags = tags
        .strip_prefix("Tags:")
        .ok_or("fourth field must start with `Tags:`")?
        .trim();
    if tags.is_empty() {
        return Err("`Tags:` field is empty".to_string());
    }
    let tags: Vec<&str> = tags.split(',').map(str::trim).collect();
    if tags.iter().any(|tag| tag.is_empty()) {
        return Err("`Tags:` field has an empty tag between commas".to_string());
    }

    Ok(DocComment {
        models: models.to_string(),
        neglects: neglects.to_string(),
        io_in: input.to_string(),
        io_out: output.to_string(),
        tags: tags.into_iter().map(String::from).collect(),
    })
}

impl fmt::Display for DocComment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Models: {} | Neglects: {} | IO: {} -> {} | Tags: {}",
            self.models,
            self.neglects,
            self.io_in,
            self.io_out,
            self.tags.join(", ")
        )
    }
}
