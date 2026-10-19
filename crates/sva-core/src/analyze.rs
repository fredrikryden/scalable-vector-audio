// Concern: answers the readings asked of a buffer nothing rendered, or refuses one needing a graph | Non-concern: decoding, routing (sva-cli) | IO: (target, Buffer, Asked) -> Answers or CliError

use sva_engine::{Buffer, PSYCHOACOUSTIC_V1, READINGS, answer_buffer};

use crate::{Answer, Asked, CliError};

/// Refuses a reading a buffer cannot answer alone.
pub fn analyzable(asked: &[Asked]) -> Result<(), CliError> {
    let Some(one) = asked.iter().find(|one| !one.off_samples(true)) else {
        return Ok(());
    };
    let answered: Vec<&str> = READINGS
        .iter()
        .filter(|row| row.reads.off_samples(true))
        .map(|row| row.name)
        .collect();
    Err(CliError::usage(
        "query.not_analyzable",
        format!(
            "`analyze` does not answer `{}` — only {} need no rendered graph",
            one.name,
            answered.join(", ")
        ),
    ))
}

/// Every reading asked of `buffer`, in order.
pub fn analyzed(target: &str, buffer: &Buffer, asked: &[Asked]) -> Result<Vec<Answer>, CliError> {
    analyzable(asked)?;
    asked
        .iter()
        .map(|one| {
            answer_buffer(
                target,
                buffer,
                one.representation(buffer.rate, None)?,
                PSYCHOACOUSTIC_V1.name,
            )
            .map_err(CliError::Engine)
        })
        .collect()
}
