// Concern: groups the readings taken off samples, and declares the three things one can consume | Non-concern: which reading consumes which (sva-engine's query.rs) | IO: none

pub mod alias;
pub mod bands;
pub mod crest;
pub mod envelope;
pub mod formants;
pub mod harmony;
pub mod ledger;
pub mod loudness;
pub mod modulation;
pub mod onsets;
pub mod peaks;
pub mod periodicity;
pub mod pitch;
pub mod self_similarity;
pub mod sliding;
pub mod spectrum;
pub mod stereo;
pub(crate) mod stiff;

pub(crate) fn held(planes: &[&[f64]]) -> usize {
    let len = planes
        .first()
        .expect("a reading reads at least one channel")
        .len();
    assert!(
        planes.iter().all(|p| p.len() == len),
        "a reading's channels are one length"
    );
    len
}

/// Every channel in power, as `envelope` reads it: summed, then divided by the count.
pub(crate) fn averaged(per_channel: Vec<Vec<f64>>) -> Vec<f64> {
    let count = per_channel.len() as f64;
    let mut each = per_channel.into_iter();
    let mut out = each.next().unwrap_or_default();
    for more in each {
        for (sum, x) in out.iter_mut().zip(more) {
            *sum += x;
        }
    }
    out.iter_mut().for_each(|sum| *sum /= count);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consumes {
    ClosedForm,
    Buffer,
    Frames,
}
