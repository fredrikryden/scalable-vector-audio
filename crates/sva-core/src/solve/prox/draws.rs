// Concern: each line's key, phase offset and noise seed per draw, the nuisance a priced fit's loss averages | Non-concern: reading or rendering either | IO: (State) -> keys; (key, draw) -> phase, seed

use std::collections::BTreeSet;

use super::price::reads;
use super::state::{Slot, State};

fn splitmix(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn fold(words: impl IntoIterator<Item = u64>) -> u64 {
    words.into_iter().fold(0, |h, w| splitmix(h ^ w))
}

/// Each line's key: its onset, its bound fields no `--vary` names that its node reads, and its
/// rank among lines agreeing on both.
pub fn keys(s: &State) -> Vec<u64> {
    let mut seen: Vec<u64> = Vec::new();
    (0..s.model.lines.len())
        .map(|l| {
            let line = &s.model.lines[l];
            let read = reads(&s.model.nodes[&line.node].body, &BTreeSet::new());
            let slots = s.slots(l);
            let mut fixed: Vec<(&str, f64)> = line
                .fields
                .iter()
                .filter(|(name, _)| {
                    read.contains(name)
                        && slots
                            .iter()
                            .any(|(n, slot)| n == name && matches!(slot, Slot::Fixed(_)))
                })
                .map(|(name, v)| (name.as_str(), *v))
                .collect();
            fixed.sort_by(|a, b| a.0.cmp(b.0));
            let placed = fold(
                std::iter::once(line.at.to_bits()).chain(
                    fixed
                        .iter()
                        .flat_map(|(name, v)| [fold(name.bytes().map(u64::from)), v.to_bits()]),
                ),
            );
            let rank = seen.iter().filter(|p| **p == placed).count() as u64;
            seen.push(placed);
            fold([placed, rank])
        })
        .collect()
}

fn mix(key: u64, draw: usize, stream: u64) -> u64 {
    fold([key, stream, draw as u64])
}

pub fn phase(key: u64, draw: usize) -> f64 {
    std::f64::consts::TAU * (mix(key, draw, 0) >> 11) as f64 / (1u64 << 53) as f64
}

pub fn seed(key: u64, draw: usize) -> u64 {
    mix(key, draw, 1) >> 33
}
