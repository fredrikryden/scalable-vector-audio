// Concern: states what one ledger entry says about the buffer under it | Non-concern: which nodes are walked (sva-engine) | IO: (buffers, deps) -> asserted entries

use std::collections::BTreeMap;

use sva_samples::measure::ledger::{Sources, Tally};
use sva_samples::{Buffer, SignalKind};

/// A tally handed all of `range` as one stretch.
fn attribute(
    buffers: &BTreeMap<String, Buffer>,
    contributed: &BTreeMap<String, Buffer>,
    deps: &BTreeMap<String, Vec<String>>,
    kinds: &BTreeMap<String, SignalKind>,
    target: &str,
    range: std::ops::Range<usize>,
    depth_limit: usize,
) -> Vec<sva_samples::LedgerEntry> {
    let held: Vec<&Buffer> = buffers.values().chain(contributed.values()).collect();
    let own: BTreeMap<String, usize> = buffers.keys().cloned().zip(0..).collect();
    let mut read = own.clone();
    read.extend(contributed.keys().cloned().zip(buffers.len()..));
    let sources = Sources {
        read: &read,
        own: &own,
        attributed: &contributed.keys().cloned().collect(),
        widths: &held.iter().map(|b| b.width()).collect::<Vec<_>>(),
    };
    let mut tally = Tally::new(&sources, deps, kinds, target, depth_limit);
    let held: Vec<Option<&Buffer>> = held.into_iter().map(Some).collect();
    tally.add(&held, range);
    tally.entries()
}

fn entry(samples: Vec<f64>) -> sva_samples::LedgerEntry {
    let len = samples.len();
    let mut buffers = BTreeMap::new();
    buffers.insert("node".to_string(), Buffer::of_planes(48_000, vec![samples]));
    let mut kinds = BTreeMap::new();
    kinds.insert("node".to_string(), SignalKind::Audio);
    let entries = attribute(
        &buffers,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &kinds,
        "node",
        0..len,
        1,
    );
    let [held] = entries.as_slice() else {
        panic!("one node, one component, one entry: {entries:?}");
    };
    held.clone()
}

/// `clipped` means strictly outside [-1, 1], the bound a `.wav` clamps to: one rule, so two
/// signals at one peak read one verdict.
#[test]
fn a_peak_of_exactly_one_is_not_clipped() {
    let held = entry(vec![0.0, 1.0, -1.0, 0.5]);
    assert_eq!(held.peak, 1.0);
    assert_eq!(
        held.clipped,
        Some(false),
        "full scale is not past full scale"
    );

    let held = entry(vec![0.0, 1.0, -f64::from_bits(1.0f64.to_bits() + 1), 0.5]);
    assert!(held.peak > 1.0, "{}", held.peak);
    assert_eq!(
        held.clipped,
        Some(true),
        "one bit past full scale is past full scale"
    );

    let held = entry(vec![0.0, 0.9999, -0.5]);
    assert_eq!(held.clipped, Some(false));
}
