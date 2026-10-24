// Concern: what self_similarity owes an A-B-A-B piece, exact and transposed repeats | Non-concern: picking partials, band filters | IO: (samples) -> asserts

use std::f64::consts::TAU;

use sva_samples::measure::self_similarity::{
    Asked, Reading, SelfSimilarity, Stripe, View, measure,
};

const RATE: f64 = 16_000.0;
const CHORD_SECS: f64 = 2.0;
/// C major pentatonic chords, no two alike.
const A: [[i32; 3]; 4] = [[60, 64, 67], [57, 60, 64], [62, 67, 69], [64, 67, 69]];
/// A's a tritone up, reversed: no class or pair in common.
const B: [[i32; 3]; 4] = [[70, 73, 75], [68, 73, 75], [63, 66, 70], [66, 70, 73]];

fn hz(midi: f64) -> f64 {
    440.0 * libm::pow(2f64, (midi - 69.0) / 12.0)
}

fn chords(sections: &[(&[[i32; 3]; 4], i32)]) -> Vec<f64> {
    let each = (CHORD_SECS * RATE) as usize;
    let fade = 0.005 * RATE;
    let mut out = Vec::new();
    for (section, up) in sections {
        for chord in section.iter() {
            out.extend((0..each).map(|i| {
                let t = i as f64 / RATE;
                let edge = (i.min(each - i) as f64 / fade).min(1.0);
                chord
                    .iter()
                    .map(|m| 0.2 * edge * libm::sin(TAU * hz(f64::from(m + up)) * t))
                    .sum::<f64>()
            }));
        }
    }
    out
}

fn asked(cell_secs: f64) -> Asked {
    Asked {
        hop_secs: 0.05,
        cell_secs,
        smooth_secs: 2.0,
        window_secs: 8.0,
        floor_db: -40.0,
        listed: 16,
        curve: false,
    }
}

fn read(samples: &[f64]) -> SelfSimilarity {
    measure(&[samples], RATE, 0.0, asked(0.5)).expect("under the cap")
}

fn view(s: &SelfSimilarity, which: View) -> &Reading {
    s.views
        .iter()
        .find(|v| v.view == which)
        .expect("every view")
}

fn near(a: f64, b: f64, by: f64) -> bool {
    (a - b).abs() <= by
}

/// A-B-A-B, 8 s each: one stripe, A/A and B/B 16 s back, ends within half the smoothing; the
/// top novelty peaks at 8, 16, 24 s.
#[test]
fn an_abab_piece_reads_its_repeats_and_boundaries() {
    let s = read(&chords(&[(&A, 0), (&B, 0), (&A, 0), (&B, 0)]));
    assert_eq!(
        (s.cell_secs, s.frames_per_cell, s.cell_count),
        (0.5, 10, 64)
    );
    let chroma = view(&s, View::Chroma);
    let lead = chroma.stripes[0];
    assert_eq!(lead.shift_semitones, 0, "{lead:?}");
    assert!(near(lead.lag_secs, 16.0, 1e-9), "{lead:?}");
    assert!(near(lead.earlier.0, 0.0, 1.0) && near(lead.earlier.1, 16.0, 1.0));
    assert!(near(lead.later.0, 16.0, 1.0) && near(lead.later.1, 32.0, 1.0));
    assert!(lead.similarity_mean > 0.99, "{lead:?}");
    let exact: Vec<&Stripe> = chroma
        .stripes
        .iter()
        .filter(|st| st.shift_semitones == 0 && st.similarity_mean > 0.9 && st.cells > 2)
        .collect();
    assert_eq!(exact, vec![&lead], "{:#?}", chroma.stripes);
    let mut top: Vec<f64> = chroma.peaks[..3].iter().map(|p| p.t_secs).collect();
    top.sort_by(f64::total_cmp);
    for (found, boundary) in top.iter().zip([8.0, 16.0, 24.0]) {
        assert!(near(*found, boundary, 0.5), "{:?}", chroma.peaks);
    }
}

/// A then A a tone up reads at shift 2, 8 s back; A then A at shift 0.
#[test]
fn a_transposed_repeat_reads_at_its_shift_and_an_exact_one_at_none() {
    let moved = read(&chords(&[(&A, 0), (&A, 2)]));
    let chroma = view(&moved, View::Chroma);
    let lead = chroma.stripes[0];
    assert_eq!(lead.shift_semitones, 2, "{lead:?}");
    assert!(near(lead.lag_secs, 8.0, 1e-9), "{lead:?}");
    assert!(lead.similarity_mean > 0.99, "{lead:?}");
    assert!(
        chroma
            .stripes
            .iter()
            .filter(|st| st.shift_semitones == 0)
            .all(|st| st.similarity_mean * st.cells as f64 <= 0.5 * lead.cells as f64),
        "{:#?}",
        chroma.stripes
    );
    let again = read(&chords(&[(&A, 0), (&A, 0)]));
    let lead = view(&again, View::Chroma).stripes[0];
    assert_eq!(lead.shift_semitones, 0, "{lead:?}");
    assert!(near(lead.lag_secs, 8.0, 1e-9), "{lead:?}");
    assert!(lead.similarity_mean > 0.99, "{lead:?}");
}
