// Concern: states what the stereo image owes antiphase and decorrelated pairs | Non-concern: level per component (src/measure/envelope.rs) | IO: (two planes) -> asserted image

use crate::helpers::drawn;
use sva_samples::measure::stereo::{RAIL_DB, StereoFrame, analyze};

fn image(l: Vec<f64>, r: Vec<f64>) -> StereoFrame {
    analyze(&[&l, &r], 2, 100.0, 0.0, 1.0).overall
}

fn noise(seed: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| drawn(seed, i as i64)).collect()
}

#[test]
fn pure_antiphase_reads_its_width_at_the_rail() {
    let n = noise(7.0, 400);
    let f = image(n.clone(), n.iter().map(|x| -x).collect());
    assert_eq!(f.mid_rms, 0.0, "nothing in the middle");
    assert_eq!(f.width, libm::pow(10f64, RAIL_DB / 20.0));
    assert_eq!(f.mono_db, -RAIL_DB);
}

#[test]
fn decorrelated_components_sit_near_zero_and_lose_3_db_to_fold_down() {
    let f = image(noise(3.0, 20000), noise(4.0, 20000));
    assert!(f.correlation.abs() < 0.05, "correlation {}", f.correlation);
    assert!(
        (f.width - 1.0).abs() < 0.05,
        "S and M are equal: {}",
        f.width
    );
    assert!(
        (f.mono_db + 3.0).abs() < 0.3,
        "summing two independent signals halves the power: {}",
        f.mono_db
    );
}
