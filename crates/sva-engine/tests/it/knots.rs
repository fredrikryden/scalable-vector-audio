// Concern: proves a knot table is its lines at every instant, its integral their exact area, and what it refuses | Non-concern: the token bound (sva-ast) | IO: (a composition) -> samples or a refusal
use crate::fixtures::graph_of;
use sva_engine::{RenderConfig, Tier, render};

const RATE: u32 = 8_000;

fn samples(body: &str, secs: f64) -> Result<Vec<f64>, sva_engine::EngineError> {
    let g = graph_of("knots", &[("node", body)]);
    let held = render(
        &g,
        "node",
        RenderConfig::seconds(RATE, secs),
        &Tier::default(),
    )?;
    let root = held.id("node").expect("the root");
    Ok(held.output(root).expect("a buffer").plane(0).to_vec())
}

/// A jump reads its right side; a read at `t - d` moves every knot `d` later.
#[test]
fn a_knot_table_is_its_lines_at_every_sample_and_zero_outside() {
    let (ts, vs) = ([0.01, 0.02, 0.05, 0.05, 0.1], [0.0, 1.0, -0.5, 0.9, 0.0]);
    let at = |t: f64, by: f64| -> f64 {
        let ts = ts.map(|x| x + by);
        if t < ts[0] || t >= ts[4] {
            return 0.0;
        }
        let k = (0..4).rev().find(|k| ts[*k] <= t).expect("a segment");
        vs[k] + (vs[k + 1] - vs[k]) * (t - ts[k]) / (ts[k + 1] - ts[k])
    };
    let table = "0.01s, 0, 0.02s, 1, 0.05s, -0.5, 0.05s, 0.9, 0.1s, 0";
    let got = samples(&format!("pwl(t, {table})\n"), 0.12).expect("a table");
    for (n, v) in got.iter().enumerate() {
        let want = at(n as f64 / f64::from(RATE), 0.0);
        assert!((v - want).abs() < 1e-12, "sample {n}: {v} against {want}");
    }
    assert_eq!(got[400], 0.9, "the jump at 0.05 s reads its right side");
    let nudged = samples(&format!("pwl(t - 0.1s, {table})\n"), 0.3).expect("a moved table");
    assert_eq!(
        nudged[1_200], 0.9,
        "0.05 s + 0.1 s is sample 1200 however doubles round"
    );
    assert!(nudged[1_199] < 0.0, "the sample before reads the left side");
    let late = samples(&format!("pwl(t - 0.25s, {table})\n"), 0.5).expect("a late table");
    for (n, v) in late.iter().enumerate() {
        let want = at(n as f64 / f64::from(RATE), 0.25);
        assert!(
            (v - want).abs() < 1e-12,
            "late sample {n}: {v} against {want}"
        );
    }
}

/// A linear glide from 440 Hz to 880 Hz over a second turns `440 t + 220 t^2` cycles.
#[test]
fn a_knot_integral_is_the_area_under_its_lines_and_a_glides_phase() {
    let glide = samples("cos(2*pi*pwl_integral(t, 0s, 440hz, 1s, 880hz))\n", 1.0).expect("a glide");
    for (n, v) in glide.iter().enumerate() {
        let t = n as f64 / f64::from(RATE);
        let want = (std::f64::consts::TAU * (440.0 * t + 220.0 * t * t)).cos();
        assert!((v - want).abs() < 1e-9, "sample {n}: {v} against {want}");
    }
    let area = samples(
        "crop(pwl_integral(t, 0s, 0, 0.5s, 2, 0.5s, -1, 1s, -1), 0s, 2s)\n",
        2.0,
    )
    .expect("an area");
    assert!((area[4_000] - 0.5).abs() < 1e-12, "{}", area[4_000]);
    assert!((area[12_000] - 0.0).abs() < 1e-12, "{}", area[12_000]);
}

#[test]
fn a_table_naming_no_curve_refuses_as_bad_knots() {
    for bad in [
        "pwl(t, 0s, 0, 1s, 1, 2s)",
        "pwl(t, 1s, 0, 0.5s, 1)",
        "pwl(t, 1s, 0, 1s, 1)",
        "pwl(t, 0s, 0, 1s, t)",
        "pwl_integral(t, 0s, 0, 1s, inf)",
    ] {
        let e = samples(&format!("crop({bad}, 0s, 1s)\n"), 0.1).expect_err(bad);
        assert_eq!(e.code(), "engine.bad_knots", "{bad}: {e}");
    }
}
