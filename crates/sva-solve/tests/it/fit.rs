// Concern: states the fit's refusal codes and that its composition reads back as a resume | Non-concern: which sources a fit finds | IO: (a clip) -> a code or a resume

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use sva_core::CliError;
use sva_engine::Never;
use sva_solve::fit::{Fit, Options, Resume, SPL_DB, fit};
use sva_solve::sources::files_at;

const RATE: u32 = 16_000;

fn now<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::noop();
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(waker))
    {
        std::task::Poll::Ready(out) => out,
        std::task::Poll::Pending => panic!("nothing to wait on with no signal"),
    }
}

fn code(e: &CliError) -> String {
    match e {
        CliError::Usage { code, .. } => code.to_string(),
        other => format!("{other:?}"),
    }
}

fn met(samples: &[f64], rate: u32, options: &Options) -> Result<Fit, String> {
    now(fit(
        samples,
        rate,
        SPL_DB,
        options,
        &mut |_| {},
        &Never,
        &|| 0.0,
    ))
    .map_err(|e| code(&e))
}

fn sine(rate: u32, secs: f64, hz: f64) -> Vec<f64> {
    let len = (secs * f64::from(rate)) as usize;
    (0..len)
        .map(|n| 0.3 * libm::sin(TAU * hz * n as f64 / f64::from(rate)))
        .collect()
}

#[test]
fn a_clip_of_no_sample_or_one_not_finite_refuses_by_name() {
    let options = Options::default();
    assert_eq!(
        met(&[], RATE, &options).err().as_deref(),
        Some("solve.empty_clip")
    );
    let mut held = sine(RATE, 0.1, 440.0);
    for bad in [f64::NAN, f64::INFINITY] {
        held[7] = bad;
        assert_eq!(
            met(&held, RATE, &options).err().as_deref(),
            Some("solve.not_finite")
        );
    }
}

/// A fit is the same on one thread as on four.
#[test]
fn a_fit_is_the_same_on_one_thread_and_on_four() {
    let samples = sine(RATE, 0.3, 440.0);
    let on = |threads: usize| {
        let options = Options {
            threads: NonZeroUsize::new(threads).expect("a thread"),
            ..Options::default()
        };
        let fit = met(&samples, RATE, &options).expect("a sine fits");
        (fit.sources, fit.source_bits, fit.residual_bits)
    };
    assert_eq!(on(1), on(4));
}

#[test]
fn a_rate_putting_no_bin_in_the_feasible_set_refuses_by_name() {
    let held = sine(100, 1.0, 20.0);
    assert_eq!(
        met(&held, 100, &Options::default()).err().as_deref(),
        Some("solve.rate_too_low")
    );
}

/// A fit's composition reads back as its sources, and a resume onto another clip, or from a
/// composition holding no score, refuses by name.
#[test]
fn a_composition_reads_back_and_a_resume_from_another_clip_refuses_by_name() {
    let samples = sine(RATE, 0.3, 440.0);
    let cut = met(&samples, RATE, &Options::default()).expect("a sine fits");
    let files = files_at(&cut.sources, &cut.shared, &cut.clip, cut.gain_db);
    let from = Resume::of(&files, cut.passes).expect("the composition reads back");
    assert_eq!(from.sources, cut.sources);
    let onto = |samples: &[f64]| {
        let options = Options {
            from: Some(from.clone()),
            ..Options::default()
        };
        met(samples, RATE, &options).err()
    };
    assert_eq!(
        onto(&samples[..samples.len() / 2]).as_deref(),
        Some("solve.from_mismatch")
    );
    let louder: Vec<f64> = samples.iter().map(|s| 2.0 * s).collect();
    assert_eq!(onto(&louder).as_deref(), Some("solve.from_mismatch"));
    let held: Vec<(String, String)> = files.into_iter().filter(|(p, _)| p == "master").collect();
    assert_eq!(
        Resume::of(&held, 0).map(|_| ()).map_err(|e| code(&e)),
        Err("solve.from_unread".to_string())
    );
}
