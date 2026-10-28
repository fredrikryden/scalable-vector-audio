// Concern: states that the fit meets every hard clip by fitting it or refusing it under a stable code, never a panic | Non-concern: which sources it finds (fit.rs) | IO: (a clip) -> a fit or a code

use std::cell::Cell;
use std::f64::consts::TAU;
use std::pin::Pin;

use sva_core::CliError;
use sva_engine::{Abandon, Never};
use sva_solve::fit::{Fit, Line, Options, SPL_DB, fit, mono};

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

/// The code a refusal names.
fn code(e: &CliError) -> String {
    match e {
        CliError::Usage { code, .. } => code.to_string(),
        other => format!("{other:?}"),
    }
}

/// `samples` at `rate` fitted at the default threshold, or the code it refuses under; each
/// pass it made lowering the DL.
fn met(samples: &[f64], rate: u32) -> Result<Fit, String> {
    let mut lines: Vec<Line> = Vec::new();
    let fitted = now(fit(
        samples,
        rate,
        SPL_DB,
        &Options::default(),
        &mut |line| lines.push(line.clone()),
        &Never,
        &|| 0.0,
    ));
    let dl: Vec<f64> = lines
        .iter()
        .map(|l| l.source_bits + l.residual_bits)
        .collect();
    assert!(dl.iter().all(|v| v.is_finite()), "{dl:?}");
    assert!(dl.windows(2).all(|w| w[1] < w[0]), "{dl:?}");
    fitted.map_err(|e| code(&e))
}

/// `samples` at `rate` fits, its DL finite.
fn fits(samples: &[f64], rate: u32) -> Fit {
    let fit = met(samples, rate).unwrap_or_else(|code| panic!("the clip fits: {code}"));
    assert!((fit.source_bits + fit.residual_bits).is_finite());
    fit
}

fn secs(rate: u32, secs: f64) -> usize {
    libm::round(f64::from(rate) * secs) as usize
}

fn sine(rate: u32, len: usize, hz: f64, amplitude: f64) -> Vec<f64> {
    (0..len)
        .map(|n| amplitude * libm::sin(TAU * hz * n as f64 / f64::from(rate)))
        .collect()
}

/// White noise of unit variance by a fixed draw (splitmix64, Box-Muller).
fn white(len: usize) -> Vec<f64> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..len)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            libm::sqrt(-2.0 * libm::log(u)) * libm::cos(TAU * v)
        })
        .collect()
}

/// A low rate, so each fit is quick; `a_clip_fits_at_each_common_rate` takes the others.
const RATE: u32 = 16_000;

#[test]
fn silence_fits_with_no_source() {
    let fit = fits(&vec![0.0; secs(RATE, 0.5)], RATE);
    assert!(fit.sources.is_empty(), "{:?}", fit.sources);
}

#[test]
fn a_dc_offset_fits() {
    fits(&vec![0.5; secs(RATE, 0.5)], RATE);
}

#[test]
fn a_full_scale_square_wave_fits() {
    let square: Vec<f64> = sine(RATE, secs(RATE, 0.3), 1_100.0, 1.0)
        .iter()
        .map(|s| if *s >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    let fit = fits(&square, RATE);
    assert!(!fit.sources.is_empty());
}

#[test]
fn a_clip_of_twenty_milliseconds_fits() {
    fits(&sine(RATE, secs(RATE, 0.02), 440.0, 0.3), RATE);
}

#[test]
fn a_stereo_clip_fits_its_channels_averaged() {
    let len = secs(RATE, 0.5);
    let planes: Vec<Vec<f32>> = [(440.0, 0.3), (660.0, 0.2)]
        .iter()
        .map(|(hz, a)| sine(RATE, len, *hz, *a).iter().map(|s| *s as f32).collect())
        .collect();
    let fit = fits(&mono(&planes), RATE);
    assert!(!fit.sources.is_empty());
}

#[test]
fn a_clip_fits_at_each_common_rate() {
    for rate in [22_050, 44_100, 48_000, 96_000] {
        let fit = fits(&sine(rate, secs(rate, 0.5), 440.0, 0.3), rate);
        assert!(!fit.sources.is_empty(), "at {rate} Hz");
    }
}

#[test]
fn white_noise_fits() {
    let noise: Vec<f64> = white(secs(RATE, 0.5)).iter().map(|s| 0.1 * s).collect();
    let fit = fits(&noise, RATE);
    assert!(!fit.sources.is_empty());
}

#[test]
fn an_impulse_fits() {
    let mut impulse = vec![0.0; secs(RATE, 0.5)];
    let middle = impulse.len() / 2;
    impulse[middle] = 1.0;
    fits(&impulse, RATE);
}

#[test]
fn a_clip_of_no_sample_or_one_not_finite_refuses_by_name() {
    assert_eq!(met(&[], RATE).err().as_deref(), Some("solve.empty_clip"));
    let mut held = sine(RATE, secs(RATE, 0.1), 440.0, 0.3);
    held[7] = f64::NAN;
    assert_eq!(met(&held, RATE).err().as_deref(), Some("solve.not_finite"));
    held[7] = f64::INFINITY;
    assert_eq!(met(&held, RATE).err().as_deref(), Some("solve.not_finite"));
}

/// A rate whose bins all lie under the feasible set's lowest Hz refuses by name.
#[test]
fn a_rate_putting_no_bin_in_the_feasible_set_refuses_by_name() {
    for rate in [1, 10, 100] {
        let held = sine(rate, rate as usize, f64::from(rate) / 5.0, 0.3);
        assert_eq!(
            met(&held, rate).err().as_deref(),
            Some("solve.rate_too_low"),
            "at {rate} Hz"
        );
    }
}

/// Stops once a pass has been heard.
struct Once<'a>(&'a Cell<bool>);

impl Abandon for Once<'_> {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(std::future::ready(self.0.get()))
    }
}

/// A minute's clip makes its first pass, the fit stopped there.
#[test]
fn a_minute_long_clip_makes_its_first_pass() {
    let rate = 16_000;
    let samples = sine(rate, secs(rate, 60.0), 440.0, 0.3);
    let passed = Cell::new(false);
    let mut heard = 0;
    let fitted = now(fit(
        &samples,
        rate,
        SPL_DB,
        &Options::default(),
        &mut |line| {
            heard += 1;
            passed.set(line.pass >= 1);
        },
        &Once(&passed),
        &|| 0.0,
    ))
    .unwrap_or_else(|e| panic!("the clip fits: {}", code(&e)));
    assert_eq!(heard, 2);
    assert_eq!(fitted.passes, 1);
}
