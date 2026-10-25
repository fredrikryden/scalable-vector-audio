// Concern: what `internal` owes silence, a tone, a wider ear, two channels, element rates, composed bins, a known ripple | Non-concern: the curves (measure_hearing.rs) | IO: (samples, s) -> asserts

use std::f64::consts::PI;

use sva_samples::measure::hearing::{
    WINDOW_SECS, WINDOW_SPLITS_HZ, cam, ear_gain_db, threshold_excitation_db,
};
use sva_samples::measure::internal::modulation::{Band, Element};
use sva_samples::measure::internal::{Ear, Internal, measure};

const RATE: f64 = 48_000.0;
const SPL: f64 = 100.0;

/// A sine at `hz`, `secs` long, `sl` dB above the excitation at threshold where it peaks.
fn tone(hz: f64, sl: f64, secs: f64) -> Vec<f64> {
    let spl = sl + threshold_excitation_db(hz) - ear_gain_db(hz);
    let amp = libm::pow(10.0, (spl - SPL) / 20.0);
    (0..(secs * RATE) as usize)
        .map(|n| amp * libm::sin(std::f64::consts::TAU * hz * n as f64 / RATE))
        .collect()
}

fn settled(read: &Internal, secs: f64) -> Vec<f64> {
    read.channels
        .iter()
        .map(|c| {
            let lowpass = &c.bands[0];
            assert_eq!(lowpass.band.element, Element::Lowpass);
            lowpass.elements[libm::round(secs * lowpass.rate_hz) as usize]
        })
        .collect()
}

#[test]
fn silence_reads_zero_in_every_element_at_each_bands_rate() {
    let read = measure(&[&vec![0.0; 24_000]], RATE, 0.0, SPL);
    assert_eq!(read.channels.len(), 149);
    for channel in &read.channels {
        for band in &channel.bands {
            let count = libm::floor(500.0 * band.rate_hz * read.hop_secs) as usize + 1;
            assert_eq!(band.elements.len(), count, "{} Hz", channel.hz);
            assert!(band.elements.iter().all(|e| *e == 0.0));
        }
    }
}

/// V is dB over the excitation at threshold: 10 dB more level reads 10 dB more, and the
/// window's spread across the filter costs a 40 dB SL tone about a dB at its own channel.
#[test]
fn a_steady_tone_reads_its_level_over_threshold_in_the_channel_centred_on_it() {
    let read = measure(&[&[0.0]], RATE, 0.0, SPL);
    let centre = read
        .channels
        .iter()
        .position(|c| c.hz > 1000.0)
        .expect("a channel above 1 kHz");
    let hz = read.channels[centre].hz;
    let level = |sl: f64| settled(&measure(&[&tone(hz, sl, 1.0)], RATE, 0.0, SPL), 0.9)[centre];
    let (low, high) = (level(40.0), level(50.0));
    assert!((high - low - 10.0).abs() < 0.2, "{low} then {high}");
    assert!((low - 40.0).abs() < 2.0, "{low}");
}

/// The Cam span within 3 dB of the excitation's peak.
fn width_cam(read: &Internal) -> f64 {
    let pattern: Vec<f64> = read
        .channels
        .iter()
        .zip(settled(read, 1.0))
        .map(|(c, v)| {
            libm::pow(10.0, threshold_excitation_db(c.hz) / 10.0)
                * (libm::pow(10.0, v / 10.0) - 1.0)
        })
        .collect();
    let cams: Vec<f64> = read.channels.iter().map(|c| cam(c.hz)).collect();
    let peak = pattern.iter().copied().fold(0.0, f64::max);
    let top = pattern.iter().position(|e| *e == peak).expect("a peak");
    let db = |i: usize| 10.0 * libm::log10(pattern[i] / peak);
    let crossing = |order: Vec<usize>| {
        let pair = order
            .windows(2)
            .find(|w| db(w[1]) < -3.0)
            .expect("a 3 dB fall");
        let (j, i) = (pair[0], pair[1]);
        cams[j] + (cams[i] - cams[j]) * (-3.0 - db(j)) / (db(i) - db(j))
    };
    crossing((top..pattern.len()).collect()) - crossing((0..=top).rev().collect())
}

/// At s = 1 the window's own spread, about an ERB, adds to the filters' width, and shrinks as
/// the windows lengthen with s, so doubling s from 2 and from 4 doubles the width.
#[test]
fn a_wider_ear_reads_a_tones_excitation_wider_by_its_scale_in_cam() {
    let x = tone(1000.0, 50.0, 1.2);
    let widths: Vec<f64> = [2.0, 4.0, 8.0]
        .iter()
        .map(|s| width_cam(&Ear::of(RATE, SPL, *s).read(&[&x], 0.0)))
        .collect();
    for pair in widths.windows(2) {
        let ratio = pair[1] / pair[0] / 2.0;
        assert!((ratio - 1.0).abs() < 0.1, "{widths:?}");
    }
}

fn response(band: &Band, window_secs: f64, nu: f64) -> f64 {
    let x = nu * window_secs;
    let sinc = |u: f64| match u == 0.0 {
        true => 1.0,
        false => libm::sin(PI * u) / (PI * u),
    };
    let w = sinc(x)
        + 2.0 / 3.0 * (sinc(x - 1.0) + sinc(x + 1.0))
        + 1.0 / 6.0 * (sinc(x - 2.0) + sinc(x + 2.0));
    let filter = match band.element {
        Element::Lowpass => 1.0 / (1.0 + libm::pow(nu / 2.5, 4.0)),
        _ => 1.0 / (1.0 + libm::pow((nu - band.hz) / (band.bandwidth_hz / 2.0), 2.0)),
    };
    w * w / (1.0 + libm::pow(nu / 150.0, 2.0)) * filter
}

fn bandwidth_by_grid(band: &Band, window_secs: f64) -> f64 {
    let top = 50.0 * (band.hz + band.bandwidth_hz + 150.0 + 2.0 / window_secs);
    let panels = 2_000_000;
    let step = top / panels as f64;
    let (mut i1, mut i2) = (0.0, 0.0);
    for k in 0..=panels {
        let weight = match k {
            0 => 1.0,
            k if k == panels => 1.0,
            k if k % 2 == 1 => 4.0,
            _ => 2.0,
        };
        let h = response(band, window_secs, k as f64 * step);
        i1 += weight * h;
        i2 += weight * h * h;
    }
    libm::pow(i1 * step / 3.0, 2.0) / (i2 * step / 3.0)
}

/// B summed here on a fixed grid; the design counts 198 k elements a second at s = 1.
#[test]
fn each_band_holds_twice_its_statistical_bandwidth_in_elements_per_second() {
    let read = measure(&[&vec![0.0; RATE as usize]], RATE, 0.0, SPL);
    let mut checked: Vec<(f64, Band)> = Vec::new();
    let mut total = 0;
    for channel in &read.channels {
        for sampled in &channel.bands {
            total += sampled.elements.len();
            let key = (channel.window_secs, sampled.band);
            if checked.contains(&key) {
                continue;
            }
            checked.push(key);
            let rate = (2.0 * bandwidth_by_grid(&sampled.band, channel.window_secs))
                .min(1.0 / read.hop_secs);
            assert!(sampled.rate_bound_hz <= 1e-9 * rate, "{sampled:?}");
            assert!(
                (sampled.rate_hz - rate).abs() <= sampled.rate_bound_hz + 1e-9 * rate,
                "{:?} in {} s: {} against {rate}",
                sampled.band,
                channel.window_secs,
                sampled.rate_hz
            );
        }
    }
    assert!((total as f64 / 198_000.0 - 1.0).abs() < 0.01, "{total}");
}

/// Every channel's power averaged, as every reading off samples takes it: a channel copied to
/// two reads as itself, and a tone beside silence as that tone 3 dB down.
#[test]
fn two_channels_read_as_their_averaged_power() {
    let x = tone(1000.0, 40.0, 0.2);
    let halved: Vec<f64> = x
        .iter()
        .map(|v| v * std::f64::consts::FRAC_1_SQRT_2)
        .collect();
    let silence = vec![0.0; x.len()];
    let alike = |a: &[&[f64]], b: &[&[f64]]| {
        let (a, b) = (measure(a, RATE, 0.0, SPL), measure(b, RATE, 0.0, SPL));
        for (p, q) in a.channels.iter().zip(&b.channels) {
            for (u, v) in p.bands.iter().zip(&q.bands) {
                for (e, f) in u.elements.iter().zip(&v.elements) {
                    assert!((e - f).abs() <= 1e-9 * e.abs().max(1.0), "{e} against {f}");
                }
            }
        }
    };
    alike(&[&x, &x], &[&x]);
    alike(&[&silence, &x], &[&halved]);
}

/// A bin on a split (500 Hz at 32 kHz) takes the upper range's window; 0 Hz to Nyquist.
#[test]
fn each_composed_bin_comes_from_its_own_ranges_window() {
    let rate = 32_000.0;
    let ear = Ear::of(rate, SPL, 1.0);
    let read = ear.compose(0, &mut |frame| {
        let secs = (frame.weights().len() + 1) as f64 / rate;
        frame.hz().iter().map(|hz| (*hz, secs)).collect()
    });
    let hz = ear.hz();
    assert_eq!(read.len(), hz.len());
    assert_eq!((hz[0], hz[hz.len() - 1]), (0.0, rate / 2.0));
    assert!(hz.windows(2).all(|w| w[0] < w[1]), "{hz:?}");
    assert!(hz.contains(&500.0));
    for ((from, secs), f) in read.iter().zip(hz) {
        let range = WINDOW_SPLITS_HZ.partition_point(|split| split <= f);
        assert_eq!(from, f);
        assert_eq!(*secs, WINDOW_SECS[range], "{f} Hz");
    }
}

/// MODEL's known limit: short windows' +f and -f sidelobes beat at 2f, aliased by the 1 ms hop.
#[test]
fn a_steady_tone_ripples_by_the_published_models_stated_limit() {
    let x = tone(1001.3, 80.0, 0.625);
    let ear = Ear::of(RATE, SPL, 1.0);
    let read = ear.read(&[&x], 0.0);
    let channel = read
        .channels
        .iter()
        .position(|c| libm::round(c.hz) == 5302.0)
        .expect("a 5302 Hz channel");
    let db: Vec<f64> = (100..520)
        .map(|m| {
            let e = ear.excite(m, &ear.compose(m, &mut |frame| frame.power(&[&x], 0)));
            10.0 * libm::log10(e[channel])
        })
        .collect();
    let (low, high) = db
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), e| (a.min(*e), b.max(*e)));
    assert!((high - low - 3.43).abs() < 0.005, "{} dB", high - low);
}

/// `elements` on the reading's own excitation answers what `read` maps through each band's
/// element, element `j` at the frame `round(j / per_frame)`, and each kernel is its band's:
/// 1 at 0 Hz for the lowpass, even for a real band, a magnitude band's over twice as large at
/// its centre as at its mirror.
#[test]
fn elements_are_the_readings_outputs_and_each_kernel_its_bands() {
    let rate = 8_000.0;
    let x: Vec<f64> = tone(1000.0, 40.0, 0.25).into_iter().step_by(6).collect();
    let ear = Ear::of(rate, SPL, 1.0);
    let read = ear.read(&[&x], 0.0);
    let mut frames = Vec::new();
    let outputs = ear.elements(x.len(), &mut |m| {
        frames.push(m);
        ear.excite(m, &ear.compose(m, &mut |frame| frame.power(&[&x], 0)))
    });
    let (first, last) = ear.frames(x.len());
    assert_eq!(frames, (first..=last).collect::<Vec<_>>());
    for (c, (heard, bands)) in read.channels.iter().zip(&outputs).enumerate() {
        for (k, (sampled, ys)) in heard.bands.iter().zip(bands).enumerate() {
            let per_frame = sampled.rate_hz * read.hop_secs;
            for (j, (e, y)) in sampled.elements.iter().zip(ys).enumerate() {
                assert_eq!(*e, sampled.band.element(*y));
                let at = (j as f64 / per_frame).round_ties_even() as i64;
                assert_eq!(ear.sampled(c, k, j), at);
            }
            let (near, far) = (ear.kernel(c, k, 3.0), ear.kernel(c, k, -3.0));
            match sampled.band.element {
                Element::Lowpass => {
                    assert!((ear.kernel(c, k, 0.0) - 1.0).abs() < 1e-12);
                    assert!((near - far).abs() <= 1e-15 * near);
                }
                Element::Real => assert!((near - far).abs() <= 1e-15 * near),
                Element::Magnitude => {
                    let hz = sampled.band.hz;
                    let (up, down) = (ear.kernel(c, k, hz), ear.kernel(c, k, -hz));
                    assert!(up > 2.0 * down, "{hz} Hz: {up} against {down}");
                }
            }
        }
    }
}
