// Concern: states what the discrete spectrum owes tones of known content | Non-concern: the FFT itself (src/fft.rs), the ERB bank (measure_bands.rs) | IO: (samples) -> asserted Hz and dB

use sva_samples::measure::spectrum::{analyze, db, magnitudes, third_octave_edges};

const RATE: f64 = 44_100.0;

fn tone(hz: f64, secs: f64, amp: f64) -> Vec<f64> {
    let len = (secs * RATE) as usize;
    (0..len)
        .map(|i| amp * (std::f64::consts::TAU * hz * i as f64 / RATE).sin())
        .collect()
}

/// A sine's RMS is `amp/sqrt(2)` and every other figure names the one partial it holds.
#[test]
fn one_sine_reads_as_its_own_frequency_level_and_centroid() {
    let s = analyze(&[&tone(1000.0, 1.0, 0.5)[..]], RATE, 4, None);
    assert!(s.frames >= 1 && s.frame_size.is_power_of_two());
    assert!(
        (s.rms - 0.5 / 2f64.sqrt()).abs() < 0.01,
        "a sine's RMS is its amplitude over root two: {}",
        s.rms
    );
    let loudest = s.peaks.first().expect("a sine has a peak");
    assert!(
        (loudest.hz - 1000.0).abs() < s.resolution_hz,
        "within one bin of 1 kHz: {loudest:?}"
    );
    assert!(
        (s.centroid_hz - 1000.0).abs() < 50.0,
        "one partial is its own centre of mass: {}",
        s.centroid_hz
    );
    assert!(
        s.rolloff85_hz >= 1000.0 - s.resolution_hz && s.rolloff85_hz < 2000.0,
        "85% of the energy is at or below the partial: {}",
        s.rolloff85_hz
    );
}

/// Two partials an octave apart, the upper half the amplitude.
#[test]
fn two_partials_are_both_found_and_ordered_by_level() {
    let low = tone(500.0, 1.0, 0.5);
    let high = tone(1000.0, 1.0, 0.25);
    let mixed: Vec<f64> = low.iter().zip(&high).map(|(a, b)| a + b).collect();
    let s = analyze(&[&mixed[..]], RATE, 4, None);

    let [first, second, ..] = s.peaks.as_slice() else {
        panic!("two partials, two peaks: {:?}", s.peaks);
    };
    assert!((first.hz - 500.0).abs() < s.resolution_hz, "{first:?}");
    assert!((second.hz - 1000.0).abs() < s.resolution_hz, "{second:?}");
    assert!(first.db > second.db, "the louder partial is named first");
    assert!(
        (first.db - second.db - 6.02).abs() < 0.5,
        "half the amplitude is 6 dB down: {first:?} {second:?}"
    );
    assert!(s.centroid_hz > 500.0 && s.centroid_hz < 1000.0);
}

/// Silence has no peak to name and no level to state, and says so rather than guessing.
#[test]
fn silence_reads_as_the_floor_with_no_peaks() {
    let s = analyze(&[&vec![0.0; 4096][..]], RATE, 4, None);
    assert_eq!(s.rms, 0.0);
    assert!(s.peaks.is_empty(), "{:?}", s.peaks);
    assert_eq!(db(0.0), db(0.0), "the floor is one value, not a NaN");
    assert!(db(0.0) < -100.0, "and it is a floor: {}", db(0.0));
}

/// Every band reading downstream is sliced on these.
#[test]
fn third_octave_edges_step_by_a_third_of_an_octave_up_to_nyquist() {
    let edges = third_octave_edges(RATE);
    let [(lo, hi), ..] = edges.as_slice() else {
        panic!("at least one band");
    };
    assert!((hi / lo - 2f64.powf(1.0 / 3.0)).abs() < 1e-9, "{lo} {hi}");
    let (_, top) = edges.last().expect("a last band");
    assert!(*top <= RATE / 2.0, "nothing above Nyquist: {top}");
    for pair in edges.windows(2) {
        assert_eq!(pair[0].1, pair[1].0, "the bands tile without a gap");
    }
}

/// A lone tone puts its energy in the band that holds it, and leaves the others far below.
#[test]
fn a_bands_level_is_the_energy_of_the_partials_inside_it() {
    let s = analyze(&[&tone(1000.0, 1.0, 0.5)[..]], RATE, 4, None);
    let holding = s
        .bands
        .iter()
        .find(|b| b.lo_hz <= 1000.0 && 1000.0 < b.hi_hz)
        .expect("a band holds 1 kHz");
    let loudest = s
        .bands
        .iter()
        .max_by(|a, b| a.db.total_cmp(&b.db))
        .expect("a loudest band");
    assert_eq!(loudest.lo_hz, holding.lo_hz, "and no other band is louder");
    for band in s.bands.iter().filter(|b| b.hi_hz <= 500.0) {
        assert!(band.db < holding.db - 40.0, "well below it: {band:?}");
    }
}

/// Each bin of samples shorter than their frame is their transform under a Hann of their length.
#[test]
fn a_frame_longer_than_its_samples_windows_the_samples_alone() {
    let samples = tone(261.6256, 0.05, 0.8);
    let (mags, bin_hz, frame, _) = magnitudes(&samples, RATE, None);
    assert_eq!((samples.len(), frame), (2205, 4096));
    let span = samples.len() as f64;
    for k in [10, 24, 25, 40, 300] {
        let w = std::f64::consts::TAU * k as f64 / frame as f64;
        let (re, im) = samples
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(re, im), (n, x)| {
                let hann = 0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / span).cos();
                let n = n as f64;
                (re + x * hann * (w * n).cos(), im - x * hann * (w * n).sin())
            });
        let wanted = 4.0 / span * re.hypot(im);
        assert!(
            (mags[k] - wanted).abs() <= 1e-12 * (1.0 + wanted),
            "bin {k} ({} Hz): {} against {wanted}",
            k as f64 * bin_hz,
            mags[k]
        );
    }
}

/// A short sine reads at its level and place, its sidelobes no peaks.
#[test]
fn a_short_sine_reads_at_its_level_with_no_sidelobe_peaks() {
    let s = analyze(&[&tone(261.6256, 0.05, 0.5)[..]], RATE, 8, None);
    let [only] = s.peaks.as_slice() else {
        panic!("one sine, one peak: {:?}", s.peaks);
    };
    assert!((only.hz - 261.6256).abs() < 0.5, "{only:?}");
    assert!((only.db - db(0.5)).abs() < 0.5, "{only:?}");
}

/// Under `db`'s floor peaks still rank by amplitude.
#[test]
fn peaks_under_the_floor_rank_by_amplitude() {
    let quiet = tone(1000.0, 8192.0 / RATE, 1e-9);
    let louder = tone(3000.0, 8192.0 / RATE, 3e-9);
    let mixed: Vec<f64> = quiet.iter().zip(&louder).map(|(a, b)| a + b).collect();
    let s = analyze(&[&mixed[..]], RATE, 2, None);
    let [first, second] = s.peaks.as_slice() else {
        panic!("two partials, two peaks: {:?}", s.peaks);
    };
    assert!((first.hz - 3000.0).abs() < s.resolution_hz, "{first:?}");
    assert!((second.hz - 1000.0).abs() < s.resolution_hz, "{second:?}");
}

/// A sine only in the last 50 ms of a second, under half a frame, is read.
#[test]
fn a_tone_in_the_last_partial_hop_is_read() {
    let mut samples = vec![0.0; RATE as usize];
    let tail = tone(1000.0, 0.05, 1.0);
    let from = samples.len() - tail.len();
    samples[from..].copy_from_slice(&tail);
    let s = analyze(&[&samples[..]], RATE, 4, None);
    let loudest = s.peaks.first().expect("the tail's sine is a peak");
    assert!((loudest.hz - 1000.0).abs() < s.resolution_hz, "{s:?}");
    assert!((s.centroid_hz - 1000.0).abs() < 50.0, "{}", s.centroid_hz);
}

/// A DC offset's Hann sidelobes are no peaks: under a short frame only the sine is one.
#[test]
fn a_dc_offsets_sidelobes_are_no_peaks() {
    let samples: Vec<f64> = tone(1000.0, 0.05, 0.1).iter().map(|x| 0.5 + x).collect();
    let s = analyze(&[&samples[..]], RATE, 8, None);
    let [only] = s.peaks.as_slice() else {
        panic!("one sine over a DC, one peak: {:?}", s.peaks);
    };
    assert!((only.hz - 1000.0).abs() < s.resolution_hz, "{only:?}");
}

/// 0.5 DC beside a unit 1 kHz sine is a third of the power, at bins 0 and 1, 2:1.
#[test]
fn a_dc_offset_weighs_its_own_power_in_the_centroid() {
    let samples: Vec<f64> = tone(1000.0, 1.0, 1.0).iter().map(|x| 0.5 + x).collect();
    let s = analyze(&[&samples[..]], RATE, 4, None);
    let wanted = (1000.0 * 0.5 + s.resolution_hz / 3.0 * 0.25) / 0.75;
    assert!(
        (s.centroid_hz - wanted).abs() < 1.0,
        "{} against {wanted}",
        s.centroid_hz
    );
}

/// A band reads its sine's amplitude however its frame pads; 900 Hz's lobe is inside its band.
#[test]
fn a_band_reads_a_sine_at_its_amplitude_at_any_length() {
    for len in [2205, 4096, 4097, 6000, 44_100] {
        let samples = tone(900.0, len as f64 / RATE, 0.5);
        let s = analyze(&[&samples[..]], RATE, 4, None);
        let holding = s
            .bands
            .iter()
            .find(|b| b.lo_hz <= 900.0 && 900.0 < b.hi_hz)
            .expect("a band holds 900 Hz");
        assert!(
            (holding.db - db(0.5)).abs() < 0.1,
            "{len} samples: {holding:?}"
        );
    }
}
