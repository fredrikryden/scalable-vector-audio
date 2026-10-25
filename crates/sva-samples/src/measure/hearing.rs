// Concern: cited hearing curves and their readers: ERB, ISO 532-2's ear tables, roex slopes, windows, JNDs at d' = 1 | Non-concern: a signal read through them (internal/) | IO: (Hz, dB) -> Hz, dB, p

pub const MODEL: &str = "the hearing curves, each a transcribed value with its source or \
     derived from one: ERB = C1 (C2 f/1000 + 1) Hz and Cam = C3 log10(C2 f/1000 + 1), C1 = \
     24.673, C2 = 4.368, C3 = 2302.6/(C1 C2) = 21.366 (ISO 532-2:2017's reference code binloud.h, \
     DIS 2015-12-08, https://standards.iso.org/iso/532/-2/ed-1/en/, its 2302.6 for 1000 ln 10 so \
     dCam/df = 1/ERB; ANSI/ASA S3.4-2007; Glasberg and Moore 1990 print 24.7 and 4.37); channels \
     from 1.8 to 38.9 Cam (binloud.h ESTART, EEND) spaced 0.25 Cam, a 1 ms hop and Hann windows \
     of 64 down to 2 ms over ranges split at 80, 500, 1250, 2540 and 4050 Hz (Glasberg and Moore \
     2002, JAES 50(5) 331-342, as Moore 2014, Trends Hear. 18, doi 10.1177/2331216514550620, pp. \
     12-13, gives them); G(f) the free field to eardrum gain less the middle ear's attenuation, \
     ISO 532-2's Shaw_cor (40 rows; Shaw 1974, Killion 1978) and mid_corr (41 rows), and E_T(f) \
     its ex_table (35 rows, July 2008) less its 0.58 dB, so 3.15 dB from 500 Hz up \
     (P_subs2a_2011.c), each linear in Hz and dB between rows and the end row beyond either end \
     as its lin_int states, transcribed as values with no code copied; the roex (1 + p g) \
     e^(-p g), g = |f - fc|/fc, p = 4 fc/ERB(fc) at and above fc and p - 0.35 (p/p(1 kHz)) (X - \
     51) below, X the component's level per ERB at the eardrum through the level-free roex \
     centred on it, p at least 0.1 (binloud.c; Moore, Glasberg and Baer 1997, JAES 45(4)), \
     summed over every component, with neither the code's g > 4 cut nor its 1e-10 floor; each \
     JND at d' = 1: the published step over the d' of its criterion, 2AFC at the 70.7 % a \
     2-down-1-up track converges on (Levitt 1971, JASA 49(2B) 467-477, doi 10.1121/1.1912375), sqrt(2) Phi^-1(2^-1/2) \
     = 0.77068 (Green and Swets 1966), Jesteadt's and Wier's adaptive 71 % read as that; level: \
     dI/I = 0.463 (I/I0)^-0.072, I0 the threshold intensity, 500 ms tones (Jesteadt, Wier and \
     Green 1977, JASA 61(1) 169-177, doi 10.1121/1.381278), its 10 log10(1 + dI/I) dB over \
     0.77068; frequency: log10 DLF = 0.0238 sqrt(F) - 0.24 + 4.295/SL, F in Hz, SL in dB, \
     Nelson, Stanton and Freyman's (1983, JASA 73(6) 2117-2123, doi 10.1121/1.389579) eq. (4) \
     and Table V fit to Wier, Jesteadt and Green (1977, doi 10.1121/1.381251), 500 ms tones, \
     over Nelson's 4AFC factor 1.49/0.78 (p. 2122), which gives Wier's 2AFC steps back, and over \
     0.77068";

pub const C1: f64 = 24.673;
pub const C2: f64 = 4.368;
pub const C3: f64 = 2302.6 / (C1 * C2);

pub const CAM_FIRST: f64 = 1.8;
pub const CAM_LAST: f64 = 38.9;
pub const CAM_SPACING: f64 = 0.25;
pub const HOP_SECS: f64 = 0.001;
pub const WINDOW_SECS: [f64; 6] = [0.064, 0.032, 0.016, 0.008, 0.004, 0.002];
pub const WINDOW_SPLITS_HZ: [f64; 5] = [80.0, 500.0, 1250.0, 2540.0, 4050.0];

pub fn erb_hz(hz: f64) -> f64 {
    C1 * (C2 * hz / 1000.0 + 1.0)
}

pub fn cam(hz: f64) -> f64 {
    C3 * libm::log10(C2 * hz / 1000.0 + 1.0)
}

pub fn cam_hz(cam: f64) -> f64 {
    (libm::pow(10.0, cam / C3) - 1.0) * 1000.0 / C2
}

const FIELD_HZ: [f64; 40] = [
    0.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 750.0, 800.0, 1000.0, 1250.0, 1500.0, 1600.0, 2000.0, 2500.0, 3000.0, 3150.0,
    4000.0, 5000.0, 6000.0, 6300.0, 8000.0, 9000.0, 10000.0, 11200.0, 12500.0, 14000.0, 15000.0,
    16000.0, 20000.0,
];
const FREE_FIELD_DB: [f64; 40] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.9, 1.4, 1.6, 1.7, 2.5, 2.7, 2.6,
    2.6, 3.2, 5.2, 6.6, 12.0, 16.8, 15.3, 15.2, 14.2, 10.7, 7.1, 6.4, 1.8, -0.9, -1.6, 1.9, 4.9,
    2.0, -2.0, 2.5, 2.5,
];
const MIDDLE_EAR_HZ: [f64; 41] = [
    0.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 750.0, 800.0, 1000.0, 1250.0, 1500.0, 1600.0, 2000.0, 2500.0, 3000.0, 3150.0,
    4000.0, 5000.0, 6000.0, 6300.0, 8000.0, 9000.0, 10000.0, 11200.0, 12500.0, 14000.0, 15000.0,
    16000.0, 18000.0, 20000.0,
];
const MIDDLE_EAR_DB: [f64; 41] = [
    50.0, 39.6, 32.0, 25.85, 21.4, 18.5, 15.9, 14.1, 12.4, 11.0, 9.6, 8.3, 7.4, 6.2, 4.8, 3.8, 3.3,
    2.9, 2.6, 2.6, 4.5, 5.4, 6.1, 8.5, 10.4, 7.3, 7.0, 6.6, 7.0, 9.2, 10.2, 12.2, 10.8, 10.1, 12.7,
    15.0, 18.2, 23.8, 32.3, 46.5, 50.0,
];
const THRESHOLD_HZ: [f64; 35] = [
    0.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 750.0,
    800.0, 1000.0, 1250.0, 1500.0, 1600.0, 2000.0, 2500.0, 3000.0, 3150.0, 4000.0, 5000.0, 6000.0,
    6300.0, 8000.0, 9000.0, 10000.0, 11200.0, 12500.0, 14000.0, 17000.0, 20000.0,
];
const THRESHOLD_TABLE_DB: [f64; 35] = [
    32.0, 28.2, 24.0, 19.0, 15.68, 12.50, 9.90, 8.00, 6.30, 5.3, 4.5, 3.73, 3.73, 3.73, 3.73, 3.73,
    3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73, 3.73,
    3.73, 3.73, 3.73,
];
const THRESHOLD_SHIFT_DB: f64 = 0.58;

fn interpolate(hz: &[f64], db: &[f64], f: f64) -> f64 {
    let past = hz.partition_point(|row| *row <= f);
    match past {
        0 => db[0],
        n if n == hz.len() => db[n - 1],
        i => db[i - 1] + (db[i] - db[i - 1]) * (f - hz[i - 1]) / (hz[i] - hz[i - 1]),
    }
}

pub fn free_field_db(hz: f64) -> f64 {
    interpolate(&FIELD_HZ, &FREE_FIELD_DB, hz)
}

pub fn middle_ear_db(hz: f64) -> f64 {
    interpolate(&MIDDLE_EAR_HZ, &MIDDLE_EAR_DB, hz)
}

pub fn ear_gain_db(hz: f64) -> f64 {
    free_field_db(hz) - middle_ear_db(hz)
}

pub fn threshold_excitation_db(hz: f64) -> f64 {
    interpolate(&THRESHOLD_HZ, &THRESHOLD_TABLE_DB, hz) - THRESHOLD_SHIFT_DB
}

pub const LOWER_SLOPE_ANCHOR_DB: f64 = 51.0;
pub const LOWER_SLOPE: f64 = 0.35;
pub const P_FLOOR: f64 = 0.1;

pub fn p51(hz: f64) -> f64 {
    4.0 * hz / erb_hz(hz)
}

pub fn roex_p(centre_hz: f64, component_hz: f64, x_db: f64) -> f64 {
    let p = p51(centre_hz);
    let slope = match component_hz < centre_hz {
        true => p - LOWER_SLOPE * (p / p51(1000.0)) * (x_db - LOWER_SLOPE_ANCHOR_DB),
        false => p,
    };
    slope.max(P_FLOOR)
}

pub fn roex(p: f64, centre_hz: f64, component_hz: f64) -> f64 {
    let pg = p * (component_hz - centre_hz).abs() / centre_hz;
    (1.0 + pg) * sva_formula::exp(-pg)
}

/// sqrt(2) Phi^-1(2^-1/2) to f64, as MODEL derives it; the hearing suite recomputes it.
pub const D_PRIME_TWO_DOWN_ONE_UP: f64 = 0.770_678_701_034_253_2;

const JESTEADT_SCALE: f64 = 0.463;
const JESTEADT_EXPONENT: f64 = -0.072;
pub const JESTEADT_TONE_SECS: f64 = 0.5;

pub fn level_jnd_db(sl_db: f64) -> f64 {
    let ratio = JESTEADT_SCALE * libm::pow(10.0, JESTEADT_EXPONENT * sl_db / 10.0);
    10.0 * libm::log10(1.0 + ratio) / D_PRIME_TWO_DOWN_ONE_UP
}

const NELSON_A: f64 = 0.0238;
const NELSON_K: f64 = -0.24;
const NELSON_M: f64 = 4.295;
pub const WIER_TONE_SECS: f64 = 0.5;
const NELSON_4AFC_FACTOR: f64 = 1.49 / 0.78;

pub fn frequency_jnd_hz(hz: f64, sl_db: f64) -> f64 {
    let log_dlf = NELSON_A * hz.sqrt() + NELSON_K + NELSON_M / sl_db;
    libm::pow(10.0, log_dlf) / (NELSON_4AFC_FACTOR * D_PRIME_TWO_DOWN_ONE_UP)
}
