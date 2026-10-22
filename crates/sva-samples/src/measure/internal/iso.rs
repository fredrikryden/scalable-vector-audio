// Concern: ISO 532-1 time-varying stages and adjoints: band responses, levels, core loudness, decay, slopes | Non-concern: the transform feeding it, modulation (mod.rs) | IO: (band power) -> N' rows

//! Tables and stages of ISO 532-1:2017 (Zwicker), its time-varying method (5.3-5.5 and the
//! annex program), as the mosqito port reads them: third-octave filters at 48 kHz (`REF_B`,
//! `REF_A`, `DIFF`, `GAIN`), the low-band weighting (`RAP`, `DLL`), threshold in quiet `LTQ`,
//! transmission `A0`, `DCB`, critical band edges `ZUP`, slopes `RNS`, `USL`.

const REF_B: [[f64; 3]; 3] = [[1.0, 2.0, 1.0], [1.0, 0.0, -1.0], [1.0, -2.0, 1.0]];
const REF_A: [[f64; 2]; 3] = [[-2.0, 1.0], [-2.0, 1.0], [-2.0, 1.0]];
const DIFF: [[[f64; 2]; 3]; 28] = [
    [
        [-0.00067026, 0.000659453],
        [-0.000375071, 0.000361926],
        [-0.000306523, 0.000297634],
    ],
    [
        [-0.000847258, 0.000830131],
        [-0.000476448, 0.000455616],
        [-0.000388773, 0.000374685],
    ],
    [
        [-0.0010721, 0.00104496],
        [-0.000606567, 0.000573553],
        [-0.000494004, 0.000471677],
    ],
    [
        [-0.00135836, 0.00131535],
        [-0.000774327, 0.000722007],
        [-0.000629154, 0.000593771],
    ],
    [
        [-0.0017238, 0.00165564],
        [-0.00099178, 0.000908866],
        [-0.000803529, 0.000747455],
    ],
    [
        [-0.00219188, 0.00208388],
        [-0.00127545, 0.00114406],
        [-0.00102976, 0.0009409],
    ],
    [
        [-0.00279386, 0.00262274],
        [-0.00164828, 0.00144006],
        [-0.0013252, 0.00118438],
    ],
    [
        [-0.00357182, 0.00330071],
        [-0.00214252, 0.00181258],
        [-0.00171397, 0.00149082],
    ],
    [
        [-0.00458305, 0.00415355],
        [-0.00280413, 0.00228135],
        [-0.00223006, 0.00187646],
    ],
    [
        [-0.00590655, 0.00522622],
        [-0.00369947, 0.00287118],
        [-0.00292205, 0.00236178],
    ],
    [
        [-0.00765243, 0.00657493],
        [-0.0049254, 0.00361318],
        [-0.00386007, 0.0029724],
    ],
    [
        [-0.0100023, 0.0082961],
        [-0.00663788, 0.00455999],
        [-0.00515982, 0.00375306],
    ],
    [
        [-0.013123, 0.010422],
        [-0.00902274, 0.00573132],
        [-0.00694543, 0.00471734],
    ],
    [
        [-0.0173693, 0.0130947],
        [-0.0124176, 0.00720526],
        [-0.00946002, 0.00593145],
    ],
    [
        [-0.0231934, 0.0164308],
        [-0.0173009, 0.00904761],
        [-0.0130358, 0.00744926],
    ],
    [
        [-0.0313292, 0.020637],
        [-0.0244342, 0.0113731],
        [-0.0182108, 0.00936778],
    ],
    [
        [-0.0428261, 0.0259325],
        [-0.0349619, 0.0143046],
        [-0.0257855, 0.0117912],
    ],
    [
        [-0.0591733, 0.0325054],
        [-0.0506072, 0.0179513],
        [-0.0369401, 0.0148094],
    ],
    [
        [-0.0826348, 0.0405894],
        [-0.0740348, 0.0224476],
        [-0.0534977, 0.0185371],
    ],
    [
        [-0.117018, 0.0508116],
        [-0.109516, 0.0281387],
        [-0.0785097, 0.0232872],
    ],
    [
        [-0.167714, 0.0637872],
        [-0.163378, 0.0353729],
        [-0.116419, 0.0293723],
    ],
    [
        [-0.242528, 0.0798576],
        [-0.245161, 0.044337],
        [-0.173972, 0.0370015],
    ],
    [
        [-0.353142, 0.099633],
        [-0.369163, 0.0553535],
        [-0.261399, 0.0465428],
    ],
    [
        [-0.516316, 0.124177],
        [-0.555473, 0.0689403],
        [-0.393998, 0.0586715],
    ],
    [
        [-0.756635, 0.155023],
        [-0.834281, 0.0858123],
        [-0.594547, 0.074396],
    ],
    [
        [-1.10165, 0.191713],
        [-1.23939, 0.105243],
        [-0.891666, 0.0940354],
    ],
    [
        [-1.58477, 0.239049],
        [-1.80505, 0.128794],
        [-1.325, 0.121333],
    ],
    [
        [-2.5063, 0.142308],
        [-2.19464, 0.27647],
        [-1.90231, 0.147304],
    ],
];
const GAIN: [f64; 28] = [
    4.30764e-11,
    8.5934e-11,
    1.71424e-10,
    3.41944e-10,
    6.82035e-10,
    1.36026e-09,
    2.71261e-09,
    5.4087e-09,
    1.07826e-08,
    2.1491e-08,
    4.28228e-08,
    8.54316e-08,
    1.70009e-07,
    3.38215e-07,
    6.7199e-07,
    1.33531e-06,
    2.65172e-06,
    5.25477e-06,
    1.0378e-05,
    2.0487e-05,
    4.05198e-05,
    7.97914e-05,
    0.000156511,
    0.000304954,
    0.000599157,
    0.00116544,
    0.00227488,
    0.00391006,
];
const RAP: [f64; 8] = [45.0, 55.0, 65.0, 71.0, 80.0, 90.0, 100.0, 120.0];
const DLL: [[f64; 11]; 8] = [
    [
        -32.0, -24.0, -16.0, -10.0, -5.0, 0.0, -7.0, -3.0, 0.0, -2.0, 0.0,
    ],
    [
        -29.0, -22.0, -15.0, -10.0, -4.0, 0.0, -7.0, -2.0, 0.0, -2.0, 0.0,
    ],
    [
        -27.0, -19.0, -14.0, -9.0, -4.0, 0.0, -6.0, -2.0, 0.0, -2.0, 0.0,
    ],
    [
        -25.0, -17.0, -12.0, -9.0, -3.0, 0.0, -5.0, -2.0, 0.0, -2.0, 0.0,
    ],
    [
        -23.0, -16.0, -11.0, -7.0, -3.0, 0.0, -4.0, -1.0, 0.0, -1.0, 0.0,
    ],
    [
        -20.0, -14.0, -10.0, -6.0, -3.0, 0.0, -4.0, -1.0, 0.0, -1.0, 0.0,
    ],
    [
        -18.0, -12.0, -9.0, -6.0, -2.0, 0.0, -3.0, -1.0, 0.0, -1.0, 0.0,
    ],
    [
        -15.0, -10.0, -8.0, -4.0, -2.0, 0.0, -3.0, -1.0, 0.0, -1.0, 0.0,
    ],
];
const LTQ: [f64; 20] = [
    30.0, 18.0, 12.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    3.0, 3.0,
];
const A0: [f64; 20] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1.6, -3.2, -5.4, -5.6, -4.0, -1.5,
    2.0, 5.0, 12.0,
];
const DCB: [f64; 20] = [
    -0.25, -0.6, -0.8, -0.8, -0.5, 0.0, 0.5, 1.1, 1.5, 1.7, 1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 0.8, 0.5,
    0.0, -0.5,
];

/// The floor ISO 532-1's program adds to a band's mean square before its level.
const FLOOR_PA2: f64 = 1e-12;
const REF_PA2: f64 = 4e-10;
const T_SHORT: f64 = 0.005;
const T_LONG: f64 = 0.015;
const T_VAR: f64 = 0.075;
/// The decay's step: 24 of each 0.5 ms frame.
const DECAY_RATE_HZ: f64 = 48_000.0;
pub const THIRDS: usize = 28;
pub const CORE: usize = 21;
pub const ROWS: usize = 240;

const ZUP: [f64; CORE] = [
    0.9, 1.8, 2.8, 3.5, 4.4, 5.4, 6.6, 7.9, 9.2, 10.6, 12.3, 13.8, 15.2, 16.7, 18.1, 19.3, 20.6,
    21.8, 22.7, 23.6, 24.0,
];
const RNS: [f64; 18] = [
    21.5, 18.0, 15.1, 11.5, 9.0, 6.1, 4.4, 3.1, 2.13, 1.36, 0.82, 0.42, 0.30, 0.22, 0.15, 0.10,
    0.035, 0.0,
];
const USL: [[f64; 8]; 18] = [
    [13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
    [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
    [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
    [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
    [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
    [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
    [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
    [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
    [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
    [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
    [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
    [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
    [0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
    [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
    [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
    [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
    [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
    [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
];

fn p10(x: f64) -> f64 {
    sva_formula::exp(x * std::f64::consts::LN_10)
}

pub fn centre_hz(b: usize) -> f64 {
    1000.0 * p10((b as f64 - 16.0) / 10.0)
}

/// `|H_b(f)|^2` of the standard's 48 kHz band filter.
pub fn response(b: usize, hz: f64) -> f64 {
    if hz >= 0.5 * DECAY_RATE_HZ {
        return 0.0;
    }
    let w = std::f64::consts::TAU * hz / DECAY_RATE_HZ;
    let (s1, c1) = libm::sincos(w);
    let (s2, c2) = libm::sincos(2.0 * w);
    let mut h2 = GAIN[b] * GAIN[b];
    for s in 0..3 {
        let poly = |x0: f64, x1: f64, x2: f64| {
            let re = x0 + x1 * c1 + x2 * c2;
            let im = -x1 * s1 - x2 * s2;
            re * re + im * im
        };
        let num = poly(REF_B[s][0], REF_B[s][1], REF_B[s][2]);
        let den = poly(
            1.0,
            REF_A[s][0] - DIFF[b][s][0],
            REF_A[s][1] - DIFF[b][s][1],
        );
        h2 *= num / den;
    }
    h2
}

pub fn smoothing(b: usize, step_secs: f64) -> f64 {
    let tau = 2.0 / (3.0 * centre_hz(b).min(1000.0));
    sva_formula::exp(-step_secs / tau)
}

pub fn level(pa2: f64) -> f64 {
    10.0 * libm::log10((pa2 + FLOOR_PA2) / REF_PA2)
}

pub fn level_slope(pa2: f64) -> f64 {
    10.0 / (std::f64::consts::LN_10 * (pa2 + FLOOR_PA2))
}

fn weighted(level: f64, band: usize) -> f64 {
    let above = |r: usize| level > RAP[r] - DLL[r][band];
    let mut dll = if above(0) { 0.0 } else { DLL[0][band] };
    for (r, row) in DLL.iter().enumerate().take(7).skip(1) {
        if above(r - 1) != above(r) {
            dll = row[band];
        }
    }
    p10((dll + level) / 10.0)
}

pub fn core_unit(k: usize) -> f64 {
    0.0635 * p10(0.025 * LTQ[k.min(19)])
}

fn core_law(le: f64, k: usize) -> (f64, f64) {
    let l = le - A0[k];
    if l <= LTQ[k] {
        return (0.0, 0.0);
    }
    let x = p10(0.1 * (l - DCB[k] - LTQ[k]));
    let base = 0.75 + 0.25 * x;
    let v = core_unit(k) * (base.sqrt().sqrt() - 1.0);
    match v > 0.0 {
        true => (
            v,
            core_unit(k) * 0.25 * libm::pow(base, -0.75) * 0.25 * x * 0.1 * std::f64::consts::LN_10,
        ),
        false => (0.0, 0.0),
    }
}

pub fn core(level: &[f64; THIRDS]) -> [f64; CORE] {
    let mut ti = [0.0; 11];
    for (b, t) in ti.iter_mut().enumerate() {
        *t = weighted(level[b], b);
    }
    let gi = [
        ti[0..6].iter().sum::<f64>(),
        ti[6..9].iter().sum::<f64>(),
        ti[9..11].iter().sum::<f64>(),
    ];
    let mut nm = [0.0; CORE];
    for k in 0..20 {
        let le = match k < 3 {
            true if gi[k] > 0.0 => 10.0 * libm::log10(gi[k]),
            true => 0.0,
            false => level[8 + k],
        };
        nm[k] = core_law(le, k).0;
    }
    let korry = 0.4 + 0.32 * libm::pow(nm[0], 0.2);
    if korry <= 1.0 {
        nm[0] *= korry;
    }
    nm
}

pub fn core_back(level: &[f64; THIRDS], dnm: &[f64; CORE]) -> [f64; THIRDS] {
    let mut ti = [0.0; 11];
    for (b, t) in ti.iter_mut().enumerate() {
        *t = weighted(level[b], b);
    }
    let gi = [
        ti[0..6].iter().sum::<f64>(),
        ti[6..9].iter().sum::<f64>(),
        ti[9..11].iter().sum::<f64>(),
    ];
    let mut out = [0.0; THIRDS];
    let mut dn = *dnm;
    let le0 = match gi[0] > 0.0 {
        true => 10.0 * libm::log10(gi[0]),
        false => 0.0,
    };
    let n0 = core_law(le0, 0).0;
    if 0.4 + 0.32 * libm::pow(n0, 0.2) <= 1.0 {
        dn[0] *= 0.4 + 0.384 * libm::pow(n0, 0.2);
    }
    let groups = [0..6, 6..9, 9..11];
    for k in 0..20 {
        let le = match k < 3 {
            true if gi[k] > 0.0 => 10.0 * libm::log10(gi[k]),
            true => 0.0,
            false => level[8 + k],
        };
        let d = dn[k] * core_law(le, k).1;
        if d == 0.0 {
            continue;
        }
        match k < 3 {
            true => {
                let dg = d * 10.0 / (gi[k] * std::f64::consts::LN_10);
                for b in groups[k].clone() {
                    out[b] += dg * ti[b] * std::f64::consts::LN_10 / 10.0;
                }
            }
            false => out[8 + k] += d,
        }
    }
    out
}

/// ISO 532-1's nonlinear decay over frames, the input linear between them; a frame reads the
/// state at its first step.
pub struct Decay {
    steps: usize,
    b: [f64; 6],
    branch: Vec<u8>,
}

impl Decay {
    pub fn of(step_secs: f64) -> Decay {
        let steps = libm::round(step_secs * DECAY_RATE_HZ).max(1.0) as usize;
        let dt = step_secs / steps as f64;
        let p = (T_VAR + T_LONG) / (T_VAR * T_SHORT);
        let q = 1.0 / (T_SHORT * T_VAR);
        let l1 = -p / 2.0 + (p * p / 4.0 - q).sqrt();
        let l2 = -p / 2.0 - (p * p / 4.0 - q).sqrt();
        let den = T_VAR * (l1 - l2);
        let e1 = sva_formula::exp(l1 * dt);
        let e2 = sva_formula::exp(l2 * dt);
        Decay {
            steps,
            b: [
                (e1 - e2) / den,
                ((T_VAR * l2 + 1.0) * e1 - (T_VAR * l1 + 1.0) * e2) / den,
                ((T_VAR * l1 + 1.0) * e1 - (T_VAR * l2 + 1.0) * e2) / den,
                (T_VAR * l1 + 1.0) * (T_VAR * l2 + 1.0) * (e1 - e2) / den,
                sva_formula::exp(-dt / T_LONG),
                sva_formula::exp(-dt / T_VAR),
            ],
            branch: Vec::new(),
        }
    }

    pub fn run(&mut self, core: &mut [f64]) {
        let frames = core.len() / CORE;
        let b = self.b;
        self.branch = vec![0; frames * self.steps * CORE];
        let input = core.to_vec();
        for k in 0..CORE {
            let (mut op, mut tp) = (0.0, 0.0);
            for c in 0..frames {
                let here = input[c * CORE + k];
                let next = match c + 1 < frames {
                    true => input[(c + 1) * CORE + k],
                    false => 0.0,
                };
                for m in 0..self.steps {
                    let u = here + (next - here) * m as f64 / self.steps as f64;
                    let up = op > tp;
                    let cand = if up { op * b[2] - tp * b[3] } else { op * b[4] };
                    let held = cand < u;
                    let uo = if held { u } else { cand };
                    let fall = op * b[0] - tp * b[1];
                    let release = u >= op && !((u - op).abs() < 1e-5 && uo <= tp);
                    let falls = !release && u < op && up && fall <= uo;
                    let u2 = match (release, falls) {
                        (true, _) => (tp - u) * b[5] + u,
                        (false, true) => fall,
                        (false, false) => uo,
                    };
                    self.branch[(k * frames + c) * self.steps + m] = u8::from(held)
                        | u8::from(up) << 3
                        | u8::from(falls) << 1
                        | u8::from(release) << 2;
                    op = uo;
                    tp = u2;
                    if m == 0 {
                        core[c * CORE + k] = uo;
                    }
                }
            }
        }
    }

    pub fn back(&self, d: &[f64]) -> Vec<f64> {
        let frames = d.len() / CORE;
        let b = self.b;
        let mut out = vec![0.0; d.len()];
        for k in 0..CORE {
            let (mut gop, mut gtp) = (0.0, 0.0);
            for c in (0..frames).rev() {
                let (mut ghere, mut gnext) = (0.0, 0.0);
                for m in (0..self.steps).rev() {
                    let br = self.branch[(k * frames + c) * self.steps + m];
                    let (held, falls, release, up) =
                        (br & 1 != 0, br & 2 != 0, br & 4 != 0, br & 8 != 0);
                    let mut guo = gop;
                    if m == 0 {
                        guo += d[c * CORE + k];
                    }
                    let (mut g_op, mut g_tp, mut g_u) = (0.0, 0.0, 0.0);
                    match (release, falls) {
                        (true, _) => {
                            g_tp += b[5] * gtp;
                            g_u += (1.0 - b[5]) * gtp;
                        }
                        (false, true) => {
                            g_op += b[0] * gtp;
                            g_tp -= b[1] * gtp;
                        }
                        (false, false) => guo += gtp,
                    }
                    match (held, up) {
                        (true, _) => g_u += guo,
                        (false, true) => {
                            g_op += b[2] * guo;
                            g_tp -= b[3] * guo;
                        }
                        (false, false) => g_op += b[4] * guo,
                    }
                    let w = m as f64 / self.steps as f64;
                    ghere += g_u * (1.0 - w);
                    gnext += g_u * w;
                    gop = g_op;
                    gtp = g_tp;
                }
                out[c * CORE + k] += ghere;
                if c + 1 < frames {
                    out[(c + 1) * CORE + k] += gnext;
                }
            }
        }
        out
    }
}

fn row_band(r: usize) -> usize {
    let z = 0.1 * (r + 1) as f64;
    ZUP.iter().position(|u| z <= *u + 1e-9).unwrap_or(CORE - 1)
}

fn range(v: f64) -> usize {
    RNS.iter().filter(|r| v < **r).count().min(17)
}

/// Each row the larger of its band's core loudness and the upper slope falling from below at
/// the steepness its level gives; beside it the band it came from and its slope in that band.
pub fn slopes(nm: &[f64; CORE], rows: &mut [f64; ROWS], from: &mut [(u8, f64); ROWS]) {
    let (mut c, mut origin, mut factor) = (0.0, 0u8, 0.0);
    for r in 0..ROWS {
        let k = row_band(r);
        if r > 0 && c > 0.0 {
            let column = k.saturating_sub(1).min(7);
            let mut i = range(c);
            let first = USL[i][column];
            let mut rest = 0.1;
            loop {
                let s = USL[i][column];
                let need = (c - RNS[i]) / s;
                if need >= rest {
                    c -= s * rest;
                    factor *= s / first;
                    break;
                }
                rest -= need;
                c = RNS[i];
                if i == 17 {
                    c = 0.0;
                    factor = 0.0;
                    break;
                }
                i += 1;
            }
        }
        if nm[k] >= c {
            c = nm[k];
            origin = k as u8;
            factor = 1.0;
        }
        rows[r] = c;
        from[r] = (origin, factor);
    }
}

pub fn upper_bark(k: usize) -> f64 {
    ZUP[k]
}
