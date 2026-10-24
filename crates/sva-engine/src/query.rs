// Concern: the one table of observations: names, arguments, defaults, what each reads, help; the answer envelope | Non-concern: taking one reading (answer.rs) | IO: (name, Given) -> Representation

use sva_formula::{C64, Line, SpectralSum, Var};
use sva_samples::measure::beating::Beating;
use sva_samples::measure::harmony::dissonance::Dissonance;
use sva_samples::measure::harmony::inharmonicity::Inharmonicity;
use sva_samples::measure::harmony::tonnetz::TonalDistance;
use sva_samples::measure::harmony::voice_leading::{Norm, VoiceLeading};
use sva_samples::measure::internal::Internal;
use sva_samples::measure::loudness_pattern::LoudnessPattern;
use sva_samples::measure::modulation::Modulation;
use sva_samples::measure::partials::Partials;
use sva_samples::measure::periodicity::Periodicity;
use sva_samples::measure::self_similarity::SelfSimilarity;
use sva_samples::measure::sharpness::Sharpness;
use sva_samples::{
    Alias, Bands, Buffer, Consumes, Crest, EnvelopeFrame, FormantFrame, Hits, LedgerEntry,
    Loudness, PitchFrame, Source, Spectrum, StereoImage,
};

use crate::arguments::Arguments;
use crate::bindings::Binding;

/// The default window a framed reading uses when the observation names none.
pub const DEFAULT_FRAME_SECS: f64 = 0.05;
pub const DEFAULT_MAX_PEAKS: usize = 16;
pub const DEFAULT_LEDGER_DEPTH: usize = 3;
pub const DEFAULT_OVERSAMPLE: u32 = 4;
/// How much audio a live device holds ahead of what it plays, where `samples` names one.
pub const DEFAULT_BUFFER_SECS: f64 = 0.05;
/// The amplitude, in dB against full scale, under which a harmonic reading drops a partial.
pub const DEFAULT_FLOOR_DB: f64 = -40.0;
/// The most partial pairs one `dissonance` frame sums before it refuses.
pub const DEFAULT_MAX_PAIRS: usize = 1 << 17;
/// BS.1770's momentary block, which `loudness` integrates over unless asked another.
pub const DEFAULT_LOUDNESS_WINDOW_SECS: f64 = sva_samples::measure::loudness::MOMENTARY_SECS;
/// The span of onset-strength envelope one `periodicity` window autocorrelates.
pub const DEFAULT_PERIODICITY_WINDOW_SECS: f64 = 6.0;
/// The span of band envelope one `modulation` window transforms.
pub const DEFAULT_MODULATION_WINDOW_SECS: f64 = 4.0;
/// The side of one `self_similarity` matrix cell.
pub const DEFAULT_CELL_SECS: f64 = 0.5;
/// The length `self_similarity` averages each diagonal over before it picks stripes.
pub const DEFAULT_SMOOTH_SECS: f64 = 4.0;
/// The whole width of `self_similarity`'s novelty kernel.
pub const DEFAULT_NOVELTY_WINDOW_SECS: f64 = 8.0;
/// A Hann this long parts partials some 20 Hz apart.
pub const DEFAULT_PARTIALS_FRAME_SECS: f64 = 0.1;
/// A short note's length, over which a noise's chance peaks average away.
pub const DEFAULT_PARTIALS_WINDOW_SECS: f64 = 0.5;
pub const DEFAULT_PARTIALS_HOP_SECS: f64 = 0.005;
pub const DEFAULT_PARTIALS_FLOOR_DB: f64 = -70.0;
pub const DEFAULT_PARTIALS_TRACKS: usize = 120;
/// A loudness pattern's frames step 10 ms: a 20 Hz level swing reads five frames a cycle.
pub const DEFAULT_PATTERN_FRAME_SECS: f64 = 0.01;
pub use sva_samples::measure::loudness_pattern::DEFAULT_SPL_DB;
/// Half a second of frames: a swing at 4 Hz turns twice in it, a note's fall stays its trend.
pub const DEFAULT_BEATING_WINDOW_SECS: f64 = 0.5;

/// Every named argument a reading may take; the engine reads only those `Given` holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    Peaks,
    Frame,
    Depth,
    Oversample,
    Brief,
    Skim,
    Node,
    Buffer,
    Floor,
    Norm,
    Pairs,
    Window,
    Curve,
    Bpm,
    Cell,
    Smooth,
    Hop,
    Spl,
}

impl Arg {
    pub fn key(self) -> &'static str {
        match self {
            Arg::Peaks => "peaks",
            Arg::Frame => "frame",
            Arg::Depth => "depth",
            Arg::Oversample => "oversample",
            Arg::Brief => "brief",
            Arg::Skim => "skim",
            Arg::Node => "node",
            Arg::Buffer => "buffer",
            Arg::Floor => "floor",
            Arg::Norm => "norm",
            Arg::Pairs => "pairs",
            Arg::Window => "window",
            Arg::Curve => "curve",
            Arg::Bpm => "bpm",
            Arg::Cell => "cell",
            Arg::Smooth => "smooth",
            Arg::Hop => "hop",
            Arg::Spl => "spl",
        }
    }
}

/// The arguments a `Representation` is built from, each at its default until written.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Given {
    /// `None` until written: each reading then takes its own default.
    pub peaks: Option<usize>,
    /// `None` until written: `spectrum` then sizes its transform to the range, and every
    /// other framed reading steps by `DEFAULT_FRAME_SECS`.
    pub frame: Option<f64>,
    pub depth: usize,
    pub oversample: u32,
    /// `None` until written: each reading then takes its own default.
    pub floor_db: Option<f64>,
    pub norm: Norm,
    pub max_pairs: usize,
    /// `None` until written: each windowed reading then takes its own default.
    pub window: Option<f64>,
    pub curve: bool,
    pub bpm: Option<f64>,
    pub cell_secs: f64,
    pub smooth_secs: f64,
    pub hop: Option<f64>,
    /// `None` until written: the level a full-scale sine plays at, `DEFAULT_SPL_DB`.
    pub spl_db: Option<f64>,
}

impl Default for Given {
    fn default() -> Given {
        Given {
            peaks: None,
            frame: None,
            depth: DEFAULT_LEDGER_DEPTH,
            oversample: DEFAULT_OVERSAMPLE,
            floor_db: None,
            norm: Norm::L1,
            max_pairs: DEFAULT_MAX_PAIRS,
            window: None,
            curve: false,
            bpm: None,
            cell_secs: DEFAULT_CELL_SECS,
            smooth_secs: DEFAULT_SMOOTH_SECS,
            hop: None,
            spl_db: None,
        }
    }
}

/// What a reading reads, which decides whether its answer is exact or measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reads {
    /// The written closed form alone: always exact.
    ClosedForm,
    /// Whichever the node holds: exact off a closed form, measured off samples.
    Either,
    /// Samples alone: always measured.
    Samples,
    /// Samples and the graph that rendered them, so no file answers it.
    Graph,
    /// The resolved instances, before anything renders; nothing composes for it.
    Structure,
}

impl Reads {
    pub fn consumes(self, closed: bool) -> Consumes {
        match self {
            Reads::ClosedForm | Reads::Structure => Consumes::ClosedForm,
            Reads::Either if closed => Consumes::ClosedForm,
            Reads::Either | Reads::Samples | Reads::Graph => Consumes::Buffer,
        }
    }

    pub fn off_samples(self, sampled: bool) -> bool {
        match self {
            Reads::Samples => true,
            Reads::Either => sampled,
            Reads::ClosedForm | Reads::Graph | Reads::Structure => false,
        }
    }
}

pub struct Reading {
    pub name: &'static str,
    pub takes: &'static [Arg],
    pub reads: Reads,
    pub help: &'static str,
    build: fn(&Given) -> Representation,
}

impl Reading {
    pub fn representation(&self, given: &Given) -> Representation {
        (self.build)(given)
    }
}

static LINES: Reading = Reading {
    name: "lines",
    takes: &[],
    reads: Reads::ClosedForm,
    help: "the exact line list of a closed form, every channel's in channel order with no \
           channel named; `atoms` names each term's",
    build: |_| Representation::Lines,
};

static ATOMS: Reading = Reading {
    name: "atoms",
    takes: &[],
    reads: Reads::ClosedForm,
    help: "each term of a closed form: its channel, sketch, line, decay, polynomial and window",
    build: |_| Representation::Atoms,
};

static SPECTRUM: Reading = Reading {
    name: "spectrum",
    takes: &[Arg::Peaks, Arg::Frame],
    reads: Reads::Either,
    help: "its lines, or its spectrum averaged over every frame and, power per bin, every \
           channel; off samples a Hann spans the samples a frame holds, zero past them, each \
           third-octave band reads as the amplitude of the one sine carrying its power, and, \
           loudest first, a local maximum is no peak where its bin holds no more than the \
           peaks kept, and a DC bin 0 holds as its maximum, can leak there, each bounded by \
           Harris's (1978) Hann sidelobe bound 1/(pi m (m^2 - 1)) at m bins, its sinusoid taken \
           anywhere within half a bin of its own: a bound where the frame is stationary \
           sinusoids at the kept peaks",
    build: |g| Representation::Spectrum {
        max_peaks: g.peaks.unwrap_or(DEFAULT_MAX_PEAKS),
        frame_secs: g.frame,
    },
};

static ENVELOPE: Reading = Reading {
    name: "envelope",
    takes: &[Arg::Frame],
    reads: Reads::Either,
    help: "off a closed form the squared modulus of its analytic signal, x^2 + H{x}^2, as \
           atoms per channel; off samples rms and peak per frame: the rms of every channel's \
           samples together, the peak of any channel",
    build: |g| Representation::Envelope {
        frame_secs: g.frame,
    },
};

static DERIVATIVE: Reading = Reading {
    name: "derivative",
    takes: &[],
    reads: Reads::Either,
    help: "the derivative's atoms per channel, each series listed as `atoms` lists it; off \
           samples the first difference, (x[n] - x[n-1]) * rate at every sample but the first",
    build: |_| Representation::Derivative,
};

static SAMPLES: Reading = Reading {
    name: "samples",
    takes: &[Arg::Buffer],
    reads: Reads::Samples,
    help: "the samples themselves; a `.wav` destination writes them as audio, an \
           `alsa:<device>` one plays them as they render",
    build: |_| Representation::Samples,
};

static LEDGER: Reading = Reading {
    name: "ledger",
    takes: &[Arg::Depth, Arg::Brief, Arg::Skim],
    reads: Reads::Graph,
    help: "rms, peak, share and clipping per node under the target",
    build: |g| Representation::Ledger { depth: g.depth },
};

static PITCH: Reading = Reading {
    name: "pitch",
    takes: &[Arg::Peaks, Arg::Frame],
    reads: Reads::Either,
    help: "the `peaks` loudest notes it holds, per frame off samples, every channel's power \
           per bin averaged; off samples its peaks are picked as `spectrum` picks them, and a series' \
           fundamental is read within half a bin of its root's peak, the rate over the samples \
           its Hann spans",
    build: |g| Representation::Pitch {
        max_notes: g.peaks.unwrap_or(DEFAULT_MAX_PEAKS),
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
    },
};

static FORMANTS: Reading = Reading {
    name: "formants",
    takes: &[Arg::Peaks, Arg::Frame],
    reads: Reads::Samples,
    help: "formant frequencies, bandwidths and levels per frame, the `peaks` lowest, off \
           every channel's autocorrelation averaged, an all-pole fit of 2 poles and 1 per kHz \
           of rate, at most 64: a frame or target holding no more samples than its order \
           refuses, and a tail frame reads zero past its end",
    build: |g| Representation::Formants {
        max_formants: g.peaks.unwrap_or(DEFAULT_MAX_PEAKS),
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
    },
};

static STEREO: Reading = Reading {
    name: "stereo",
    takes: &[Arg::Frame],
    reads: Reads::Samples,
    help: "correlation, width and balance between exactly two channels, per frame; any other \
           count refuses",
    build: |g| Representation::Stereo {
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
    },
};

static BANDS: Reading = Reading {
    name: "bands",
    takes: &[],
    reads: Reads::Samples,
    help: "each auditory band's level over time, every channel's band energy averaged, at 2 kHz",
    build: |_| Representation::Bands,
};

static CREST: Reading = Reading {
    name: "crest",
    takes: &[],
    reads: Reads::Samples,
    help: "peak against rms, broadband and per band: any channel's peak against the rms of \
           every channel's samples together",
    build: |_| Representation::Crest,
};

static LOUDNESS: Reading = Reading {
    name: "loudness",
    takes: &[Arg::Window],
    reads: Reads::Samples,
    help: "integrated, momentary and short-term loudness in LUFS, the momentary blocks `window` \
           long and the integrated gated over them",
    build: |g| Representation::Loudness {
        window_secs: g.window.unwrap_or(DEFAULT_LOUDNESS_WINDOW_SECS),
    },
};

static HITS: Reading = Reading {
    name: "hits",
    takes: &[],
    reads: Reads::Samples,
    help: "its hits: every peak of an attack novelty, each one's onset, placed at the sample \
           its strike rises, its end and the next, new, a re-strike or held from the start, \
           its attack and level, its energy in its first 120 ms below, between and above \
           `bands_hz`, and its strength, how far the peak stands over the novelty about it; \
           no pitch; every channel's samples averaged",
    build: |_| Representation::Hits,
};

static LOUDNESS_PATTERN: Reading = Reading {
    name: "loudness_pattern",
    takes: &[Arg::Frame, Arg::Spl],
    reads: Reads::Samples,
    help: "specific loudness in sone per Cam, band by band every `frame`, and each frame's \
           total, after the ear's transfer and the excitation's spread, so a sound another \
           masks adds little, a full-scale sine at `spl` dB SPL; every channel's power per \
           bin averaged",
    build: |g| Representation::LoudnessPattern {
        frame_secs: g.frame.unwrap_or(DEFAULT_PATTERN_FRAME_SECS),
        spl_db: g.spl_db.unwrap_or(DEFAULT_SPL_DB),
    },
};

static SHARPNESS: Reading = Reading {
    name: "sharpness",
    takes: &[Arg::Frame, Arg::Spl],
    reads: Reads::Samples,
    help: "sharpness in acum every `frame`, DIN 45692's weighting over the specific loudness \
           `loudness_pattern` reads at the same `frame` and `spl`, null where the frame is \
           silent; every channel's power per bin averaged",
    build: |g| Representation::Sharpness {
        frame_secs: g.frame.unwrap_or(DEFAULT_PATTERN_FRAME_SECS),
        spl_db: g.spl_db.unwrap_or(DEFAULT_SPL_DB),
    },
};

static BEATING: Reading = Reading {
    name: "beating",
    takes: &[Arg::Frame, Arg::Window, Arg::Spl],
    reads: Reads::Samples,
    help: "how far each band's level swings about its own quadratic trend over a `window` \
           centred on every `frame`, in phon, the bands weighted by their loudness: beating \
           and tremolo, a note's rise and fall left in the trend; off the specific loudness \
           `loudness_pattern` reads at the same `frame` and `spl`, every channel's power per \
           bin averaged",
    build: |g| Representation::Beating {
        frame_secs: g.frame.unwrap_or(DEFAULT_PATTERN_FRAME_SECS),
        window_secs: g.window.unwrap_or(DEFAULT_BEATING_WINDOW_SECS),
        spl_db: g.spl_db.unwrap_or(DEFAULT_SPL_DB),
    },
};

static INTERNAL: Reading = Reading {
    name: "internal",
    takes: &[Arg::Spl],
    reads: Reads::Samples,
    help: "the ear's internal representation at 1 Bark by 10 ms: specific loudness by ISO \
           532-1's time-varying stages and each band's level through a modulation filterbank, \
           every element over its JND, so two subtract into an ideal observer's d' by `norm`; \
           a full-scale sine at `spl` dB SPL; every channel averaged",
    build: |g| Representation::Internal {
        spl_db: g.spl_db.unwrap_or(DEFAULT_SPL_DB),
    },
};

static ALIAS: Reading = Reading {
    name: "alias",
    takes: &[Arg::Oversample],
    reads: Reads::Graph,
    help: "what folded, against a re-render at a multiple of the rate",
    build: |g| Representation::Alias {
        oversample: g.oversample,
    },
};

static DISSONANCE: Reading = Reading {
    name: "dissonance",
    takes: &[Arg::Frame, Arg::Floor, Arg::Pairs],
    reads: Reads::Either,
    help: "Sethares' sensory roughness per frame of its partials over a floor, with its mean, \
           max and min; measured partials off every channel's power per bin averaged",
    build: |g| Representation::Harmony {
        reading: Harmonic::Dissonance {
            max_pairs: g.max_pairs,
        },
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        floor_db: g.floor_db.unwrap_or(DEFAULT_FLOOR_DB),
    },
};

static VOICE_LEADING: Reading = Reading {
    name: "voice_leading",
    takes: &[Arg::Frame, Arg::Floor, Arg::Norm],
    reads: Reads::Either,
    help: "its chords over time, and the smallest voice leading between each two, l1 or l2 \
           in semitones; measured partials off every channel's power per bin averaged",
    build: |g| Representation::Harmony {
        reading: Harmonic::VoiceLeading { norm: g.norm },
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        floor_db: g.floor_db.unwrap_or(DEFAULT_FLOOR_DB),
    },
};

static TONAL_DISTANCE: Reading = Reading {
    name: "tonal_distance",
    takes: &[Arg::Frame, Arg::Floor],
    reads: Reads::Either,
    help: "its chords over time, the Tonnetz steps between each two triads and each one's \
           from the first triad; measured partials off every channel's power per bin averaged",
    build: |g| Representation::Harmony {
        reading: Harmonic::TonalDistance,
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        floor_db: g.floor_db.unwrap_or(DEFAULT_FLOOR_DB),
    },
};

static INHARMONICITY: Reading = Reading {
    name: "inharmonicity",
    takes: &[Arg::Frame, Arg::Floor],
    reads: Reads::Either,
    help: "its harmonic series per frame: each member's cents off n f1, the stiff-string B each \
           fits, and the energy in no series; measured partials off every channel's power per \
           bin averaged",
    build: |g| Representation::Harmony {
        reading: Harmonic::Inharmonicity,
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        floor_db: g.floor_db.unwrap_or(DEFAULT_FLOOR_DB),
    },
};

static PERIODICITY: Reading = Reading {
    name: "periodicity",
    takes: &[Arg::Frame, Arg::Window, Arg::Curve],
    reads: Reads::Samples,
    help: "how strongly its onset strength repeats per sliding window, broadband, under 150 Hz \
           and over: the leading periods up to 2 s, or half the window, in seconds and bpm \
           with their bounds, their \
           autocorrelation heights, the drift between windows; every channel's band energy \
           averaged",
    build: |g| Representation::Periodicity {
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        window_secs: g.window.unwrap_or(DEFAULT_PERIODICITY_WINDOW_SECS),
        curve: g.curve,
    },
};

static MODULATION: Reading = Reading {
    name: "modulation",
    takes: &[Arg::Frame, Arg::Window, Arg::Bpm],
    reads: Reads::Samples,
    help: "how its band envelopes move per sliding window, under 150 Hz, to 2 kHz and over: \
           the leading modulation rates from 0.5 Hz, or the slowest two of whose periods the \
           window holds, to 20 Hz, in beats where `bpm` is given, \
           their depth as a modulation index, each with its bounds, the level, and the \
           spectral centroid's mean and its own swing in Hz; every channel's band energy \
           averaged",
    build: |g| Representation::Modulation {
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        window_secs: g.window.unwrap_or(DEFAULT_MODULATION_WINDOW_SECS),
        bpm: g.bpm,
    },
};

static SELF_SIMILARITY: Reading = Reading {
    name: "self_similarity",
    takes: &[
        Arg::Frame,
        Arg::Cell,
        Arg::Smooth,
        Arg::Window,
        Arg::Floor,
        Arg::Peaks,
        Arg::Curve,
    ],
    reads: Reads::Samples,
    help: "how its stretches resemble one another, by chroma and by band levels, over cells \
           of `cell`: the repeated stretches as stripes, each pair of spans with its \
           similarity and, by chroma, its transposition in semitones; Foote's novelty at every \
           cell boundary and its peaks; `curve=1` adds the cells and the matrix; every \
           channel's power per bin and band energy averaged",
    build: |g| Representation::SelfSimilarity {
        frame_secs: g.frame.unwrap_or(DEFAULT_FRAME_SECS),
        cell_secs: g.cell_secs,
        smooth_secs: g.smooth_secs,
        window_secs: g.window.unwrap_or(DEFAULT_NOVELTY_WINDOW_SECS),
        floor_db: g.floor_db.unwrap_or(DEFAULT_FLOOR_DB),
        listed: g.peaks.unwrap_or(DEFAULT_MAX_PEAKS),
        curve: g.curve,
    },
};

static PARTIALS: Reading = Reading {
    name: "partials",
    takes: &[Arg::Frame, Arg::Window, Arg::Hop, Arg::Floor, Arg::Peaks],
    reads: Reads::Samples,
    help: "each partial as a track: its frequency and amplitude as knot tables for `pwl` and \
           `pwl_integral`, tonal or breathy by its amplitude's flutter and its band's \
           coherence, each with its resolution and the residual its tables leave; peaks \
           of `frame` Hann frames' power averaged over `window`, found every `hop` over the \
           floor, the `peaks` tracks of most energy kept; every channel's samples averaged",
    build: |g| Representation::Partials {
        frame_secs: g.frame.unwrap_or(DEFAULT_PARTIALS_FRAME_SECS),
        window_secs: g.window.unwrap_or(DEFAULT_PARTIALS_WINDOW_SECS),
        hop_secs: g.hop.unwrap_or(DEFAULT_PARTIALS_HOP_SECS),
        floor_db: g.floor_db.unwrap_or(DEFAULT_PARTIALS_FLOOR_DB),
        tracks: g.peaks.unwrap_or(DEFAULT_PARTIALS_TRACKS),
    },
};

static BINDINGS: Reading = Reading {
    name: "bindings",
    takes: &[Arg::Node],
    reads: Reads::Structure,
    help: "the bindings one instance resolved, each with its source",
    build: |_| Representation::Bindings,
};

static NUMBERS: Reading = Reading {
    name: "numbers",
    takes: &[],
    reads: Reads::Structure,
    help: "every number the composition below the target writes: each literal of a reached \
           node's body and default lines, each field of a reached score's lines, with the name \
           it binds; in a solve's cost it stands alone, its bits each named number's \
           `--vary` range over its step, log2, a number no `--vary` names costing none",
    build: |_| Representation::Numbers,
};

static ARGUMENTS: Reading = Reading {
    name: "arguments",
    takes: &[],
    reads: Reads::Structure,
    help: "each builtin call's named arguments as the numbers it was lowered with",
    build: |_| Representation::Arguments,
};

/// Every reading a caller may ask for, in the order a listing prints them.
pub static READINGS: [&Reading; 30] = [
    &LINES,
    &ATOMS,
    &SPECTRUM,
    &ENVELOPE,
    &DERIVATIVE,
    &SAMPLES,
    &LEDGER,
    &PITCH,
    &FORMANTS,
    &STEREO,
    &BANDS,
    &CREST,
    &LOUDNESS,
    &HITS,
    &ALIAS,
    &DISSONANCE,
    &VOICE_LEADING,
    &TONAL_DISTANCE,
    &INHARMONICITY,
    &PERIODICITY,
    &MODULATION,
    &SELF_SIMILARITY,
    &PARTIALS,
    &LOUDNESS_PATTERN,
    &SHARPNESS,
    &BEATING,
    &INTERNAL,
    &NUMBERS,
    &BINDINGS,
    &ARGUMENTS,
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Representation {
    Lines,
    Atoms,
    Spectrum {
        max_peaks: usize,
        frame_secs: Option<f64>,
    },
    Envelope {
        frame_secs: Option<f64>,
    },
    Derivative,
    Samples,
    Ledger {
        depth: usize,
    },
    Pitch {
        max_notes: usize,
        frame_secs: f64,
    },
    Formants {
        max_formants: usize,
        frame_secs: f64,
    },
    Stereo {
        frame_secs: f64,
    },
    Bands,
    Crest,
    /// Momentary blocks of `window_secs`, the integrated loudness gated over them.
    Loudness {
        window_secs: f64,
    },
    Hits,
    Alias {
        oversample: u32,
    },
    /// The partials it is read from step by `frame_secs` and stop at `floor_db`.
    Harmony {
        reading: Harmonic,
        frame_secs: f64,
        floor_db: f64,
    },
    /// Windows of `window_secs` centred every `frame_secs`; `curve` adds each one's heights.
    Periodicity {
        frame_secs: f64,
        window_secs: f64,
        curve: bool,
    },
    /// Windows of `window_secs` centred every `frame_secs`; `bpm` adds each rate in beats.
    Modulation {
        frame_secs: f64,
        window_secs: f64,
        bpm: Option<f64>,
    },
    /// Cells of `cell_secs` over frames every `frame_secs`; `listed` stripes and peaks are
    /// listed, and `curve` adds the cells and matrices.
    SelfSimilarity {
        frame_secs: f64,
        cell_secs: f64,
        smooth_secs: f64,
        window_secs: f64,
        floor_db: f64,
        listed: usize,
        curve: bool,
    },
    Partials {
        frame_secs: f64,
        window_secs: f64,
        hop_secs: f64,
        floor_db: f64,
        tracks: usize,
    },
    LoudnessPattern {
        frame_secs: f64,
        spl_db: f64,
    },
    Sharpness {
        frame_secs: f64,
        spl_db: f64,
    },
    /// Windows of `window_secs` centred on every frame of the pattern `frame_secs` steps.
    Beating {
        frame_secs: f64,
        window_secs: f64,
        spl_db: f64,
    },
    Internal {
        spl_db: f64,
    },
    Numbers,
    Bindings,
    Arguments,
}

/// The readings taken off one curve of partials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Harmonic {
    Dissonance { max_pairs: usize },
    VoiceLeading { norm: Norm },
    TonalDistance,
    Inharmonicity,
}

/// What a caller asked of one node.
#[derive(Clone, Debug, PartialEq)]
pub struct Ask {
    pub node: String,
    pub representation: Representation,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Output {
    Lines(Vec<Line>),
    Atoms(Vec<Atom>),
    Symbolic(Box<SpectralSum>),
    Samples(Box<Buffer>),
    Spectrum(Box<Spectrum>),
    Envelope(Vec<EnvelopeFrame>),
    Pitch(Vec<PitchFrame>),
    Formants(Vec<FormantFrame>),
    Stereo(Box<StereoImage>),
    Bands(Box<Bands>),
    Crest(Box<Crest>),
    Loudness(Box<Loudness>),
    Hits(Box<Hits>),
    Alias(Box<Alias>),
    Ledger(Vec<LedgerEntry>),
    Dissonance(Box<Dissonance>),
    VoiceLeading(Box<VoiceLeading>),
    TonalDistance(Box<TonalDistance>),
    Inharmonicity(Box<Inharmonicity>),
    Periodicity(Box<Periodicity>),
    Modulation(Box<Modulation>),
    SelfSimilarity(Box<SelfSimilarity>),
    Partials(Box<Partials>),
    LoudnessPattern(Box<LoudnessPattern>),
    Sharpness(Box<Sharpness>),
    Beating(Box<Beating>),
    Internal(Box<Internal>),
    Numbers(Box<crate::Written>),
    Bindings(Vec<Binding>),
    Arguments(Vec<Arguments>),
}

/// `amp (x-polynomial_at)^degree e^(-decay (x-decay_from)) e^(2pi i turns x)` on `[start, end)`
#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub var: Var,
    /// The lane, or channel, it stands in.
    pub channel: usize,
    pub sketch: String,
    pub amp: C64,
    pub turns: f64,
    pub decay: f64,
    pub decay_from: f64,
    pub degree: u16,
    pub polynomial_at: f64,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

/// Every answer says which reading ran and under which profile, so a caller never has to
/// infer either from the shape of the value.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer {
    pub value: Output,
    pub source: Source,
    pub profile: &'static str,
    pub rate: Option<u32>,
    /// A series answers the terms it keeps, the tail beside them.
    pub dropped: Vec<sva_formula::Line>,
    pub tail_db: Option<f64>,
}

impl Answer {
    pub fn whole(
        value: Output,
        source: Source,
        profile: &'static str,
        rate: Option<u32>,
    ) -> Answer {
        Answer {
            value,
            source,
            profile,
            rate,
            dropped: Vec::new(),
            tail_db: None,
        }
    }
}

impl Representation {
    /// Exhaustive, so a variant with no row is a compile error.
    pub fn reading(self) -> &'static Reading {
        match self {
            Representation::Lines => &LINES,
            Representation::Atoms => &ATOMS,
            Representation::Spectrum { .. } => &SPECTRUM,
            Representation::Envelope { .. } => &ENVELOPE,
            Representation::Derivative => &DERIVATIVE,
            Representation::Samples => &SAMPLES,
            Representation::Ledger { .. } => &LEDGER,
            Representation::Pitch { .. } => &PITCH,
            Representation::Formants { .. } => &FORMANTS,
            Representation::Stereo { .. } => &STEREO,
            Representation::Bands => &BANDS,
            Representation::Crest => &CREST,
            Representation::Loudness { .. } => &LOUDNESS,
            Representation::Hits => &HITS,
            Representation::Alias { .. } => &ALIAS,
            Representation::Harmony { reading, .. } => match reading {
                Harmonic::Dissonance { .. } => &DISSONANCE,
                Harmonic::VoiceLeading { .. } => &VOICE_LEADING,
                Harmonic::TonalDistance => &TONAL_DISTANCE,
                Harmonic::Inharmonicity => &INHARMONICITY,
            },
            Representation::Periodicity { .. } => &PERIODICITY,
            Representation::Modulation { .. } => &MODULATION,
            Representation::SelfSimilarity { .. } => &SELF_SIMILARITY,
            Representation::Partials { .. } => &PARTIALS,
            Representation::LoudnessPattern { .. } => &LOUDNESS_PATTERN,
            Representation::Sharpness { .. } => &SHARPNESS,
            Representation::Beating { .. } => &BEATING,
            Representation::Internal { .. } => &INTERNAL,
            Representation::Numbers => &NUMBERS,
            Representation::Bindings => &BINDINGS,
            Representation::Arguments => &ARGUMENTS,
        }
    }

    pub fn name(self) -> &'static str {
        self.reading().name
    }

    pub fn consumes(self, closed: bool) -> Consumes {
        self.reading().reads.consumes(closed)
    }

    /// Whether a node's samples alone answer this reading, `sampled` where it holds samples.
    pub fn off_samples(self, sampled: bool) -> bool {
        self.reading().reads.off_samples(sampled)
    }

    pub fn from_name(name: &str) -> Option<Representation> {
        READINGS
            .iter()
            .find(|row| row.name == name)
            .map(|row| row.representation(&Given::default()))
    }
}
