// Concern: a solve's JS shapes and run over a page's clip, fresh or resumed, a composition per pass; `seed` refused | Non-concern: the method's binding (lib.rs) | IO: (clip, options) -> `data`

use std::pin::Pin;

use sva_core::solve::atoms::files_at;
use sva_core::solve::fit::{Options, Resume, SPL_DB, fit, mono};
use sva_core::solve::optimize::TOLERANCE;
use sva_core::solve::report::{answer, composition, progress};
use sva_core::{Abandon, success_envelope};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};

use crate::analyze::planes_of;
use crate::{aborted, parse, refuse, thrown, yielded};

#[wasm_bindgen(typescript_custom_section)]
const SOLVE_CLIP: &str = r#"
/** A recording passed whole: one Float32Array per channel, at `rate` Hz. */
export interface SolveClip {
    rate: number;
    channels: Float32Array[];
}
/**
 * `spl`, a full-scale sine's dB SPL (default 103.95), as `sva-cli solve` takes `--spl`;
 * `tolerance`, the stopping threshold on a pass's relative drop, as `--tolerance`: the fit
 * ends after a pass lowering the description length by no more than `tolerance` times the
 * larger of its value before and after (and 1), `ended` then `"tolerance"`; default 0.001, a
 * resumed fit's 0, so it passes on till no move lowers it or the page stops it;
 * `from`, the fit to continue, as `--from`: a previous answer's `data` as it stands, or its
 * files and the passes that made them, the pass numbers going on from there;
 * `progress` hears each line the CLI prints on stderr, one per pass, each beside the
 * composition that pass leaves; `signal`, read before each pass and between its moves once
 * the page has had its turn: aborted, the fit ends there, the pass under way dropped,
 * `ended` `"stopped"`.
 */
export interface SolveOptions {
    spl?: number;
    tolerance?: number;
    from?: SolveFrom;
    progress?: (line: SolveProgress) => void;
    signal?: AbortSignal;
}
/** One file of a composition. */
export interface SolveFile {
    path: string;
    text: string;
}
/** A fit to continue: an answer's `data`, or its `files` and `passes` (default 0). */
export type SolveFrom =
    | SolveAnswer
    | { files: SolveFile[] | { items: SolveFile[] }; passes?: number };
"#;

#[wasm_bindgen(typescript_custom_section)]
const SOLVE_PROGRESS: &str = r#"
/** A solve's `progress` line: the start, then one per pass. */
export type SolveProgress = SolveStart | SolveStep;
/** What a pass added or adjusted; `null` where its kind has none. */
export interface SolveAtom {
    key: number;
    kind: "partial" | "harmonic" | "noise" | "shaped";
    hz: number | null;
    /** A partial's or voice's glide in nepers per second, its Hz `hz` e^(glide t); 0 fixed. */
    glide: number;
    /** A voice's B: harmonic k at k sqrt(1 + B k^2) times `hz`. */
    stretch: number | null;
    harmonics: number;
    /** A noise band's edges; a shaped noise's lowest and highest. */
    lo_hz: number | null;
    hi_hz: number | null;
    /** A shaped noise's bands, each its gain in dB over its one shared level. */
    bands: { lo_hz: number; hi_hz: number; gain_db: number }[] | null;
}
/**
 * One pass, one move: `action` an add of a new atom or an adjust of a kept one, `adjust` what
 * that adjust changed (null for an add), `move` both in words (`"add partial"`, `"level
 * harmonic"`, `"start"`), `from_db` and `to_db` the atom's level before and
 * after (null where it was not, or an adjust to nought removed it); `delta_dl_bits` the
 * pass's change to the description length, `drop_bits` its negation, `total_drop_bits` the
 * drop from the empty fit's description length to `dl_bits`, `share` the pass's drop over that
 * (null where it is none), `dl_bits` the total, `atom_bits` plus
 * `residual_bits`; `residual_db[frame][band]` the power
 * the fit leaves in each octave band in dB, null where none; `envelopes` each kept atom's
 * written level per frame in dB; `skipped` the atoms of better moves passed over, their nodes
 * refused; `seconds` the clock since the solve began as it ended, `pass_secs` its pass's own
 * time; `files` the composition it leaves, as the answer's `files`, to render as it arrives.
 * `pass` counts from 1, a fresh start's 0, a resumed start's the pass it resumes from.
 */
export interface SolveStep {
    pass: number;
    action: "start" | "add" | "adjust";
    adjust: "level" | "remove" | "frequency" | "rethin" | null;
    move: string;
    atom: SolveAtom | null;
    from_db: number | null;
    to_db: number | null;
    delta_dl_bits: number;
    drop_bits: number;
    total_drop_bits: number;
    share: number | null;
    dl_bits: number;
    atom_bits: number;
    residual_bits: number;
    residual_db: (number | null)[][];
    envelopes: { items: { key: number; db: number[] }[] };
    skipped: { items: SolveAtom[] };
    seconds: number;
    pass_secs: number;
    files: { items: SolveFile[] };
}
/** The first line, before any pass: the grid's shape beside the line of the fit it starts from. */
export interface SolveStart extends SolveStep {
    action: "start";
    rate_hz: number;
    frame_secs: number;
    first_centre_secs: number;
    band_edges_hz: [number, number][];
    band_floor_db: number[];
    gain_db: number;
}
/** The answer's `data`: the fit's counts and description length, and its composition. */
export interface SolveAnswer {
    fit: {
        rate_hz: number;
        passes: number;
        adds: number;
        adjusts: number;
        adjusts_by: { level: number; remove: number; frequency: number; rethin: number };
        partials: number;
        harmonics: number;
        noises: number;
        shaped: number;
        dl_bits: number;
        atom_bits: number;
        residual_bits: number;
        gain_db: number;
        /** The pass it resumed from, 0 for a fresh fit; `passes` the last it made. */
        from_pass: number;
        tolerance: number;
        ended: "converged" | "tolerance" | "stopped";
        stopped: boolean;
        secs: number;
        atoms: { items: SolveAtom[] };
        wrote: string[];
    };
    files: { items: SolveFile[] };
}
"#;

/// A solve's `signal`, asked before each pass once the page has had its turn, so the page
/// paints its progress and an abort it queued arrives mid-fit.
struct Paced(Option<JsValue>);

impl Abandon for Paced {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(async move {
            yielded().await;
            self.0.as_ref().is_some_and(aborted)
        })
    }
}

fn number(options: &JsValue, key: &str) -> Option<f64> {
    js_sys::Reflect::get(options, &key.into())
        .ok()
        .and_then(|v| v.as_f64())
}

fn get(object: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(object, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

/// The fit `from` names, as `SolveFrom` states.
fn resume(from: &JsValue) -> Result<Resume, JsValue> {
    let help = "pass a previous answer's `data`, or `{ files, passes }`";
    let files = get(from, "files");
    let files = match js_sys::Array::is_array(&files) {
        true => files,
        false => get(&files, "items"),
    };
    if !js_sys::Array::is_array(&files) {
        return Err(refuse("`from.files` is no list of files".into(), help));
    }
    let files = js_sys::Array::from(&files)
        .iter()
        .map(|f| get(&f, "path").as_string().zip(get(&f, "text").as_string()))
        .collect::<Option<Vec<(String, String)>>>()
        .ok_or_else(|| refuse("a file in `from.files` is no `{ path, text }`".into(), help))?;
    let passes = [get(from, "passes"), get(&get(from, "fit"), "passes")]
        .into_iter()
        .find(|v| !v.is_undefined())
        .map_or(Some(0.0), |v| v.as_f64())
        .filter(|n| n.fract() == 0.0 && *n >= 0.0 && *n <= u32::MAX as f64)
        .ok_or_else(|| refuse("`from.passes` is no whole count".into(), help))?;
    Resume::of(&files, passes as usize).map_err(|e| thrown(&e))
}

/// `sva-cli solve`'s envelope over `clip`, its channels averaged, under `options`
/// (`SolveOptions`, already checked by name).
pub(crate) async fn solved(clip: &JsValue, options_js: &JsValue) -> Result<JsValue, JsValue> {
    let rate = number(clip, "rate")
        .filter(|r| r.fract() == 0.0 && *r >= 1.0 && *r <= f64::from(u32::MAX))
        .ok_or_else(|| {
            refuse(
                "`clip.rate` is no whole Hz".into(),
                "pass { rate: 48000, channels }",
            )
        })? as u32;
    let channels = js_sys::Reflect::get(clip, &"channels".into())
        .ok()
        .filter(js_sys::Array::is_array)
        .map(|v| js_sys::Array::from(&v))
        .ok_or_else(|| {
            refuse(
                "`clip.channels` is no array".into(),
                "pass one Float32Array per channel",
            )
        })?;
    let planes = planes_of(channels)?;
    let samples = mono(&planes);
    let spl = get(options_js, "spl");
    let spl_db = match spl.is_undefined() {
        true => SPL_DB,
        false => spl.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            refuse(
                "`spl` is no finite number".into(),
                "pass a full-scale sine's dB SPL, or leave it out",
            )
        })?,
    };
    let from = get(options_js, "from");
    let from = match from.is_undefined() {
        true => None,
        false => Some(resume(&from)?),
    };
    let tolerance = get(options_js, "tolerance");
    let tolerance = match tolerance.is_undefined() {
        true => match from {
            Some(_) => 0.0,
            None => TOLERANCE,
        },
        false => tolerance
            .as_f64()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .ok_or_else(|| {
                refuse(
                    "`tolerance` is no finite number from 0".into(),
                    "pass a pass's least drop as a share of the description length, or leave it out",
                )
            })?,
    };
    let options = Options {
        tolerance,
        from,
        threads: crate::threads::limit(None),
    };
    let hear = js_sys::Reflect::get(options_js, &"progress".into())
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
    let signal = js_sys::Reflect::get(options_js, &"signal".into())
        .ok()
        .filter(|s| !s.is_undefined());
    let started = js_sys::Date::now() / 1000.0;
    let clock = || js_sys::Date::now() / 1000.0 - started;
    let mut gain_db = 0.0;
    let fitted = fit(
        &samples,
        rate,
        spl_db,
        &options,
        &mut |line| {
            if let Some(g) = &line.geometry {
                gain_db = g.gain_db;
            }
            if let Some(hear) = &hear {
                let line_js = parse(&progress(line)).unwrap_or(JsValue::NULL);
                let files = files_at(&line.atoms, &line.clip, gain_db);
                let files = parse(&composition(&files)).unwrap_or(JsValue::NULL);
                let _ = js_sys::Reflect::set(&line_js, &"files".into(), &files);
                let _ = hear.call1(&JsValue::NULL, &line_js);
            }
        },
        &Paced(signal),
        &clock,
    )
    .await
    .map_err(|e| thrown(&e))?;
    let files = files_at(&fitted.atoms, &fitted.clip, fitted.gain_db);
    parse(&success_envelope(
        &answer(&fitted, clock(), &[], Some(&files)),
        &[],
    ))
}

/// `limits`: `{ harmonics, lone }`, each optional, as `new --from` takes `--harmonics` and
/// `--lone`. This build checks them and throws `seed.unbuilt`.
#[wasm_bindgen]
pub fn seed(
    rate: u32,
    channels: js_sys::Array,
    clip: &str,
    limits: Option<js_sys::Object>,
) -> Result<JsValue, JsValue> {
    planes_of(channels)?;
    if let Some(limits) = limits {
        for key in ["harmonics", "lone"] {
            let value = js_sys::Reflect::get(&limits, &key.into()).unwrap_or(JsValue::UNDEFINED);
            let least = f64::from(u8::from(key == "harmonics"));
            if !value.is_undefined()
                && !value
                    .as_f64()
                    .is_some_and(|n| n.fract() == 0.0 && n >= least)
            {
                return Err(refuse(
                    format!("`limits.{key}` is no whole number from {least}"),
                    "pass `{ harmonics: 48, lone: 8 }` or leave a key out for its default",
                ));
            }
        }
    }
    let _ = (rate, clip);
    Err(thrown(&sva_core::solve::seed_unbuilt()))
}
