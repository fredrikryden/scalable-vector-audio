// Concern: a solve's JS shapes and its run over a clip a page holds; `seed` refused while rebuilt | Non-concern: the method's binding (lib.rs) | IO: (clip, options) -> `data`

use std::pin::Pin;

use sva_core::solve::pyramid::{Asked, LEVELS, SPL_DB, mono, pyramid};
use sva_core::solve::report::{answer, progress};
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
 * `levels` (default 4) and `spl`, a full-scale sine's dB SPL (default 103.95), as `sva-cli
 * solve` takes `--levels` and `--spl`; `progress` hears each line the CLI prints on stderr;
 * `signal`, read before each descent's step once the page has had its turn: aborted, the run
 * ends with the stage it interrupts, measured as one that finished.
 */
export interface SolveOptions {
    levels?: number;
    spl?: number;
    progress?: (line: SolveProgress) => void;
    signal?: AbortSignal;
}
"#;

#[wasm_bindgen(typescript_custom_section)]
const SOLVE_PROGRESS: &str = r#"
/** A solve's `progress` line: a descent's last step, an activity, or a stage as it finishes. */
export type SolveProgress = SolveStep | SolveActivity | SolveStageFinished;
/** The step a descent ends at, `ended` why it did; `step` its steps. */
export interface SolveStep {
    stage: string;
    phase: "solve";
    price: number;
    step: number;
    draws: number;
    residual_bits: number;
    numbers: number;
    bits: number;
    ended: "converged" | "cap" | "signal";
}
/** What a solve is doing. */
export type SolveAction = "reading_reference" | "ranking_proposals" | "descending" | "measuring";
/**
 * Heard as an activity starts. `stage` is `level k of n`; `edit` the atom a proposal or a
 * descent concerns; `count` the activity's own, `of` null as nothing bounds it; `part` the
 * level's place, `of` the levels. `work.comparisons`: elements of the clip's `internal`
 * reading compared with the model's since the solve began, `per_second` null.
 */
export interface SolveActivity {
    action: SolveAction;
    stage: string;
    edit: string | null;
    count: { done: number; of: null; unit: "atom" | "proposal" | "evaluation" } | null;
    part: { at: number; of: number };
    work: { comparisons: number; per_second: null };
    seconds: number;
}
/**
 * A level heard as it finishes: `length_bits` its residual bits at s = 1 plus its B(x) at the
 * precisions s = 1 reads for this level's atoms, which price every level so far alike;
 * `residual_bits` and `bits` at its own scale; `files` its composition, `least` true where its
 * `length_bits` is at most every earlier level's so priced. `data.stages` prices every level
 * at the last one's, so the last `least` there is `data.chosen`.
 */
export interface SolveStageFinished {
    stage: string;
    finished: true;
    stopped: boolean;
    length_bits: number;
    residual_bits: number;
    bits: number;
    numbers: number;
    predicted_at_cap: null;
    least: boolean;
    seconds: number;
    files: { items: { path: string; text: string }[] };
}
"#;

/// A solve's `signal`, asked before each descent's step once the page has had its turn, so the
/// page paints its progress and an abort it queued arrives mid-level.
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

/// `sva-cli solve`'s envelope over `clip`, its channels averaged, under `options`
/// (`SolveOptions`, already checked by name).
pub(crate) async fn solved(clip: &JsValue, options: &JsValue) -> Result<JsValue, JsValue> {
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
    let levels = match number(options, "levels") {
        None => LEVELS,
        Some(l) if l.fract() == 0.0 && l >= 1.0 => l as usize,
        Some(_) => {
            return Err(refuse(
                "`levels` is no whole number from 1".into(),
                "pass 4, or leave it out",
            ));
        }
    };
    let spl = js_sys::Reflect::get(options, &"spl".into()).unwrap_or(JsValue::UNDEFINED);
    let spl_db = match spl.is_undefined() {
        true => SPL_DB,
        false => spl.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
            refuse(
                "`spl` is no finite number".into(),
                "pass a full-scale sine's dB SPL, or leave it out",
            )
        })?,
    };
    let hear = js_sys::Reflect::get(options, &"progress".into())
        .ok()
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
    let signal = js_sys::Reflect::get(options, &"signal".into())
        .ok()
        .filter(|s| !s.is_undefined());
    let stages = pyramid(
        &samples,
        rate,
        &Asked {
            levels,
            spl_db,
            threads: sva_engine::default_threads(),
        },
        &mut |event| {
            if let Some(hear) = &hear {
                let line = parse(&progress(&event)).unwrap_or(JsValue::NULL);
                let _ = hear.call1(&JsValue::NULL, &line);
            }
        },
        &Paced(signal),
        &|| js_sys::Date::now() / 1000.0,
    )
    .await;
    parse(&success_envelope(&answer(&stages, levels, &[]), &[]))
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
