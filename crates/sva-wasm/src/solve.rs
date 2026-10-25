// Concern: a solve's JS shapes, and the `seed` export refused while the solver is rebuilt | Non-concern: the solve method (lib.rs), decoding audio | IO: (rate, planes, name, limits) -> a refusal

use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::analyze::planes_of;
use crate::{refuse, thrown};

#[wasm_bindgen(typescript_custom_section)]
const SOLVE_COST: &str = r#"
/** A recording passed whole: one Float32Array per channel, at `rate` Hz. */
export interface Samples {
    samples: { rate: number; channels: Float32Array[] };
}
/**
 * One node of a solve's cost; each node holds one key. A number, or a weight vector `[1, 0.5]`;
 * a leaf `{ repr: "envelope(frame=10ms)", target: "@lead([0, 5s])" }`, the reading as `render`'s
 * representations spell it and its target a render expression or a recording passed whole;
 * `{ value }`, a number, a list of numbers or a reading's `data` as `analyze` answers it,
 * checked against the leaf it subtracts from; `add`, `sub`, `mul` and `div` over two or more,
 * left to right; `neg`; the reducers `norm`, `sum`, `max`, `abs`, and `dot: [weights, v]`.
 * `{ repr: "numbers", target: "@master" }` is the bits a composition writes, which
 * `method: "prox"` prices.
 */
export type SolveCost =
    | number
    | number[]
    | { repr: string; target: string | Samples }
    | { value: unknown }
    | { add: SolveCost[] } | { sub: SolveCost[] } | { mul: SolveCost[] } | { div: SolveCost[] }
    | { neg: SolveCost }
    | { norm: SolveCost } | { sum: SolveCost } | { max: SolveCost } | { abs: SolveCost }
    | { dot: [number[], SolveCost] };
/**
 * The recording each `sample("<path>")` of a cost's text reads, under that path:
 * `{ "clip.wav": { rate: 48000, channels: [plane] } }`, as the CLI reads the file there.
 */
export type SolveRecordings = Record<string, Samples["samples"]>;
/** One variable a solve moves, as `--vary <name>=<lo>..<hi>[/<step>][,log]`, or that text. */
export type SolveVary = string | SolveVaryFields;
export interface SolveVaryFields {
    name: string;
    lo: string | number;
    hi: string | number;
    step?: string | number;
    log?: boolean;
}
"#;

#[wasm_bindgen(typescript_custom_section)]
const SOLVE_PROGRESS: &str = r#"
/**
 * A priced solve's `progress` line: a step, an activity, a stage as it finishes, or the activity
 * a refusal stopped, heard before the solve throws.
 */
export type SolveProgress = SolveStep | SolveActivity | SolveStageFinished | SolveFailed;
/**
 * Every 50th step and the step a phase converges or reaches its cap at, `ended` why it did,
 * null otherwise. A trial's `stage` is `<stage> trial <edit>`.
 */
export interface SolveStep {
    stage: string;
    phase: string;
    price: number;
    step: number;
    draws: number;
    residual_bits: number;
    numbers: number;
    bits: number;
    ended: "converged" | "cap" | "signal" | null;
}
/** What a priced solve is doing. */
export type SolveAction =
    | "seeding"
    | "reading_reference"
    | "rendering_noise"
    | "descending"
    | "debiasing"
    | "ranking_proposals"
    | "trying_proposal"
    | "evaluating_at_cap"
    | "measuring";
/**
 * Heard as an activity starts, as the one it ran inside resumes, and between its units once
 * `heartbeat` seconds (default 1) have passed since the last line; 0 hears one between every
 * unit. `edit` is the structure edit a trial tries. `count` is the activity's own, `of` at most
 * that many steps or trials. `part` is the stage's place in the scheme, `of` at most that many.
 * `work.comparisons`: cell comparisons since the solve began, a counted cell of the reference's
 * `internal` reading against the same cell of one draw's reading of the model, an evaluation
 * making cells times its draws and measuring the same off its renders; `per_second` its rate
 * over the last window at least `heartbeat` long, null before one.
 */
export interface SolveActivity {
    action: SolveAction;
    stage: string | null;
    edit: string | null;
    count: { done: number; of: number; unit: "render" | "step" | "proposal" | "trial" | "draw" } | null;
    part: { at: number; of: number } | null;
    work: { comparisons: number; per_second: number | null };
    seconds: number;
}
/** A residual as `data.stages` lists it. */
export interface SolveResidual {
    residual_bits: number;
    standard_error: number;
    d_prime: number;
    per_cell: number;
}
/**
 * A stage as `data.stages` will list it, heard as it finishes, with its terms: `length_bits` is
 * `predicted_at_cap.residual_bits` plus `bits`, null where stopped; `residual_bits` the residual
 * at the draws it was solved at. `files` the `{ path, text }`s `write` would insert for it,
 * `least` true where its `length_bits` is below every earlier stage's, so the last such is
 * `data.chosen`. Writing them loads the stage's composition once and renders nothing, cheap
 * next to its solve.
 */
export interface SolveStageFinished {
    stage: string;
    finished: true;
    stopped: boolean;
    length_bits: number | null;
    residual_bits: number;
    bits: number;
    numbers: number;
    predicted_at_cap: SolveResidual | null;
    least: boolean;
    files: { items: { path: string; text: string }[] };
}
/**
 * The activity a refusal stopped, `action` null where it came before any; `code` the thrown
 * refusal's.
 */
export type SolveFailed = { failed: true; code: string } & (SolveActivity | { action: null });
"#;

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
