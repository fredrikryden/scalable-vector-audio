// Concern: reads a solve's cost tree, recordings and variables as the plain objects a page builds, references as values | Non-concern: typing or running it (sva-core) | IO: (JsValue) -> Tree, Vec<Vary>

use std::sync::Arc;

use sva_ast::Recording;
use sva_core::solve::{OPERATORS, REDUCERS, REPR, Target, Tree, Vary, takes, vary};
use wasm_bindgen::{JsCast, JsValue};

use crate::analyze::planes_of;
use crate::{refuse, thrown};

#[wasm_bindgen::prelude::wasm_bindgen(typescript_custom_section)]
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

const FORMS: &str = "a number, a list of weights `[1, 0.5]`, `{ repr: \"envelope(frame=10ms)\", target: \
     \"@lead([0, 5s])\" }`, `{ value: <a number, numbers or a reading's data> }`, `{ add: [a, b] }`, \
     `{ sub: [a, b] }`, `{ mul: [a, b] }`, `{ div: [a, b] }`, `{ neg: a }`, `{ norm: v }`, \
     `{ sum: v }`, `{ max: v }`, `{ abs: x }` or `{ dot: [[1, 0.5], v] }`";

fn bad(at: &str, why: String) -> JsValue {
    refuse(
        format!("`{at}` {why}"),
        &format!("each node of a cost is one of {FORMS}"),
    )
}

/// An object's own keys and values.
fn entries(value: &JsValue) -> Option<Vec<(String, JsValue)>> {
    let object = value.dyn_ref::<js_sys::Object>()?;
    if js_sys::Array::is_array(value) {
        return None;
    }
    Some(
        js_sys::Object::entries(object)
            .iter()
            .map(|pair| {
                let pair = js_sys::Array::from(&pair);
                (pair.get(0).as_string().unwrap_or_default(), pair.get(1))
            })
            .collect(),
    )
}

fn list(value: &JsValue, at: &str, least: usize) -> Result<Vec<JsValue>, JsValue> {
    if !js_sys::Array::is_array(value) {
        return Err(bad(at, format!("is no list of {least} or more")));
    }
    let items: Vec<JsValue> = js_sys::Array::from(value).iter().collect();
    match items.len() >= least {
        true => Ok(items),
        false => Err(bad(
            at,
            format!("holds {}, and takes {least} or more", items.len()),
        )),
    }
}

fn weights(value: &JsValue) -> Option<Vec<f64>> {
    js_sys::Array::is_array(value)
        .then(|| {
            js_sys::Array::from(value)
                .iter()
                .map(|v| v.as_f64())
                .collect()
        })
        .flatten()
}

/// `value` as one node of a cost, `at` its place, as `cost.norm.sub[1]`.
pub(crate) fn tree_of(value: &JsValue, at: &str) -> Result<Tree, JsValue> {
    if let Some(n) = value.as_f64() {
        return Ok(Tree::Num(n));
    }
    if js_sys::Array::is_array(value) {
        return weights(value)
            .map(Tree::Weights)
            .ok_or_else(|| bad(at, "is a list holding more than numbers".to_string()));
    }
    let Some(fields) = entries(value) else {
        return Err(bad(at, "is no number, list or object".to_string()));
    };
    let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
    let field = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    if keys.contains(&REPR) {
        if let Some(extra) = keys.iter().find(|k| ![REPR, "target"].contains(k)) {
            return Err(bad(
                &format!("{at}.{extra}"),
                "is no field of a leaf, which holds `repr` and `target`".to_string(),
            ));
        }
        let call = field(REPR).and_then(JsValue::as_string).ok_or_else(|| {
            bad(
                &format!("{at}.{REPR}"),
                "is no reading's call, as `\"envelope(frame=10ms)\"`".to_string(),
            )
        })?;
        let target = field("target")
            .ok_or_else(|| bad(&format!("{at}.target"), "is missing".to_string()))?;
        return Ok(Tree::Repr {
            call,
            target: target_of(target, &format!("{at}.target"))?,
        });
    }
    let [key] = keys.as_slice() else {
        return Err(bad(
            at,
            format!("holds the keys {}, and a node holds one", keys.join(", ")),
        ));
    };
    let inner = field(key).expect("listed");
    let here = format!("{at}.{key}");
    let one = |v: &JsValue| tree_of(v, &here);
    Ok(match *key {
        "value" => {
            let text = js_sys::JSON::stringify(inner)
                .ok()
                .and_then(|t| t.as_string())
                .ok_or_else(|| bad(&here, "is no JSON value".to_string()))?;
            let held = serde_json::from_str(&text)
                .map_err(|e| bad(&here, format!("is no JSON value: {e}")))?;
            Tree::Value(held)
        }
        op if OPERATORS.iter().any(|(name, _)| *name == op) => {
            let (_, op) = *OPERATORS
                .iter()
                .find(|(name, _)| *name == op)
                .expect("found");
            let items = list(inner, &here, 2)?;
            let mut items = items
                .iter()
                .enumerate()
                .map(|(k, v)| tree_of(v, &format!("{here}[{k}]")));
            let first = items.next().expect("two or more")?;
            items.try_fold(first, |a, b| {
                Ok::<_, JsValue>(Tree::Bin(op, Box::new(a), Box::new(b?)))
            })?
        }
        "neg" => Tree::Neg(Box::new(one(inner)?)),
        reducer if REDUCERS.contains(&reducer) => {
            let reducer = *REDUCERS.iter().find(|r| **r == reducer).expect("found");
            let args = match takes(reducer) {
                1 => vec![one(inner)?],
                n => {
                    let items = list(inner, &here, n)?;
                    if items.len() != n {
                        return Err(bad(&here, format!("takes {n}, as `[[1, 0.5], v]`")));
                    }
                    items
                        .iter()
                        .enumerate()
                        .map(|(k, v)| tree_of(v, &format!("{here}[{k}]")))
                        .collect::<Result<Vec<_>, _>>()?
                }
            };
            Tree::Reduce(reducer, args)
        }
        "samples" => {
            return Err(bad(
                &here,
                "is a recording, which only a leaf's `target` reads".to_string(),
            ));
        }
        other => {
            return Err(bad(
                &format!("{at}.{other}"),
                "is no node of a cost".to_string(),
            ));
        }
    })
}

/// A render expression, or `{ samples: { rate, channels } }`, a recording passed whole.
fn target_of(value: &JsValue, at: &str) -> Result<Target, JsValue> {
    if let Some(text) = value.as_string() {
        return Ok(Target::Text(text));
    }
    let inner = entries(value)
        .and_then(|f| match f.as_slice() {
            [(key, inner)] if key == "samples" => Some(inner.clone()),
            _ => None,
        })
        .ok_or_else(|| {
            bad(
                at,
                "is no render expression and no `{ samples: { rate, channels } }`".to_string(),
            )
        })?;
    let (rate, planes) = samples_of(&inner, &format!("{at}.samples"), &bad)?;
    Ok(Target::Samples { rate, planes })
}

/// `{ rate, channels }`, one Float32Array per channel at `rate` Hz; `bad` refuses one field.
fn samples_of(
    value: &JsValue,
    at: &str,
    bad: &dyn Fn(&str, String) -> JsValue,
) -> Result<(u32, Vec<Vec<f32>>), JsValue> {
    let samples = entries(value).ok_or_else(|| {
        bad(
            at,
            "is no recording, which holds `rate` and `channels`".to_string(),
        )
    })?;
    let field = |key: &str| {
        samples
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    if let Some((extra, _)) = samples
        .iter()
        .find(|(k, _)| !["rate", "channels"].contains(&k.as_str()))
    {
        return Err(bad(
            &format!("{at}.{extra}"),
            "is no field of a recording, which holds `rate` and `channels`".to_string(),
        ));
    }
    let rate = field("rate")
        .and_then(|r| r.as_f64())
        .filter(|r| *r >= 1.0 && r.fract() == 0.0 && *r <= f64::from(u32::MAX))
        .ok_or_else(|| {
            bad(
                &format!("{at}.rate"),
                "is no rate in whole hertz, as `AudioBuffer.sampleRate`".to_string(),
            )
        })?;
    let channels = field("channels")
        .filter(js_sys::Array::is_array)
        .ok_or_else(|| {
            bad(
                &format!("{at}.channels"),
                "is no list of Float32Arrays".to_string(),
            )
        })?;
    let channels = js_sys::Array::from(&channels);
    for (c, one) in channels.iter().enumerate() {
        if !one.is_instance_of::<js_sys::Float32Array>() {
            return Err(bad(
                &format!("{at}.channels[{c}]"),
                "is no Float32Array, as `AudioBuffer.getChannelData(c)`".to_string(),
            ));
        }
    }
    Ok((rate as u32, planes_of(channels)?))
}

/// `{ "clip.wav": { rate, channels } }`: the recording each `sample("<path>")` of a cost's text
/// reads, as the CLI reads the file at that path.
pub(crate) fn recordings_of(value: &JsValue) -> Result<Vec<Arc<Recording>>, JsValue> {
    let held =
        entries(value).ok_or_else(|| bad_recording("recordings", "is no object".to_string()))?;
    held.iter()
        .map(|(path, one)| {
            let (rate, planes) =
                samples_of(one, &format!("recordings[\"{path}\"]"), &bad_recording)?;
            Ok(Arc::new(Recording {
                path: path.clone(),
                rate,
                planes: Arc::new(planes),
            }))
        })
        .collect()
}

fn bad_recording(at: &str, why: String) -> JsValue {
    refuse(
        format!("`{at}` {why}"),
        "`recordings` is `{ \"clip.wav\": { rate: 48000, channels: [Float32Array] } }`, each \
         path one `sample(\"<path>\")` of the cost reads",
    )
}

/// `{ name, lo, hi, step?, log? }`, as `--vary <name>=<lo>..<hi>[/<step>][,log]` reads it.
pub(crate) fn vary_of(values: &js_sys::Array) -> Result<Vec<Vary>, JsValue> {
    values
        .iter()
        .enumerate()
        .map(|(k, value)| {
            let at = format!("vary[{k}]");
            if let Some(written) = value.as_string() {
                return vary(&written).map_err(|e| thrown(&e));
            }
            let fields = entries(&value).ok_or_else(|| bad_vary(&at, "is no object or text"))?;
            let mut held = (None, None, None, None, false);
            for (key, v) in &fields {
                let text = || {
                    v.as_string()
                        .or_else(|| v.as_f64().map(|n| n.to_string()))
                        .ok_or_else(|| bad_vary(&format!("{at}.{key}"), "is no number or text"))
                };
                match key.as_str() {
                    "name" => {
                        held.0 = Some(
                            v.as_string()
                                .ok_or_else(|| bad_vary(&format!("{at}.name"), "is no name"))?,
                        )
                    }
                    "lo" => held.1 = Some(text()?),
                    "hi" => held.2 = Some(text()?),
                    "step" => held.3 = Some(text()?),
                    "log" => {
                        held.4 = v
                            .as_bool()
                            .ok_or_else(|| bad_vary(&format!("{at}.log"), "is not true or false"))?
                    }
                    other => {
                        return Err(bad_vary(
                            &format!("{at}.{other}"),
                            "is no field of a variable",
                        ));
                    }
                }
            }
            let (Some(name), Some(lo), Some(hi)) = (held.0, held.1, held.2) else {
                return Err(bad_vary(&at, "needs `name`, `lo` and `hi`"));
            };
            let step = held.3.map_or(String::new(), |s| format!("/{s}"));
            let log = if held.4 { ",log" } else { "" };
            vary(&format!("{name}={lo}..{hi}{step}{log}")).map_err(|e| thrown(&e))
        })
        .collect()
}

fn bad_vary(at: &str, why: &str) -> JsValue {
    refuse(
        format!("`{at}` {why}"),
        "a variable is `{ name: \"cutoff\", lo: \"200hz\", hi: \"8000hz\", step: 1.25, log: true }`, \
         `step` and `log` optional, as `--vary` reads them",
    )
}
