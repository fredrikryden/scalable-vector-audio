// Concern: reads PCM a page holds as `sva-cli analyze` reads a file | Non-concern: decoding audio (the page), taking a reading (sva-core) | IO: (target, rate, Float32Arrays, readings) -> `data`

use sva_core::{Printed, Report, analyzed, query_data};
use sva_engine::{Buffer, PSYCHOACOUSTIC_V1};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};

use crate::{options_of, parse, refuse, representations_of, thrown};

/// One Float32Array per channel, every one as long as the first.
pub(crate) fn planes_of(channels: js_sys::Array) -> Result<Vec<Vec<f32>>, JsValue> {
    let planes = channels
        .iter()
        .enumerate()
        .map(|(c, one)| {
            one.dyn_into::<js_sys::Float32Array>()
                .map(|plane| plane.to_vec())
                .map_err(|_| {
                    refuse(
                        format!("channel {c} is no Float32Array"),
                        "pass one Float32Array per channel, as `AudioBuffer.getChannelData(c)`",
                    )
                })
        })
        .collect::<Result<Vec<Vec<f32>>, JsValue>>()?;
    let len = planes.first().map(Vec::len).ok_or_else(|| {
        refuse(
            "no channel was passed".to_string(),
            "pass one Float32Array per channel, as `[left, right]`",
        )
    })?;
    if let Some(c) = planes.iter().position(|p| p.len() != len) {
        return Err(refuse(
            format!(
                "channel {c} holds {} samples and channel 0 holds {len}",
                planes[c].len()
            ),
            "pass channels of one length, cut from one span",
        ));
    }
    Ok(planes)
}

/// The object `sva-cli analyze` puts under `data` for these channels at `rate`, refusing as
/// it refuses. `options.limit`: how many values each array holds, or `null` for every
/// value, as a reading `sva-cli` writes to a file.
#[wasm_bindgen]
pub fn analyze(
    target: &str,
    rate: u32,
    channels: js_sys::Array,
    representations: Vec<String>,
    options: JsValue,
) -> Result<JsValue, JsValue> {
    let options = options_of(&options, &["limit"])?;
    if rate == 0 {
        return Err(refuse(
            "a rate of 0 Hz places no sample".to_string(),
            "pass the rate the channels were sampled at, as `48000`",
        ));
    }
    let planes: Vec<Vec<f64>> = planes_of(channels)?
        .into_iter()
        .map(|plane| plane.into_iter().map(f64::from).collect())
        .collect();
    let len = planes[0].len();
    let asked = representations_of(&representations)?;
    let buffer = Buffer::of_planes(rate, planes);
    let taken = analyzed(target, &buffer, &asked).map_err(|e| thrown(&e))?;
    let answers: Vec<Printed> = asked
        .iter()
        .zip(taken)
        .map(|(one, answer)| Printed {
            name: one.name.clone(),
            answer,
            skim: one.skim,
        })
        .collect();
    parse(&query_data(&Report {
        target,
        rate,
        bits: None,
        interval: Some((0.0, len as f64 / f64::from(rate))),
        profile: PSYCHOACOUSTIC_V1.name,
        label: None,
        written: &[],
        played: None,
        answers: &answers,
        limit: options.limit.0,
    }))
}
