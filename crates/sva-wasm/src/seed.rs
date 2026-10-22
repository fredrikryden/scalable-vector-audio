// Concern: hands a page the priced fit's starter for a clip it holds, as `new --from` writes it | Non-concern: the starter itself, decoding audio | IO: (rate, planes, name, limits) -> files, dropped

use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::analyze::planes_of;
use crate::parse;

/// `limits`: `{ harmonics, lone }`, each optional, as `new --from` takes `--harmonics` and
/// `--lone`.
#[wasm_bindgen]
pub fn seed(
    rate: u32,
    channels: js_sys::Array,
    clip: &str,
    limits: Option<js_sys::Object>,
) -> Result<JsValue, JsValue> {
    let planes = planes_of(channels)?;
    let mut held = sva_core::seed::LIMITS;
    if let Some(limits) = limits {
        for (key, slot) in [("harmonics", &mut held.harmonics), ("lone", &mut held.lone)] {
            let value = js_sys::Reflect::get(&limits, &key.into()).unwrap_or(JsValue::UNDEFINED);
            if value.is_undefined() {
                continue;
            }
            *slot = value
                .as_f64()
                .filter(|n| n.fract() == 0.0 && *n >= 1.0)
                .ok_or_else(|| {
                    crate::refuse(
                        format!("`limits.{key}` is no whole number above 0"),
                        "pass `{ harmonics: 48, lone: 8 }` or leave a key out for its default",
                    )
                })? as usize;
        }
    }
    let seeded = sva_core::seed::seed(&planes, f64::from(rate), clip, held);
    let esc = sva_core::json::escape;
    let files = sva_core::json::list(&seeded.files, |(path, text)| {
        format!(
            "{{ \"path\": \"{}\", \"text\": \"{}\" }}",
            esc(path),
            esc(text)
        )
    });
    let varies = sva_core::json::list(&seeded.varies, |v| format!("\"{}\"", esc(v)));
    let dropped = sva_core::json::list(&seeded.dropped, |d| format!("\"{}\"", esc(d)));
    parse(&format!(
        "{{ \"files\": {files}, \"varies\": {varies}, \"dropped\": {dropped}, \"cost\": \"{}\" }}",
        esc(&seeded.cost)
    ))
}
