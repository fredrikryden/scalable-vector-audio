// Concern: reads one reading's `value`, as answer.rs writes it, back into that reading | Non-concern: where the JSON came from, fitting a render (solve/) | IO: (Representation, JSON) -> Output

use serde_json::Value;
use sva_engine::bands::BandFloor;
use sva_engine::pitch::Note;
use sva_engine::spectrum::{Band, Peak};
use sva_engine::{
    BandCrest, BandTrack, Bands, Crest, EnvelopeFrame, IoiBucket, Loudness, LoudnessFrame, Onset,
    Onsets, Output, PEAK_NOTE, PitchFrame, Representation, Spectrum,
};

/// One place in the JSON, so a refusal names the key it stopped at.
struct At<'v> {
    value: &'v Value,
    path: String,
}

type Read<T> = Result<T, String>;

impl<'v> At<'v> {
    fn key(&self, key: &str) -> Read<At<'v>> {
        let path = format!("{}.{key}", self.path);
        match self.value.get(key) {
            Some(value) => Ok(At { value, path }),
            None => Err(format!("`{path}` is missing")),
        }
    }

    fn num(&self) -> Read<f64> {
        match self.value {
            Value::Number(n) => n
                .as_f64()
                .ok_or_else(|| format!("`{}` is no number", self.path)),
            // `answer.rs` writes a non-finite measurement as null.
            Value::Null => Ok(f64::NAN),
            _ => Err(self.not("a number")),
        }
    }

    fn maybe(&self) -> Read<Option<f64>> {
        match self.value {
            Value::Null => Ok(None),
            _ => self.num().map(Some),
        }
    }

    fn count(&self) -> Read<usize> {
        let held = self.value.as_u64().and_then(|n| usize::try_from(n).ok());
        held.ok_or_else(|| self.not("a count"))
    }

    fn flag(&self) -> Read<bool> {
        self.value
            .as_bool()
            .ok_or_else(|| self.not("true or false"))
    }

    fn text(&self) -> Read<String> {
        let held = self.value.as_str().map(str::to_string);
        held.ok_or_else(|| self.not("a string"))
    }

    fn not(&self, wanted: &str) -> String {
        format!("`{}` is {}, not {wanted}", self.path, shape(self.value))
    }

    /// A collection's items, refused where it holds fewer than it counts.
    fn items<T>(&self, mut each: impl FnMut(At) -> Read<T>) -> Read<Vec<T>> {
        let items = self.key("items")?;
        let Value::Array(list) = items.value else {
            return Err(items.not("a list"));
        };
        let count = self.key("pagination")?.key("count")?.count()?;
        if count != list.len() {
            return Err(format!(
                "`{}` holds {} of the {count} items it counts, so it was cut short",
                items.path,
                list.len()
            ));
        }
        list.iter()
            .enumerate()
            .map(|(k, value)| {
                each(At {
                    value,
                    path: format!("{}[{k}]", items.path),
                })
            })
            .collect()
    }
}

/// What a JSON value is, in a refusal's words.
pub fn shape(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "an object",
    }
}

/// `field!(at, "key")` is the `At` under `key`.
macro_rules! field {
    ($at:expr, $key:expr) => {
        $at.key($key)?
    };
}

fn spectrum(at: At) -> Read<Output> {
    Ok(Output::Spectrum(Box::new(Spectrum {
        frame_size: field!(at, "frame_size").count()?,
        frames: field!(at, "frames").count()?,
        resolution_hz: field!(at, "resolution_hz").num()?,
        rms: field!(at, "rms").num()?,
        centroid_hz: field!(at, "centroid_hz").num()?,
        rolloff85_hz: field!(at, "rolloff85_hz").num()?,
        peaks: field!(at, "peaks").items(|p| {
            Ok(Peak {
                hz: field!(p, "hz").num()?,
                db: field!(p, "db").num()?,
            })
        })?,
        bands: field!(at, "bands").items(|b| {
            Ok(Band {
                lo_hz: field!(b, "lo_hz").num()?,
                hi_hz: field!(b, "hi_hz").num()?,
                db: field!(b, "db").num()?,
            })
        })?,
    })))
}

fn envelope(at: At) -> Read<Output> {
    Ok(Output::Envelope(at.items(|f| {
        Ok(EnvelopeFrame {
            t_secs: field!(f, "t").num()?,
            rms: field!(f, "rms").num()?,
            peak: field!(f, "peak").num()?,
        })
    })?))
}

/// A note's partial number is not written, so it reads back as 0.
fn pitch(at: At) -> Read<Output> {
    Ok(Output::Pitch(at.items(|f| {
        Ok(PitchFrame {
            t_secs: field!(f, "t").num()?,
            notes: field!(f, "notes").items(|n| {
                Ok(Note {
                    hz: field!(n, "hz").num()?,
                    name: field!(n, "note").text()?,
                    cents: field!(n, "cents").num()?,
                    db: field!(n, "db").num()?,
                    harmonic_of: field!(n, "harmonic_of").maybe()?,
                    n: 0,
                })
            })?,
        })
    })?))
}

fn floor(at: At) -> Read<BandFloor> {
    Ok(BandFloor {
        peak: field!(at, "peak").num()?,
        time_to_peak_secs: field!(at, "time_to_peak_secs").num()?,
        rise_10_90_secs: field!(at, "rise_10_90_secs").maybe()?,
    })
}

fn bands(at: At) -> Read<Output> {
    Ok(Output::Bands(Box::new(Bands {
        rate_hz: field!(at, "rate_hz").num()?,
        start_secs: field!(at, "start_secs").num()?,
        bands: field!(at, "bands").items(|b| {
            Ok(BandTrack {
                centre_hz: field!(b, "centre_hz").num()?,
                cam: field!(b, "cam").num()?,
                erb_hz: field!(b, "erb_hz").num()?,
                q: field!(b, "q").num()?,
                peak: field!(b, "peak").num()?,
                time_to_peak_secs: field!(b, "time_to_peak_secs").maybe()?,
                rise_10_90_secs: field!(b, "rise_10_90_secs").maybe()?,
                floor: floor(field!(b, "floor"))?,
                rms: field!(b, "rms").items(|v| v.num())?,
            })
        })?,
    })))
}

fn loudness(at: At) -> Read<Output> {
    let frames = |at: At| {
        at.items(|f| {
            Ok(LoudnessFrame {
                t: field!(f, "t").num()?,
                lufs: field!(f, "lufs").num()?,
            })
        })
    };
    Ok(Output::Loudness(Box::new(Loudness {
        integrated_lufs: field!(at, "integrated_lufs").maybe()?,
        range_lu: field!(at, "range_lu").maybe()?,
        momentary_max_lufs: field!(at, "momentary_max_lufs").maybe()?,
        short_term_max_lufs: field!(at, "short_term_max_lufs").maybe()?,
        sample_peak: field!(at, "sample_peak").num()?,
        sample_peak_dbfs: field!(at, "sample_peak_dbfs").maybe()?,
        peak_note: PEAK_NOTE,
        momentary: frames(field!(at, "momentary"))?,
        short_term: frames(field!(at, "short_term"))?,
    })))
}

fn crest(at: At) -> Read<Output> {
    Ok(Output::Crest(Box::new(Crest {
        broadband_crest_db: field!(at, "broadband_crest_db").num()?,
        spread_db: field!(at, "spread_db").maybe()?,
        widest_band_hz: field!(at, "widest_band_hz").maybe()?,
        tightest_band_hz: field!(at, "tightest_band_hz").maybe()?,
        counted_under_db: field!(at, "counted_under_db").num()?,
        bands: field!(at, "bands").items(|b| {
            Ok(BandCrest {
                lo_hz: field!(b, "lo_hz").num()?,
                hi_hz: field!(b, "hi_hz").num()?,
                centre_hz: field!(b, "centre_hz").num()?,
                peak: field!(b, "peak").num()?,
                rms: field!(b, "rms").num()?,
                crest_db: field!(b, "crest_db").num()?,
                counted: field!(b, "counted").flag()?,
            })
        })?,
    })))
}

fn onsets(at: At) -> Read<Output> {
    Ok(Output::Onsets(Box::new(Onsets {
        onsets: field!(at, "onsets").items(|o| {
            Ok(Onset {
                t_secs: field!(o, "t_secs").num()?,
                strength: field!(o, "strength").num()?,
            })
        })?,
        resolution_secs: field!(at, "resolution_secs").num()?,
        ioi_histogram: field!(at, "ioi_histogram").items(|b| {
            Ok(IoiBucket {
                lo_secs: field!(b, "lo_secs").num()?,
                hi_secs: field!(b, "hi_secs").num()?,
                count: field!(b, "count").count()?,
            })
        })?,
    })))
}

/// `value` read as `representation` writes it; `path` names where it sits, for a refusal.
/// The reading's own form decides the schema, never the JSON's.
pub fn reading(representation: Representation, value: &Value, path: &str) -> Read<Output> {
    let at = At {
        value,
        path: path.to_string(),
    };
    match representation {
        Representation::Spectrum { .. } => spectrum(at),
        Representation::Envelope { .. } => envelope(at),
        Representation::Pitch { .. } => pitch(at),
        Representation::Bands => bands(at),
        Representation::Loudness => loudness(at),
        Representation::Crest => crest(at),
        Representation::Onsets => onsets(at),
        other => Err(format!("`{other:?}` has no JSON reader")),
    }
}
