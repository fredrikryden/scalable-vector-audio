// Concern: keeps N0's calibrations on disk per build, rate, scale and spl, so each is fitted once | Non-concern: fitting one (calibrate.rs), the render store | IO: (CacheAt, rates, scales) -> warnings

use std::hash::{DefaultHasher, Hasher};

use sva_core::solve::calibrate::{Calibration, Condition, keep, kept};
use sva_core::{Backend, Diagnostic, Severity};

use crate::args::CacheAt;
use crate::directory::Directory;
use crate::store::{platform_dir, wait};

/// Bumped by any change to what a file holds.
const FORMAT: &str = "n0-v1";

const DEFAULT_DIR: &str = "sva-n0";

fn warning(message: String) -> Diagnostic {
    Diagnostic::new("solve.calibration_cache", message)
        .with_severity(Severity::Warning)
        .helped("pass `--cache <path>` elsewhere, or `--cache none` to fit N0 afresh")
}

/// This build's directory, keyed by the executable's bytes, and the rates and scales read
/// from it.
pub struct Kept {
    medium: Directory,
    read: Vec<(u32, u64)>,
}

fn name(rate: u32, scale: f64, spl_db: f64) -> String {
    format!("{rate}-{:016x}-{:016x}", scale.to_bits(), spl_db.to_bits())
}

/// Each number as its bits in hex, read back bit for bit.
fn text(c: &Calibration) -> String {
    let mut lines = vec![
        c.rate.to_string(),
        format!("{:016x}", c.scale.to_bits()),
        format!("{:016x}", c.spl_db.to_bits()),
        format!("{:016x}", c.n0.to_bits()),
    ];
    for k in &c.conditions {
        lines.push(
            [k.hz, k.sl_db, k.floor_db, k.unit_dl_db]
                .map(|v| format!("{:016x}", v.to_bits()))
                .join(" "),
        );
    }
    lines.join("\n") + "\n"
}

fn parsed(text: &str) -> Option<Calibration> {
    let bits = |s: &str| u64::from_str_radix(s, 16).ok().map(f64::from_bits);
    let mut lines = text.lines();
    let rate = lines.next()?.parse().ok()?;
    let scale = bits(lines.next()?)?;
    let spl_db = bits(lines.next()?)?;
    let n0 = bits(lines.next()?)?;
    let conditions = lines
        .map(|line| {
            let v: Vec<f64> = line.split(' ').map(bits).collect::<Option<_>>()?;
            let [hz, sl_db, floor_db, unit_dl_db] = v.as_slice() else {
                return None;
            };
            Some(Condition {
                hz: *hz,
                sl_db: *sl_db,
                floor_db: *floor_db,
                unit_dl_db: *unit_dl_db,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Calibration {
        rate,
        scale,
        spl_db,
        n0,
        conditions,
    })
}

/// Holds each calibration this build kept under `at` at each of `levels`' rate and scale and
/// at `spl_db`, where it reads as one a fit makes at the key its name holds; `None` where `at`
/// keeps none.
pub fn held(
    at: &CacheAt,
    (levels, spl_db): (&[(u32, f64)], f64),
    warnings: &mut Vec<Diagnostic>,
) -> Option<Kept> {
    let root = match at {
        CacheAt::Off => return None,
        CacheAt::Path(path) => Ok(path.clone()),
        CacheAt::Platform => platform_dir(DEFAULT_DIR).map_err(|e| e.message()),
    };
    let built = std::env::current_exe()
        .and_then(std::fs::read)
        .map_err(|e| format!("this executable could not be read: {e}"));
    let (root, bytes) = match root.and_then(|r| built.map(|b| (r, b))) {
        Ok(both) => both,
        Err(why) => {
            warnings.push(warning(format!("no calibration was kept: {why}")));
            return None;
        }
    };
    let mut hasher = DefaultHasher::new();
    hasher.write(&bytes);
    let medium = Directory::at(root.join(FORMAT).join(format!("{:016x}", hasher.finish())));
    let mut read = Vec::new();
    for (rate, scale) in levels.iter().copied() {
        let Ok(Some(bytes)) = wait(medium.get(&name(rate, scale, spl_db))) else {
            continue;
        };
        let keyed = |c: &Calibration| {
            (c.rate, c.scale.to_bits(), c.spl_db.to_bits())
                == (rate, scale.to_bits(), spl_db.to_bits())
        };
        let calibration = String::from_utf8(bytes).ok().and_then(|t| parsed(&t));
        if calibration.filter(keyed).is_some_and(keep) {
            read.push((rate, scale.to_bits()));
        }
    }
    Some(Kept { medium, read })
}

impl Kept {
    /// Writes each calibration held and not read, over a bad file, whole or not at all.
    pub fn keep_all(&self, (levels, spl_db): (&[(u32, f64)], f64), warnings: &mut Vec<Diagnostic>) {
        let unread = levels
            .iter()
            .copied()
            .filter(|(rate, s)| !self.read.contains(&(*rate, s.to_bits())));
        for (rate, scale) in unread {
            let Some(c) = kept(rate, scale, spl_db) else {
                continue;
            };
            if let Err(why) = wait(
                self.medium
                    .put(&name(rate, scale, spl_db), text(&c).as_bytes()),
            ) {
                warnings.push(warning(format!("a calibration was not kept: {why}")));
            }
        }
    }
}
