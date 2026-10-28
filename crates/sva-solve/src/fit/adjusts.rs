// Concern: a kept source's adjusts: its level scaled or removed, its level curves rethinned, its frequencies retuned | Non-concern: adding one (adds.rs) | IO: (Problem, kept, source) -> Scored

use std::f64::consts::LOG2_E;

use super::Tweak;
use super::adds::{Glide, golden, rate_floor, refined, slope, spread};
use super::peaks::{moves, pathed};
use super::problem::{Problem, Reading};
use super::render::{Held, Template, change, db, glided};
use super::shared::{self, linked};
use super::written::{Written, frequency_steps, rewrite, write};
use crate::optimize::code::Price;
use crate::optimize::{self, Change, Edit, Ledger, Scored};
use crate::sources::{Link, Sound, Trajectory};

/// The DL change replacing kept source `a` by `after`, or removing it where `None`, each shared
/// variable it references repriced.
fn swapped(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    a: usize,
    after: Option<&Written>,
    missed: f64,
) -> f64 {
    let rest = || {
        let others = kept.iter().enumerate().filter(move |(i, _)| *i != a);
        others.map(|(_, h)| &h.written).chain(after)
    };
    let prices: Vec<(usize, Price)> = kept[a]
        .links()
        .iter()
        .map(|l| (l.to, shared::price(p, rest(), l.kind, l.to)))
        .collect();
    let uses: Vec<usize> =
        after.map_or_else(Vec::new, |w| w.source.links.iter().map(|l| l.to).collect());
    let change = Change {
        added: None,
        removed: after.is_none().then_some(a),
        replaced: after.map_or_else(Vec::new, |w| vec![(a, &w.price, &uses[..])]),
        shared: prices.iter().map(|(s, price)| (*s, price)).collect(),
    };
    ledger.changed(&change, missed)
}

/// Source `a`'s level adjust, as `MODEL` states: its DL change, its tweak and what it leaves.
fn scaled(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    a: usize,
) -> (f64, Tweak, Option<Held>) {
    let (held, residual) = (&kept[a], &reading.residual);
    let render = held.render(p, &reading.shared);
    let (energy, pull) = (0..p.frames()).fold((0.0, 0.0), |(e, c), f| {
        render[f]
            .iter()
            .zip(&residual[f])
            .zip(&p.weight[f])
            .fold((e, c), |(e, c), ((u, r), w)| (e + w * u * u, c + w * u * r))
    });
    // Removing it changes the residual by +u: sum w (u^2 + 2 r u).
    let gone = swapped(
        p,
        kept,
        ledger,
        a,
        None,
        0.5 * LOG2_E * (energy + 2.0 * pull),
    );
    let gain = match energy > 0.0 {
        true => 1.0 + pull / energy,
        false => 1.0,
    };
    if gain <= 0.0 {
        return (gone, Tweak::Remove, None);
    }
    let trajectories = held
        .written
        .source
        .levels
        .iter()
        .map(|t| Trajectory {
            base: t.base + db(gain),
            knots: t.knots.clone(),
        })
        .collect();
    let written = rewrite(
        p,
        (held.sound(), held.written.step),
        held.key(),
        trajectories,
        held.links(),
    );
    // The residual changes by -(gain - 1) u: sum w (d^2 u^2 - 2 d r u).
    let d = gain - 1.0;
    let missed = 0.5 * LOG2_E * (d * d * energy - 2.0 * d * pull);
    let rescaled = swapped(p, kept, ledger, a, Some(&written), missed);
    match gone < rescaled {
        true => (gone, Tweak::Remove, None),
        false => (
            rescaled,
            Tweak::Level,
            Some(Held::new(p, held.templates.clone(), written)),
        ),
    }
}

/// Column `c` of `held` as heard through its links at `shared`, frame by frame.
fn heard(p: &Problem, held: &Held, shared: &[optimize::Shared], c: usize) -> Template {
    let t = &held.templates[c];
    let mut each: Vec<Vec<f64>> = (0..p.frames()).map(|f| t.at(f).to_vec()).collect();
    linked(p, held.links(), shared, &mut each);
    Template(each)
}

/// Source `a`'s rethin, as `MODEL` states: its DL change beside the source it leaves.
fn rethinned(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    a: usize,
) -> (f64, Held) {
    let (held, residual) = (&kept[a], &reading.residual);
    let before = &held.written.gains;
    let fitted: Vec<Vec<f64>> = (0..held.templates.len())
        .map(|c| {
            let t = match held.links().is_empty() {
                true => held.templates[c].clone(),
                false => heard(p, held, &reading.shared, c),
            };
            (0..p.frames())
                .map(|f| {
                    let (energy, pull) = p.moments(f, t.at(f), &residual[f]);
                    let g = before[c][f];
                    match energy > 0.0 {
                        true => (g + pull / energy).max(0.0),
                        false => g,
                    }
                })
                .collect()
        })
        .collect();
    let step = (held.sound(), held.written.step);
    let written = write(p, step, held.key(), &fitted, held.links());
    let mut delta = p.summed(&held.templates, &|c, f| written.gains[c][f] - before[c][f]);
    linked(p, held.links(), &reading.shared, &mut delta);
    let missed = 0.5 * LOG2_E * change(p, residual, &delta);
    let change = swapped(p, kept, ledger, a, Some(&written), missed);
    (change, Held::new(p, held.templates.clone(), written))
}

/// Source `a`'s frequency adjust, as `MODEL` states, `None` for a noise.
fn retuned(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    a: usize,
) -> Option<(f64, Held)> {
    let (held, residual) = (&kept[a], &reading.residual);
    if matches!(held.sound(), Sound::Noise { .. } | Sound::Shaped { .. }) {
        return None;
    }
    let (key, links, shared) = (held.key(), held.links(), &reading.shared[..]);
    let render = held.render(p, shared);
    let alone: Vec<Vec<f64>> = residual
        .iter()
        .zip(render.iter())
        .map(|(r, u)| r.iter().zip(u).map(|(r, u)| r + u).collect())
        .collect();
    let columns = p.columns(held.sound());
    let tracks: Vec<Vec<(f64, f64)>> = columns
        .paths
        .iter()
        .map(|path| pathed(p, &alone, path))
        .collect();
    let trajectories = &held.written.source.levels;
    let score = |sound: &Sound, step: f64| -> (f64, Written, Vec<Template>) {
        let templates = p.templates(sound);
        let written = rewrite(p, (sound, step), key, trajectories.clone(), links);
        let mut delta: Vec<Vec<f64>> = (0..p.frames())
            .map(|f| {
                let mut u = vec![0.0; p.range.len()];
                for ((after, before), g) in
                    templates.iter().zip(&held.templates).zip(&written.gains)
                {
                    for ((u, x), y) in u.iter_mut().zip(after.at(f)).zip(before.at(f)) {
                        *u += g[f] * (x - y);
                    }
                }
                u
            })
            .collect();
        linked(p, links, shared, &mut delta);
        let missed = 0.5 * LOG2_E * change(p, residual, &delta);
        let change = swapped(p, kept, ledger, a, Some(&written), missed);
        (change, written, templates)
    };
    let value = |sound: &Sound, step: f64| score(sound, step).0;
    let current = Glide {
        rate: held.sound().glide(),
        step: held.written.step,
    };
    let fitted = slope(&tracks, rate_floor(p, &columns, &held.written.levels));
    let mut glides = vec![current];
    for g in [Glide::default(), fitted] {
        let heard = g.rate == 0.0 || moves(p, &glided(held.sound(), g.rate, p.middle()));
        if heard && !glides.iter().any(|h| h.rate == g.rate) {
            glides.push(g);
        }
    }
    let others = ledger.without(a);
    let (change, written, templates) = glides
        .into_iter()
        .map(|glide| {
            let sound = match glided(held.sound(), glide.rate, p.middle()) {
                Sound::Partial { hz, .. } => tuned(p, hz, glide, &value, held, shared),
                voice => refined(p, &others, &alone, (voice, glide.step), key).0,
            };
            score(&sound, glide.step)
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))?;
    Some((change, Held::new(p, templates, written)))
}

/// A partial's Hz by golden section against `value` over half an analysis bin of `hz`, its
/// levels `held`'s, to the finer of its written precision and its standard deviation.
fn tuned(
    p: &Problem,
    hz: f64,
    glide: Glide,
    value: &dyn Fn(&Sound, f64) -> f64,
    held: &Held,
    shared: &[optimize::Shared],
) -> Sound {
    let bin = p.grid.rate / p.grid.n as f64;
    let h = bin * libm::cbrt(f64::EPSILON);
    let at = |hz: f64| Sound::Partial {
        hz,
        glide: glide.rate,
    };
    let written = &held.written;
    let step = frequency_steps(p, &p.columns(&at(hz)), &written.levels).0;
    let links: (&[Link], _) = (held.links(), shared);
    let sigma = spread(p, (&at(hz - h), &at(hz + h)), h, &written.gains, links);
    let score = |hz: f64| value(&at(hz), glide.step);
    let range = (hz - bin / 2.0, hz + bin / 2.0);
    at(golden(&score, range, step.min(sigma), (hz, score(hz))).0)
}

/// Source `a`'s adjusts, as `MODEL` states: its level, frequency and rethin.
pub(super) fn adjusts(
    p: &Problem,
    kept: &[Held],
    ledger: &Ledger,
    reading: &Reading,
    a: usize,
) -> Vec<Scored<Problem>> {
    let (change, tweak, held) = scaled(p, kept, ledger, reading, a);
    let edit = match held {
        Some(held) => Edit::Replace(a, held, tweak),
        None => Edit::Remove(a, tweak),
    };
    let mut out = vec![Scored { change, edit }];
    let retune =
        retuned(p, kept, ledger, reading, a).filter(|(_, h)| p.apart(reading, h.sound(), Some(a)));
    if let Some((change, held)) = retune {
        out.push(Scored {
            change,
            edit: Edit::Replace(a, held, Tweak::Frequency),
        });
    }
    let (change, held) = rethinned(p, kept, ledger, reading, a);
    out.push(Scored {
        change,
        edit: Edit::Replace(a, held, Tweak::Rethin),
    });
    out
}
