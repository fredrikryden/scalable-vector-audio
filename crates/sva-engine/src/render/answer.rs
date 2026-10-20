// Concern: takes one reading off the representation a node declares, a ledger edge by edge | Non-concern: naming the observations (query.rs), the arithmetic of one | IO: (&Render, node) -> Answer

use sva_formula::closed_form::Edge;
use sva_formula::spectral_sum::atom::{Singular, SpectralAtom};
use sva_formula::{C64, Line, SpectralSum, Var, d_dt, envelope_read, line_atoms_read};
use sva_samples::measure::harmony::dissonance::Crowded;
use sva_samples::measure::harmony::{
    self, Framing, Moment, chords, dissonance, inharmonicity, measured as framed, tonnetz,
    voice_leading,
};
use sva_samples::measure::{modulation, partials, periodicity, self_similarity, sliding};
use sva_samples::{
    Buffer, Consumes, Extent, Peak, PitchFrame, Profile, Source, measure::bands, measure::crest,
    measure::envelope, measure::formants, measure::loudness, measure::loudness_pattern,
    measure::panel, measure::pitch, measure::score, measure::spectrum, measure::stereo,
    measure_alias,
};

use crate::error::{Diagnostic, EngineError, Located};
use crate::query::{Answer, Atom, DEFAULT_FRAME_SECS, Harmonic, Output, Representation};
use crate::refs;
use crate::render::Render;
use crate::typing::Value;

/// A closed form answers off its spectral sum and a buffer off its samples; its own `Ty`
/// decides which, never a flag on the observation.
pub fn answer(
    render: &Render,
    node: sva_formula::NodeId,
    representation: Representation,
) -> Result<Answer, EngineError> {
    let closed = render.tys.ty(node).is_closed_form();
    let profile = render.config.profile.name;
    if representation == Representation::Arguments {
        return Ok(Answer::whole(
            Output::Arguments(arguments_under(render, node)),
            Source::Exact,
            profile,
            None,
        ));
    }
    if representation == Representation::Bindings {
        return Ok(Answer::whole(
            Output::Bindings(render.bindings.get(&node).cloned().unwrap_or_default()),
            Source::Exact,
            profile,
            None,
        ));
    }
    match representation.consumes(closed) {
        Consumes::ClosedForm => {
            let found;
            let sum = match render.symbolic.get(&node) {
                Some(held) => held,
                None => match refs::spectral_sum_of(&render.tys, node, render.tys.var(node)) {
                    Ok(held) => {
                        found = held;
                        &found
                    }
                    Err(left) => return Err(unsummed(representation, left)),
                },
            };
            let (value, listed, source) = exact(render, node, representation, sum)
                .map_err(|left| not_exact(representation, left))?;
            Ok(Answer {
                value,
                source,
                profile,
                rate: None,
                tail_db: listed.tail_db(),
                dropped: listed.dropped,
            })
        }
        _ => {
            if render.buffer(node).is_none() {
                return Err(unmaterialized(render, node, representation));
            }
            let buffer = render.output(node)?;
            Ok(Answer::whole(
                measured(render, node, representation, &buffer)?,
                Source::Measured,
                profile,
                Some(render.config.rate),
            ))
        }
    }
}

fn exact(
    render: &Render,
    node: sva_formula::NodeId,
    representation: Representation,
    sum: &SpectralSum,
) -> Result<(Output, Listed, Source), EngineError> {
    let source = Source::Exact;
    let enumerated = || listed(render, node, sum, &render.config.profile);
    let (value, listed) = match representation {
        Representation::Lines => {
            let listed = enumerated()?;
            let held = lines(render, node, sum.var, &listed.atoms)?;
            (Output::Lines(held), listed)
        }
        Representation::Spectrum {
            frame_secs: Some(_),
            ..
        } => return Err(unframed(render, node, "spectrum is its lines", "the lines")),
        Representation::Envelope {
            frame_secs: Some(_),
        } => {
            return Err(unframed(
                render,
                node,
                "envelope is its exact squared magnitude",
                "the exact envelope",
            ));
        }
        // A pair's spectrum is its whole line list; no estimate, no peaks.
        Representation::Spectrum {
            frame_secs: None, ..
        } => {
            let each = per_channel(render, node, sum)?;
            let mut held = match each.as_slice() {
                [one] => lines(render, node, sum.var, &one.atoms)?,
                _ => in_power(
                    each.iter()
                        .map(|one| lines(render, node, sum.var, &one.atoms))
                        .collect::<Result<_, _>>()?,
                ),
            };
            held.sort_by(|a, b| a.hz.total_cmp(&b.hz));
            (Output::Lines(held), Listed::joined(each))
        }
        Representation::Atoms => {
            let each = per_channel(render, node, sum)?;
            let found = each
                .iter()
                .enumerate()
                .flat_map(|(c, one)| one.atoms.iter().map(move |a| atom_of(sum.var, a, c)))
                .collect();
            (Output::Atoms(found), Listed::joined(each))
        }
        // A pair states its partials; `max_notes` is the question, not a cut answer.
        Representation::Pitch { max_notes, .. } => {
            let each = per_channel(render, node, sum)?;
            let partials = each
                .iter()
                .map(|one| {
                    let held = lines(render, node, sum.var, &one.atoms)?;
                    Ok(harmony::partials(held.iter().map(|l| (l.hz, l.amp))))
                })
                .collect::<Result<_, EngineError>>()?;
            let mut found: Vec<Peak> = harmony::averaged(partials)
                .into_iter()
                .map(|p| Peak {
                    hz: p.hz,
                    db: 20.0 * libm::log10(p.amplitude),
                })
                .collect();
            found.sort_by(|a, b| b.db.total_cmp(&a.db));
            let start = render.read_from.ok_or_else(|| {
                refused(
                    render,
                    node,
                    "engine.not_materialized",
                    "an exact `pitch` frame starts at the sample the render reads its root \
                     from, and this render, not asked for `pitch`, decided none"
                        .to_string(),
                )
            })?;
            let frame = PitchFrame {
                t_secs: start as f64 / f64::from(render.config.rate),
                notes: pitch::name_peaks(&found, max_notes),
            };
            (Output::Pitch(vec![frame]), Listed::joined(each))
        }
        // Each series is listed first, as `atoms` lists it, so no term is differentiated away.
        Representation::Derivative => {
            let each = per_channel(render, node, sum)?;
            let series_free = SpectralSum::of(
                sum.var,
                each.iter()
                    .map(|one| sva_formula::Lane::of(one.atoms.clone()))
                    .collect(),
            );
            let slope = d_dt(&series_free).map_err(|left| {
                EngineError::of_closed_form(
                    &left.refusal(),
                    render.tys.locate(left.origin),
                    "read the derivative of a lower order or degree",
                )
            })?;
            let dropped = Listed::joined(each).dropped.into_iter().map(|l| Line {
                amp: l.amp * C64::new(0.0, std::f64::consts::TAU * l.hz),
                ..l
            });
            let differentiated = Listed {
                atoms: slope.atoms().copied().collect(),
                dropped: dropped.collect(),
            };
            (Output::Symbolic(Box::new(slope)), differentiated)
        }
        Representation::Envelope { frame_secs: None } => {
            let held =
                refs::read_through(&render.tys, |t| envelope_read(sum, t)).map_err(|left| {
                    EngineError::of_closed_form(
                        &left.refusal(),
                        render.tys.locate(left.origin),
                        "read the envelope off sample(...) for a measured one",
                    )
                })?;
            (Output::Symbolic(Box::new(held.squared)), Listed::NONE)
        }
        Representation::Harmony {
            reading,
            frame_secs,
            floor_db,
        } => {
            let (moments, framing, listed, end_secs) =
                exact_moments(render, node, sum, frame_secs)?;
            let value = harmonic(reading, &moments, (framing, end_secs), floor_db)
                .map_err(|held| crowded(render.tys.name(node), representation, held))?;
            (value, listed)
        }
        other => return Err(not_a_closed_form(render, node, other)),
    };
    Ok((value, listed, source))
}

/// Each instant on the frame grid over the range, each channel's partials exactly: every atom
/// a partial at its carrier, `omega/2pi`, its complex value there merged with the channel's
/// others at that carrier, so a decay, polynomial, Gaussian or window weighs it as it stands;
/// then every channel in power.
fn exact_moments(
    render: &Render,
    node: sva_formula::NodeId,
    sum: &SpectralSum,
    frame_secs: f64,
) -> Result<(Vec<Moment>, Framing, Listed, f64), EngineError> {
    let range = match (render.range, &render.unranged) {
        (Some(range), _) => range,
        (None, Some(refused)) => return Err(refused.clone()),
        (None, None) => unreachable!("a harmonic reading decides its range before it answers"),
    };
    let rate = f64::from(render.config.rate);
    let hop = ((frame_secs * rate).round() as usize).max(1);
    let each = per_channel(render, node, sum)?;
    let instants: Vec<f64> = (range.start..range.end)
        .step_by(hop)
        .map(|n| n as f64 / rate)
        .collect();
    let moments = match sum.var {
        Var::F => {
            let partials = harmony::averaged(
                each.iter()
                    .map(|one| {
                        let held = lines(render, node, Var::F, &one.atoms)?;
                        Ok(harmony::partials(held.iter().map(|l| (l.hz, l.amp))))
                    })
                    .collect::<Result<_, EngineError>>()?,
            );
            instants
                .iter()
                .map(|t| Moment {
                    t_secs: *t,
                    partials: partials.clone(),
                })
                .collect()
        }
        Var::T => instants
            .iter()
            .map(|t| {
                let at = |one: &Listed| {
                    let carried = one.atoms.iter().map(|a| {
                        let value = (!a.is_delta()).then(|| a.smooth_at(*t)).flatten();
                        let hz = a.exp.map_or(0.0, |e| e.omega / std::f64::consts::TAU);
                        value
                            .map(|v| (hz, v))
                            .ok_or_else(|| no_partial(render, node, a, *t))
                    });
                    Ok(harmony::partials(carried.collect::<Result<Vec<_>, _>>()?))
                };
                Ok(Moment {
                    t_secs: *t,
                    partials: harmony::averaged(each.iter().map(at).collect::<Result<_, _>>()?),
                })
            })
            .collect::<Result<_, EngineError>>()?,
    };
    let framing = Framing {
        hop_secs: hop as f64 / rate,
        window: None,
    };
    let end_secs = range.end as f64 / rate;
    Ok((moments, framing, Listed::joined(each), end_secs))
}

/// One harmonic reading off a curve of partials, whichever source took them, over a render
/// ending at `end_secs`.
fn harmonic(
    reading: Harmonic,
    moments: &[Moment],
    (framing, end_secs): (Framing, f64),
    floor_db: f64,
) -> Result<Output, Crowded> {
    let chords = || chords::segment(moments, framing.hop_secs, end_secs, floor_db);
    Ok(match reading {
        Harmonic::Dissonance { max_pairs } => Output::Dissonance(Box::new(dissonance::measure(
            moments, framing, floor_db, max_pairs,
        )?)),
        Harmonic::VoiceLeading { norm } => Output::VoiceLeading(Box::new(voice_leading::analyze(
            chords(),
            framing,
            floor_db,
            norm,
        ))),
        Harmonic::TonalDistance => {
            Output::TonalDistance(Box::new(tonnetz::analyze(chords(), framing, floor_db)))
        }
        Harmonic::Inharmonicity => {
            Output::Inharmonicity(Box::new(inharmonicity::measure(moments, framing, floor_db)))
        }
    })
}

/// An impulse stands at no frequency, and a pole has no value at its own instant.
fn no_partial(
    render: &Render,
    node: sva_formula::NodeId,
    atom: &SpectralAtom,
    t: f64,
) -> EngineError {
    let mut at = render.tys.locate(atom.origin);
    if at.node.is_empty() {
        at = Located::at(render.tys.name(node), None);
    }
    let message = match atom.is_delta() {
        true => "this term is an impulse in `t`, which stands at no one frequency, so it is \
                 no partial"
            .to_string(),
        false => {
            format!("this term has a pole at {t}s, an instant of the curve, where it has no value")
        }
    };
    EngineError::refused(Diagnostic {
        code: "read.no_partial_at_instant".to_string(),
        message,
        location: at,
        help: "a harmonic reading of `sample(...)` of it, as `sample(@x([0, 1s]))`, is \
               measured off frames of its samples instead"
            .to_string(),
    })
}

fn crowded(name: &str, representation: Representation, held: Crowded) -> EngineError {
    let reading = representation.name();
    EngineError::refused(Diagnostic {
        code: "read.too_many_partial_pairs".to_string(),
        message: format!(
            "at {}s, {} partials may stand over the floor, {} pairs, past the {} asked for",
            held.t_secs, held.partials, held.pairs, held.cap
        ),
        location: Located::at(name, None),
        help: format!(
            "ask for more pairs, as `{reading}(pairs={})`, or raise the floor so fewer partials \
             pair, as `{reading}(floor=-30db)`",
            held.pairs
        ),
    })
}

/// Each names the window or rate that would read, `longer` how to hand it a target that
/// long and `faster` how to hand it samples at a rate over that.
fn unread(
    name: &str,
    representation: Representation,
    why: sliding::Unread,
    longer: impl Fn(f64) -> String,
    faster: impl Fn(f64) -> String,
) -> EngineError {
    let reading = representation.name();
    let (code, message, help) = match why {
        sliding::Unread::RateUnderBands { rate_hz, least_hz } => {
            under_bands(reading, rate_hz, least_hz, faster)
        }
        sliding::Unread::WindowUnderTwoPeriods {
            window_secs,
            least_secs,
        } => (
            "read.window_under_two_periods",
            format!(
                "a {window_secs}s window holds under two of the shortest period `{reading}` \
                 reads, {}s",
                least_secs / 2.0
            ),
            format!(
                "ask a window of {least_secs}s or longer, as `{reading}(window={least_secs}s)`"
            ),
        ),
        sliding::Unread::TargetShorterThanWindow {
            held_secs,
            window_secs,
            least_secs,
        } => (
            "read.target_under_window",
            format!("the target holds {held_secs}s, under the {window_secs}s window"),
            match least_secs <= held_secs {
                true => format!(
                    "{}, or ask a window as long as the target, as \
                     `{reading}(window={held_secs}s)`",
                    longer(window_secs)
                ),
                false => format!(
                    "{}; no window under {least_secs}s holds two of the shortest period",
                    longer(window_secs)
                ),
            },
        ),
    };
    EngineError::refused(Diagnostic {
        code: code.to_string(),
        message,
        location: Located::at(name, None),
        help,
    })
}

/// `faster` says how to hand a reading of ERB bands samples at the least rate that places one.
fn under_bands(
    reading: &str,
    rate_hz: f64,
    least_hz: f64,
    faster: impl Fn(f64) -> String,
) -> (&'static str, String, String) {
    (
        "read.rate_under_bands",
        format!(
            "`{reading}` reads ERB bands, and {rate_hz} Hz samples place none: the first band's \
             centre stands at {:.2} Hz, over {} of the rate",
            least_hz * bands::HIGHEST_CENTRE_OF_RATE,
            bands::HIGHEST_CENTRE_OF_RATE
        ),
        faster((least_hz * 100.0).ceil() / 100.0),
    )
}

/// Names the least rate and, through `faster`, how to hand it samples at one.
fn rate_under(
    name: &str,
    held: sva_samples::RateUnderWeighting,
    faster: impl Fn(f64) -> String,
) -> EngineError {
    EngineError::refused(Diagnostic {
        code: "read.rate_under_weighting".to_string(),
        message: format!(
            "BS.1770's K-weighting shelf stands at {} Hz, at or past the Nyquist of {} Hz \
             samples, where its bilinear design has no filter",
            held.least_hz / 2.0,
            held.rate_hz
        ),
        location: Located::at(name, None),
        help: faster(held.least_hz),
    })
}

fn block_under(name: &str, (window_secs, samples, rate): (f64, usize, u32)) -> EngineError {
    let least = sva_samples::measure::loudness::LEAST_BLOCK_SAMPLES;
    let secs = least as f64 / f64::from(rate);
    EngineError::refused(Diagnostic {
        code: "read.block_under_samples".to_string(),
        message: format!(
            "a {window_secs}s `loudness` window holds {samples} samples at {rate} Hz, under the \
             {least} a block a quarter of itself apart needs"
        ),
        location: Located::at(name, None),
        help: format!("ask a window of {secs}s or longer, as `loudness(window={secs}s)`"),
    })
}

/// Names the least frame or target an all-pole fit of the order reads, `longer` how to hand
/// it a target that long.
fn under_order(
    name: &str,
    why: formants::UnderOrder,
    longer: impl Fn(f64) -> String,
) -> EngineError {
    let (code, message, help) = match why {
        formants::UnderOrder::Frame {
            frame_secs,
            held,
            order,
            least_secs,
        } => (
            "read.frame_under_order",
            format!(
                "a {frame_secs}s frame holds {held} samples, and an order-{order} all-pole fit \
                 reads {} lags",
                order + 1
            ),
            format!("ask a frame of {least_secs}s or longer, as `formants(frame={least_secs}s)`"),
        ),
        formants::UnderOrder::Target {
            held,
            order,
            least_secs,
        } => (
            "read.target_under_order",
            format!(
                "the target holds {held} samples, and an order-{order} all-pole fit reads {} \
                 lags",
                order + 1
            ),
            longer(least_secs),
        ),
    };
    EngineError::refused(Diagnostic {
        code: code.to_string(),
        message,
        location: Located::at(name, None),
        help,
    })
}

/// Names the rate, cell or window that fits and, through `shorter` and `faster`, how to hand
/// it a target that short or samples that fast.
fn unfit(
    name: &str,
    why: self_similarity::Unfit,
    shorter: impl Fn(f64) -> String,
    faster: impl Fn(f64) -> String,
) -> EngineError {
    let (code, message, help) = match why {
        self_similarity::Unfit::RateUnderBands { rate_hz, least_hz } => {
            under_bands("self_similarity", rate_hz, least_hz, faster)
        }
        self_similarity::Unfit::TooManyCells(held) => (
            "read.too_many_cells",
            format!(
                "the target's {}s at cell={}s is {} cells a side, past the {} a matrix holds",
                held.held_secs, held.cell_secs, held.cells, held.cap
            ),
            format!(
                "ask a coarser cell, as `self_similarity(cell={}s)`, or {}",
                held.least_cell_secs,
                shorter(held.most_secs)
            ),
        ),
        self_similarity::Unfit::CellUnderBandStep {
            cell_secs,
            least_cell_secs,
            band_hz,
        } => (
            "read.cell_under_band_step",
            format!(
                "a {cell_secs}s cell holds no sample of the {band_hz} Hz band envelopes its \
                 band levels are read off"
            ),
            format!(
                "ask a cell of {least_cell_secs}s or coarser, as \
                 `self_similarity(cell={least_cell_secs}s)`"
            ),
        ),
        self_similarity::Unfit::KernelPastCap {
            window_secs,
            kernel_cells,
            cap,
            cell_secs,
            widest_secs,
        } => (
            "read.kernel_past_cap",
            format!(
                "a {window_secs}s novelty window at cell={cell_secs}s is {kernel_cells} cells \
                 a side, past the {cap} a matrix holds"
            ),
            format!(
                "ask a window of at most {widest_secs}s, as \
                 `self_similarity(window={widest_secs}s)`, or a coarser cell"
            ),
        ),
    };
    EngineError::refused(Diagnostic {
        code: code.to_string(),
        message,
        location: Located::at(name, None),
        help,
    })
}

/// A closed form's envelope is exact or refused, never measured unasked: a refusal of its
/// sum says so, and names `sample(...)` as the way to ask for the measured one.
fn not_exact(representation: Representation, left: EngineError) -> EngineError {
    match (representation, left) {
        (Representation::Envelope { frame_secs: None }, EngineError::Refused(cause)) => {
            EngineError::refused(Diagnostic {
                message: format!("this closed form has no exact envelope: {}", cause.message),
                help: "a measured envelope is asked for explicitly: read `envelope` of \
                       `sample(...)` of it, as `sample(@x([0, 1s]))`"
                    .to_string(),
                ..*cause
            })
        }
        (_, left) => left,
    }
}

/// A closed form with no atom sum has no exact partials; the measured ones are asked for
/// explicitly, never taken in their place.
fn unsummed(representation: Representation, left: EngineError) -> EngineError {
    match (representation, left) {
        (Representation::Harmony { .. }, EngineError::Refused(cause)) => {
            EngineError::refused(Diagnostic {
                message: format!(
                    "`{}` of a closed form reads its exact partials, and it has none: {}",
                    representation.name(),
                    cause.message
                ),
                help: format!(
                    "a measured `{}` is asked for explicitly: read it of `sample(...)` of it, as \
                     `sample(@x([0, 1s]))`",
                    representation.name()
                ),
                ..*cause
            })
        }
        (_, left) => not_exact(representation, left),
    }
}

/// A closed form holds no frames, so a written `frame` is refused rather than ignored.
fn unframed(render: &Render, node: sva_formula::NodeId, is: &str, bare: &str) -> EngineError {
    EngineError::refused(Diagnostic {
        code: "engine.observation_needs_samples".to_string(),
        message: format!("a closed form's {is}, and it has no frames to take it across"),
        location: Located::at(render.tys.name(node), None),
        help: format!("drop `frame` for {bare}, or read `sample(...)` of it to measure frames"),
    })
}

/// Every atom the spectral sum stands for, beside the terms its series truncated away or left
/// above the ceiling: `atoms` alone holds only the terms already written out.
struct Listed {
    atoms: Vec<SpectralAtom>,
    dropped: Vec<Line>,
}

impl Listed {
    /// A reading that enumerates no series leaves nothing out.
    const NONE: Listed = Listed {
        atoms: Vec::new(),
        dropped: Vec::new(),
    };

    fn joined(each: Vec<Listed>) -> Listed {
        let mut held = Listed::NONE;
        for one in each {
            held.atoms.extend(one.atoms);
            held.dropped.extend(one.dropped);
        }
        held
    }

    /// The tail is the loudest line left out against the loudest one kept, so
    /// it is one ratio over the whole answer, never a per-series one carried upward.
    fn tail_db(&self) -> Option<f64> {
        let loudest = |set: &mut dyn Iterator<Item = f64>| set.fold(0.0f64, f64::max);
        let gone = loudest(&mut self.dropped.iter().map(|l| l.amp.abs()));
        let kept = loudest(&mut self.atoms.iter().map(|a| a.c.abs()));
        (gone > 0.0 && kept > 0.0).then(|| 20.0 * libm::log10(gone / kept))
    }
}

/// Each lane's series enumerated under the band a rate-free reading is taken in. A series
/// whose term no line closed form reads yields nothing, and nothing is not an exact empty answer.
fn listed(
    render: &Render,
    node: sva_formula::NodeId,
    sum: &SpectralSum,
    profile: &Profile,
) -> Result<Listed, EngineError> {
    let ceiling = profile.ceiling_hz;
    let band = (ceiling, profile.floor(ceiling), profile.half_lsb());
    let mut held = Listed::NONE;
    for lane in &sum.lanes {
        held.atoms.extend(lane.clone().expanded().atoms);
        for series in &lane.series {
            let found = refs::read_through(&render.tys, |t| line_atoms_read(series, band, t));
            let Some(found) = found else {
                return Err(unenumerable(render, node));
            };
            held.atoms.extend(found.atoms);
            held.dropped.extend(found.dropped);
        }
    }
    Ok(held)
}

/// Each channel's lane listed on its own, in channel order.
fn per_channel(
    render: &Render,
    node: sva_formula::NodeId,
    sum: &SpectralSum,
) -> Result<Vec<Listed>, EngineError> {
    sum.lanes
        .iter()
        .map(|lane| {
            let one = SpectralSum::of(sum.var, vec![lane.clone()]);
            listed(render, node, &one, &render.config.profile)
        })
        .collect()
}

/// Every channel in power, as a measured spectrum reads them: a channel's lines at one
/// frequency summed, their squared modulus summed over the channels, divided by the count.
/// Channels share no phase, so each line is that power's root, real.
fn in_power(per_channel: Vec<Vec<Line>>) -> Vec<Line> {
    let count = per_channel.len() as f64;
    let mut power: Vec<(Line, f64)> = Vec::new();
    for lines in per_channel {
        let mut channel: Vec<Line> = Vec::new();
        for line in lines {
            match channel
                .iter_mut()
                .find(|l| l.hz.to_bits() == line.hz.to_bits())
            {
                Some(held) => held.amp = held.amp + line.amp,
                None => channel.push(line),
            }
        }
        for line in channel {
            let squared = line.amp.abs() * line.amp.abs();
            match power
                .iter_mut()
                .find(|(l, _)| l.hz.to_bits() == line.hz.to_bits())
            {
                Some((_, sum)) => *sum += squared,
                None => power.push((line, squared)),
            }
        }
    }
    power
        .into_iter()
        .filter(|(_, sum)| *sum > 0.0)
        .map(|(line, sum)| Line {
            amp: C64::real((sum / count).sqrt()),
            ..line
        })
        .collect()
}

/// The exact line list of a pair: in `t` a bare turning exponential, in `f` the delta it
/// duals to. A list every other atom was dropped from answers a spectrum this node has not.
fn lines(
    render: &Render,
    node: sva_formula::NodeId,
    var: Var,
    atoms: &[SpectralAtom],
) -> Result<Vec<Line>, EngineError> {
    atoms
        .iter()
        .map(|a| line_of(var, a).ok_or_else(|| not_a_line(render, node, var, a)))
        .collect()
}

fn line_of(var: Var, a: &SpectralAtom) -> Option<Line> {
    match (var, a.sing) {
        (Var::F, Singular::Delta { at, order: 0 }) => Some(Line::bare(at, a.c)),
        (Var::T, Singular::Regular) => {
            if !a.poly.is_one() || a.gauss.is_some() || a.ind.is_some() || a.pole.is_some() {
                return None;
            }
            match a.exp {
                // A constant turns at no rate, which is the line at zero hertz.
                None => Some(Line::bare(0.0, a.c)),
                Some(e) if e.sigma == 0.0 => Some(Line::bare(e.omega / std::f64::consts::TAU, a.c)),
                Some(_) => None,
            }
        }
        _ => None,
    }
}

/// What gave this atom a width; a turning exponential is the line itself and never among it.
fn widening(a: &SpectralAtom) -> Vec<&'static str> {
    let mut held = Vec::new();
    if !a.poly.is_one() {
        held.push("a polynomial");
    }
    if a.exp.is_some_and(|e| e.sigma != 0.0) {
        held.push("a decaying exponential");
    }
    if a.gauss.is_some() {
        held.push("a Gaussian");
    }
    if a.pole.is_some() {
        held.push("a pole");
    }
    if held.is_empty() {
        held.push("a delta");
    }
    held
}

/// Convolved into a line, each shape beside the turning exponential answers a band.
fn not_a_line(
    render: &Render,
    node: sva_formula::NodeId,
    var: Var,
    atom: &SpectralAtom,
) -> EngineError {
    let mut at = render.tys.locate(atom.origin);
    if at.node.is_empty() {
        at = Located::at(render.tys.name(node), None);
    }
    let message = match atom.ind {
        _ if var == Var::F => "this term spreads over `f` rather than standing at one \
                               frequency, and a line in `f` is a delta"
            .to_string(),
        Some(window) => format!(
            "this term is windowed to [{}s, {}s), and a windowed line is that line convolved \
             with the window's transform, which has a width and is no line",
            window.l.value(),
            window.r.value()
        ),
        None => format!(
            "this term carries {}, and a line is a bare turning exponential: each of those \
             convolves the line with a shape of its own width",
            widening(atom).join(", ")
        ),
    };
    EngineError::refused(Diagnostic {
        code: "read.lines_need_unwindowed_lines".to_string(),
        message,
        location: at,
        help: "`atoms` states each term as it stands, `lines` of the node under the window \
               or envelope lists the lines it multiplies, and `pitch` or `spectrum` of \
               `sample(...)` of it measures it frame by frame"
            .to_string(),
    })
}

/// One atom as the six factors it is present in, which is what `atoms` answers with.
fn sketch_atom(a: &SpectralAtom) -> String {
    let factors: Vec<&'static str> = a.present().iter().map(|f| f.as_str()).collect();
    format!("{} x {}", a.c.abs(), factors.join(" times "))
}

/// Each atom of every lane as `atoms` answers it, the lane its channel.
pub fn atoms_of(sum: &SpectralSum) -> Vec<Atom> {
    sum.lanes
        .iter()
        .enumerate()
        .flat_map(|(c, lane)| lane.atoms.iter().map(move |a| atom_of(sum.var, a, c)))
        .collect()
}

/// The atom's own fields, read off as it holds them; a constant turns at zero and decays at
/// none. Adding zero writes a negative zero as the zero it equals.
fn atom_of(var: Var, a: &SpectralAtom, channel: usize) -> Atom {
    let (sigma, omega, mu) = a.exp.map_or((0.0, 0.0, 0.0), |e| (e.sigma, e.omega, e.mu));
    let bounded = |edge: Edge| Some(edge.value() + 0.0).filter(|x| x.is_finite());
    Atom {
        var,
        channel,
        sketch: sketch_atom(a),
        amp: a.c,
        turns: omega / std::f64::consts::TAU + 0.0,
        decay: 0.0 - sigma,
        decay_from: mu + 0.0,
        degree: a.poly.degree,
        polynomial_at: a.poly.at + 0.0,
        start: a.ind.and_then(|i| bounded(i.l)),
        end: a.ind.and_then(|i| bounded(i.r)),
    }
}

fn measured(
    render: &Render,
    node: sva_formula::NodeId,
    representation: Representation,
    buffer: &Buffer,
) -> Result<Output, EngineError> {
    Ok(match representation {
        Representation::Alias { oversample } => {
            let reference = oversampled(render, behind(render, node)?, oversample)?;
            Output::Alias(Box::new(worst_alias(
                render, node, buffer, &reference, oversample,
            )?))
        }
        Representation::Ledger { depth } => Output::Ledger(attributed(render, node, depth)?),
        other => {
            let name = render.tys.name(node);
            let longer = |secs: f64| {
                format!("render an interval of at least {secs}s, as `@x([0, {secs}s])` writes one")
            };
            let shorter = |secs: f64| {
                format!("render an interval of at most {secs}s, as `@x([0, {secs}s])` writes one")
            };
            let faster = |hz: f64| format!("render at a rate over {hz} Hz, as `--rate 48000` asks");
            return off_buffer(buffer, other).map_err(|fault| match fault {
                NoReading::NeedsAGraph => not_a_reading(render, node, other),
                NoReading::WidthMismatch { need, held } => width_mismatch(render, node, need, held),
                NoReading::Crowded(held) => crowded(name, other, held),
                NoReading::Unread(why) => unread(name, other, why, longer, faster),
                NoReading::RateUnder(held) => rate_under(name, held, faster),
                NoReading::BlockUnder {
                    window_secs,
                    samples,
                    rate,
                } => block_under(name, (window_secs, samples, rate)),
                NoReading::UnderOrder(why) => under_order(name, why, longer),
                NoReading::Unfit(why) => unfit(name, why, shorter, faster),
            });
        }
    })
}

/// Why a buffer answered nothing: the reading needs the tree the node was built from, or
/// the buffer holds other than the components the reading needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoReading {
    NeedsAGraph,
    WidthMismatch {
        need: usize,
        held: usize,
    },
    Crowded(Crowded),
    Unread(sliding::Unread),
    Unfit(self_similarity::Unfit),
    RateUnder(sva_samples::RateUnderWeighting),
    BlockUnder {
        window_secs: f64,
        samples: usize,
        rate: u32,
    },
    UnderOrder(formants::UnderOrder),
}

/// Every reading a buffer answers on its own, with no graph behind it.
pub fn off_buffer(buffer: &Buffer, representation: Representation) -> Result<Output, NoReading> {
    let sr = f64::from(buffer.rate);
    let start = buffer.origin_secs();
    let planes = buffer.slices();
    Ok(match representation {
        Representation::Samples => Output::Samples(Box::new(buffer.clone())),
        Representation::Spectrum {
            max_peaks,
            frame_secs,
        } => Output::Spectrum(Box::new(spectrum::analyze(
            &planes, sr, max_peaks, frame_secs,
        ))),
        Representation::Envelope { frame_secs } => Output::Envelope(envelope::trace(
            &buffer.slices(),
            sr,
            start,
            frame_secs.unwrap_or(DEFAULT_FRAME_SECS),
        )),
        Representation::Derivative => Output::Samples(Box::new(difference(buffer))),
        Representation::Pitch {
            max_notes,
            frame_secs,
        } => Output::Pitch(pitch::track(&planes, sr, start, frame_secs, max_notes)),
        Representation::Formants {
            max_formants,
            frame_secs,
        } => Output::Formants(
            formants::track(
                &planes,
                sr,
                start,
                frame_secs,
                formants::default_order(sr),
                max_formants,
            )
            .map_err(NoReading::UnderOrder)?,
        ),
        Representation::Stereo { frame_secs } => {
            if buffer.width() != 2 {
                return Err(NoReading::WidthMismatch {
                    need: 2,
                    held: buffer.width(),
                });
            }
            Output::Stereo(Box::new(stereo::analyze(
                &planes,
                buffer.width(),
                sr,
                start,
                frame_secs,
            )))
        }
        Representation::Bands => Output::Bands(Box::new(bands::analyze(&planes, sr, start))),
        Representation::Crest => Output::Crest(Box::new(crest::analyze(&planes, sr))),
        Representation::Score => Output::Score(Box::new(score::measure(&planes, sr, start))),
        Representation::LoudnessPattern { frame_secs, spl_db } => {
            Output::LoudnessPattern(Box::new(loudness_pattern::measure(
                &planes, sr, start, frame_secs, spl_db,
            )))
        }
        Representation::Panel { spl_db } => {
            Output::Panel(Box::new(panel::measure(&planes, sr, start, spl_db)))
        }
        Representation::Loudness { window_secs } => Output::Loudness(Box::new(
            loudness::analyze(&planes, sr, start, window_secs).map_err(|why| match why {
                sva_samples::Unweighted::Rate(held) => NoReading::RateUnder(held),
                sva_samples::Unweighted::Block {
                    window_secs,
                    samples,
                } => NoReading::BlockUnder {
                    window_secs,
                    samples,
                    rate: buffer.rate,
                },
            })?,
        )),
        Representation::Harmony {
            reading,
            frame_secs,
            floor_db,
        } => {
            let (moments, framing) = framed::moments(&planes, sr, start, frame_secs);
            let end_secs = buffer.end() as f64 / sr;
            harmonic(reading, &moments, (framing, end_secs), floor_db)
                .map_err(NoReading::Crowded)?
        }
        Representation::Periodicity {
            frame_secs,
            window_secs,
            curve,
        } => Output::Periodicity(Box::new(
            periodicity::measure(&planes, sr, start, frame_secs, window_secs, curve)
                .map_err(NoReading::Unread)?,
        )),
        Representation::Modulation {
            frame_secs,
            window_secs,
            bpm,
        } => Output::Modulation(Box::new(
            modulation::measure(&planes, sr, start, frame_secs, window_secs, bpm)
                .map_err(NoReading::Unread)?,
        )),
        Representation::SelfSimilarity {
            frame_secs,
            cell_secs,
            smooth_secs,
            window_secs,
            floor_db,
            listed,
            curve,
        } => {
            let asked = self_similarity::Asked {
                hop_secs: frame_secs,
                cell_secs,
                smooth_secs,
                window_secs,
                floor_db,
                listed,
                curve,
            };
            Output::SelfSimilarity(Box::new(
                self_similarity::measure(&planes, sr, start, asked).map_err(NoReading::Unfit)?,
            ))
        }
        Representation::Partials {
            frame_secs,
            window_secs,
            hop_secs,
            floor_db,
            tracks,
        } => {
            let asked = partials::Asked {
                frame_secs,
                window_secs,
                hop_secs,
                floor_db,
                tracks,
            };
            Output::Partials(Box::new(partials::measure(&planes, sr, start, asked)))
        }
        _ => return Err(NoReading::NeedsAGraph),
    })
}

/// The ledger the pull tallied of `node`, refused where none was.
fn attributed(
    render: &Render,
    node: sva_formula::NodeId,
    depth: usize,
) -> Result<Vec<sva_samples::LedgerEntry>, EngineError> {
    super::ledger::entries(render, node, depth).unwrap_or_else(|| {
        Err(EngineError::refused(Diagnostic {
            code: "engine.not_materialized".to_string(),
            message: format!(
                "a ledger is tallied block by block as the render pulls, and this render was \
                 not asked for `ledger` of `{}` to depth {depth}",
                render.tys.name(node)
            ),
            location: Located::at(render.tys.name(node), None),
            help: "ask for the ledger with the render it reads, as `--representation ledger`"
                .to_string(),
        }))
    })
}

/// What every instance under `node` was lowered with, `node` first, then breadth first. The
/// walk is over what each node was lowered to, so it needs no buffer.
fn arguments_under(render: &Render, node: sva_formula::NodeId) -> Vec<crate::Arguments> {
    let under = |id: sva_formula::NodeId| -> Vec<sva_formula::NodeId> {
        match render.tys.value(id) {
            Value::ClosedForm(form) => refs::nodes_in(&form.body),
            Value::Read { source, .. } | Value::Cast(_, source) => vec![*source],
            Value::Op { args, .. } => args.clone(),
            Value::Filter {
                x, cutoff, q, gain, ..
            } => vec![*x, *cutoff, *q, *gain],
            Value::Solver { varying, .. } => varying.iter().map(|(_, a)| *a).collect(),
            Value::SelfAt { .. } | Value::Noise(_) | Value::Recorded(_) => Vec::new(),
        }
    };
    let mut names: Vec<&str> = Vec::new();
    let mut seen = std::collections::BTreeSet::from([node]);
    let mut level = vec![node];
    while !level.is_empty() {
        let mut next = Vec::new();
        for at in level {
            let name = render.tys.name(at);
            if !names.contains(&name) {
                names.push(name);
            }
            next.extend(under(at).into_iter().filter(|c| seen.insert(*c)));
        }
        level = next;
    }
    names
        .into_iter()
        .filter_map(|name| render.tys.arguments(name).cloned())
        .collect()
}

/// The closed form an alias score oversamples: the node itself where it is one, and the operand of
/// the `sample(...)` that collapsed it where it is not.
fn behind(render: &Render, node: sva_formula::NodeId) -> Result<sva_formula::NodeId, EngineError> {
    if render.tys.ty(node).is_closed_form() {
        return Ok(node);
    }
    match render.tys.value(node) {
        crate::typing::Value::Cast(crate::cast::Cast::Sample, source) => Ok(*source),
        _ => Err(refused(
            render,
            node,
            "engine.alias_needs_a_closed_form",
            "an alias score is a render against the same closed form oversampled, and this node \
             is samples with no closed form behind it"
                .to_string(),
        )),
    }
}

/// The score a point sampling's label carries where a reading asks for one, at its multiple.
pub(super) fn alias_db(
    render: &Render,
    node: sva_formula::NodeId,
    buffer: &Buffer,
    oversample: u32,
) -> Result<f64, EngineError> {
    let reference = oversampled(render, behind(render, node)?, oversample)?;
    Ok(worst_alias(render, node, buffer, &reference, oversample)?.asr_db)
}

/// Each component scored against its own reference; the worst answers.
fn worst_alias(
    render: &Render,
    node: sva_formula::NodeId,
    buffer: &Buffer,
    reference: &Buffer,
    oversample: u32,
) -> Result<sva_samples::Alias, EngineError> {
    debug_assert_eq!(
        buffer.width(),
        reference.width(),
        "the reference is the same form at another rate"
    );
    let sr = f64::from(buffer.rate);
    let width = buffer.width().min(reference.width());
    let scored = (0..width).map(|c| {
        measure_alias(
            buffer.plane(c),
            reference.plane(c),
            oversample as usize,
            sr,
            buffer.origin_secs(),
        )
    });
    let scored = scored.collect::<Result<Vec<_>, _>>().map_err(|short| {
        let secs = (short.frame as f64 / sr * 1000.0).ceil() / 1000.0;
        EngineError::refused(Diagnostic {
            code: "read.target_under_frame".to_string(),
            message: format!(
                "the target holds {} samples, fewer than the {}-sample frame an alias score reads",
                short.held, short.frame
            ),
            location: Located::at(render.tys.name(node), None),
            help: format!(
                "render an interval of at least {secs}s, as `@x([0, {secs}s])` writes one"
            ),
        })
    })?;
    Ok(sva_samples::worst_alias(scored).expect("a buffer holds at least one component"))
}

/// The node read at a multiple of the rate by its own route: what an alias score is against.
fn oversampled(
    render: &Render,
    node: sva_formula::NodeId,
    oversample: u32,
) -> Result<Buffer, EngineError> {
    let range = render.range.expect("an alias score reads a decided range");
    let k = i64::from(oversample);
    let extent = Extent::new(range.start * k, range.end * k);
    super::finer(render, node, oversample, extent)
}

/// A first difference on the grid, which is what a derivative is once the closed form is gone:
/// `(x[n] - x[n-1]) * rate` at each sample from the second, as the first has no sample before it.
fn difference(buffer: &Buffer) -> Buffer {
    let step = f64::from(buffer.rate);
    let planes = (0..buffer.width())
        .map(|c| {
            let plane = buffer.plane(c);
            plane.windows(2).map(|x| (x[1] - x[0]) * step).collect()
        })
        .collect();
    let mut out = Buffer::of_planes(buffer.rate, planes);
    out.start = buffer.start + buffer.plane(0).len().min(1) as i64;
    out
}

fn refused(render: &Render, node: sva_formula::NodeId, code: &str, message: String) -> EngineError {
    EngineError::refused(Diagnostic {
        code: code.to_string(),
        message,
        location: Located::at(render.tys.name(node), None),
        help: "ask for a reading this representation answers".to_string(),
    })
}

fn unenumerable(render: &Render, node: sva_formula::NodeId) -> EngineError {
    refused(
        render,
        node,
        "read.series_not_enumerable",
        "a series whose term is no line lists no lines, and an empty list would read as \
         a node with none"
            .to_string(),
    )
}

fn not_a_closed_form(render: &Render, node: sva_formula::NodeId, r: Representation) -> EngineError {
    refused(
        render,
        node,
        "engine.observation_needs_samples",
        format!(
            "`{}` reads samples, and this node is a closed form",
            r.name()
        ),
    )
}

fn not_a_reading(render: &Render, node: sva_formula::NodeId, r: Representation) -> EngineError {
    refused(
        render,
        node,
        "engine.observation_not_wired",
        format!("`{}` takes no reading off a buffer here", r.name()),
    )
}

fn width_mismatch(
    render: &Render,
    node: sva_formula::NodeId,
    need: usize,
    held: usize,
) -> EngineError {
    EngineError::refused(Diagnostic {
        code: "type.width_mismatch".to_string(),
        message: width_message(need, held, "node"),
        location: Located::at(render.tys.name(node), None),
        help: format!("read a node of exactly {need} components, as `join(@a, @b)` writes one"),
    })
}

/// More components than a reading reads refuse too, so none goes unread under its answer.
fn width_message(need: usize, held: usize, holder: &str) -> String {
    format!("this reading needs exactly {need} components and the {holder} holds {held}")
}

pub(super) fn unheld(render: &Render, node: sva_formula::NodeId) -> EngineError {
    refused(
        render,
        node,
        "engine.not_materialized",
        "this render held no samples of it".to_string(),
    )
}

pub(super) fn unmaterialized(
    render: &Render,
    node: sva_formula::NodeId,
    r: Representation,
) -> EngineError {
    refused(
        render,
        node,
        "engine.not_materialized",
        format!("`{}` reads samples this render never held", r.name()),
    )
}

/// One reading off a buffer nothing rendered — an external file, say — under the profile the
/// caller names. Every representation needing the graph behind it refuses here.
pub fn answer_buffer(
    name: &str,
    buffer: &Buffer,
    representation: Representation,
    profile: &'static str,
) -> Result<Answer, EngineError> {
    let refused = |code: &str, message: String| {
        EngineError::refused(Diagnostic {
            code: code.to_string(),
            message,
            location: Located::at(name, None),
            help: "ask for a reading a buffer answers on its own".to_string(),
        })
    };
    let longer = |secs: f64| format!("analyze a file holding at least {secs}s");
    let shorter = |secs: f64| format!("analyze a file holding at most {secs}s");
    let faster = |hz: f64| format!("analyze a file sampled over {hz} Hz");
    let value = off_buffer(buffer, representation).map_err(|fault| match fault {
        NoReading::NeedsAGraph => refused(
            "engine.observation_needs_a_graph",
            format!(
                "`{}` reads the tree a node was built from, and a file has none",
                representation.name()
            ),
        ),
        NoReading::WidthMismatch { need, held } => EngineError::refused(Diagnostic {
            code: "type.width_mismatch".to_string(),
            message: width_message(need, held, "file"),
            location: Located::at(name, None),
            help: format!("analyze a file of exactly {need} channels"),
        }),
        NoReading::Crowded(held) => crowded(name, representation, held),
        NoReading::Unread(why) => unread(name, representation, why, longer, faster),
        NoReading::Unfit(why) => unfit(name, why, shorter, faster),
        NoReading::RateUnder(held) => rate_under(name, held, faster),
        NoReading::BlockUnder {
            window_secs,
            samples,
            rate,
        } => block_under(name, (window_secs, samples, rate)),
        NoReading::UnderOrder(why) => under_order(name, why, longer),
    })?;
    Ok(Answer::whole(
        value,
        Source::Measured,
        profile,
        Some(buffer.rate),
    ))
}
