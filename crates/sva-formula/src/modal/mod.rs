// Concern: the geometries and the hammer a bank is built from, and the atoms it expands to | Non-concern: any geometry's own eigenfrequencies (the siblings) | IO: (&ModalBank) -> Vec<SpectralAtom>

pub mod bar;
pub mod bore;
pub mod damping;
mod grid;
pub mod membrane;
pub mod room;
pub mod string;

use std::f64::consts::TAU;

use crate::closed_form::{Edge, Excitation, ModalBank, Mode};
use crate::complex::C64;
use crate::modal::damping::Damping;
use crate::origin::Origin;
use crate::spectral_sum::atom::{Exp, Factors, Indicator, Poly, Singular, SpectralAtom};
use crate::spectral_sum::image::raised_cosine;

/// An impulse rings each mode from `t0`; a hammer's force rings it by convolution.
pub fn atoms(bank: &ModalBank, origin: Origin) -> Vec<SpectralAtom> {
    let rings = bank.modes.iter().flat_map(|m| {
        [1.0, -1.0].map(|sign| {
            let onset = C64::new(0.0, sign * m.phase).exp().scale(m.amp * 0.5);
            (onset, C64::new(-1.0 / m.tau, sign * m.omega))
        })
    });
    match bank.excite {
        Excitation::Impulse { t0 } => rings
            .map(|(a, alpha)| {
                onset_atom(
                    a,
                    alpha,
                    t0,
                    (Edge::at(t0), Edge::PosInf),
                    Poly::ONE,
                    origin,
                )
            })
            .collect(),
        Excitation::HammerPulse { f0, t0, contact } => {
            struck(rings.collect(), Force { f0, t0, contact }, origin)
        }
    }
}

/// The felt's force.
struct Force {
    f0: f64,
    t0: f64,
    contact: f64,
}

impl Force {
    fn lines(&self) -> [(C64, C64); 3] {
        let w = TAU / self.contact;
        let side = C64::real(-self.f0 * 0.25);
        [
            (C64::real(self.f0 * 0.5), C64::ZERO),
            (side, C64::new(0.0, w)),
            (side, C64::new(0.0, -w)),
        ]
    }
}

/// Series terms where `|d|T <= 1/2`; the rest is under 1e-23 of `|a q| T`.
const NEAR: usize = 18;

/// `a e^(alpha u)` driven by `q e^(beta u)` over `T`: `a q (e^(beta u) - e^(alpha u)) / d`
/// while pressed, each term under `2|aq|T` past `|d|T = 1/2`, a series within; then a ring.
fn struck(rings: Vec<(C64, C64)>, force: Force, origin: Origin) -> Vec<SpectralAtom> {
    let (t0, contact) = (force.t0, force.contact);
    if contact <= 0.0 || !contact.is_finite() {
        return Vec::new();
    }
    let t1 = t0 + contact;
    let pressing = (Edge::at(t0), Edge::at(t1));
    let lines = force.lines();
    let mut driven = [C64::ZERO; 3];
    let mut out = Vec::with_capacity(3 * rings.len() + lines.len());
    for (a, alpha) in rings {
        let ringing = alpha.scale(contact).exp();
        let (mut head, mut released) = (C64::ZERO, C64::ZERO);
        let mut near = [C64::ZERO; NEAR];
        for (k, (q, beta)) in lines.iter().enumerate() {
            let aq = a * *q;
            let d = *beta - alpha;
            released = released
                + aq.scale(contact) * apart(beta.scale(contact), ringing, d.scale(contact));
            if d.abs() * contact > 0.5 {
                let over = aq / d;
                driven[k] = driven[k] + over;
                head = head - over;
                continue;
            }
            let mut term = aq;
            for (n, held) in near.iter_mut().enumerate() {
                *held = *held + term;
                term = (term * d).over((n + 2) as f64);
            }
        }
        out.push(onset_atom(head, alpha, t0, pressing, Poly::ONE, origin));
        for (n, c) in near.into_iter().enumerate() {
            let power = Poly {
                degree: (n + 1) as u16,
                at: t0,
            };
            out.push(onset_atom(c, alpha, t0, pressing, power, origin));
        }
        let after = (Edge::at(t1), Edge::PosInf);
        out.push(onset_atom(released, alpha, t1, after, Poly::ONE, origin));
    }
    for ((_, beta), c) in lines.iter().zip(driven) {
        out.push(onset_atom(c, *beta, t0, pressing, Poly::ONE, origin));
    }
    out.retain(|atom| !atom.c.is_zero());
    out
}

/// `(e^(beta T) - e^(alpha T)) / (d T)`, uncancelled near `d = 0`.
fn apart(beta_t: C64, ringing: C64, d_t: C64) -> C64 {
    if d_t.re == 0.0 && d_t.im == 0.0 {
        return ringing;
    }
    let lifted = match d_t.re.abs() <= 1.0 && d_t.im.abs() <= 1.0 {
        true => ringing * d_t.exp_m1(),
        false => beta_t.exp() - ringing,
    };
    lifted / d_t
}

/// `c e^(alpha (t - t0))` over `[l, r)`, from `t0` where from zero it overflows.
fn onset_atom(
    c: C64,
    alpha: C64,
    t0: f64,
    (l, r): (Edge, Edge),
    poly: Poly,
    origin: Origin,
) -> SpectralAtom {
    let from_zero = c * (-alpha.scale(t0)).exp();
    let (c, exp) = match (alpha.re == 0.0 && alpha.im == 0.0, from_zero.is_finite()) {
        (true, _) => (c, None),
        (false, true) => (from_zero, Some(Exp::at(alpha.re, alpha.im))),
        (false, false) => (
            c * C64::new(0.0, -alpha.im * t0).exp(),
            Some(Exp {
                sigma: alpha.re,
                omega: alpha.im,
                mu: t0,
            }),
        ),
    };
    SpectralAtom::new(
        c,
        Factors {
            poly,
            exp,
            ind: Some(Indicator { l, r }),
            ..Factors::NONE
        },
        Singular::Regular,
        origin,
    )
}

/// Which geometry a modal builtin names.
pub enum Geometry {
    String(string::StringGeom),
    Membrane(membrane::MembraneGeom),
    Bar(bar::BarGeom),
    Bore(bore::BoreGeom),
    Room(room::RoomGeom),
}

pub fn mode(hz: f64, amp: f64, damping: Damping) -> Mode {
    Mode {
        omega: TAU * hz,
        tau: damping.tau(hz),
        amp,
        phase: 0.0,
    }
}

pub fn modes(g: &Geometry, count: usize) -> Vec<Mode> {
    match g {
        Geometry::String(s) => string::modes(s, count),
        Geometry::Membrane(m) => membrane::modes(m, count),
        Geometry::Bar(b) => bar::modes(b, count),
        Geometry::Bore(b) => bore::modes(b, count),
        Geometry::Room(r) => room::modes(r, count),
    }
}

/// Hertzian felt is `p = 1.5`, piano felt about 2.5; Chaigne & Askenfelt, JASA 95 (1994) 1112.
pub struct HammerGeom {
    pub reference_velocity: f64,
    pub reference_contact: f64,
    pub stiffness_exponent: f64,
    pub force: f64,
}

pub fn contact_time(velocity: f64, h: &HammerGeom) -> f64 {
    let p = h.stiffness_exponent;
    h.reference_contact * (velocity / h.reference_velocity).powf((1.0 - p) / (1.0 + p))
}

/// `F0 * Tc` is the hammer's momentum, so the peak force carries what `Tc` does not.
pub fn strike(velocity: f64, h: &HammerGeom, at: f64) -> Excitation {
    let p = h.stiffness_exponent;
    let ratio = velocity / h.reference_velocity;
    Excitation::HammerPulse {
        f0: h.force * ratio.powf(2.0 * p / (p + 1.0)),
        t0: at,
        contact: contact_time(velocity, h),
    }
}

/// A whole turn of a raised cosine, upside down, over the contact; none over no contact.
pub fn pulse(f0: f64, t0: f64, contact: f64, origin: Origin) -> Vec<SpectralAtom> {
    if contact <= 0.0 || !contact.is_finite() {
        return Vec::new();
    }
    let [(level, _), (side, turn), _] = Force { f0, t0, contact }.lines();
    raised_cosine(t0, t0 + contact, turn.im, level.re, side.re, origin)
}
