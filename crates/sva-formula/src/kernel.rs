// Concern: this crate's readings of a form over each number, built once here | Non-concern: what a reading computes (affine.rs, build.rs) | IO: (&Body<N>) -> its reading over N

use crate::affine::{self, Coeff};
use crate::closed_form::{Body, ClosedForm, Unary};
use crate::complex::Cx;
use crate::num::{Dual, Num};
use crate::refusal::Left;
use crate::spectral_sum::SpectralSum;
use crate::spectral_sum::atom::SpectralAtom;
use crate::spectral_sum::build;

/// Another crate's generic code reads through these: one `f64` copy, not one per crate.
pub trait Kernel: Num {
    fn normal(form: &ClosedForm<Self>) -> Result<SpectralSum<Self>, Left>;
    fn written_out(f: &Body<Self>) -> Option<Body<Self>>;
    fn polynomial(f: &Body<Self>) -> Option<Vec<Coeff<Self>>>;
    fn slide(at: &Body<Self>) -> Option<Coeff<Self>>;
    fn exact_affine(f: &Body<Self>) -> Option<(Cx<Self>, Cx<Self>)>;
    fn exact_constant(f: &Body<Self>) -> Option<Cx<Self>>;
    fn apply_scalar(op: Unary, x: Cx<Self>) -> Cx<Self>;
    fn smooth_at(a: &SpectralAtom<Self>, x: Self) -> Option<Cx<Self>>;
    fn smooth_inside(a: &SpectralAtom<Self>, x: Self) -> Option<Cx<Self>>;
}

macro_rules! kernel {
    ($n:ty) => {
        impl Kernel for $n {
            fn normal(form: &ClosedForm<$n>) -> Result<SpectralSum<$n>, Left> {
                build::normalize_closed_form_over(form)
            }
            fn written_out(f: &Body<$n>) -> Option<Body<$n>> {
                crate::series::written_out_over(f)
            }
            fn polynomial(f: &Body<$n>) -> Option<Vec<Coeff<$n>>> {
                affine::polynomial_over(f)
            }
            fn slide(at: &Body<$n>) -> Option<Coeff<$n>> {
                affine::slide_over(at)
            }
            fn exact_affine(f: &Body<$n>) -> Option<(Cx<$n>, Cx<$n>)> {
                affine::exact_affine_over(f)
            }
            fn exact_constant(f: &Body<$n>) -> Option<Cx<$n>> {
                affine::exact_constant_over(f)
            }
            fn apply_scalar(op: Unary, x: Cx<$n>) -> Cx<$n> {
                affine::apply_scalar_over(op, x)
            }
            fn smooth_at(a: &SpectralAtom<$n>, x: $n) -> Option<Cx<$n>> {
                a.smooth_at_over(x)
            }
            fn smooth_inside(a: &SpectralAtom<$n>, x: $n) -> Option<Cx<$n>> {
                a.smooth_inside_over(x)
            }
        }
    };
}

kernel!(f64);
kernel!(Dual);
