// Concern: the number a node's lowering folds in, and the step into a render's own lowering | Non-concern: the arithmetic of a number (sva-formula num.rs) | IO: (Lowering<N>) -> Lowering<f64>

use sva_formula::Kernel;

use crate::error::EngineError;
use crate::lower::{Lowering, Piece};

pub trait Number: Kernel {
    fn render<'s, 'a>(low: &'s mut Lowering<'a, Self>) -> &'s mut Lowering<'a, f64>;
    fn lift(piece: Piece<f64>) -> Piece<Self>;
    fn value(piece: Piece<Self>) -> Piece<f64>;
}

impl Number for f64 {
    #[inline(always)]
    fn render<'s, 'a>(low: &'s mut Lowering<'a, f64>) -> &'s mut Lowering<'a, f64> {
        low
    }
    #[inline(always)]
    fn lift(piece: Piece<f64>) -> Piece<f64> {
        piece
    }
    #[inline(always)]
    fn value(piece: Piece<f64>) -> Piece<f64> {
        piece
    }
}

impl<N: Number> Lowering<'_, N> {
    pub(super) fn rendered(
        &mut self,
        step: impl FnOnce(&mut Lowering<'_, f64>) -> Result<Piece<f64>, EngineError>,
    ) -> Result<Piece<N>, EngineError> {
        step(N::render(self)).map(N::lift)
    }
}
