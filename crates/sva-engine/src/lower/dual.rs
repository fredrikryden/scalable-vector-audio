// Concern: the lowering's numbers, f64 for a render and a dual for a fit, and one instance lowered with tangents | Non-concern: which numbers a fit varies | IO: (graph, instance, varied) -> ClosedForm

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use sva_ast::Graph;
use sva_formula::{Body, ClosedForm, Cx, Dual, Left, Num, Origin, Sparse, SpectralSum, Var};

use crate::error::{Diagnostic, EngineError, Located};
use crate::instantiate::ScopeId;
use crate::lower::{Lowering, Piece, SelfMode};

/// A render's `f64` builds every node a value needs; a dual lowers one closed form and
/// refuses by name whatever would make a node of its own, which no tangent reaches.
pub trait Number: Num {
    fn render<'s, 'a>(low: &'s mut Lowering<'a, Self>) -> Option<&'s mut Lowering<'a, f64>>;
    fn lift(piece: Piece<f64>) -> Piece<Self>;
    fn value(piece: Piece<Self>) -> Piece<f64>;
    /// The normal form, a render's built once in sva-formula.
    fn normal(form: &ClosedForm<Self>) -> Result<SpectralSum<Self>, Left>;
    fn seeded(held: Self, tangent: &[(u32, f64)]) -> Self;
}

impl Number for f64 {
    #[inline(always)]
    fn render<'s, 'a>(low: &'s mut Lowering<'a, f64>) -> Option<&'s mut Lowering<'a, f64>> {
        Some(low)
    }
    #[inline(always)]
    fn lift(piece: Piece<f64>) -> Piece<f64> {
        piece
    }
    #[inline(always)]
    fn value(piece: Piece<f64>) -> Piece<f64> {
        piece
    }
    #[inline(always)]
    fn normal(form: &ClosedForm) -> Result<SpectralSum, Left> {
        sva_formula::normalize_closed_form(form)
    }
    fn seeded(_: f64, _: &[(u32, f64)]) -> f64 {
        unreachable!("a render lowers with no seeds")
    }
}

impl Number for Dual {
    fn render<'s, 'a>(_: &'s mut Lowering<'a, Dual>) -> Option<&'s mut Lowering<'a, f64>> {
        None
    }
    fn lift(piece: Piece<f64>) -> Piece<Dual> {
        match piece {
            Piece::ClosedForm(body) => Piece::ClosedForm(sva_formula::closed_form::lifted(&body)),
            Piece::Value(id) => Piece::Value(id),
        }
    }
    fn value(piece: Piece<Dual>) -> Piece<f64> {
        match piece {
            Piece::ClosedForm(body) => Piece::ClosedForm(sva_formula::closed_form::value(&body)),
            Piece::Value(id) => Piece::Value(id),
        }
    }
    fn normal(form: &ClosedForm<Dual>) -> Result<SpectralSum<Dual>, Left> {
        sva_formula::spectral_sum::build::normalize_closed_form_over(form)
    }
    fn seeded(held: Dual, tangent: &[(u32, f64)]) -> Dual {
        held + Dual::seeded(0.0, tangent)
    }
}

/// Each varied parameter by the scope binding it and its name.
#[derive(Debug, Default)]
pub struct Seeds {
    at: HashMap<(ScopeId, String), Sparse>,
}

impl Seeds {
    pub(crate) fn at(&self, scope: ScopeId, name: &str) -> Option<&[(u32, f64)]> {
        self.at.get(&(scope, name.to_string())).map(Vec::as_slice)
    }
}

pub struct Varied<'a> {
    pub instance: &'a str,
    pub name: &'a str,
    pub tangent: &'a [(u32, f64)],
}

/// The render's own lowering of `instance`, each number beside its tangent. Inside
/// [`sva_formula::Tangents::active`].
pub fn tangents(
    graph: &Graph,
    root: &str,
    rate: u32,
    instance: &str,
    varied: &[Varied],
) -> Result<ClosedForm<Dual>, EngineError> {
    let inst = crate::instantiate::instantiate(graph, root, rate)?;
    let path = inst.instance_of(instance)?;
    let order = crate::schedule::schedule_from(&inst, std::slice::from_ref(&path))?;
    let mut typing = crate::typing::infer_all(&inst, &order)?;
    let mut seeds = Seeds::default();
    for v in varied {
        let held = inst.instance_of(v.instance)?;
        let scope = inst.scope_of(&held).expect("an instance it named");
        if inst.binds(scope, v.name).is_none() {
            return Err(EngineError::refused(Diagnostic {
                code: "dual.unbound_parameter".to_string(),
                message: format!("`{held}` binds no parameter `{}` to vary", v.name),
                location: Located::at(&held, None),
                help: "vary a parameter the instance binds, by a default or its caller".to_string(),
            }));
        }
        seeds
            .at
            .insert((scope, v.name.to_string()), v.tangent.to_vec());
    }
    let (expr, cx) = inst
        .at(&path)
        .ok_or_else(|| EngineError::UnknownNode(path.clone()))?;
    let refuse = |what: &str| untangented(&path, what);
    if inst.reads_self(&path) {
        return Err(refuse("a node reading itself"));
    }
    let var = super::axis_of(&inst, &typing, expr, cx, &path)?;
    let node: Arc<str> = Arc::from(path.as_str());
    let mut low = Lowering {
        inst: &inst,
        typing: &mut typing,
        node,
        indices: Vec::new(),
        mode: SelfMode::Absent,
        own: Vec::new(),
        grid: inst.grid(),
        seeds: Some(&seeds),
        number: PhantomData,
    };
    match low.walk(expr, cx, var)? {
        Piece::ClosedForm(body) => Ok(ClosedForm {
            var,
            body,
            origin: Origin::UNKNOWN,
        }),
        Piece::Value(_) => Err(refuse("a value of its own")),
    }
}

pub(crate) fn untangented(node: &str, what: &str) -> EngineError {
    EngineError::refused(Diagnostic {
        code: "dual.no_tangent".to_string(),
        message: format!(
            "`{node}` passes through {what}, which carries no tangent: only a closed form \
             lowers beside its derivatives"
        ),
        location: Located::at(node, None),
        help: "vary numbers inside a closed form, or leave this node's numbers fixed".to_string(),
    })
}

impl<'a, N: Number> Lowering<'a, N> {
    pub(super) fn rendered(
        &mut self,
        what: &str,
        step: impl FnOnce(&mut Lowering<'_, f64>) -> Result<Piece<f64>, EngineError>,
    ) -> Result<Piece<N>, EngineError> {
        match N::render(self) {
            Some(low) => step(low).map(N::lift),
            None => Err(untangented(&self.node, what)),
        }
    }

    pub(super) fn seed(&self, scope: ScopeId, name: &str) -> Option<&'a [(u32, f64)]> {
        self.seeds.and_then(|s| s.at(scope, name))
    }

    pub(super) fn varied(
        &self,
        piece: Piece<N>,
        tangent: &[(u32, f64)],
        var: Var,
    ) -> Result<Piece<N>, EngineError> {
        let held = match &piece {
            Piece::ClosedForm(body) => super::constant_value(body, var),
            Piece::Value(_) => None,
        };
        match held {
            Some(x) => Ok(Piece::ClosedForm(Body::Const(Cx::real(N::seeded(
                x, tangent,
            ))))),
            None => Err(untangented(
                &self.node,
                "a varied parameter bound to no number",
            )),
        }
    }
}
