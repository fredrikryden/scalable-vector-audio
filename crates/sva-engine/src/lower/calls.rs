// Concern: the formula each written call lowers to | Non-concern: the recursion over an Expr (mod.rs) | IO: (name, args) -> a Piece

use sva_ast::{Arg, ByteSpan, Expr, Literal};
use sva_formula::{
    Body, C64, Codomain, Cx as Complex, Edge, Fold, Held, NodeId, Num, Origin, Part, Ty, Unary,
    Var, hash,
};
use sva_samples::physics::Varies;

use crate::arguments::{Argument, Called, Chosen};
use crate::cast::Cast;
use crate::error::EngineError;
use crate::instantiate::Cx;
use crate::lower::number::Number;
use crate::lower::{Lowering, Piece};
use crate::refs::{PerNode, written_form};
use crate::time::Lattice;
use crate::typing::{Value, When};
use sva_ast::SERIES;

/// A crop's two edges.
type Window<N> = (Edge<N>, Edge<N>);

impl<N: Number> Lowering<'_, N> {
    pub(super) fn call(
        &mut self,
        name: &str,
        args: &[Arg],
        span: ByteSpan,
        cx: Cx,
        var: Var,
    ) -> Result<Piece<N>, EngineError> {
        if name == SERIES {
            return self.sum(args, span, cx, var);
        }
        if name == sva_ast::INDEX {
            return Err(self.refused_at(
                "type.index_outside_read",
                "idx(...) names a sample index, which only an index read takes".to_string(),
                "read by index, as @x[idx(t - 0.5b)], or write the time itself",
                Some(span),
            ));
        }
        let written = args.iter().filter(|a| matches!(a, Arg::Pos(_))).count();
        let keys: Vec<String> = args
            .iter()
            .filter_map(|a| match a {
                Arg::Named(k, _) => Some(k.clone()),
                Arg::Pos(_) => None,
            })
            .collect();
        crate::overload::check_arity(name, written, &keys)
            .map_err(|m| self.refuse(name, &m, Some(span)))?;
        if name == sva_ast::PWL || name == sva_ast::PWL_INTEGRAL {
            return self.knots(name, args, span, cx, var);
        }
        if name == "draw" {
            return self.keyed(args, span, cx, var);
        }
        let mut chosen = Vec::new();
        let named = self.named_values(name, args, span, cx, &mut chosen)?;
        let view: Vec<(&str, N)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let values: Vec<(String, f64)> = named.iter().map(|(k, v)| (k.clone(), v.v())).collect();
        let solved = sva_ast::FINITE_DIFFERENCE.contains(&name);
        if solved || sva_ast::MODAL.contains(&name) {
            return self
                .rendered(|low| low.physical(name, args, span, (values, chosen), (cx, var)));
        }
        if name == "noise" {
            return self.rendered(|low| low.noise(args, span, (values, chosen), cx));
        }
        N::render(self).note_call(name, span, written_named(&values), chosen);
        if let Some(cast) = Cast::from_name(name) {
            return self.rendered(|low| {
                let cast = match cast {
                    Cast::Stft { .. } => Cast::Stft {
                        window: low.step_count(args, "window", span, cx)?,
                        hop: low.step_count(args, "hop", span, cx)?,
                    },
                    other => other,
                };
                low.cast(cast, args, span, cx, var)
            });
        }
        if let Some(shape) = crate::vocabulary::shape(name) {
            return self.rendered(|low| low.filter(shape, args, span, cx, var));
        }
        if name == "rand" {
            return self.rendered(|low| low.drawn(args, span, cx, var));
        }
        let positional: Vec<&Expr> = args
            .iter()
            .filter_map(|a| match a {
                Arg::Pos(x) => Some(x),
                Arg::Named(..) => None,
            })
            .collect();
        if let Some(piece) = self.wave(name, &positional, &view, cx, var)? {
            return Ok(piece);
        }
        let mut pieces = Vec::with_capacity(positional.len());
        for (at, x) in positional.iter().enumerate() {
            pieces.push(match (name, at, x) {
                ("crop", 1 | 2, Expr::Lit(Literal::Samples(n)))
                    if let Some(edge) = self.counted(*n) =>
                {
                    Piece::ClosedForm(Body::Const(Complex::real(N::of(edge))))
                }
                _ => self.walk(x, cx, var)?,
            });
        }
        if let Some(held) = beside_infinity(name, &mut pieces) {
            return Ok(held);
        }
        if pieces.iter().any(|p| matches!(p, Piece::Value(_))) {
            let pieces: Vec<Piece> = pieces.into_iter().map(N::value).collect();
            return self.rendered(|low| {
                let view: Vec<(&str, f64)> = values.iter().map(|(k, v)| (k.as_str(), *v)).collect();
                match (name, view.iter().find(|(k, _)| *k == "drive")) {
                    ("sat", Some((_, drive))) => low.driven(pieces, *drive, span, var),
                    ("crop", _) => low.sampled_crop(pieces, &view, span, var),
                    _ => low.operation(name, pieces, Some(span), var),
                }
            });
        }
        let bodies: Vec<Body<N>> = pieces
            .into_iter()
            .map(|p| match p {
                Piece::ClosedForm(f) => f,
                Piece::Value(_) => unreachable!("every piece here stayed inside the closed form"),
            })
            .collect();
        self.image(name, bodies, &positional, &view, span, var)
    }

    /// An edge written in `sp` is a count of samples: it opens on the first sample at or past
    /// the count, whatever decimal the instant prints as.
    fn counted(&self, n: f64) -> Option<f64> {
        let at = n.ceil();
        (at.abs() < 2f64.powi(53)).then_some(())?;
        self.grid.edge_at(at as i64)
    }

    /// The atoms a written name lowers to, once every operand is inside the same closed form.
    fn image(
        &mut self,
        name: &str,
        bodies: Vec<Body<N>>,
        written: &[&Expr],
        named: &[(&str, N)],
        span: ByteSpan,
        var: Var,
    ) -> Result<Piece<N>, EngineError> {
        let read = |key: &str, fallback: f64| named_or(named, key, N::of(fallback));
        let arity = |want: usize| -> Result<(), EngineError> {
            match bodies.len() == want {
                true => Ok(()),
                false => Err(EngineError::BadArity(name.to_string())),
            }
        };
        if numeric(name) {
            let origin = self.typing.mark(&self.node, Some(span));
            let two = bodies.len() == 2;
            return match arithmetic(name, bodies, named, var, origin) {
                Some(body) => self.folded(body, var),
                None if name == "pow" && two => Err(self.non_integer_power(written, span)),
                None => Err(EngineError::BadArity(name.to_string())),
            };
        }
        match name {
            "crop" => {
                arity(3)?;
                let (rise, fall) = shoulders_of(named);
                let Some((l, r)) = self.cropped(&bodies[1], &bodies[2], (rise, fall), var, span)?
                else {
                    return Ok(Piece::ClosedForm(Body::Const(Complex::ZERO)));
                };
                let fall = match r {
                    Edge::PosInf => N::ZERO,
                    _ => fall,
                };
                let of = self.part(first(bodies), Some(span));
                Ok(Piece::ClosedForm(Body::Crop {
                    of,
                    l,
                    r,
                    rise,
                    fall,
                }))
            }
            "delta" => {
                arity(1)?;
                let at = self.part(first(bodies), Some(span));
                Ok(Piece::ClosedForm(Body::Delta {
                    at,
                    order: read("k", 0.0).v() as u16,
                }))
            }
            "pv" => {
                arity(1)?;
                let at = self.part(first(bodies), Some(span));
                Ok(Piece::ClosedForm(Body::Pv(at)))
            }
            "join" => {
                let parts = self.parts(bodies, span);
                Ok(Piece::ClosedForm(Body::Join(parts)))
            }
            "ch" => {
                arity(2)?;
                let Some(index) = super::constant_value(&bodies[1], var) else {
                    return Err(EngineError::BadArity(name.to_string()));
                };
                let of = self.part(first(bodies), Some(span));
                Ok(Piece::ClosedForm(Body::Channel(of, index.v() as u8)))
            }
            other => Err(EngineError::UnknownBuiltin(other.to_string())),
        }
    }

    fn non_integer_power(&self, written: &[&Expr], span: ByteSpan) -> EngineError {
        let [_, exponent] = written else {
            unreachable!("pow checked its arity before it lowered either operand")
        };
        self.refused_at(
            "type.non_integer_power",
            format!(
                "`pow` raises a signal to `{}`, which names no polynomial power.",
                sva_ast::render_expr(exponent)
            ),
            "write a whole exponent within 65535, or a positive constant base, which raises \
             as exp(x*ln(base))",
            Some(span),
        )
    }

    /// A window's bounds are two numbers, and a constant expression is one: every ref
    /// naming a number folds first, so `@variables/bar*3` is the second a bar names. Only a
    /// bound that moves with the free variable has no number to be.
    fn window(
        &mut self,
        l: &Body<N>,
        r: &Body<N>,
        var: Var,
        name: &str,
        span: ByteSpan,
    ) -> Result<Window<N>, EngineError> {
        Ok((
            self.edge(l, var, name, "start", span)?,
            self.edge(r, var, name, "end", span)?,
        ))
    }

    /// A crop's window and shoulders, for either representation of its operand; `None`
    /// where the window opens at or after it closes, which is zero whatever it crops.
    fn cropped(
        &mut self,
        l: &Body<N>,
        r: &Body<N>,
        (rise, fall): (N, N),
        var: Var,
        span: ByteSpan,
    ) -> Result<Option<Window<N>>, EngineError> {
        let (l, r) = self.window(l, r, var, "crop", span)?;
        if l >= r {
            return Ok(None);
        }
        self.shoulders(rise.v(), fall.v(), r.value() - l.value(), span)?;
        Ok(Some((l, r)))
    }

    /// A shoulder opens inside the window it belongs to. Zero is a hard edge, a negative one
    /// opens before the window starts, and two that overlap add energy the term never had.
    fn shoulders(&self, rise: f64, fall: f64, span: f64, at: ByteSpan) -> Result<(), EngineError> {
        let refuse = |what: String| {
            Err(self.refused_at(
                "engine.bad_crop_shoulder",
                what,
                "write a rise and a fall at or above zero that together fit the window",
                Some(at),
            ))
        };
        if rise < 0.0 || fall < 0.0 {
            return refuse(format!(
                "`crop` reads rise={rise} and fall={fall}; a shoulder \
                 opens forward in time."
            ));
        }
        if rise + fall > span {
            return refuse(format!(
                "`crop`'s rise={rise} and fall={fall} are {} together, longer than the {span} \
                 the window itself runs.",
                rise + fall
            ));
        }
        Ok(())
    }

    fn edge(
        &self,
        body: &Body<N>,
        var: Var,
        name: &str,
        which: &str,
        span: ByteSpan,
    ) -> Result<Edge<N>, EngineError> {
        let folded = crate::refs::fold_constants(self.typing, body);
        let infinite = match *folded {
            Body::Const(c) if c.im.is_zero() && c.re.v().is_infinite() => Some(c.re),
            _ => None,
        };
        match infinite.or_else(|| super::constant_value(&folded, var)) {
            Some(x) => Ok(Edge::at(x)),
            None => Err(self.refused_at(
                "engine.non_constant_argument",
                format!("`{name}` reads `{which}` as a number, and this one moves."),
                "write a constant there; a bound that moves with t names no window",
                Some(span),
            )),
        }
    }

    /// `draw(k, seed=)` is the draw `noise` gives its line `k` under that seed: one number at
    /// any rate, as both fold to numbers when the node lowers and neither reads `t`.
    fn keyed(
        &mut self,
        args: &[Arg],
        span: ByteSpan,
        cx: Cx,
        var: Var,
    ) -> Result<Piece<N>, EngineError> {
        let bad = || EngineError::BadArity("draw".to_string());
        let (key, seed) = draw_arguments(args).ok_or_else(bad)?;
        let k = self.key_number(key, "k", (cx, var), span)?;
        let seed = self.key_number(seed, "seed", (cx, var), span)?;
        match keyed_draw(k.v(), seed.v()) {
            Some(drawn) => Ok(Piece::ClosedForm(Body::Const(Complex::real(N::of(drawn))))),
            None => Err(self.refused_at(
                "engine.draw_past_index",
                format!(
                    "`draw` reads `k={}`, past the last whole step an i64 holds; a \
                     correctness limit, with no override",
                    k.v()
                ),
                "key the draw by a whole number within an i64",
                Some(span),
            )),
        }
    }

    /// A draw's key or seed, folded to the number it names when the node lowers.
    fn key_number(
        &mut self,
        written: &Expr,
        key: &str,
        (cx, var): (Cx, Var),
        span: ByteSpan,
    ) -> Result<N, EngineError> {
        let folded = match self.walk(written, cx, var)? {
            Piece::ClosedForm(body) => {
                let folded = crate::refs::fold_constants(self.typing, &body);
                if super::holds_infinite(&folded) {
                    return Err(self.infinite(&format!(
                        "`draw` reads `{key}={}`, which is infinite, and a draw is keyed by a \
                         finite number",
                        sva_ast::render_expr(written)
                    )));
                }
                super::constant_value(&folded, var)
            }
            Piece::Value(_) => None,
        };
        folded.ok_or_else(|| {
            self.refused_at(
                "engine.non_constant_argument",
                format!(
                    "`draw` reads `{key}={}` as one number when its node lowers, and this one \
                     moves: a draw is keyed, never a function of time.",
                    sva_ast::render_expr(written)
                ),
                "write a number, a parameter its row binds, or a finite sum's index; \
                 `rand(t, seed=k)` is noise in time",
                Some(span),
            )
        })
    }

    fn parts(&mut self, bodies: Vec<Body<N>>, span: ByteSpan) -> Vec<Part<N>> {
        bodies
            .into_iter()
            .map(|b| self.part(b, Some(span)))
            .collect()
    }

    /// An argument that folds to inf, or to no number, names none any builtin reads.
    pub(super) fn finite_argument(
        &self,
        name: &str,
        key: &str,
        written: &Expr,
        folded: Option<f64>,
    ) -> Result<(), EngineError> {
        match folded {
            Some(v) if !v.is_finite() => Err(self.infinite(&format!(
                "`{name}` reads `{key}={}`, which is {v}, and an argument is a finite number",
                sva_ast::render_expr(written)
            ))),
            _ => Ok(()),
        }
    }

    /// A level is the amplitude ratio `db` writes, so one that folds to zero or below names no
    /// decibel: `-6`, a level written as a bare number in dB, is refused, never read as a ratio.
    pub(super) fn ratio_argument(
        &self,
        name: &str,
        key: &str,
        written: &Expr,
        cx: Cx,
        span: ByteSpan,
    ) -> Result<(), EngineError> {
        match crate::loops::amount(self.inst, written, cx) {
            Some(v) if v <= 0.0 => Err(self.refused_at(
                "engine.level_not_a_ratio",
                format!(
                    "`{name}` reads `{key}={}`: `{key}` is an amplitude ratio, and {v} is at \
                     or below zero, where no level in dB is",
                    sva_ast::render_expr(written)
                ),
                &format!("write the level in db, as `{key}=-6db`; a bare number is a ratio"),
                Some(span),
            )),
            _ => Ok(()),
        }
    }

    /// Every named number a call reads. One that folds to inf is refused; one the vocabulary
    /// lets move is its builtin's to route when it folds to none; any other is refused.
    fn named_values(
        &self,
        name: &str,
        args: &[Arg],
        span: ByteSpan,
        cx: Cx,
        chosen: &mut Vec<Chosen>,
    ) -> Result<Vec<(String, N)>, EngineError> {
        let mut out = Vec::new();
        for arg in args {
            let Arg::Named(key, value) = arg else {
                continue;
            };
            let folded: Option<N> = crate::loops::amount_choosing(self.inst, value, cx, chosen);
            self.finite_argument(name, key, value, folded.map(N::v))?;
            match folded.and_then(crate::loops::plain) {
                Some(v) => out.push((key.clone(), v)),
                None if crate::vocabulary::named_may_move(name, key) => {}
                None => {
                    return Err(self.refused_at(
                        "engine.non_constant_argument",
                        format!(
                            "`{name}` reads `{key}={}` as one number, and it names none.",
                            sva_ast::render_expr(value)
                        ),
                        "write a constant there: a number, a unit, or a ref naming one",
                        Some(span),
                    ));
                }
            }
        }
        Ok(out)
    }

    /// A number a call reads as an argument. A grid count is not one: `1sp` names no
    /// cutoff, so the same fold that reads a delay answers `None` for it here.
    pub(super) fn named_value(&self, e: &Expr, cx: Cx) -> Option<N> {
        crate::loops::plain(crate::loops::amount_in(self.inst, e, cx)?)
    }
}

impl Lowering<'_> {
    /// A finite-difference solver or a modal bank, each of plain numbers.
    fn physical(
        &mut self,
        name: &str,
        args: &[Arg],
        span: ByteSpan,
        (named, mut chosen): (Vec<(String, f64)>, Vec<Chosen>),
        (cx, var): (Cx, Var),
    ) -> Result<Piece, EngineError> {
        let view: Vec<(&str, f64)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let Some(numbers) = positional_values(self, args, cx, &mut chosen) else {
            return Err(EngineError::BadArity(name.to_string()));
        };
        if !sva_ast::FINITE_DIFFERENCE.contains(&name) {
            self.note_call(name, span, positional_named(name, &numbers, &named), chosen);
            return self.modal(name, &numbers, &view);
        }
        self.solver(name, (&numbers, &view, args), span, chosen, (cx, var))
    }

    fn noise(
        &mut self,
        args: &[Arg],
        span: ByteSpan,
        (named, mut chosen): (Vec<(String, f64)>, Vec<Chosen>),
        cx: Cx,
    ) -> Result<Piece, EngineError> {
        let name = "noise";
        let view: Vec<(&str, f64)> = named.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let Some(numbers) = positional_values(self, args, cx, &mut chosen) else {
            return Err(EngineError::BadArity(name.to_string()));
        };
        let [seed, ..] = numbers.as_slice() else {
            return Err(EngineError::BadArity(name.to_string()));
        };
        if let Some(Arg::Named(_, color)) = args
            .iter()
            .find(|a| matches!(a, Arg::Named(k, _) if k == "color"))
        {
            self.ratio_argument(name, "color", color, cx, span)?;
        }
        let period = named_or(&view, "period", 1.0);
        let lines = self.noise_band(&view, period, span)?;
        self.note_call(name, span, positional_named(name, &numbers, &named), chosen);
        Ok(Piece::ClosedForm(Body::Series(Box::new(
            sva_formula::noise_in(seed_of(*seed), period, named_or(&view, "color", 1.0), lines),
        ))))
    }

    /// The lines `lo` and `hi` keep, every line from the first where neither is written. Each
    /// already folded to a finite number, so only its sign, order and lines are left to refuse.
    fn noise_band(
        &self,
        view: &[(&str, f64)],
        period: f64,
        span: ByteSpan,
    ) -> Result<(i64, sva_formula::Bound), EngineError> {
        let edge = |key: &str| view.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);
        let (lo, hi) = (edge("lo"), edge("hi"));
        if lo.is_none() && hi.is_none() {
            return Ok(sva_formula::noise::EVERY_LINE);
        }
        let refused = |code: &str, message: String, help: &str| {
            Err(self.refused_at(code, message, help, Some(span)))
        };
        for (key, v) in [("lo", lo), ("hi", hi)] {
            if let Some(v) = v.filter(|v| *v < 0.0) {
                return refused(
                    "engine.noise_band_negative",
                    format!(
                        "`noise` reads `{key}={v}`, and a band edge is a frequency in Hz, at or above 0"
                    ),
                    "write the edge as 0 or more hertz, as `lo=200hz`",
                );
            }
        }
        if let (Some(lo), Some(hi)) = (lo, hi)
            && lo >= hi
        {
            return refused(
                "engine.noise_band_inverted",
                format!(
                    "`noise` reads `lo={lo}` and `hi={hi}`, and a band's lower edge is under its upper"
                ),
                "write `lo` under `hi`, as `lo=200hz, hi=400hz`",
            );
        }
        let Some((first, last)) = sva_formula::noise::band(period, lo.unwrap_or(0.0), hi) else {
            return refused(
                "engine.noise_band_past_index",
                format!(
                    "`noise` over `period={period}` reads an edge past line {}, the last index a \
                     line holds; a correctness limit, with no override",
                    i64::MAX
                ),
                "lower the edge or the period",
            );
        };
        if let sva_formula::Bound::Finite(last) = last
            && first > last
        {
            return refused(
                "engine.noise_band_empty",
                format!(
                    "`noise` holds a line every {} Hz, and none lies in its band from {} to {} Hz",
                    1.0 / period,
                    lo.unwrap_or(0.0),
                    hi.unwrap_or(f64::INFINITY)
                ),
                "widen the band to hold a multiple of 1/period, or lengthen the period",
            );
        }
        Ok((first, last))
    }

    /// The product `arithmetic` writes, so the drive reaches the renderer and the identity.
    fn driven(
        &mut self,
        mut pieces: Vec<Piece>,
        drive: f64,
        span: ByteSpan,
        var: Var,
    ) -> Result<Piece, EngineError> {
        pieces.push(Piece::ClosedForm(Body::Const(C64::real(drive))));
        let product = self.operation("*", pieces, Some(span), var)?;
        self.operation("sat", vec![product], Some(span), var)
    }

    /// A closed form's window: its shoulders, refused alike, ride along as two numbers.
    fn sampled_crop(
        &mut self,
        mut pieces: Vec<Piece>,
        named: &[(&str, f64)],
        span: ByteSpan,
        var: Var,
    ) -> Result<Piece, EngineError> {
        let (rise, fall) = shoulders_of(named);
        if let [_, Piece::ClosedForm(l), Piece::ClosedForm(r)] = pieces.as_slice()
            && self.cropped(l, r, (rise, fall), var, span)?.is_none()
        {
            return Ok(Piece::ClosedForm(Body::Const(C64::ZERO)));
        }
        if rise > 0.0 || fall > 0.0 {
            for shoulder in [rise, fall] {
                pieces.push(Piece::ClosedForm(Body::Const(C64::real(shoulder))));
            }
        }
        self.operation("crop", pieces, Some(span), var)
    }

    fn chosen_value(&self, e: &Expr, cx: Cx, chosen: &mut Vec<Chosen>) -> Option<f64> {
        crate::loops::plain(crate::loops::amount_choosing(self.inst, e, cx, chosen)?)
    }

    /// The numbers a call was lowered with, where it was written, and what each constant
    /// `min`/`max` inside them chose.
    fn note_call(
        &mut self,
        name: &str,
        at: ByteSpan,
        arguments: Vec<Argument>,
        chosen: Vec<Chosen>,
    ) {
        if arguments.is_empty() && chosen.is_empty() || !self.grid.is_rate() {
            return;
        }
        let call = (!arguments.is_empty()).then(|| Called {
            name: name.to_string(),
            at,
            arguments,
        });
        self.typing.note(&self.node, call, chosen);
    }

    /// Checked in range before its grid is sized from it, then noted and registered. A
    /// varying parameter that names no number is a node the site reads every sample.
    fn solver(
        &mut self,
        name: &str,
        (numbers, view, args): (&[f64], &[(&str, f64)], &[Arg]),
        span: ByteSpan,
        chosen: Vec<Chosen>,
        (cx, var): (Cx, Var),
    ) -> Result<Piece, EngineError> {
        let [first, ..] = numbers else {
            return Err(EngineError::BadArity(name.to_string()));
        };
        let mut params = super::solvers::params(name, *first, view);
        let mut varying = Vec::new();
        for (key, varies) in sva_samples::physics::varying(name) {
            let written = args.iter().find_map(|a| match a {
                Arg::Named(k, x) if k == key => Some(x),
                _ => None,
            });
            let Some(written) = written.filter(|_| !view.iter().any(|(k, _)| k == key)) else {
                continue;
            };
            let id = self.automation((name, key), written, cx, var)?;
            if let Value::ClosedForm(form) = self.typing.value(id)
                && let Some(v) = super::constant_value(&form.body, form.var)
            {
                *super::field(&mut params, key).expect("a varying field") = v + 0.0;
                continue;
            }
            if *varies == Varies::Piecewise && !self.piecewise(id) {
                return Err(self.moving_energy(name, key, span));
            }
            varying.push((*key, id));
        }
        if !params.valid() {
            return Err(EngineError::refused(crate::error::Diagnostic {
                code: "engine.physics_out_of_range".to_string(),
                message: format!("`{name}` was given an argument outside the range it models"),
                location: crate::error::Located::at(name, Some(span)),
                help: "`sva-cli builtins` names every argument each solver takes".to_string(),
            }));
        }
        let handed = super::solvers::handed(&params)
            .into_iter()
            .filter(|(key, _)| !varying.iter().any(|(k, _)| k == key))
            .enumerate()
            .map(|(at, (key, value))| Argument {
                written: at == 0 || view.iter().any(|(k, _)| *k == key),
                name: key,
                value,
            })
            .collect();
        self.note_call(name, span, handed, chosen);
        let value = Value::Solver {
            params: Box::new(params),
            varying,
        };
        let ty = Ty::discrete(Held::Sampled, Codomain::Real);
        Ok(Piece::Value(self.register(value, ty, var)))
    }

    /// A stored-energy coefficient holds between jumps: numbers under windows that open and
    /// close without shoulders, and arithmetic over them.
    fn piecewise(&self, id: NodeId) -> bool {
        let Value::ClosedForm(form) = self.typing.value(id) else {
            return false;
        };
        let typing: &crate::typing::Typing = self.typing;
        crate::refs::reads_through(typing, &form.body, form.var)
            && Steps {
                typing,
                steps: PerNode::new(),
                constant: PerNode::new(),
            }
            .steps(&form.body)
    }

    fn moving_energy(&self, name: &str, key: &str, span: ByteSpan) -> EngineError {
        self.refused_at(
            "engine.moving_energy_parameter",
            format!(
                "`{name}` stores energy in `{key}`, which may only jump: a coefficient that \
                 moves between jumps can pump energy into the scheme"
            ),
            "write it as numbers under crops that open and close without shoulders, as \
             k*step(t - t0)",
            Some(span),
        )
    }

    /// A duration a cast counts in steps of the rate in use, which has to be a whole number of
    /// them; zero where it is not written.
    fn step_count(
        &self,
        args: &[Arg],
        key: &str,
        span: ByteSpan,
        cx: Cx,
    ) -> Result<usize, EngineError> {
        let Some(written) = args.iter().find_map(|a| match a {
            Arg::Named(k, x) if k == key => Some(x),
            _ => None,
        }) else {
            return Ok(0);
        };
        let count = crate::loops::time_of(self.inst, written, cx)
            .filter(|at| at.scale.is_zero())
            .and_then(|at| at.shift.div(self.grid.steps(crate::time::Q::ONE)?))
            .filter(|n| n.is_integer() && n.num() > 0);
        match count {
            Some(n) => Ok(n.num() as usize),
            None => Err(self.refused_at(
                "cast.window_off_the_lattice",
                format!(
                    "`stft`'s `{key}` is no whole number of steps at this rate: {}",
                    sva_ast::render_expr(written)
                ),
                "write it in sp, as window=2048sp",
                Some(span),
            )),
        }
    }

    /// `rand(key, seed=)` is white noise, one draw per step of the rate in use keyed by that
    /// step's index, read at the step nearest the key: a constant key is one number, `t` the
    /// noise itself, and any other time a read of it.
    fn drawn(
        &mut self,
        args: &[Arg],
        span: ByteSpan,
        cx: Cx,
        var: Var,
    ) -> Result<Piece, EngineError> {
        let bad = || EngineError::BadArity("rand".to_string());
        let (key, seed) = rand_arguments(args, |x| self.named_value(x, cx)).ok_or_else(bad)?;
        let ty = sva_formula::Ty::discrete(sva_formula::Held::Sampled, sva_formula::Codomain::Real);
        let (grid, at) = match crate::loops::time_of(self.inst, key, cx) {
            Some(at) if at.scale.is_zero() => {
                let drawn = noise_at(seed, at.shift, self.inst.rate());
                return Ok(Piece::ClosedForm(Body::Const(C64::real(drawn))));
            }
            Some(at) => {
                let unplaced = || EngineError::BadArity("rand".to_string());
                (self.grid.speed(at).ok_or_else(unplaced)?, When::At(at))
            }
            None => (self.grid, When::Moving(self.time(key, cx)?)),
        };
        let noise = self.typing.push(
            crate::typing::Node {
                name: std::sync::Arc::clone(&self.node),
                ty,
                var,
                value: Value::Noise(seed),
                grid,
            },
            None,
        );
        match at {
            When::At(crate::time::Affine::NOW) => Ok(Piece::Value(noise)),
            at => Ok(Piece::Value(self.reading(noise, at, span, var))),
        }
    }
}

/// The key and the seed of `rand(key, seed)` or `rand(key, seed=)`.
pub(crate) fn rand_arguments(
    args: &[Arg],
    number: impl Fn(&Expr) -> Option<f64>,
) -> Option<(&Expr, u64)> {
    let mut positional = args.iter().filter_map(|a| match a {
        Arg::Pos(x) => Some(x),
        Arg::Named(..) => None,
    });
    let key = positional.next()?;
    let seed = match positional.next() {
        Some(x) => number(x)?,
        None => match args.iter().find_map(|a| match a {
            Arg::Named(k, x) if k == "seed" => Some(x),
            _ => None,
        }) {
            Some(x) => number(x)?,
            None => 0.0,
        },
    };
    Some((key, seed_of(seed)))
}

/// The key and the seed of `draw(k, seed=)`, each by position or by name.
pub(crate) fn draw_arguments(args: &[Arg]) -> Option<(&Expr, &Expr)> {
    let written = |at: usize, key: &str| {
        args.iter()
            .filter_map(|a| match a {
                Arg::Pos(x) => Some(x),
                Arg::Named(..) => None,
            })
            .nth(at)
            .or_else(|| {
                args.iter().find_map(|a| match a {
                    Arg::Named(k, x) if k == key => Some(x),
                    _ => None,
                })
            })
    };
    Some((written(0, "k")?, written(1, "seed")?))
}

/// The one keyed draw `noise` gives its line `key` under `seed`, the seed read as noise
/// reads its own; `None` past any whole step, or for a seed no finite number names.
pub(crate) fn keyed_draw(key: f64, seed: f64) -> Option<f64> {
    seed.is_finite().then_some(())?;
    sva_formula::draw_nearest(seed_of(seed), key)
}

/// A seed is the whole part of its number, as every keyed builtin reads it.
pub(crate) fn seed_of(seed: f64) -> u64 {
    seed as u64
}

/// The noise's draw at the step of `rate` nearest an instant, ties to even.
pub(crate) fn noise_at(seed: u64, at: crate::time::Q, rate: u32) -> f64 {
    match at
        .mul(crate::time::Q::int(i64::from(rate)))
        .and_then(crate::time::nearest)
    {
        Some(step) => hash::draw(seed, step),
        None => f64::NAN,
    }
}

/// FORMAT 3.3's arithmetic row: a closed form of its operands alone, so one image serves a
/// lowered call and a folded constant.
pub(super) fn numeric(name: &str) -> bool {
    Unary::from_name(name).is_some() || matches!(name, "max" | "min" | "pow")
}

/// The closed form a numeric call names, built once for the lowering and for the constant fold, so
/// an argument folded before a call and the same argument lowered agree by construction.
pub(super) fn arithmetic<N: Number>(
    name: &str,
    bodies: Vec<Body<N>>,
    named: &[(&str, N)],
    var: Var,
    origin: Origin,
) -> Option<Body<N>> {
    let bodies = match <[Body<N>; 1]>::try_from(bodies) {
        Ok([x]) => return unary(name, x, named, origin),
        Err(bodies) => bodies,
    };
    let [a, b] = <[Body<N>; 2]>::try_from(bodies).ok()?;
    let fold = |op: Fold, a: Body<N>, b: Body<N>| {
        Body::Fold(op, vec![Part::new(origin, a), Part::new(origin, b)])
    };
    match name {
        "max" => Some(fold(Fold::Max, a, b)),
        "min" => Some(fold(Fold::Min, a, b)),
        "pow" => power(a, b, var, origin),
        _ => None,
    }
}

fn unary<N: Number>(
    name: &str,
    x: Body<N>,
    named: &[(&str, N)],
    origin: Origin,
) -> Option<Body<N>> {
    let part = |f: Body<N>| Part::new(origin, f);
    if name == "sat" {
        let drive = Body::Const(Complex::real(named_or(named, "drive", N::ONE)));
        let driven = Body::Mul(vec![part(x), part(drive)]);
        return Some(Body::Apply(Unary::Sat, part(driven)));
    }
    if name == "step"
        && let Some(at) = rising_zero(&x)
    {
        return Some(Body::Crop {
            of: part(Body::Const(Complex::ONE)),
            l: Edge::at(at),
            r: Edge::PosInf,
            rise: N::ZERO,
            fall: N::ZERO,
        });
    }
    Unary::from_name(name).map(|op| Body::Apply(op, part(x)))
}

/// The one operand a call of one argument reads, moved out of the operands it was lowered in.
fn first<N: Num>(bodies: Vec<Body<N>>) -> Body<N> {
    bodies
        .into_iter()
        .next()
        .expect("arity checked before the operand is read")
}

/// `min(x, inf)` is `x` and `min(x, -inf)` is `-inf`, `max` alike, whatever `x` holds.
fn beside_infinity<N: Num>(name: &str, pieces: &mut Vec<Piece<N>>) -> Option<Piece<N>> {
    let absorbs = match name {
        "min" => f64::NEG_INFINITY,
        "max" => f64::INFINITY,
        _ => return None,
    };
    let infinite = |p: &Piece<N>| match p {
        Piece::ClosedForm(Body::Const(c)) if c.im.is_zero() && c.re.v().is_infinite() => {
            Some(c.re.v())
        }
        _ => None,
    };
    let at = pieces.iter().position(|p| infinite(p).is_some())?;
    if pieces.len() != 2 {
        return None;
    }
    let held = infinite(&pieces[at]).expect("an infinite operand");
    match held == absorbs {
        true => Some(pieces.swap_remove(at)),
        false => Some(pieces.swap_remove(1 - at)),
    }
}

/// Where `a*t + b` with `a > 0` crosses zero, so `step` of it is the crop opening there.
fn rising_zero<N: Number>(x: &Body<N>) -> Option<N> {
    let (a, b) = N::exact_affine(x)?;
    let (a, b) = (a.is_real().then_some(a.re)?, b.is_real().then_some(b.re)?);
    let at = -b / a;
    (a.v() > 0.0 && !at.v().is_nan()).then_some(at)
}

/// An integer exponent is a polynomial factor and stays one. A real exponent is
/// `exp(x*ln(b))`, which only a positive base has: a constant one folds here, a signal one
/// becomes that exponential, and anything else has no value to approximate.
fn power<N: Number>(base: Body<N>, exponent: Body<N>, var: Var, origin: Origin) -> Option<Body<N>> {
    let part = |f: Body<N>| Part::new(origin, f);
    let n = super::constant_value(&exponent, var);
    if let Some(n) = n.filter(|n| whole(*n)) {
        return Some(Body::Pow(part(base), n.v() as i32));
    }
    let b = super::constant_value(&base, var).filter(|b| b.v() > 0.0)?;
    if let Some(n) = n {
        return Some(Body::Const(Complex::real(b.pow(n))));
    }
    let rate = Body::Const(Complex::real(b.ln()));
    let scaled = Body::Mul(vec![part(exponent), part(rate)]);
    Some(Body::Apply(Unary::Exp, part(scaled)))
}

/// Whether a real exponent names a polynomial power an atom's u16 order holds.
fn whole<N: Num>(n: N) -> bool {
    n.is_constant() && n.v().fract() == 0.0 && n.v().abs() <= f64::from(u16::MAX)
}

fn shoulders_of<N: Num>(named: &[(&str, N)]) -> (N, N) {
    (
        named_or(named, "rise", N::ZERO),
        named_or(named, "fall", N::ZERO),
    )
}

/// A named argument's number, or what the builtin takes when the call left it out.
pub(super) fn named_or<N: Num>(named: &[(&str, N)], key: &str, fallback: N) -> N {
    named
        .iter()
        .find(|(k, _)| *k == key)
        .map_or(fallback, |(_, v)| *v)
}

/// Every positional argument as a number, in written order. A positional this cannot fold
/// is not skipped: skipping one would move every argument after it into the wrong field.
fn positional_values(
    low: &Lowering,
    args: &[Arg],
    cx: Cx,
    chosen: &mut Vec<Chosen>,
) -> Option<Vec<f64>> {
    args.iter()
        .filter_map(|a| match a {
            Arg::Pos(x) => Some(low.chosen_value(x, cx, chosen)),
            Arg::Named(..) => None,
        })
        .collect()
}

fn written_named(named: &[(String, f64)]) -> Vec<Argument> {
    named
        .iter()
        .map(|(name, value)| Argument {
            name: name.clone(),
            value: *value,
            written: true,
        })
        .collect()
}

/// A modal bank's positionals under the names its signature gives them, then its named ones.
fn positional_named(name: &str, numbers: &[f64], named: &[(String, f64)]) -> Vec<Argument> {
    let params = crate::overload::signature(name).map_or(&[][..], |s| s.params);
    let positional = params.iter().zip(numbers).map(|(p, value)| Argument {
        name: p.name.to_string(),
        value: *value,
        written: true,
    });
    positional.chain(written_named(named)).collect()
}

/// Whether a written form holds numbers under hard windows and arithmetic over them, each ref
/// read as the form it names.
struct Steps<'a> {
    typing: &'a crate::typing::Typing,
    steps: PerNode<bool>,
    constant: PerNode<bool>,
}

impl Steps<'_> {
    fn steps(&self, f: &Body) -> bool {
        match f {
            Body::Node(id) => self
                .steps
                .of(*id, || self.steps(written_form(self.typing, *id))),
            Body::Const(_) => true,
            Body::Crop { of, rise, fall, .. } => {
                *rise == 0.0 && *fall == 0.0 && self.steps(&of.body)
            }
            Body::Add(parts) | Body::Mul(parts) | Body::Fold(Fold::Max | Fold::Min, parts) => {
                parts.iter().all(|p| self.steps(&p.body))
            }
            Body::Div(a, b) => self.steps(&a.body) && self.constant(&b.body),
            _ => false,
        }
    }

    fn constant(&self, f: &Body) -> bool {
        let node = |id: NodeId| {
            self.constant
                .of(id, || self.constant(written_form(self.typing, id)))
        };
        super::constant::constant_with(f, &node)
    }
}
