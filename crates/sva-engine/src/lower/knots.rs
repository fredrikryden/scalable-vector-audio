// Concern: the knot table a `pwl` or `pwl_integral` call lowers to | Non-concern: evaluating one (sva-samples), any other call (calls.rs) | IO: (args) -> a Piece

use std::sync::Arc;

use sva_ast::{Arg, ByteSpan, Expr, Literal, PWL_INTEGRAL};
use sva_formula::{Body, Knots, Unknotted, Var};

use crate::error::EngineError;
use crate::instantiate::Cx;
use crate::lower::{Lowering, Piece};

const HELP: &str = "write the time a table is read at, then each knot as a time and a value, \
                    times never falling: pwl(t, 0s, 0, 10ms, 1, 0.5s, 0)";

impl Lowering<'_> {
    pub(super) fn knots(
        &mut self,
        name: &str,
        args: &[Arg],
        span: ByteSpan,
        cx: Cx,
        var: Var,
    ) -> Result<Piece, EngineError> {
        let refuse =
            |low: &Self, what: String| low.refused_at("engine.bad_knots", what, HELP, Some(span));
        let mut written = args.iter().filter_map(|a| match a {
            Arg::Pos(x) => Some(x),
            Arg::Named(..) => None,
        });
        let at = written.next().ok_or_else(|| {
            refuse(
                self,
                format!("`{name}` reads a time first, and none is written"),
            )
        })?;
        let mut numbers = Vec::with_capacity(args.len());
        for (k, x) in written.enumerate() {
            let number = match x {
                Expr::Lit(Literal::Num(v)) => Some(*v),
                other => crate::loops::amount(self.inst, other, cx),
            };
            numbers.push(number.ok_or_else(|| {
                refuse(
                    self,
                    format!(
                        "`{name}` reads knot {} as a number, and `{}` moves",
                        k / 2 + 1,
                        sva_ast::render_expr(x)
                    ),
                )
            })?);
        }
        if !numbers.len().is_multiple_of(2) {
            return Err(refuse(
                self,
                format!(
                    "`{name}` reads knots as time, value pairs, and {} numbers follow its time",
                    numbers.len()
                ),
            ));
        }
        let (times, values) = numbers.chunks(2).map(|p| (p[0], p[1])).unzip();
        let table = Knots::new(times, values, name == PWL_INTEGRAL).map_err(|why| {
            refuse(
                self,
                match why {
                    Unknotted::Odd | Unknotted::Short => {
                        format!("`{name}` holds fewer than two knots")
                    }
                    Unknotted::NotFinite(k) => {
                        format!("`{name}`'s knot {} is no finite number", k + 1)
                    }
                    Unknotted::Backwards(k) => format!(
                        "`{name}`'s knot {} comes before knot {}, and times never fall",
                        k + 1,
                        k
                    ),
                    Unknotted::NoSpan => format!("`{name}`'s knots all sit at one time"),
                },
            )
        })?;
        let at = match self.walk(at, cx, var)? {
            Piece::ClosedForm(body) => body,
            Piece::Value(_) => {
                return Err(refuse(
                    self,
                    format!(
                        "`{name}` reads its table at a time a closed form names, and this one is \
                     samples"
                    ),
                ));
            }
        };
        let at = self.part(at, Some(span));
        Ok(Piece::ClosedForm(Body::Knots {
            at,
            table: Arc::new(table),
        }))
    }
}
