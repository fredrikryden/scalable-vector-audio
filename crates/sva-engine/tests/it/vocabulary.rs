// Concern: proves constant powers fold exactly and the vocabulary names what the engine lowers | Non-concern: the numbers each produces (sva-formula) | IO: (a composition) -> Ty

use crate::fixtures::graph_of;
use sva_ast::BUILTINS;
use sva_engine::{Held, Value, Var, types};

fn ty_of(name: &str, body: &str) -> sva_engine::Ty {
    let g = graph_of(name, &[("node", body)]);
    let typing = types(&g, "node").unwrap_or_else(|e| panic!("{name}: {e}"));
    typing.ty(typing.id("node").expect("the root"))
}

/// The one number a constant body normalizes to.
fn folded(name: &str, body: &str) -> f64 {
    let g = graph_of(name, &[("node", body)]);
    let typing = types(&g, "node").unwrap_or_else(|e| panic!("{name}: {e}"));
    let Value::ClosedForm(form) = typing.value(typing.id("node").expect("the root")) else {
        panic!("{name}: a law");
    };
    let sum = sva_formula::normalize_closed_form(form).unwrap_or_else(|_| panic!("{name}: a sum"));
    let [lane] = sum.lanes.as_slice() else {
        panic!("{name}: one lane");
    };
    match lane.atoms.as_slice() {
        [atom] if atom.is_bare() => atom.c.re,
        other => panic!("{name}: expected one bare atom, got {other:?}"),
    }
}

/// A real exponent over a positive constant base is that number, never a truncated power.
#[test]
fn a_non_integer_constant_power_folds_exactly() {
    assert_eq!(
        folded("root-of-two", "pow(2, 1.5)\n"),
        libm::pow(2f64, 1.5),
        "pow(2, 1.5) is not 2"
    );
    assert_eq!(
        ty_of("real-exponent-of-a-signal", "pow(2, t)\n").held,
        Held::Form(Var::T),
        "a base raised to a moving exponent is exp(t*ln 2), which grows"
    );
}

/// `st` and a written power of two are the same transposition.
#[test]
fn pow_two_to_n_over_twelve_transposes_exactly() {
    assert_eq!(
        folded("written-power", "440*pow(2, 7/12)\n"),
        folded("semitones", "440*7st\n"),
        "a fifth is a fifth however it is written"
    );
}

#[test]
fn the_vocabulary_names_exactly_what_the_engine_lowers() {
    for name in sva_ast::FINITE_DIFFERENCE {
        assert!(
            BUILTINS.contains(&name),
            "`{name}` lowers to a solver but is not callable"
        );
    }
    let mut held: Vec<&str> = BUILTINS.to_vec();
    held.sort_unstable();
    let mut once = held.clone();
    once.dedup();
    assert_eq!(held.len(), once.len(), "one name, one entry: {held:?}");
}
