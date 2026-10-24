// Concern: proves written text and printed text name one expression, units and precedence included | Non-concern: what an expression denotes (sva-formula) | IO: (text) -> a Diag or the text printed back

use sva_ast::{DiagCode, Expr, Literal, parse_expr, render_expr};

fn round_trip(src: &str) -> String {
    render_expr(&parse_expr(src).expect("text that parses"))
}

#[test]
fn printed_text_reparses_to_the_same_expression() {
    for src in [
        "sin(2*pi*50*t)*exp(0 - t*30)",
        "1 + 2*3",
        "(1 + 2)*3",
        "@drums/kick(t - 0.02s)*0.9",
        "@fx/comb(t, x=@kick, delay=0.0297, g=0.805)",
        "self(t - 1sp) + rand(0, seed=3)",
        "t % 1 - 4/(2 - t)",
        "sum(k, 1, inf, sin(2*pi*k*220*t)/k)",
        "exp(i*2*pi*440*t)",
        "delta(f)/2 + pv(2*pi*i*f)",
        "delta(t, k=1)",
    ] {
        let printed = round_trip(src);
        assert_eq!(
            parse_expr(&printed).unwrap(),
            parse_expr(src).unwrap(),
            "{src} printed as {printed}"
        );
    }
}

#[test]
fn a_bare_number_in_a_joined_time_slot_refuses_like_a_plain_one() {
    let plain = parse_expr("@src(t - 0.01)").expect_err("a bare time");
    assert_eq!(plain.code, DiagCode::BareTime);
    let joined = parse_expr("@src(join(t - 0.01, t - 0.02s))").expect_err("a bare time");
    assert_eq!(joined.code, DiagCode::BareTime);
    assert_eq!(
        joined.message, plain.message,
        "one slot, one refusal, however the time is written"
    );
    parse_expr("@src(join(t - 0.01s, t - 0.02s))").expect("every component carries its unit");
}

/// A negative time has no spelling but the minus, so the printer's `-0.5s` has to be the one
/// the parser reads back as that same literal.
#[test]
fn a_negative_shift_round_trips() {
    for src in [
        "@kick(-0.5s)",
        "@kick(-2b)",
        "@kick(-64sp)",
        "crop(@a, -1s, 1s)",
    ] {
        let printed = round_trip(src);
        assert_eq!(printed, src, "{src} printed as {printed}");
        assert_eq!(
            parse_expr(&printed).unwrap(),
            parse_expr(src).unwrap(),
            "{src} printed as {printed}"
        );
    }
}

/// 2^(11/12) sits by a rounding tie, where `powf` and `exp2` round apart.
#[test]
fn a_pitch_unit_reads_the_same_bits_in_every_build() {
    let eleven_twelfths = libm::exp2(std::hint::black_box(11.0f64 / 12.0));
    for src in ["1100ct", "11st"] {
        assert_eq!(
            parse_expr(src).unwrap(),
            Expr::Lit(Literal::Num(eleven_twelfths)),
            "{src}"
        );
    }
}

/// Knot times, a crop's shoulders and a product in a time slot keep their unit.
#[test]
fn a_time_slot_keeps_its_unit_through_knots_shoulders_and_products() {
    for src in [
        "pwl(k*1s, 1s, 0.5, 2s, 1)",
        "pwl_integral(t, 0s, 0, 1s, 2)",
        "crop(@a, 0s, len, fall=min(rel, len))",
        "crop(@a, 0s, 2*1s, rise=0.01s)",
        "pwl(2*1s*k, 1s, 0.5, 2s, 1)",
        "pwl(k*(2*1s), 1s, 0.5, 2s, 1)",
        "pwl(1s % 2s, 1s, 0.5, 2s, 1)",
        "pwl(k*1s/2, 1s, 0.5, 2s, 1)",
    ] {
        assert_eq!(round_trip(src), src);
    }
}
