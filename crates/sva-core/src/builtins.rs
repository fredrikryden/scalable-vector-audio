// Concern: assembles the language's callable/syntactic vocabulary off the engine's own tables, as the `data` object | Non-concern: what a builtin renders | IO: none -> Builtins, JSON

use sva_ast::BUILTINS;
use sva_engine::overload::{notation, signature};
use sva_engine::{
    Cast, Codomain, Held, Meaning, REGISTRY, Ty, Var, meaning, named_may_move, recognized_named,
    shape_name,
};
use sva_formula::filter::{ALL_SHAPES, Shape};
use sva_formula::{FAMILIES, FOURIER_DUAL_RULES_VERSION};

use crate::json::{NONE, escape, list, pair_list, strings};

pub struct Callable {
    pub name: &'static str,
    pub required: usize,
    /// `None` where no count is too many.
    pub max_positional: Option<usize>,
    pub named: &'static [&'static str],
    /// A subset of `named` with no default — a caller who omits one gets a refusal, not a
    /// fallback. Every builtin leaves this empty today.
    pub required_named: &'static [&'static str],
    pub takes_gain: Option<bool>,
    /// Each of `named`, in its order, beside what it means and whether it may move with `t`.
    pub arguments: Vec<(&'static str, Meaning, bool)>,
    /// The signature's positional names, each beside its meaning where the model states one.
    pub positional: Vec<(&'static str, Option<Meaning>)>,
}

fn positional(builtin: &str) -> Vec<(&'static str, Option<Meaning>)> {
    signature(builtin)
        .map_or(&[][..], |s| s.params)
        .iter()
        .map(|p| (p.name, meaning(builtin, p.name)))
        .collect()
}

fn meanings(builtin: &str, named: &'static [&'static str]) -> Vec<(&'static str, Meaning, bool)> {
    named
        .iter()
        .map(|key| {
            let held = meaning(builtin, key)
                .unwrap_or_else(|| unreachable!("`{builtin}` takes `{key}` with no meaning"));
            (*key, held, named_may_move(builtin, key))
        })
        .collect()
}

/// `(spelling, meaning)`, in the lexer's own longest-match-first order.
pub const UNIT_SUFFIXES: [(&str, &str); 11] = [
    ("khz", "kilohertz (x1000)"),
    ("ms", "milliseconds (seconds x0.001)"),
    ("sp", "samples"),
    ("hz", "hertz"),
    ("db", "decibels (log)"),
    ("ct", "cents (log)"),
    ("st", "semitones (log)"),
    ("b", "bars, resolved against bpm/meter"),
    ("s", "seconds"),
    ("m", "minutes (seconds x60)"),
    ("h", "hours (seconds x3600)"),
];

pub const NOTE_GRAMMAR: &str = "a bare identifier: a letter A-G, an optional accidental spelled \
    s (sharp) or b (flat) -- never #, then a whole-number octave, e.g. A4, Cs4, Db4; A4 = 440 Hz, \
    twelve-tone equal temperament, up to G9 where MIDI's 128 notes end";

/// `(name, its call shape)`.
pub const SPECIAL_FORMS: [(&str, &str); 10] = [
    (
        "deg",
        "deg(n) -- the tonic in variables/key times the ratio of scale degree n, a whole \
         number folded from literals, counted from 1 through the scale variables/scale \
         names, a built-in as `minor` or a scale file by ref; deg(7)*1st is a raised seventh, deg(5)/2 the fifth an \
         octave down, and each follows every edit of the key or its scale",
    ),
    (
        "sum",
        "sum(index, lo, hi, expr) -- index is a name the series binds, not a value; a finite \
         sum is its terms, whatever each reads, and one whose index a node's binding reads is \
         written out one node per index between whole-number bounds; hi may be inf, which \
         makes it a series of closed forms",
    ),
    (
        "crop",
        "crop(x, start, end) -- windows x to [start, end) seconds, zero outside",
    ),
    (
        "join",
        "join(a, b, ...) -- builds one wide value from 2..=8 mono args",
    ),
    (
        "pwl",
        "pwl(x, t0, v0, t1, v1, ...) -- the straight lines through the knots (tk, vk), read \
         at the time x: zero before t0 and from the last knot on, two knots at one time a jump \
         read from its right. Each time is a constant with a unit, never falling, and each \
         value a constant, so it is linear in its values; any number of knots, found by \
         bisection or by stepping forward sample to sample, so its length costs nothing per \
         sample, and a knot table counts against no body's size bound. Exact at every \
         instant; its support is [t0, tn) and its bound from an instant on the largest |vk| \
         from there. It holds no atoms: `lines`, `atoms`, `spectrum`, `envelope`, `pitch` \
         and `derivative` of a form holding one refuse as cast.left_algebra, and read off \
         sample(...) they are measured",
    ),
    (
        "pwl_integral",
        "pwl_integral(x, t0, v0, ...) -- the integral of the same pwl from t0 to x, exact: \
         zero before t0, a parabola on each segment, the whole area from the last knot on. \
         Its knots in hertz make a phase in cycles: cos(2*pi*pwl_integral(t, 0s, 440hz, 1s, \
         880hz)) glides an octave in one second; its bound from an instant on is the largest \
         it reaches from there, and it holds no atoms either",
    ),
    (
        "ch",
        "ch(x, index) -- extracts one component of a wide value by a literal index",
    ),
    (
        "x[i]",
        "@x[i], self[i] -- the stored sample at integer index i, at the rate in use; @x(e) \
         reads the instant e",
    ),
    (
        "idx",
        "idx(time), idx(time, floor), idx(time, ceil) -- the sample index nearest a time, \
         ties to even, or the one below or above it; written only inside an index",
    ),
    (
        "noise",
        "noise(seed, period=, color=, lo=, hi=) -- a line every 1/period hertz from 1/period \
         up, or only those at or above lo and at or under hi Hz, either edge alone or both, each \
         at a phase its seed draws; its level is a density in hertz that the period never \
         moves: the line at f Hz has amplitude f^log2(color)/sqrt(period), so the lines hold \
         power f^(2*log2(color))/2 a hertz, a flat noise B/2 over any B Hz, an RMS of \
         sqrt(B/2), 100 under a 20 kHz ceiling; scale it to the level the piece wants",
    ),
];

/// Confirmed against the lexer's and parser's own tables, not asserted from memory: no token
/// exists for any of these, so none can appear in an expression at all.
pub const NOT_SUPPORTED: [&str; 4] = [
    "comparison operators: < > <= >= == !=",
    "boolean operators: && || !",
    "conditionals: if/else, a ternary",
    "exponentiation `^`: write pow(base, exponent)",
];

/// One written cast, and every representation it takes to which.
pub struct Crossing {
    pub name: &'static str,
    pub rows: Vec<(&'static str, &'static str)>,
}

pub struct Builtins {
    pub callables: Vec<Callable>,
    pub casts: Vec<Crossing>,
    pub fourier_dual_rules_version: u64,
    pub families: &'static [(&'static str, &'static str)],
    pub refusals: &'static [(&'static str, &'static str)],
    pub unit_suffixes: &'static [(&'static str, &'static str)],
    pub note_names: &'static str,
    pub scales: &'static [(&'static str, &'static [f64], &'static str)],
    pub reserved: &'static [(&'static str, &'static str)],
    pub special_forms: &'static [(&'static str, &'static str)],
    pub not_supported: &'static [&'static str],
}

fn plain_callable(name: &'static str) -> Callable {
    let sig = signature(name)
        .unwrap_or_else(|| unreachable!("{name} is in BUILTINS but SIGNATURES does not cover it"));
    Callable {
        name,
        required: sig.required(),
        max_positional: sig.most(),
        named: sig.named(),
        required_named: &[],
        takes_gain: None,
        arguments: meanings(name, sig.named()),
        positional: positional(name),
    }
}

/// `q` is refused outright on `Shape::OnePole` rather than merely unused, so its own positional
/// ceiling sits below every other shape's.
fn filter_callable(shape: Shape) -> Callable {
    let gain = shape.takes_gain();
    let max_positional = if shape == Shape::OnePole {
        2
    } else if gain {
        4
    } else {
        3
    };
    let named = recognized_named(shape_name(shape)).unwrap_or(&[]);
    Callable {
        name: shape_name(shape),
        required: 2,
        max_positional: Some(max_positional),
        named,
        required_named: &[],
        takes_gain: Some(gain),
        arguments: meanings(shape_name(shape), named),
        positional: positional(shape_name(shape)),
    }
}

/// Every state a value's form can be in, so a crossing table shows which duals a cast needs.
const HELD: [(Held, bool); 6] = [
    (Held::Form(Var::T), false),
    (Held::Form(Var::T), true),
    (Held::Form(Var::F), false),
    (Held::Form(Var::F), true),
    (Held::Sampled, false),
    (Held::Frames, false),
];

/// Read off `Cast::resolve` rather than restated: a row exists where the cast answers.
fn crossings() -> Vec<Crossing> {
    Cast::NAMES
        .into_iter()
        .map(|name| {
            let cast = match Cast::from_name(name) {
                Some(Cast::Stft { .. }) => Cast::Stft {
                    window: 1024,
                    hop: 256,
                },
                Some(cast) => cast,
                None => unreachable!("{name} is one of Cast::NAMES"),
            };
            Crossing {
                name,
                rows: HELD
                    .into_iter()
                    .filter_map(|(held, dual)| {
                        let ty = Ty {
                            dual,
                            ..Ty::discrete(held, Codomain::Real)
                        };
                        cast.resolve(&[ty])
                            .ok()
                            .map(|out| (notation(ty), notation(out)))
                    })
                    .collect(),
            }
        })
        .collect()
}

pub fn builtins() -> Builtins {
    let mut callables: Vec<Callable> = BUILTINS.into_iter().map(plain_callable).collect();
    callables.extend(ALL_SHAPES.into_iter().map(filter_callable));
    Builtins {
        callables,
        casts: crossings(),
        fourier_dual_rules_version: FOURIER_DUAL_RULES_VERSION,
        families: &FAMILIES,
        refusals: &REGISTRY,
        unit_suffixes: &UNIT_SUFFIXES,
        note_names: NOTE_GRAMMAR,
        scales: &sva_ast::BUILTIN_SCALES,
        reserved: &sva_ast::RESERVED,
        special_forms: &SPECIAL_FORMS,
        not_supported: &NOT_SUPPORTED,
    }
}

pub fn builtins_data(b: &Builtins) -> String {
    let callables = list(&b.callables, |c| {
        format!(
            "\n    {{ \"name\": \"{}\", \"required\": {}, \"max_positional\": {}, \"named\": {}, \
             \"required_named\": {}, \"takes_gain\": {}, \"arguments\": {}, \"positional\": {} }}",
            escape(c.name),
            c.required,
            c.max_positional
                .map_or_else(|| NONE.to_string(), |n| n.to_string()),
            strings(c.named),
            strings(c.required_named),
            c.takes_gain
                .map_or_else(|| NONE.to_string(), |g| g.to_string()),
            list(&c.arguments, argument_json),
            list(&c.positional, positional_json)
        )
    });
    let casts = list(&b.casts, |c| {
        format!(
            "\n    {{ \"name\": \"{}\", \"crossings\": {} }}",
            escape(c.name),
            list(&c.rows, |(from, to)| format!(
                "{{ \"from\": \"{from}\", \"to\": \"{to}\" }}"
            ))
        )
    });
    format!(
        "{{\n  \"callables\": {callables},\n  \"casts\": {casts},\n  \
         \"rule_table\": {{ \"version\": {}, \"families\": {} }},\n  \
         \"refusals\": {},\n  \"unit_suffixes\": {},\n  \"note_names\": \"{}\",\n  \
         \"scales\": {},\n  \"reserved\": {},\n  \"special_forms\": {},\n  \
         \"not_supported\": {}\n}}",
        b.fourier_dual_rules_version,
        pair_list(b.families, "name", "duals"),
        pair_list(b.refusals, "code", "when"),
        pair_list(b.unit_suffixes, "suffix", "meaning"),
        escape(b.note_names),
        list(b.scales, scale_json),
        pair_list(b.reserved, "name", "note"),
        pair_list(b.special_forms, "name", "shape"),
        strings(b.not_supported)
    )
}

fn scale_json((name, cents, tuning): &(&str, &[f64], &str)) -> String {
    let cents: Vec<String> = cents.iter().map(f64::to_string).collect();
    format!(
        "\n    {{ \"name\": \"{}\", \"cents\": [{}], \"tuning\": \"{}\" }}",
        escape(name),
        cents.join(", "),
        escape(tuning)
    )
}

fn positional_json((name, m): &(&str, Option<Meaning>)) -> String {
    let text =
        |t: Option<&str>| t.map_or_else(|| NONE.to_string(), |t| format!("\"{}\"", escape(t)));
    format!(
        "{{ \"name\": \"{}\", \"meaning\": {}, \"unit\": {}, \"part\": {} }}",
        escape(name),
        text(m.map(|m| m.text)),
        text(m.map(|m| m.unit)),
        text(m.and_then(|m| m.part))
    )
}

fn argument_json((name, m, moves): &(&str, Meaning, bool)) -> String {
    format!(
        "{{ \"name\": \"{}\", \"meaning\": \"{}\", \"unit\": \"{}\", \"part\": {}, \
         \"moves\": {moves} }}",
        escape(name),
        escape(m.text),
        escape(m.unit),
        m.part
            .map_or_else(|| NONE.to_string(), |p| format!("\"{}\"", escape(p)))
    )
}

#[cfg(test)]
mod tests {

    #[test]
    fn every_operator_this_dump_says_does_not_exist_actually_refuses_to_parse() {
        for bad in [
            "1 < 2",
            "1 > 2",
            "1 <= 2",
            "1 >= 2",
            "1 == 2",
            "1 != 2",
            "1 && 1",
            "1 || 1",
            "!1",
            "1 ? 2 : 3",
            "2 ^ 3",
        ] {
            assert!(
                sva_ast::parse_expr(bad).is_err(),
                "`{bad}` parsed, so this dump's not_supported claim is stale"
            );
        }
    }

    #[test]
    fn each_reserved_name_still_behaves_the_way_this_dump_says_it_does() {
        for (name, _) in sva_ast::RESERVED {
            assert!(
                sva_ast::is_reserved(name),
                "`{name}` is listed as reserved but a parameter may still take it"
            );
            if name == "self" {
                continue;
            }
            assert!(
                sva_ast::parse_expr(name).is_ok(),
                "`{name}` should parse as the special name this dump claims"
            );
        }
        assert!(
            !sva_ast::is_reserved("cutoff"),
            "an ordinary parameter name must not be reserved, or the claim says nothing"
        );
        assert!(
            sva_ast::parse_expr("self").is_err(),
            "`self` alone must still require a call form"
        );
        assert!(
            sva_ast::parse_expr("self(t - 1sp)").is_ok(),
            "`self(...)` must still parse as a bounded self-reference"
        );
    }
}
