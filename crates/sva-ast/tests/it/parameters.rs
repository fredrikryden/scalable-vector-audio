// Concern: proves which parameters one node reads that no default line of its own binds | Non-concern: what a caller binds them to (sva-engine) | IO: (node text) -> names

use sva_ast::{free_parameters, parse_file};

/// One node's text, with no node beside it.
fn alone(text: &str) -> Vec<String> {
    let parsed = parse_file(text).unwrap_or_else(|d| panic!("{text}: {}", d.message));
    free_parameters(&parsed.expr, &parsed.defaults, &[], |_| false)
}

#[test]
fn a_default_line_sees_only_the_lines_above_it() {
    assert_eq!(alone("base = 1\nf0 = base*2\nsin(f0*t)\n"), [] as [&str; 0]);
    assert_eq!(alone("f0 = base*2\nbase = 1\nsin(f0*t + base)\n"), ["base"]);
}

#[test]
fn the_languages_own_names_are_no_parameters() {
    let text = "gain = -6db\ncrop(lowpass(sample(sin(2*pi*A4*t) + sum(k, 1, 3, sin(2*pi*k*C4*t)/k)), \
                cutoff=800hz, q=0.7)*gain + 0.5*self[idx(t) - 1] + f*i*pi, 0s, inf)\n";
    assert_eq!(alone(text), [] as [&str; 0]);
}

#[test]
fn a_name_the_caller_binds_is_bound_and_its_default_line_is_not_read() {
    let parsed = parse_file("f0 = base*2\namp*sin(f0*t)\n").expect("parses");
    let free = |given: &[&str]| free_parameters(&parsed.expr, &parsed.defaults, given, |_| false);
    assert_eq!(free(&[]), ["amp", "base"]);
    assert_eq!(free(&["amp", "f0"]), [] as [&str; 0]);
    assert_eq!(free(&["base"]), ["amp"]);
}
