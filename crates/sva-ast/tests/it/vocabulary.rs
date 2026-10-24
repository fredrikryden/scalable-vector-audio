// Concern: proves which names the language answers itself, notes with their pitch and builtins included | Non-concern: what a builtin evaluates to (sva-engine) | IO: (name) -> bool, MIDI number or Hz

use sva_ast::note_midi;

/// The octave is parsed off a user's string, so the name has to run out where MIDI does
/// instead of wrapping its arithmetic into some other note.
#[test]
fn an_octave_beyond_midi_is_no_note() {
    assert_eq!(note_midi("G9"), Some(127), "MIDI 127 is still a note");
    assert_eq!(note_midi("As9"), None, "MIDI 130 is past the range");
    assert_eq!(note_midi("C10"), None);
    assert_eq!(
        note_midi("C178956971"),
        None,
        "an octave that would overflow"
    );
}

#[test]
fn a_note_sounds_at_its_equal_tempered_frequency() {
    assert_eq!(sva_ast::note_hz("A4"), Some(440.0));
    assert_eq!(
        sva_ast::note_hz("A5"),
        Some(880.0),
        "an octave is a doubling"
    );
    assert!((sva_ast::note_hz("C4").unwrap() - 261.6255653005986).abs() < 1e-9);
    assert!((sva_ast::note_hz("As4").unwrap() / 440.0 - libm::pow(2f64, 1.0 / 12.0)).abs() < 1e-12);
}
