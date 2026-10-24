// Concern: proves an absolute ref reads the composition that holds it | Non-concern: what the render then says | IO: (cwd, target) -> a directory, a target

use crate::helpers::{put, scratch};
use sva_cli::located;

#[test]
fn an_absolute_ref_reads_the_composition_that_holds_it() {
    let song = scratch("absolute");
    put(&song, "variables/bpm", "120\n");
    put(&song, "voice/note", "sin(2*pi*220*t)\n");
    let elsewhere = scratch("elsewhere");
    let written = format!("@{}([0, 1b], f0=C4)", song.join("voice/note").display());
    let (root, target) = located(&elsewhere, &written).expect("a target");
    assert_eq!(root, song);
    assert_eq!(target, "@voice/note([0, 1b], f0=C4)");
}
