// Concern: states what a target sampling its one ref read, `sample(@x([a, b]))`, answers | Non-concern: `sample` inside a node | IO: (target) -> a reading, its source, memory's lookups

use sva_core::{Asked, CliError, Job, Output, asked, call, execute};
use sva_engine::{Source, Tier};

const RAMP: &str = "crop(tanh(4*(2*t - 1)), 0s, 1s)\n";
const NOTE: &str = "f0 = 220\ncrop(tanh(4*sin(2*pi*f0*t)), 0s, 4s)\n";

fn composition() -> sva_ast::Composition {
    [("n", RAMP), ("note", NOTE)].into_iter().collect()
}

fn pitch() -> Asked {
    asked(&call("pitch").expect("a call")).expect("a reading")
}

fn pitched(target: &str, tier: &Tier) -> Result<sva_engine::Answer, CliError> {
    let held = composition();
    let asking = [pitch()];
    let job = Job {
        asked: &asking,
        ..Job::over(&held, target)
    };
    execute(job, tier)?.answered(&asking[0])
}

/// A reader samples the one ref it reads and measures it over that ref's interval.
#[test]
fn a_sampled_read_measures_the_pitch_a_plain_one_refuses() {
    let answer = pitched("sample(@n([0, 1s]))", &Tier::default()).expect("measured");
    assert_eq!(answer.source, Source::Measured);
    let Output::Pitch(frames) = answer.value else {
        panic!("pitch answers frames");
    };
    assert_eq!(frames.len(), 20, "one frame per 50ms across the second");
}
