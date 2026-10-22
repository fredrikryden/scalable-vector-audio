// Concern: the nuisance draws a priced fit's loss is the expectation over: each line's phase offset and noise seed per draw | Non-concern: reading or rendering either | IO: (line, draw) -> phase, seed

/// How many draws of the lines' phases and noise realisations the expected loss is read over.
pub const DRAWS: usize = 16;

fn mix(line: usize, draw: usize, stream: u64) -> u64 {
    let mut z = ((line as u64) << 32 | draw as u64)
        .wrapping_add(stream)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Line `line`'s phase offset in draw `draw`: `2 pi` times splitmix64 of `line << 32 | draw`
/// over `2^64`, the same every solve and target.
pub fn phase(line: usize, draw: usize) -> f64 {
    std::f64::consts::TAU * (mix(line, draw, 0) >> 11) as f64 / (1u64 << 53) as f64
}

/// Line `line`'s noise seed in draw `draw`: splitmix64 of the same key on its own stream, under
/// `2^31`.
pub fn seed(line: usize, draw: usize) -> u64 {
    mix(line, draw, 1 << 62) >> 33
}
