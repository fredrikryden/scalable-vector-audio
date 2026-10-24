// Concern: gathers every sva-cli integration suite into one test binary | Non-concern: what any suite asserts (its own module) | IO: none

mod analyze_integration;
mod bare;
mod composition_dir;
mod envelope;
mod helpers;
mod lint;
mod lint_verdict;
mod panics;
mod pinned;
mod prox;
mod render_integration;
mod score;
mod score_rows;
mod solve;
mod song_reads;
mod store;
mod trace;
