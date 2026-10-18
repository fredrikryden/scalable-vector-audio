// Concern: proves a song read by several paths or at a delay keeps one set of lanes in song time, its notes sounding d later | Non-concern: one lane's refusals | IO: (a composition) -> samples, events

use std::path::{Path, PathBuf};

use crate::helpers::{put, scratch};
use sva_core::probe;

/// At 120 bpm: a key lift, an energy ramp, a sound lane and a tempo ramp; a note a bar.
fn song(name: &str) -> PathBuf {
    let dir = scratch(&format!("reads-{name}"));
    for (path, text) in [
        ("variables/bpm", "120\n"),
        ("variables/meter", "4/4\n"),
        ("variables/key", "A3\n"),
        ("variables/scale", "major\n"),
        ("variables/energy", "0.5\n"),
        (
            "voice",
            "f0 = 220hz\nlen = 0.5s\ncut = 0.5\n\
             crop(@variables/energy*sin(2*pi*f0*t + cut*sin(2*pi*3*t)), 0s, len)\n",
        ),
        (
            "song.jsonl",
            "{\"at\": \"0b\", \"node\": \"@voice\", \"lane\": \"cut\", \"to\": 0}\n\
             {\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 1, \"len\": \"0.25b\"}\n\
             {\"at\": \"1b\", \"lane\": \"key\", \"to\": \"key*2st\"}\n\
             {\"at\": \"1b\", \"node\": \"@voice\", \"deg\": 1, \"len\": \"0.25b\"}\n\
             {\"at\": \"2b\", \"lane\": \"energy\", \"to\": 1, \"len\": \"1b\", \"shape\": \"lin\"}\n\
             {\"at\": \"2b\", \"node\": \"@voice\", \"deg\": 3, \"len\": \"0.5b\"}\n\
             {\"at\": \"3b\", \"lane\": \"bpm\", \"to\": 90, \"len\": \"1b\", \"shape\": \"lin\"}\n\
             {\"at\": \"3b\", \"node\": \"@voice\", \"lane\": \"cut\", \"to\": 2}\n\
             {\"at\": \"3b\", \"node\": \"@voice\", \"deg\": 5, \"len\": \"0.25b\"}\n\
             {\"at\": \"4b\", \"node\": \"@voice\", \"deg\": 1, \"len\": \"0.25b\"}\n",
        ),
    ] {
        put(&dir, path, text);
    }
    dir
}

fn samples(dir: &Path, target: &str) -> Vec<f64> {
    let rendered =
        probe(dir, &format!("{target}([0, 12s])")).unwrap_or_else(|e| panic!("{target}: {e}"));
    let out = rendered
        .render
        .output(rendered.render.root)
        .unwrap_or_else(|e| panic!("{target}: {e}"));
    let plane = out.plane(0).to_vec();
    assert_eq!(plane.len(), 12 * 44_100, "{target} at 44.1 kHz");
    assert!(plane.iter().any(|x| x.abs() > 0.1), "{target} sounds");
    plane
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("samples");
    assert!(
        worst.1 < 1e-9,
        "{what}: sample {} is {} off",
        worst.0,
        worst.1
    );
}

/// A dry read plus a send reach each move by two paths, and each is one move.
#[test]
fn a_song_read_by_two_paths_moves_each_lane_once() {
    let dir = song("twice");
    put(&dir, "send", "0.5*sample(@song(t))\n");
    put(&dir, "master", "@song(t) + @send(t)\n");
    let alone = samples(&dir, "@song");
    let both = samples(&dir, "@master");
    let want: Vec<f64> = alone.iter().map(|x| 1.5 * x).collect();
    assert_close(&both, &want, "dry plus a send");
}

/// A read at `t - d` is the song as it was d earlier: an echo keeps its note's key and lanes.
#[test]
fn a_delayed_read_hears_the_song_as_it_was_d_earlier() {
    let dir = song("delayed");
    put(&dir, "master", "@song(t) + 0.5*@song(t - 1b)\n");
    let alone = samples(&dir, "@master");
    let dry = samples(&dir, "@song");
    let d = 88_200;
    let want: Vec<f64> = (0..dry.len())
        .map(|i| dry[i] + 0.5 * i.checked_sub(d).map_or(0.0, |j| dry[j]))
        .collect();
    assert_close(
        &alone,
        &want,
        "dry plus an echo a home bar later, across the key lift",
    );
}

/// Two distinct moves that collide still refuse, once.
#[test]
fn distinct_moves_that_collide_still_refuse_under_two_reads() {
    let dir = song("collide");
    put(
        &dir,
        "song.jsonl",
        "{\"at\": \"0b\", \"node\": \"@voice\", \"deg\": 1, \"len\": \"4b\"}\n\
         {\"at\": \"1b\", \"lane\": \"energy\", \"to\": 1}\n\
         {\"at\": \"1b\", \"lane\": \"energy\", \"to\": 0.2}\n",
    );
    put(&dir, "master", "@song(t) + @song(t - 0.5b)\n");
    let Err(refused) = probe(&dir, "@master([0, 1s])") else {
        panic!("a tie on one lane renders");
    };
    let found = refused.diagnostics();
    assert_eq!(found.len(), 1, "{}", refused.message());
    assert_eq!(found[0].code, "bad-lane", "{}", found[0].message);
    assert!(
        found[0].message.contains("this line and song.jsonl:2"),
        "{}",
        found[0].message
    );
}

/// Each placed note's song time and how long after it it sounds, in seconds, in song order.
fn heard(dir: &Path, target: &str) -> Vec<(f64, f64)> {
    let (_, song) = sva_core::song_of(&sva_ast::Dir::at(dir.to_path_buf()), target)
        .unwrap_or_else(|e| panic!("{target}: {e}"));
    let curve = song.curve();
    let mut out: Vec<(f64, f64)> = song
        .instances()
        .iter()
        .filter(|i| i.line.is_some())
        .map(|i| {
            let (at, sounds) = (
                i.at.expect("a song time"),
                i.sounds(curve).expect("a sound"),
            );
            (at.seconds(curve), sounds.seconds(curve) - at.seconds(curve))
        })
        .collect();
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    out
}

/// A read of a score at `t - d` sounds d later, nested reads adding up, and keeps its song time.
#[test]
fn a_delayed_read_sounds_d_later_at_its_song_time() {
    let dir = song("sounds");
    put(&dir, "echo", "@song(t - 1s)\n");
    put(
        &dir,
        "master",
        "@song(t) + 0.5*@song(t - 1b) + 0.25*@echo(t - 0.5s)\n",
    );
    let dry: Vec<f64> = heard(&dir, "@song").iter().map(|(at, _)| *at).collect();
    assert_eq!(dry.len(), 5, "a note a bar");
    let want: Vec<(f64, f64)> = dry
        .iter()
        .flat_map(|at| [(*at, 0.0), (*at, 1.5), (*at, 2.0)])
        .collect();
    assert_eq!(heard(&dir, "@master"), want);
}

/// A note no view delays sounds at its song time, exactly, still in bars.
#[test]
fn an_undelayed_note_sounds_at_its_song_time_exactly() {
    let dir = song("undelayed");
    put(&dir, "master", "@song(t)\n");
    let (_, song) = sva_core::song_of(&sva_ast::Dir::at(dir), "@master").expect("a song");
    let curve = song.curve();
    let notes: Vec<_> = song
        .instances()
        .iter()
        .filter(|i| i.line.is_some())
        .collect();
    assert_eq!(notes.len(), 5);
    for note in notes {
        assert!(matches!(note.at, Some(sva_ast::song::Pos::Bars(_))));
        assert_eq!(note.sounds(curve), note.at);
    }
}

/// `outline` prints a delayed note's song time and, beside it, when it sounds.
#[test]
fn outline_prints_when_a_delayed_note_sounds() {
    let dir = song("outline");
    put(&dir, "late", "0.5*@song(t - 1b)\n");
    put(&dir, "top.jsonl", "{\"at\": \"0b\", \"node\": \"@late\"}\n");
    let printed =
        sva_core::song_data(&sva_ast::Dir::at(dir), "@top").unwrap_or_else(|e| panic!("{e}"));
    assert!(
        printed.contains(
            "\"at\": { \"bars\": 0, \"seconds\": 0 }, \"sounds\": { \"bars\": 1, \"seconds\": 2 }"
        ),
        "{printed}"
    );
}
