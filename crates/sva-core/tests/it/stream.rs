// Concern: a stream edit takes in what it newly reads, retunes on a variable, keeps a song's lanes | Non-concern: what the stream plays | IO: (a source, changes) -> what each took in

use std::cell::RefCell;

use sva_ast::Composition;
use sva_core::{Job, Placed, Stream};
use sva_engine::Tier;

/// A future over no store: one poll finishes it.
fn now<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::noop();
    match future
        .as_mut()
        .poll(&mut std::task::Context::from_waker(waker))
    {
        std::task::Poll::Ready(out) => out,
        std::task::Poll::Pending => panic!("nothing to wait on over no store"),
    }
}

/// An edit whose held node now reads a node the stream never held takes that node in, and
/// plays from there the samples a render of the edited source writes.
#[test]
fn an_edit_takes_in_a_node_a_held_node_newly_reads() {
    let mut held = Composition::new();
    held.insert("a", "0.1*sin(2*pi*220*t)\n");
    held.insert("master", "@a\n");
    let job = Job::over(&held, "@master");
    let stream = now(sva_core::stream(&job, (64, None), &Tier::default())).expect("a stream");
    let stream = RefCell::new(stream);
    stream.borrow_mut().read(0, 64).expect("a block");
    held.insert("b", "0.1*sin(2*pi*330*t)\n");
    held.insert("master", "@a + @b\n");
    now(sva_core::edit(&stream, &held, "@master", &Tier::default())).expect("an edit");
    let at = stream.borrow().position();
    let block = stream.borrow_mut().read(at, 64).expect("a block");
    let played = block.expect("samples").plane(0).to_vec();
    let job = Job::over(&held, "@master([0, 0.01s])");
    let rendered = sva_core::execute(job, &Tier::default()).expect("a render");
    let whole = rendered
        .render
        .output(rendered.render.root)
        .expect("the root");
    let span = at as usize..at as usize + 64;
    assert_eq!(played, whole.plane(0)[span].to_vec());
}

/// Each next block of `stream` after an edit, beside the same samples of a fresh render of
/// the edited source.
fn edited(stream: &RefCell<Stream>, held: &Composition) -> (Vec<f64>, Vec<f64>) {
    now(sva_core::edit(stream, held, "@master", &Tier::default())).expect("an edit");
    let at = stream.borrow().position();
    let block = stream.borrow_mut().read(at, 64).expect("a block");
    let played = block.expect("samples").plane(0).to_vec();
    let job = Job::over(held, "@master([0, 1s])");
    let rendered = sva_core::execute(job, &Tier::default()).expect("a render");
    let whole = rendered
        .render
        .output(rendered.render.root)
        .expect("the root");
    (
        played,
        whole.plane(0)[at as usize..at as usize + 64].to_vec(),
    )
}

/// An edit to a reserved variable lowers again every held node that reads it, though no
/// node's own text changed, and a value set back is followed back.
#[test]
fn an_edit_to_a_reserved_variable_retunes_and_retimes_what_the_stream_holds() {
    let mut held = Composition::new();
    held.insert("variables/bpm", "120\n");
    held.insert("variables/meter", "4/4\n");
    held.insert("variables/key", "C4\n");
    held.insert("variables/scale", "@../scales/major\n");
    held.insert("scales/major", "200\n400\n500\n700\n900\n1100\n1200\n");
    held.insert("scales/minor", "200\n300\n500\n700\n800\n1000\n1200\n");
    held.insert("lead", "0.1*sin(2*pi*deg(3)*t)\n");
    held.insert("a-0.001b", "0.05*sin(2*pi*330*t)\n");
    held.insert("b-0.001b", "0.05*sin(2*pi*495*t)\n");
    held.insert("arranged", "concat(@a-0.001b, @b-0.001b)\n");
    held.insert("master", "crop(@lead(t - 0.001b) + @arranged, 0s, 1s)\n");
    let job = Job::over(&held, "@master");
    let stream = now(sva_core::stream(&job, (64, None), &Tier::default())).expect("a stream");
    let stream = RefCell::new(stream);
    stream.borrow_mut().read(0, 64).expect("a block");
    for (path, text) in [
        ("variables/scale", "@../scales/minor\n"),
        ("variables/scale", "@../scales/major\n"),
        ("variables/key", "A3\n"),
        ("scales/major", "200\n300\n500\n700\n900\n1100\n1200\n"),
        ("variables/bpm", "60\n"),
    ] {
        held.insert(path, text);
        let (played, rendered) = edited(&stream, &held);
        assert_eq!(played, rendered, "after {path} = {text:?}");
    }
}

/// A song moving a lane, its note reading the lane, and a master the stream plays.
fn laned() -> Composition {
    let mut held = Composition::new();
    held.insert("variables/bpm", "120\n");
    held.insert("variables/meter", "4/4\n");
    held.insert("variables/energy", "0.5\n");
    held.insert(
        "tone",
        "f0 = 220\ncrop(@variables/energy*sin(2*pi*f0*t), 0s, 0.5s)\n",
    );
    held.insert(
        "song.jsonl",
        "{\"at\": \"0b\", \"node\": \"@tone\"}\n{\"at\": \"0.125b\", \"lane\": \"energy\", \"to\": 1}\n",
    );
    held.insert(
        "moving.jsonl",
        "{\"at\": \"0b\", \"lane\": \"energy\", \"to\": 0.2}\n",
    );
    held.insert("master", "crop(@song, 0s, 1s)\n");
    held
}

/// A term at a written time reads the song's lanes where it sits; one reaching a move of its
/// own refuses, as does one with no song time that a lane reaches.
#[test]
fn a_term_reads_the_songs_lanes_and_never_moves_one() {
    let held = laned();
    let job = Job::over(&held, "@notes + @master");
    let stream = now(sva_core::stream(&job, (64, None), &Tier::default())).expect("a stream");
    let stream = RefCell::new(stream);
    let term = ("@tone(t - 0.375s)", Placed::Written);
    now(sva_core::add(&stream, &held, term, &Tier::default())).expect("a term at a written time");
    let at = 26_460;
    let block = stream.borrow_mut().read(at, 64).expect("a block");
    let played = block.expect("samples").plane(0).to_vec();
    let job = Job::over(&held, "crop(@master + @tone(t - 0.375s), 0s, 1s)");
    let rendered = sva_core::execute(job, &Tier::default()).expect("a render");
    let whole = rendered
        .render
        .output(rendered.render.root)
        .expect("the root");
    assert_eq!(
        played,
        whole.plane(0)[at as usize..at as usize + 64].to_vec(),
        "the term reads energy at 1 where it sits, as the song's own note would"
    );
    for (term, placed) in [("@moving", Placed::Written), ("@tone", Placed::Landing)] {
        let refused = now(sva_core::add(
            &stream,
            &held,
            (term, placed),
            &Tier::default(),
        ))
        .expect_err("a term that moves a lane, or that a lane reaches with no song time");
        let found = refused.diagnostics();
        assert_eq!(found[0].code, "bad-lane", "{term}: {}", found[0].message);
    }
}
