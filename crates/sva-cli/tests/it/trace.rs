// Concern: states what a trace must answer about a node's position, whatever the implementation | Non-concern: how any pass computes it (sva-engine) | IO: (fixtures/*) -> asserted Traced or CliError

use std::path::{Path, PathBuf};

use crate::helpers::{put, scratch};
use sva_cli::{Traceable, trace};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn traced(name: &str, target: &str) -> Traceable {
    trace(&fixture(name), target)
        .unwrap_or_else(|e| panic!("tracing `{target}` in {name}: {}", e.message()))
}

fn up(t: &Traceable) -> Vec<&str> {
    t.traced.up.iter().map(|u| u.node.as_str()).collect()
}

#[test]
fn down_is_one_hop_and_up_reaches_every_reader_to_an_entry_point() {
    let t = traced("reverb", "src");
    assert_eq!(t.traced.down, Vec::<String>::new(), "`src` reads nothing");
    assert_eq!(t.traced.entry, ["master"], "the only unreferenced node");
    let readers = up(&t);
    assert!(readers.contains(&"master"), "{readers:?}");
    assert!(
        readers.iter().filter(|n| n.starts_with("fx/comb(")).count() == 8,
        "every comb reads `src` through one invocation: {readers:?}"
    );
    assert!(
        readers.iter().any(|n| n.starts_with("fx/diffuse(")),
        "a reader four hops up is still a reader: {readers:?}"
    );
    assert!(
        !readers.contains(&"src"),
        "a node is not its own reader: {readers:?}"
    );
}

/// A bare file name is the file on its own terms, however many call sites bound it
/// otherwise: three verbs, one written name, one instance.
#[test]
fn three_verbs_agree_on_an_instance_name() {
    let dir = scratch("two-instances");
    let head = "; Models: a probe | Neglects: everything | IO: (t) -> amplitude | Tags: probe\n";
    put(
        &dir,
        "motif",
        &format!("{head}third = 15\nsin(2*pi*third*29*t)\n"),
    );
    put(
        &dir,
        "master",
        &format!("{head}@motif(t) + @motif(t, third=16)\n"),
    );

    let source = sva_ast::Dir::at(&dir);
    let held = sva_core::execute(
        sva_core::Job::over(&source, "@motif([0, 1s])"),
        &sva_core::Tier::default(),
    )
    .expect("`render @motif` reads the file on its own terms");
    let rendered = held
        .render
        .tys
        .paths()
        .map(|(path, _)| path)
        .find(|path| path.starts_with("motif("))
        .expect("the instance it read")
        .to_string();

    let traced = trace(&dir, "motif")
        .expect("`trace motif` reads the same file the same way")
        .traced;
    assert_eq!(traced.node, rendered, "one written name, one instance");
    assert_eq!(
        traced.node, "motif(third=15)",
        "the file's own defaults, not a call site's override"
    );
    assert_eq!(
        traced.entry,
        vec!["master".to_string()],
        "reading a file on its own terms does not make it an entry point"
    );

    let report =
        sva_cli::lint(&dir, Some("@motif([0, 1s])")).expect("`lint @motif` names the same file");
    assert_eq!(report.nodes, 1, "one node checked");
}
