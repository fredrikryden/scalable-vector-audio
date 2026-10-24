// Concern: pins the bytes and store keys fixed compositions render to, and two cases' readings | Non-concern: `atoms`, why any value is right | IO: (fixtures/*) -> hashes

use std::path::Path;
use std::process::Command;

use crate::helpers::{fixture, scratch};

/// Each case: a fixture, a target over a closed interval, and the one instance `bindings` reads.
const CASES: [(&str, &str, &str); 17] = [
    ("arranged", "@master([0, 1s])", "@master"),
    ("bar-grid", "@master([0, 1s])", "@master"),
    ("basic", "@master([0, 1s])", "@master"),
    ("basic", "@loop-1b([0, 0.5s])", "@loop-1b"),
    ("cost", "@bowl([0, 0.1s])", "@bowl"),
    ("no-tempo", "@master([0, 1s])", "@master"),
    ("reverb", "@master([0, 0.5s])", "@fx/reverb"),
    ("stereo", "@master([0, 0.06s])", "@pan"),
    ("stereo", "@pingpong([0, 0.6s])", "@pingpong"),
    ("two-entries", "@master([0, 1s])", "@master"),
    ("two-entries", "@bench-2s([0, 1s])", "@bench-2s"),
    (
        "closure-1a4989c6",
        "@sva1a4989c6fb22b2969fa478e6ee47b16aee6a20df52c484d241618c6b112876bb([0, 1s])",
        "@sva4065c0245b85ba8686d4da12124b41e3d0ea09ebb3000a229f931ee6194a9558(f0=D2)",
    ),
    (
        "closure-1e362f88",
        "@sva1e362f88072bf4bb3c0edb42fda6269ffc00ee12c4f3f4e07bbde0269c103ded([0, 1s])",
        "@sva5f25539e1ff16edd2ae8228c54964a2afe65f15881c456fdcaf0a4d36ec52082(f0=A1, release=0.8, vel=1)",
    ),
    (
        "closure-1f15ca3d",
        "@sva1f15ca3db2096f385f391721c5c0fff2210e4c6b3ed996ef063529631ac3bda6([0, 1s])",
        "@svad163ed7735e112fc74e76ed73b689901236c80717584dcc13d14fb72cd3f0e4e(seed=1)",
    ),
    (
        "closure-db781e56",
        "@svadb781e56ae75d7bb4802f257ff1b7c509b9cce9dcd53f4cad13450110d3a4a8f([0, 0.5s])",
        "@sva8062dfa9b6df789c110d873e7fe5164b6ab95b0c7517ca5dcb7afce047d0f9bc(f0=261.6255653006, release=3.0833333333, vel=0.7087)",
    ),
    (
        "closure-db781e56",
        "@sva046a5fc7c03eb08d8ac01e517b4ce72a7765d820d9966e676c197a21b3b39045([0, 0.1s], tc=0.08)",
        "@sva046a5fc7c03eb08d8ac01e517b4ce72a7765d820d9966e676c197a21b3b39045",
    ),
    (
        "closure-f5a155b1",
        "@svaf5a155b1d5f640ea99aa441093bdb8c793a166a697f1f1eb8c032160feb4cd81([0, 0.5s])",
        "@sva3828c5b584260ff15968d7fd155d5d8856184bd78099c66fca8b2b39f671b8b5(f0=A2, release=2, vel=0.35)",
    ),
];

/// The cases whose readings are pinned: a mono loop and a stereo one answer most of them.
const READ_CASES: [usize; 2] = [3, 8];

/// Every reading but `atoms`, and `samples`, which the `.wav` pins.
const READINGS: [&str; 15] = [
    "lines",
    "spectrum",
    "envelope",
    "derivative",
    "ledger",
    "pitch",
    "formants",
    "stereo",
    "bands",
    "crest",
    "loudness",
    "hits",
    "alias",
    "bindings",
    "arguments",
];

/// One line per case and item, `<case> <item> <hash, or the codes it refused>`.
const PINNED: &str = r#"
0 stdout e17e3c4311440757
0 wav 665924a7a274d695
0 store:3620e0612b5d84925fa1bd15f1b7716e 64065f1fd529e141
1 stdout e17e3c4311440757
1 wav 26291e77cc78e829
1 store:90af903a105917870539e1943a4b72c7 1a0ed0f32991d363
2 stdout e17e3c4311440757
2 wav 968e2e9858e90ed9
2 store:29c7e17ebaef85e9ff784832c3108192 950494b9265a06d4
3 stdout 71f4f9a9b3cd9d30
3 wav 64318e943f03cc25
3 store:273d32d1fbc24d73b322a79899449566 81afe0dc42e2e1ef
3 lines ef35f29de2cc8837
3 spectrum d76d6d5aa41272bc
3 envelope 508733913bd67342
3 derivative 9685ab0be14bd4b4
3 ledger 1cdfc7f7172a3d74
3 pitch bc6a1749d2e897fa
3 formants 32150671bce8281a
3 stereo refused "type.width_mismatch"
3 bands 8d18c12abb247ca1
3 crest 871492938378bf0f
3 loudness 691ca4a23b72fb64
3 hits 118a6129db0a790e
3 alias 57b3de39acaa9baf
3 bindings f15293e1080c5049
3 arguments eb3ed92e88f435e1
4 stdout 5b546f69552c55f0
4 wav d8d646db74dc9cb2
4 store:7d17584ff33a1d5714e10c35ba698e01 045cf64ccca37967
5 stdout e17e3c4311440757
5 wav 177aed0c28183ef5
5 store:52690610e792cbd9ec87410effce6ac0 2522be41483fae0d
6 stdout 9108e37d0b7327c6
6 wav 72bce77b8424e0c6
6 store:0aa625a325ad009e42736fb0b55da15e 68bd8715d82f4ed3
6 store:11c136c343d5887a356999a4fc86e8f7 fe0f34b51c8c37f3
6 store:14f6414df6fcdd24e3ba1efa6b601e02 a582b63bfbbc32f9
6 store:2a812a178319a3d66188660d033eee56 120c430ae633cbf8
6 store:2b1dfb572460ba3c3e8a08891ad3feda 36ba97a808c358e6
6 store:2cc1e056c0cd5c6013c986cb46ea8680 17797924c40c6396
6 store:57a6b91a9d6ee92c9f220a471c761060 004a2bb993fdcb9c
6 store:6e22a9b54e73a673f2f7252c439f5d3c 7248e5306bab0bd7
6 store:6f9bcd032cb9677f874eba226b581950 e985045b7824543c
6 store:d18c8f806552ff35a1cabe95f566fe2e caad63ef99df71df
6 store:d4b554f1adbd1e2708edf6cef39dd919 e24590f2523b063c
6 store:dfbe3e234fa8c68c2717f78df327e90f 990f0c62cf2fbc57
6 store:e41ecc22b1d8d907ef9ba75bd804e1e7 ebd3d21ba7afaf4c
6 store:e51a27e86bcbd36a2c884364cb358a1e ef95a11a26dc8f91
7 stdout 71454a132807dddb
7 wav 3372a403bff9f10a
7 store:48cc333eda6f48499d525bb87ae7a03c 5479ab3a4006d5f9
8 stdout bb69bdf6de9adeb8
8 wav 083e3fdfa6fd2b2d
8 store:0435c2987fec5f6fa5f7bc60ed42587e 589ca7315988d01f
8 lines refused "type.samples_in_closed_form"
8 spectrum 7c295c6f1eda017e
8 envelope 318124550bf22b6d
8 derivative 2f23088a9ea8b53e
8 ledger e426234cdb721fd8
8 pitch a1bb7d37afebe1e6
8 formants 90ff837832c308ba
8 stereo 314da35cb65574d5
8 bands 02245c806e0fde9a
8 crest 7b7793adeb78eb3e
8 loudness b2f79549e9145e00
8 hits 08e304e92adcc9e1
8 alias refused "engine.alias_needs_a_closed_form"
8 bindings 5ce608c3a8b79a86
8 arguments b642f09606176b94
9 stdout e17e3c4311440757
9 wav 3a9ca1a53a5fa42a
9 store:3180ee7f5ba55875b73514cbc66c3107 ec4c5c56a52d8ef2
10 stdout 5687cc05a27c2ec3
10 wav a9866247325c001f
10 store:6655845ef1b852c2dfba6667b7a17cb7 1d0dd4db508178d9
11 stdout a05558226a6cba5a
11 wav c187b7408a55ebf7
11 store:49f4a6817fb359e1ca2669494fe63af2 008d37c5eed9f05e
11 store:d974ac29049427884f6c2e4f1edbb778 7f26570614c665f7
11 store:dc788d72ef609cbfbd531912e37ffe4d 454727a4cc31b522
12 stdout 24a0009133ff612c
12 wav 3fc9f322aaf6fc5f
12 store:a754efa011f428d52010e3099bc14cb1 00ef65f3c39f7583
12 store:b75918a010c072bb16f03119a7dcbf64 eb8f05273ec5ed57
13 stdout 189201d106182ae6
13 wav 256b6dfecab93743
13 store:995a04465e3289abd658d5254fb949f3 e9a98203e890cddb
13 store:e8cfef4afb2349bb0fe8c2e3f1cea74f 7786da5294625d8f
14 stdout b6ad884dff66b63b
14 wav 0c1d8e0ad4b6a019
14 store:194d57ecf1b72fa62c9b17ec02129d13 b093ccc1b22afc47
14 store:32547f9d11c1e39ed00836d8ff4dbff2 a851941ef22e4f93
14 store:4d307b3a5d05232779527d608b4de3ad 78ac776f298094df
14 store:52d1c7bcf24fd65556b3a64eddbd2e59 fb6abdddd01f7477
15 stdout ea658b80403f05b8
15 wav 82773cac15ff1cca
15 store:b5146c74a37c137f36a90b33af43d7d6 deebcf6f4e565737
16 stdout 6d9beba6f3f07c0c
16 wav 9007d5d55c305e67
16 store:043708e3e666563747bf4fa06e8134c9 717b52524979bc3b
16 store:1523d3727e3560f4c087cc26bcdb92c1 b9dcfd9f8afa197c
16 store:56d9a9664634a8707c8206366f7ace03 745d2d564d63e78e
16 store:b00e0698a765f996175369b4040463f1 780ddf85338b2a88
16 store:e98c8e1e8b33053c9a7f3b9e64d00f51 15c2b643f04af58b
"#;

/// FNV-1a: stable across toolchains, which `DefaultHasher` does not promise.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The envelope before its `meta`, which differs per run.
fn before_meta(printed: &str) -> &str {
    &printed[..printed.find("\"meta\"").expect("an envelope")]
}

/// One `sva-cli render` of `target` in `dir` over the store at `store`: whether it answered,
/// and its stdout before `meta`.
fn render(dir: &Path, target: &str, store: &Path, representation: &str) -> (bool, String) {
    let ran = Command::new(env!("CARGO_BIN_EXE_sva-cli"))
        .current_dir(dir)
        .args([
            "render",
            target,
            "--representation",
            representation,
            "--cache",
        ])
        .arg(store)
        .output()
        .expect("the binary runs");
    let printed = String::from_utf8_lossy(&ran.stdout);
    (ran.status.success(), before_meta(&printed).to_string())
}

/// The codes a refusal names, which a caller branches on, and never the wording around them.
fn codes(printed: &str) -> &str {
    let opening = "\"codes\": { \"items\": [";
    let from = printed.find(opening).expect("a refusal's codes") + opening.len();
    let listed = &printed[from..];
    &listed[..listed.find(']').expect("a closed array")]
}

/// Each value the store holds, under its key.
fn stored(store: &Path) -> Vec<(String, String)> {
    let mut keys: Vec<String> = std::fs::read_dir(store)
        .expect("a store")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.len() == 32 && name.bytes().all(|b| b.is_ascii_hexdigit()))
        .collect();
    keys.sort();
    keys.into_iter()
        .map(|key| {
            let bytes = std::fs::read(store.join(&key)).expect("a stored value");
            (format!("store:{key}"), format!("{:016x}", fnv(&bytes)))
        })
        .collect()
}

/// The CLI's render of one case: its stdout with the scratch path taken out, its samples as a
/// `.wav`, the store that render left, then, for `READ_CASES`, each reading off that store or
/// what it refused.
fn pinned(case: usize, dir: &Path, target: &str, instance: &str) -> Vec<(String, String)> {
    let out = scratch(&format!("pinned-{case}"));
    let wav = out.join("out.wav");
    let store = out.join("store");
    let (answered, stdout) = render(dir, target, &store, &format!("samples={}", wav.display()));
    let samples = std::fs::read(&wav).unwrap_or_default();
    let mut held = vec![
        (
            "stdout".to_string(),
            format!(
                "{:016x}",
                fnv(stdout
                    .replace(&wav.display().to_string(), "<wav>")
                    .as_bytes())
            ),
        ),
        ("wav".to_string(), format!("{:016x}", fnv(&samples))),
    ];
    held.extend(stored(&store));
    let readings = if READ_CASES.contains(&case) {
        &READINGS[..]
    } else {
        &[]
    };
    for &name in readings {
        let call = match name {
            "bindings" => format!("bindings(node={instance})"),
            other => other.to_string(),
        };
        let (read, printed) = render(dir, target, &store, &call);
        let value = match read {
            true => format!("{:016x}", fnv(printed.as_bytes())),
            false => format!("refused {}", codes(&printed)),
        };
        held.push((name.to_string(), value));
    }
    let _ = std::fs::remove_dir_all(&out);
    assert!(answered, "{target}: {stdout}");
    held
}

/// Bytes and stored values per case, and every reading but `atoms` for `READ_CASES`. A failure prints what moved
/// and the whole table, which replaces `PINNED` after a change meant to move it.
#[test]
fn renders_stores_and_readings_answer_what_they_answered() {
    let mut table = String::new();
    for (case, (name, target, instance)) in CASES.iter().enumerate() {
        for (item, value) in pinned(case, &fixture(name), target, instance) {
            table.push_str(&format!("{case} {item} {value}\n"));
        }
    }
    let moved: Vec<&str> = table
        .lines()
        .filter(|line| !PINNED.trim().lines().any(|p| p == *line))
        .collect();
    assert!(
        moved.is_empty() && table.lines().count() == PINNED.trim().lines().count(),
        "moved:\n{}\nwhole table:\n{table}",
        moved.join("\n")
    );
}
