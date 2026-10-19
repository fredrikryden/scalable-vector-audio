// Concern: pins the bytes, store keys and readings fixed compositions render to | Non-concern: `atoms`, why any value is right | IO: (fixtures/*) -> hashes

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
    "onsets",
    "alias",
    "bindings",
    "arguments",
];

/// One line per case and item, `<case> <item> <hash, or the codes it refused>`.
const PINNED: &str = r#"
0 stdout e17e3c4311440757
0 wav 665924a7a274d695
0 store:3620e0612b5d84925fa1bd15f1b7716e 6e7a3b8c4323869b
0 lines refused "read.lines_need_unwindowed_lines"
0 spectrum refused "read.lines_need_unwindowed_lines"
0 envelope refused "cast.left_algebra"
0 derivative 1bc92e818f2bdc1f
0 ledger 24f68452f636868a
0 pitch refused "read.lines_need_unwindowed_lines"
0 formants 7937fae6964809c1
0 stereo refused "type.width_mismatch"
0 bands 29cdf02ba6c90f94
0 crest d19e8705213694a2
0 loudness 9d458582a7d5015c
0 onsets 8591adf828334d9e
0 alias 07e069f8ef972de1
0 bindings cca84563df793651
0 arguments ac479335c0844369
1 stdout e17e3c4311440757
1 wav 26291e77cc78e829
1 store:90af903a105917870539e1943a4b72c7 dd23718686b0019d
1 lines refused "read.lines_need_unwindowed_lines"
1 spectrum refused "read.lines_need_unwindowed_lines"
1 envelope refused "cast.left_algebra"
1 derivative 2f6bf49684a384e3
1 ledger c149748fe5702c33
1 pitch refused "read.lines_need_unwindowed_lines"
1 formants 75e330384c1b438d
1 stereo refused "type.width_mismatch"
1 bands 7c8f5f5a2f91277c
1 crest d0c45ca2c5895ee0
1 loudness 334a1c882bbca322
1 onsets 4edc5e2b278c6d64
1 alias 4575666c5c7a4278
1 bindings cca84563df793651
1 arguments ac479335c0844369
2 stdout e17e3c4311440757
2 wav 968e2e9858e90ed9
2 store:29c7e17ebaef85e9ff784832c3108192 dbaee90c4cd14203
2 lines refused "read.no_spectral_sum"
2 spectrum refused "read.no_spectral_sum"
2 envelope refused "read.no_spectral_sum"
2 derivative refused "read.no_spectral_sum"
2 ledger ddba3b6e4cba44fd
2 pitch refused "read.no_spectral_sum"
2 formants 6cd848881180f5d1
2 stereo refused "type.width_mismatch"
2 bands 903beb4df06e1c61
2 crest 3d7b669b681b4bff
2 loudness 1978180633eb2812
2 onsets 1d134e40445adf6d
2 alias 99ea71c420cadac9
2 bindings cca84563df793651
2 arguments 639edc93953a6808
3 stdout 71f4f9a9b3cd9d30
3 wav 64318e943f03cc25
3 store:273d32d1fbc24d73b322a79899449566 d655ced2289ed136
3 lines ef35f29de2cc8837
3 spectrum 6406f9a85eff56a2
3 envelope 508733913bd67342
3 derivative 9685ab0be14bd4b4
3 ledger 365ee65e8121cdc2
3 pitch 842cd98f51eb847c
3 formants c17cad9385229390
3 stereo refused "type.width_mismatch"
3 bands 38faef5d6d5b6179
3 crest 730b41a3cb8b4197
3 loudness 446c16dcd8e3b41d
3 onsets bc5f8c0bb2b64780
3 alias c6b2263b05850daf
3 bindings f15293e1080c5049
3 arguments eb3ed92e88f435e1
4 stdout 5b546f69552c55f0
4 wav d8d646db74dc9cb2
4 store:7d17584ff33a1d5714e10c35ba698e01 487a1b3fda336215
4 lines fba1b60d66f0321f
4 spectrum 675faaa10dc9cdee
4 envelope c554c0bf3802fe1a
4 derivative b27c2b62c8a14697
4 ledger 5aa686392a3dc064
4 pitch 2c103980502add4f
4 formants d49fc45d5ce771cb
4 stereo refused "type.width_mismatch"
4 bands 4cc3dd440cd69438
4 crest 993fdce04a235b38
4 loudness 38f36754ea3daf9f
4 onsets 30c23f7db5058c85
4 alias fa6f16576f7c3641
4 bindings 5a6d3cadc8d1b9e7
4 arguments f74b8084fd1b914d
5 stdout e17e3c4311440757
5 wav 177aed0c28183ef5
5 store:52690610e792cbd9ec87410effce6ac0 c5488abce5d848a7
5 lines 65ef389a409cc7a3
5 spectrum 6d04057b9eba60ce
5 envelope 8c1c2bc6da6a2f6d
5 derivative 09ebc5558a19d4d5
5 ledger 740fc598aaecd8ff
5 pitch a6e21c5a2c21ec31
5 formants 38712182b78e7bef
5 stereo refused "type.width_mismatch"
5 bands b81c9b480a2d59ed
5 crest 22091637c552e761
5 loudness df9a933b71dc654f
5 onsets 79ee9a65f18ed5e7
5 alias 2dc76e74a6175a0c
5 bindings cca84563df793651
5 arguments ac479335c0844369
6 stdout 9108e37d0b7327c6
6 wav 72bce77b8424e0c6
6 store:0aa625a325ad009e42736fb0b55da15e c5942df6f6705a52
6 store:11c136c343d5887a356999a4fc86e8f7 0cd034c8419e36e9
6 store:14f6414df6fcdd24e3ba1efa6b601e02 9b49ca9f4338685b
6 store:2a812a178319a3d66188660d033eee56 dddb4966aa54c1e2
6 store:2b1dfb572460ba3c3e8a08891ad3feda c455643137e1666c
6 store:2cc1e056c0cd5c6013c986cb46ea8680 45eea96b7c077254
6 store:57a6b91a9d6ee92c9f220a471c761060 070c72473fe22f3e
6 store:6e22a9b54e73a673f2f7252c439f5d3c 0b91058235937250
6 store:6f9bcd032cb9677f874eba226b581950 be30a3f2d69e0a1f
6 store:d18c8f806552ff35a1cabe95f566fe2e 74803c93aea52c1f
6 store:d4b554f1adbd1e2708edf6cef39dd919 200d07b6e91dfa95
6 store:dfbe3e234fa8c68c2717f78df327e90f 6187182b7dda3f2e
6 store:e41ecc22b1d8d907ef9ba75bd804e1e7 1dbdd6ed6f7b9789
6 store:e51a27e86bcbd36a2c884364cb358a1e b13e933cd4588103
6 lines refused "type.samples_in_closed_form"
6 spectrum 42bed4c0b2069043
6 envelope 87ffd87017994a9c
6 derivative 4a59aeeefeec93cb
6 ledger a0defb360ca48bd6
6 pitch 89b36bedafb61961
6 formants 9d1b35f94c0063dc
6 stereo refused "type.width_mismatch"
6 bands bd1fd745fa510446
6 crest e33bd368de79b61b
6 loudness 45a4943a1e029c95
6 onsets ca75dc267e968000
6 alias refused "engine.alias_needs_a_closed_form"
6 bindings f836b683b117cd32
6 arguments db19184c92016629
7 stdout 71454a132807dddb
7 wav 3372a403bff9f10a
7 store:48cc333eda6f48499d525bb87ae7a03c 413a510a27a5c41f
7 lines refused "read.lines_need_unwindowed_lines"
7 spectrum refused "read.lines_need_unwindowed_lines"
7 envelope refused "cast.left_algebra"
7 derivative d910c16d47459556
7 ledger 46e5787a8425e680
7 pitch refused "read.lines_need_unwindowed_lines"
7 formants 79b876e987cc1962
7 stereo 253d0243cde6d3f1
7 bands abe00ce3fd0d466c
7 crest 1519d39532b67500
7 loudness 7ed282b028e76da9
7 onsets f9e35661305846c9
7 alias eff5d014d695a316
7 bindings b63bdcd55385e2fe
7 arguments 7c1d152f8c0e3714
8 stdout bb69bdf6de9adeb8
8 wav 083e3fdfa6fd2b2d
8 store:0435c2987fec5f6fa5f7bc60ed42587e 31fc1d6de3a78d9f
8 lines refused "type.samples_in_closed_form"
8 spectrum 263622d4e39fb4bb
8 envelope 318124550bf22b6d
8 derivative 7e5fe43cb629a9ad
8 ledger 113e91649540786a
8 pitch be81fe72bcbb47d0
8 formants acc60cc71b6f3c7e
8 stereo be4b410523c86ec8
8 bands 99f5a79b571ea1ce
8 crest 88972155a19a1f8a
8 loudness 9de3a82615bc2604
8 onsets 51f50b95db85dea6
8 alias refused "engine.alias_needs_a_closed_form"
8 bindings 5ce608c3a8b79a86
8 arguments b642f09606176b94
9 stdout e17e3c4311440757
9 wav 3a9ca1a53a5fa42a
9 store:3180ee7f5ba55875b73514cbc66c3107 ee52b899077cfc7b
9 lines refused "read.lines_need_unwindowed_lines"
9 spectrum refused "read.lines_need_unwindowed_lines"
9 envelope refused "cast.left_algebra"
9 derivative d877ae4e9f8f299f
9 ledger e407bd03bfaef3a1
9 pitch refused "read.lines_need_unwindowed_lines"
9 formants d435c1d2fea457a5
9 stereo refused "type.width_mismatch"
9 bands fea1ea74be44e3b9
9 crest 4f9afb40d9743f97
9 loudness 9dd07af91a7b9f8f
9 onsets 3252466726aa0382
9 alias 443c203e9bbc5053
9 bindings cca84563df793651
9 arguments ac479335c0844369
10 stdout 5687cc05a27c2ec3
10 wav a9866247325c001f
10 store:6655845ef1b852c2dfba6667b7a17cb7 b0709c7476956377
10 lines refused "read.lines_need_unwindowed_lines"
10 spectrum refused "read.lines_need_unwindowed_lines"
10 envelope refused "cast.left_algebra"
10 derivative 89dc769fdd11eaeb
10 ledger 8928906b58d6379e
10 pitch refused "read.lines_need_unwindowed_lines"
10 formants 9251ff645e063e26
10 stereo refused "type.width_mismatch"
10 bands 0bbb7134b8b65965
10 crest d4ef22eb0b7406c3
10 loudness 8f019b20224e1f2b
10 onsets 31a1ed87ec02eba6
10 alias 2711379113f90a29
10 bindings 978c10e7996dc9a1
10 arguments b5457effabd50fe5
11 stdout a05558226a6cba5a
11 wav c187b7408a55ebf7
11 store:05527852d844ba60e2f03bfc763cbeda a988306ede2eb6b7
11 store:b6774780ba26bd24289ee5a5a363539e 7509bfcfccb25771
11 store:fef920a29d25f7e69ab776f1c1aa2df8 5a1a23698fff0512
11 lines refused "type.samples_in_closed_form"
11 spectrum e17c3de46a83bf13
11 envelope 3da21efb2e62895a
11 derivative 600eca219ca588e6
11 ledger e0197332a2be33aa
11 pitch a24d12fe870ae755
11 formants cfbdddc52c266a39
11 stereo refused "type.width_mismatch"
11 bands ca4c55556f72549f
11 crest 52cff5a7ddc6a16f
11 loudness 57cbc72ef803b964
11 onsets 71fe4202833e1ed8
11 alias refused "engine.alias_needs_a_closed_form"
11 bindings 193483886e1b97eb
11 arguments fbff5288f7442249
12 stdout 24a0009133ff612c
12 wav 3fc9f322aaf6fc5f
12 store:05fd9b34f0c3d96a4508e7fe93cef407 99763ac1b4f08b05
12 store:eefd7456adda71f95410a477bc38064f 25998e4cba244b7c
12 lines refused "type.samples_in_closed_form"
12 spectrum 6410df2d712681d7
12 envelope 881172fade4b59a6
12 derivative ae083f2d3b99ae73
12 ledger 54a1099db8ef3b2e
12 pitch 368288732129c0cc
12 formants 781b8dbfe0c77903
12 stereo refused "type.width_mismatch"
12 bands 9eb16a404f3b9c03
12 crest 957a6a8b36086e5d
12 loudness 7089b5f34107daff
12 onsets 6b7b7e25a9172fd8
12 alias refused "engine.alias_needs_a_closed_form"
12 bindings 4a6686d8e212fcf1
12 arguments 58316c250a5fe29f
13 stdout 189201d106182ae6
13 wav 256b6dfecab93743
13 store:7766d37a0cdcc294553055f59415b4ee ecce9080780fff1a
13 store:b636f740c7f5c69c7d133075cf6253c3 2fdd7da48324e1b7
13 lines refused "type.samples_in_closed_form"
13 spectrum 36d1f673c6e75634
13 envelope a3f6ab528e89b8e1
13 derivative 17d1bac70612fdbc
13 ledger fe04d8495d5f3a93
13 pitch f79f08371959d89a
13 formants becdd7ce84e5d5e0
13 stereo refused "type.width_mismatch"
13 bands 711c028584af69c4
13 crest 84b6249cd8946dd5
13 loudness 66f7c487262b6f4c
13 onsets e6f989780d6dc866
13 alias refused "engine.alias_needs_a_closed_form"
13 bindings 0255ef5c70e74e5e
13 arguments 0adbeca7c937da37
14 stdout b6ad884dff66b63b
14 wav 0c1d8e0ad4b6a019
14 store:254945568a381e7c3e5471da1d74728f ffdf9d3456bc9823
14 store:9799ab7ba6f836a107fd71777f42e6f0 651bf83dd5fdb31f
14 store:979e8f53e59fb8d66035a41b9132d6a7 e5655bc3498b0d57
14 store:f9f5071eb92a34c3a0ca9fe84c15c685 50d78de808a824d3
14 lines refused "type.samples_in_closed_form"
14 spectrum 512f76f46b5beaa0
14 envelope 8a05d06c94a4ebd2
14 derivative b295446119f7bca0
14 ledger 64db111d31d55f60
14 pitch 23af9b8ff9cfdad5
14 formants 662ba8798a965aa0
14 stereo refused "type.width_mismatch"
14 bands 7d74a7db8b28aeb0
14 crest b6a0116e3a0d48b2
14 loudness b3ee23a298f89c67
14 onsets f6f528e1301c1970
14 alias refused "engine.alias_needs_a_closed_form"
14 bindings 72871833b56a7987
14 arguments 566a07fc7cfce2ce
15 stdout ea658b80403f05b8
15 wav 82773cac15ff1cca
15 store:b5146c74a37c137f36a90b33af43d7d6 8acfacd50445baef
15 lines refused "read.lines_need_unwindowed_lines"
15 spectrum refused "read.lines_need_unwindowed_lines"
15 envelope refused "cast.left_algebra"
15 derivative da1daf2367fd7eb8
15 ledger e2317f40fa08a55c
15 pitch refused "read.lines_need_unwindowed_lines"
15 formants 3d80f4c9879a46af
15 stereo refused "type.width_mismatch"
15 bands 0501bbe986c0382f
15 crest c8567555d4d503f3
15 loudness 2fca583e2138915e
15 onsets 000ddb63a79e2c23
15 alias 1f69bad5069d6d25
15 bindings 1b8e53e12cdbbcab
15 arguments b6603b7216e51daa
16 stdout 6d9beba6f3f07c0c
16 wav 9007d5d55c305e67
16 store:26ad75243b2f79c4c406c6f5bf4ec486 c418462b254b42d5
16 store:3c3e02ce6997bbbc519549daeed604f4 4ba0d4a350140222
16 store:721873409668da2d147b4a20e9a0d6e3 2a704778eabb22b0
16 store:7d2a34d66251dbf649d696726b2f90f8 85083b1a5bdc5bf4
16 store:7febd1dbb83c13997bf57fbce8e72f50 0d28f9abd7177441
16 lines refused "type.samples_in_closed_form"
16 spectrum 85484af2ed2da1d6
16 envelope 43e9dc89486b18d8
16 derivative 3a46884beab051bb
16 ledger 32a4f87e73d8cfd5
16 pitch 5fd42880b14ddf9a
16 formants bbe361f47832f132
16 stereo refused "type.width_mismatch"
16 bands 0291059863af6f29
16 crest 3a46431242db5356
16 loudness b606acf7d400387a
16 onsets 007a30876cb59a51
16 alias refused "engine.alias_needs_a_closed_form"
16 bindings 01944f73abd4ed2b
16 arguments 79995a358f9efc0a
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
/// `.wav`, the store that render left, then each reading off that store, or what it refused.
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
    for name in READINGS {
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

/// Bytes, stored values and every reading but `atoms`, per case. A failure prints what moved
/// and the whole table, which replaces `PINNED` after a change meant to move it. Pinned on
/// x86_64 Linux and the pinned toolchain alone: another libm may round a sample differently.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
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
