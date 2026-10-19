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
0 derivative 35f9147e138d1ec8
0 ledger 953e7682e9f5d4c1
0 pitch refused "read.lines_need_unwindowed_lines"
0 formants 983636d7910212a6
0 stereo refused "type.width_mismatch"
0 bands a3c4e4842f07b9ff
0 crest 50d7f89020924fc1
0 loudness aad22677dfef8ef7
0 onsets eadc77c6b241cda2
0 alias 0b6c2f5a2648dea7
0 bindings b2361ce7b3484da7
0 arguments 2629850681a29371
1 stdout e17e3c4311440757
1 wav 26291e77cc78e829
1 store:90af903a105917870539e1943a4b72c7 dd23718686b0019d
1 lines refused "read.lines_need_unwindowed_lines"
1 spectrum refused "read.lines_need_unwindowed_lines"
1 envelope refused "cast.left_algebra"
1 derivative 8423ad0fc11dacf4
1 ledger 55d3200279ae6a40
1 pitch refused "read.lines_need_unwindowed_lines"
1 formants 2418ca8430c4ae82
1 stereo refused "type.width_mismatch"
1 bands 581d97526f370b89
1 crest a40f595369c03423
1 loudness 67e6d87e85c5ae31
1 onsets bd56bda86318eaa0
1 alias 1cfd6304d031e5b2
1 bindings b2361ce7b3484da7
1 arguments 2629850681a29371
2 stdout e17e3c4311440757
2 wav 968e2e9858e90ed9
2 store:29c7e17ebaef85e9ff784832c3108192 dbaee90c4cd14203
2 lines refused "read.no_spectral_sum"
2 spectrum refused "read.no_spectral_sum"
2 envelope refused "read.no_spectral_sum"
2 derivative refused "read.no_spectral_sum"
2 ledger 1b8703e67f289b62
2 pitch refused "read.no_spectral_sum"
2 formants fcac63275adf6866
2 stereo refused "type.width_mismatch"
2 bands 25fbbb79f504c832
2 crest 568687cfa8372b2c
2 loudness ef6bdab214946b77
2 onsets 3e0f68c558cc9c59
2 alias 23ed6fa6ba0ae52f
2 bindings b2361ce7b3484da7
2 arguments 626f051e3a2fc700
3 stdout 71f4f9a9b3cd9d30
3 wav 64318e943f03cc25
3 store:273d32d1fbc24d73b322a79899449566 d655ced2289ed136
3 lines 1b91625484873716
3 spectrum 72763f1617db3c1e
3 envelope 5ff2738827a015bc
3 derivative 2d6f0bddc0248089
3 ledger a3131877d3588ff5
3 pitch 31eb906bd2504ddd
3 formants 6c7ec0375f7527d9
3 stereo refused "type.width_mismatch"
3 bands cd2b43ad7e82e04c
3 crest 394c48d576cb4064
3 loudness 63f90b5f698895e4
3 onsets ba601020a6d5bff4
3 alias a01b032e9f59f819
3 bindings de379a53261f899f
3 arguments 963d7e681d3a7159
4 stdout 5b546f69552c55f0
4 wav d8d646db74dc9cb2
4 store:7d17584ff33a1d5714e10c35ba698e01 487a1b3fda336215
4 lines 856b8c259fc6ee26
4 spectrum 5875cbec9753bd9a
4 envelope cdefdc3ad777a6cc
4 derivative 1335b1af4f591c60
4 ledger 3800a54550f862d9
4 pitch 12415b8ef26ea22e
4 formants 5027acd5284610c8
4 stereo refused "type.width_mismatch"
4 bands 6a8de2cef787b25b
4 crest ab5fe8e6a6061aa3
4 loudness 9d31c064f71fa416
4 onsets c63f26d67c44a261
4 alias c5efa380d6d97797
4 bindings a066a34dcc83fcfb
4 arguments 9eaefeccb255ec5d
5 stdout e17e3c4311440757
5 wav 177aed0c28183ef5
5 store:52690610e792cbd9ec87410effce6ac0 c5488abce5d848a7
5 lines 4745742b132f215a
5 spectrum cfc00a7586fe2dc2
5 envelope 3cab08eac882effb
5 derivative fc27dc342d808d4a
5 ledger 46dedfaaf55adc98
5 pitch 6c4088af788a0ba4
5 formants fc004c90d9c293a8
5 stereo refused "type.width_mismatch"
5 bands f10e77d7db921e34
5 crest a932607c764185fc
5 loudness 687de527e5e5d00a
5 onsets 78d3e0708fa4ceab
5 alias 33ba447bfecaa33a
5 bindings b2361ce7b3484da7
5 arguments 2629850681a29371
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
6 spectrum c810bd60c71870ef
6 envelope 1661858ad14eae30
6 derivative b6f4dab7583e3972
6 ledger 435fd75fa50172ef
6 pitch aec67c16e7d191f0
6 formants a13a2fb073fb7799
6 stereo refused "type.width_mismatch"
6 bands c5291a9e33381897
6 crest a82403e0f7dffe5e
6 loudness b3a39f5e29654c6e
6 onsets c8983eec7d79b214
6 alias refused "engine.alias_needs_a_closed_form"
6 bindings 61d25ef94a8f6635
6 arguments f96cc4a110318b3d
7 stdout 71454a132807dddb
7 wav 3372a403bff9f10a
7 store:48cc333eda6f48499d525bb87ae7a03c 413a510a27a5c41f
7 lines refused "read.lines_need_unwindowed_lines"
7 spectrum refused "read.lines_need_unwindowed_lines"
7 envelope refused "cast.left_algebra"
7 derivative 17cd66f01d871437
7 ledger 730b05495dac8f1d
7 pitch refused "read.lines_need_unwindowed_lines"
7 formants 0441cddf43c7f501
7 stereo 8a5684e6ff71476b
7 bands 0867793a27647753
7 crest 2a16a80ec7c0c3e7
7 loudness bf3a0c8d346a3c22
7 onsets 778560ca33dc77bd
7 alias 9ac22474372ae010
7 bindings e78c34f33117e96f
7 arguments bcbf8809de18a142
8 stdout bb69bdf6de9adeb8
8 wav 083e3fdfa6fd2b2d
8 store:0435c2987fec5f6fa5f7bc60ed42587e 31fc1d6de3a78d9f
8 lines refused "type.samples_in_closed_form"
8 spectrum 7dd9b2b9ac2f9923
8 envelope 07f0f666a6e99af1
8 derivative ea4e8903e65bfb18
8 ledger 2b1afbbc3ee1ab27
8 pitch 88e045fa36fc3b75
8 formants e72976350c77aa9d
8 stereo 486eb5f8b21ac2ac
8 bands e700cbdd0ae4e73b
8 crest 126fecd1871771cd
8 loudness 8145e7b3ec2b68cf
8 onsets 80c3b599b5099972
8 alias refused "engine.alias_needs_a_closed_form"
8 bindings 79396b2f23896766
8 arguments a5fe2d9cfe6ecec2
9 stdout e17e3c4311440757
9 wav 3a9ca1a53a5fa42a
9 store:3180ee7f5ba55875b73514cbc66c3107 ee52b899077cfc7b
9 lines refused "read.lines_need_unwindowed_lines"
9 spectrum refused "read.lines_need_unwindowed_lines"
9 envelope refused "cast.left_algebra"
9 derivative d50dd96bf045c44e
9 ledger 2f2582948cd21ea0
9 pitch refused "read.lines_need_unwindowed_lines"
9 formants 9b1d29e622fcffc6
9 stereo refused "type.width_mismatch"
9 bands 28533ab369bc9f04
9 crest 25d543f30d2108ba
9 loudness 6a8304e371b811b8
9 onsets 51de735bebd75bfe
9 alias ea118cd4571258ed
9 bindings b2361ce7b3484da7
9 arguments 2629850681a29371
10 stdout 5687cc05a27c2ec3
10 wav a9866247325c001f
10 store:6655845ef1b852c2dfba6667b7a17cb7 b0709c7476956377
10 lines refused "read.lines_need_unwindowed_lines"
10 spectrum refused "read.lines_need_unwindowed_lines"
10 envelope refused "cast.left_algebra"
10 derivative b672d107adb53ae4
10 ledger a96398225c085d3d
10 pitch refused "read.lines_need_unwindowed_lines"
10 formants 8dce37b8a2e84765
10 stereo refused "type.width_mismatch"
10 bands b6d1fe39fca514b8
10 crest c013488117efe476
10 loudness cdf17b0d3173aed4
10 onsets 1f1eac0113d7c212
10 alias 94a42da567e5550f
10 bindings 5d16a4e7b16c54d3
10 arguments 13a9d91a5de370e5
11 stdout a05558226a6cba5a
11 wav c187b7408a55ebf7
11 store:05527852d844ba60e2f03bfc763cbeda a988306ede2eb6b7
11 store:b6774780ba26bd24289ee5a5a363539e 7509bfcfccb25771
11 store:fef920a29d25f7e69ab776f1c1aa2df8 5a1a23698fff0512
11 lines refused "type.samples_in_closed_form"
11 spectrum 6899acb84aeffb17
11 envelope 6c931b1f349957a6
11 derivative 62dbab03c01e6bdf
11 ledger 4bd1d6c3295f267d
11 pitch 8b57e29205603402
11 formants b6f534fc1b8a380e
11 stereo refused "type.width_mismatch"
11 bands 4e68e078dbeb60f2
11 crest b5958d3d831c48c2
11 loudness 198f16927cb33c3b
11 onsets 44eb672b2295c234
11 alias refused "engine.alias_needs_a_closed_form"
11 bindings 6187812054881a9c
11 arguments 00523829ab775155
12 stdout 24a0009133ff612c
12 wav 3fc9f322aaf6fc5f
12 store:05fd9b34f0c3d96a4508e7fe93cef407 99763ac1b4f08b05
12 store:eefd7456adda71f95410a477bc38064f 25998e4cba244b7c
12 lines refused "type.samples_in_closed_form"
12 spectrum 0f735fad5af3fccf
12 envelope 89aa29b0e7945432
12 derivative ef9c265b9c8067a6
12 ledger 962519bfc611720f
12 pitch b91b2a6eba707769
12 formants 6e86abc02ad56196
12 stereo refused "type.width_mismatch"
12 bands 181e16cbacbe1a60
12 crest c9c38dcd38cd9734
12 loudness 679e34af23d3c93e
12 onsets 87be6e14ba469a74
12 alias refused "engine.alias_needs_a_closed_form"
12 bindings 2246221dde6bf19e
12 arguments 31ac7203e7c98531
13 stdout 189201d106182ae6
13 wav 256b6dfecab93743
13 store:7766d37a0cdcc294553055f59415b4ee ecce9080780fff1a
13 store:b636f740c7f5c69c7d133075cf6253c3 2fdd7da48324e1b7
13 lines refused "type.samples_in_closed_form"
13 spectrum d8d62b66bbe2f650
13 envelope cbe5799e796912e5
13 derivative a07fe3e0b3147f23
13 ledger ed3f2d2fedf9b7ee
13 pitch f02ca28a94b385c7
13 formants 8e70948be83ecd83
13 stereo refused "type.width_mismatch"
13 bands 26c97f7e9f2d2257
13 crest 9da74715fcec4df0
13 loudness e2db4b0db779cf41
13 onsets 35cc465fb21cf8d2
13 alias refused "engine.alias_needs_a_closed_form"
13 bindings 4d733f3c0aabb4f4
13 arguments c4f0bea827f62da3
14 stdout b6ad884dff66b63b
14 wav 0c1d8e0ad4b6a019
14 store:254945568a381e7c3e5471da1d74728f ffdf9d3456bc9823
14 store:9799ab7ba6f836a107fd71777f42e6f0 651bf83dd5fdb31f
14 store:979e8f53e59fb8d66035a41b9132d6a7 e5655bc3498b0d57
14 store:f9f5071eb92a34c3a0ca9fe84c15c685 50d78de808a824d3
14 lines refused "type.samples_in_closed_form"
14 spectrum 733faf6b92c904c4
14 envelope 3e0738aa56781044
14 derivative 52a3a8baf8eb9c97
14 ledger ee685635515d0f5b
14 pitch 8388ebf4fa2a10d2
14 formants 697fe6c1994ce521
14 stereo refused "type.width_mismatch"
14 bands 87eb918959d9ad95
14 crest cfd72ab34716b2cd
14 loudness 2a2e183dcfe173b2
14 onsets 6baee3bb2fcc28dc
14 alias refused "engine.alias_needs_a_closed_form"
14 bindings 93b2bd559b5e577e
14 arguments 7309f5cc8a34fe8a
15 stdout ea658b80403f05b8
15 wav 82773cac15ff1cca
15 store:b5146c74a37c137f36a90b33af43d7d6 8acfacd50445baef
15 lines refused "read.lines_need_unwindowed_lines"
15 spectrum refused "read.lines_need_unwindowed_lines"
15 envelope refused "cast.left_algebra"
15 derivative 9e12de6ee9602f21
15 ledger e0e6637e9532e481
15 pitch refused "read.lines_need_unwindowed_lines"
15 formants 01555b83d7bdbc9e
15 stereo refused "type.width_mismatch"
15 bands e65cd30b30a6a050
15 crest e1d0a33b7254410e
15 loudness 5d753b49eb5fbe55
15 onsets 32ecde13574c7d6f
15 alias 47fc371a3c809533
15 bindings f758b54150146934
15 arguments 6fe5313a42a03884
16 stdout 6d9beba6f3f07c0c
16 wav 9007d5d55c305e67
16 store:26ad75243b2f79c4c406c6f5bf4ec486 c418462b254b42d5
16 store:3c3e02ce6997bbbc519549daeed604f4 4ba0d4a350140222
16 store:721873409668da2d147b4a20e9a0d6e3 2a704778eabb22b0
16 store:7d2a34d66251dbf649d696726b2f90f8 85083b1a5bdc5bf4
16 store:7febd1dbb83c13997bf57fbce8e72f50 0d28f9abd7177441
16 lines refused "type.samples_in_closed_form"
16 spectrum 5716ec96c81875be
16 envelope 9b5d5c2be783ec9c
16 derivative 5a8e9e56e2181334
16 ledger b12246ec1b32d966
16 pitch fbb3b0a9308e8b85
16 formants d61e398c61fe2891
16 stereo refused "type.width_mismatch"
16 bands 1b6c33eaf5143142
16 crest 740c04aec461692d
16 loudness ea322255b852d039
16 onsets ad49f47e611dbaad
16 alias refused "engine.alias_needs_a_closed_form"
16 bindings 76ea2f9c67cc302d
16 arguments ef9f32c3d73615ee
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
