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
0 derivative d81eaec883ddddf1
0 ledger a81a9f2e22e8741c
0 pitch refused "read.lines_need_unwindowed_lines"
0 formants 44f6bef6e80c183a
0 stereo refused "type.width_mismatch"
0 bands 5e0d7ee54333e671
0 crest 31febdc5b5efcd5e
0 loudness 9589ec89c9116b2a
0 onsets 659aba033429aae6
0 alias 49dbfaacc7111935
0 bindings 55996104faf15d37
0 arguments 06ad74b6d91890af
1 stdout e17e3c4311440757
1 wav 26291e77cc78e829
1 store:90af903a105917870539e1943a4b72c7 dd23718686b0019d
1 lines refused "read.lines_need_unwindowed_lines"
1 spectrum refused "read.lines_need_unwindowed_lines"
1 envelope refused "cast.left_algebra"
1 derivative c7d6759c5f29f901
1 ledger d92bf37606cd1add
1 pitch refused "read.lines_need_unwindowed_lines"
1 formants 2e50d4674151ea3e
1 stereo refused "type.width_mismatch"
1 bands eaa00e16e72f1753
1 crest 818bbbbdd82eb044
1 loudness c864dc170e6116f4
1 onsets ec9e1b76c07af9ec
1 alias 2fc95bc509118fe4
1 bindings 55996104faf15d37
1 arguments 06ad74b6d91890af
2 stdout e17e3c4311440757
2 wav 968e2e9858e90ed9
2 store:29c7e17ebaef85e9ff784832c3108192 dbaee90c4cd14203
2 lines refused "read.no_spectral_sum"
2 spectrum refused "read.no_spectral_sum"
2 envelope refused "read.no_spectral_sum"
2 derivative refused "read.no_spectral_sum"
2 ledger c995310c12fa68a3
2 pitch refused "read.no_spectral_sum"
2 formants fd100173ce103b1a
2 stereo refused "type.width_mismatch"
2 bands 0f05094358b6fde8
2 crest 2b84826d28a0b6b3
2 loudness b6b84e9f8100e960
2 onsets cf813379ef2fec15
2 alias cee166be742c4d0d
2 bindings 55996104faf15d37
2 arguments 790c5280d59cfece
3 stdout 71f4f9a9b3cd9d30
3 wav 64318e943f03cc25
3 store:273d32d1fbc24d73b322a79899449566 d655ced2289ed136
3 lines 345b2d307841d7af
3 spectrum a08a82096272a79b
3 envelope 1ef543230cd63132
3 derivative 3ca3819eb0d402de
3 ledger b976266740b39fd4
3 pitch 61c0849f1534d989
3 formants 5272b351a938e17f
3 stereo refused "type.width_mismatch"
3 bands 1a49cf23331df86e
3 crest a604dc1b7ca7ab89
3 loudness acc44ca10758d4e1
3 onsets a3efceecac637df2
3 alias da4a7342653ab8e3
3 bindings 33a0538bf6cc55df
3 arguments 2bb69b3f172c7467
4 stdout 5b546f69552c55f0
4 wav d8d646db74dc9cb2
4 store:7d17584ff33a1d5714e10c35ba698e01 487a1b3fda336215
4 lines 8b47effd51e53bd7
4 spectrum 81a5f4ddb1e26467
4 envelope 8531b8e451db620a
4 derivative bc60fcde3dd17d39
4 ledger 30a8b19632a98f02
4 pitch 13220774c80e981a
4 formants c84fce7c35c39ce6
4 stereo refused "type.width_mismatch"
4 bands d08c02f179af9785
4 crest dfeeafcf7370571a
4 loudness 44b60083cdc2dceb
4 onsets 092f6e1051e9dfe7
4 alias bc5983701760cabd
4 bindings 886231a95a5940a9
4 arguments 99604d4875310723
5 stdout e17e3c4311440757
5 wav 177aed0c28183ef5
5 store:52690610e792cbd9ec87410effce6ac0 c5488abce5d848a7
5 lines 7ba603ba80887d7b
5 spectrum 3ac993de57db8987
5 envelope 77712c5be8088231
5 derivative f20dfa51dc8b466f
5 ledger 705e79f7e432d859
5 pitch ab0013621325eae0
5 formants 3ed393da2d26350c
5 stereo refused "type.width_mismatch"
5 bands cf84c38e39164bc6
5 crest 023b1ead9cbf7915
5 loudness 3a959dbc136de81d
5 onsets b762a5eb6ba2985f
5 alias 39efe26b5318af14
5 bindings 55996104faf15d37
5 arguments 06ad74b6d91890af
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
6 spectrum 8001fc059a8a6f36
6 envelope d43c84b73b8206ee
6 derivative ae60bfb0fe6b1de7
6 ledger 8fd3ac88458b7b84
6 pitch 5001916fc623ed38
6 formants 886190cf60b63553
6 stereo refused "type.width_mismatch"
6 bands da76e2aca5d234d9
6 crest 54756ec0106e52cd
6 loudness d0bd8668dc98f381
6 onsets 586ea9d1107f7236
6 alias refused "engine.alias_needs_a_closed_form"
6 bindings 5a85a77bcf2bd8aa
6 arguments ac172d9792d62ec3
7 stdout 71454a132807dddb
7 wav 3372a403bff9f10a
7 store:48cc333eda6f48499d525bb87ae7a03c 413a510a27a5c41f
7 lines refused "read.lines_need_unwindowed_lines"
7 spectrum refused "read.lines_need_unwindowed_lines"
7 envelope refused "cast.left_algebra"
7 derivative cc76a0e774cc51ba
7 ledger 7c32c669dc25b0ae
7 pitch refused "read.lines_need_unwindowed_lines"
7 formants 59c416a1329519cd
7 stereo bc35ab4ce9795993
7 bands 32290f3560d14d05
7 crest 9727518a4cf6f844
7 loudness 23b19f24881076b7
7 onsets 739a824c18d871e1
7 alias 4e801f2133d7321a
7 bindings 2e2ad2185f66425e
7 arguments 88ff2c1d9aff3054
8 stdout bb69bdf6de9adeb8
8 wav 083e3fdfa6fd2b2d
8 store:0435c2987fec5f6fa5f7bc60ed42587e 31fc1d6de3a78d9f
8 lines refused "type.samples_in_closed_form"
8 spectrum 93eca2d7a211f364
8 envelope ff3b9e9e5f46971f
8 derivative c04ab153e20b79d1
8 ledger 7dd06c77df7b9950
8 pitch 2c4e9c93b799f475
8 formants ce36cf4a14faa69f
8 stereo 9b200ef45ecf425e
8 bands bb7567eae63ccf85
8 crest 5a7e82ebdc001324
8 loudness e33e86e8b4017620
8 onsets c9981bf05d4f4c18
8 alias refused "engine.alias_needs_a_closed_form"
8 bindings 18568f7211560d60
8 arguments 72d3fde01dccb1d4
9 stdout e17e3c4311440757
9 wav 3a9ca1a53a5fa42a
9 store:3180ee7f5ba55875b73514cbc66c3107 ee52b899077cfc7b
9 lines refused "read.lines_need_unwindowed_lines"
9 spectrum refused "read.lines_need_unwindowed_lines"
9 envelope refused "cast.left_algebra"
9 derivative 7198ccd074c44b3d
9 ledger 73d6c2700c5860c3
9 pitch refused "read.lines_need_unwindowed_lines"
9 formants 16203ffe2c8f2a82
9 stereo refused "type.width_mismatch"
9 bands 20f94541db30b27a
9 crest 7e59103c3fec4f03
9 loudness 228be9e253953f55
9 onsets 3754e0d2e95381aa
9 alias bd22409d56e6ae9f
9 bindings 55996104faf15d37
9 arguments 06ad74b6d91890af
10 stdout 5687cc05a27c2ec3
10 wav a9866247325c001f
10 store:6655845ef1b852c2dfba6667b7a17cb7 b0709c7476956377
10 lines refused "read.lines_need_unwindowed_lines"
10 spectrum refused "read.lines_need_unwindowed_lines"
10 envelope refused "cast.left_algebra"
10 derivative ec6473d98cddedd1
10 ledger 95a6291f7ee4b27c
10 pitch refused "read.lines_need_unwindowed_lines"
10 formants 4e9aa3f45847bf61
10 stereo refused "type.width_mismatch"
10 bands c9435ab0dfb7c9e2
10 crest 64aa3f609bf558bf
10 loudness b9a9634459e0a32d
10 onsets 7b2c4e9c5fc74fbe
10 alias 884c587dcee3c255
10 bindings 28b228f6e37c1197
10 arguments f9bda12a70a3e05b
11 stdout a05558226a6cba5a
11 wav c187b7408a55ebf7
11 store:05527852d844ba60e2f03bfc763cbeda a988306ede2eb6b7
11 store:b6774780ba26bd24289ee5a5a363539e 7509bfcfccb25771
11 store:fef920a29d25f7e69ab776f1c1aa2df8 5a1a23698fff0512
11 lines refused "type.samples_in_closed_form"
11 spectrum 6318f55b64a355fa
11 envelope d395babf410809ec
11 derivative 0a7e36ce179478f2
11 ledger d5c3e547a5948b24
11 pitch 016516f4b75637ea
11 formants 42481dc66a7bd810
11 stereo refused "type.width_mismatch"
11 bands 6d5786094833eabc
11 crest 532d528f12104f25
11 loudness 93e9d29a32476488
11 onsets fdcef50b4d02b662
11 alias refused "engine.alias_needs_a_closed_form"
11 bindings 83b658f985bfc0f7
11 arguments dd01d0a6a5c9affb
12 stdout 24a0009133ff612c
12 wav 3fc9f322aaf6fc5f
12 store:05fd9b34f0c3d96a4508e7fe93cef407 99763ac1b4f08b05
12 store:eefd7456adda71f95410a477bc38064f 25998e4cba244b7c
12 lines refused "type.samples_in_closed_form"
12 spectrum feb5e9600123ccaa
12 envelope 592d49278f068ff4
12 derivative 8709c6ff1a0f096f
12 ledger 7f46deac4fa524c0
12 pitch 1d262515628d8879
12 formants c302e659dbf8f984
12 stereo refused "type.width_mismatch"
12 bands f1cf8ae16a426ff2
12 crest 25205dcbff3bbac7
12 loudness c52ab891011e3473
12 onsets 62d40b7c72b2d3fe
12 alias refused "engine.alias_needs_a_closed_form"
12 bindings c00fcdde09976205
12 arguments 0dc132871417c7bf
13 stdout 189201d106182ae6
13 wav 256b6dfecab93743
13 store:7766d37a0cdcc294553055f59415b4ee ecce9080780fff1a
13 store:b636f740c7f5c69c7d133075cf6253c3 2fdd7da48324e1b7
13 lines refused "type.samples_in_closed_form"
13 spectrum 1244be27935ed8b9
13 envelope 6fe8d8e9c072a88f
13 derivative cecadb73169ac2d8
13 ledger 0c11e8e9d388dd39
13 pitch 22b42b0c0deb46ef
13 formants 5a6205f4a2fd7d75
13 stereo refused "type.width_mismatch"
13 bands 3b2fe8075b914ed9
13 crest 9bb85256699e51bb
13 loudness f8bb6aa1abd75f88
13 onsets 0cc8a8a12011b04c
13 alias refused "engine.alias_needs_a_closed_form"
13 bindings 55d8ef34036f9814
13 arguments fd4f588c17d9b49d
14 stdout b6ad884dff66b63b
14 wav 0c1d8e0ad4b6a019
14 store:254945568a381e7c3e5471da1d74728f ffdf9d3456bc9823
14 store:9799ab7ba6f836a107fd71777f42e6f0 651bf83dd5fdb31f
14 store:979e8f53e59fb8d66035a41b9132d6a7 e5655bc3498b0d57
14 store:f9f5071eb92a34c3a0ca9fe84c15c685 50d78de808a824d3
14 lines refused "type.samples_in_closed_form"
14 spectrum c3a35add1d0a97a9
14 envelope b81470ab6f78554c
14 derivative 524757bb3d37ddc6
14 ledger 5891dcca1c0fd662
14 pitch 48059796cd127f7e
14 formants 5bd9d7b769cbce25
14 stereo refused "type.width_mismatch"
14 bands 629d3b53835e3a9f
14 crest 788aa2a6b8f58ba6
14 loudness b4bd339cb6ad301d
14 onsets 140bd376304b47f8
14 alias refused "engine.alias_needs_a_closed_form"
14 bindings d7871f6c1319e88f
14 arguments 3952891c8f49f350
15 stdout ea658b80403f05b8
15 wav 82773cac15ff1cca
15 store:b5146c74a37c137f36a90b33af43d7d6 8acfacd50445baef
15 lines refused "read.lines_need_unwindowed_lines"
15 spectrum refused "read.lines_need_unwindowed_lines"
15 envelope refused "cast.left_algebra"
15 derivative 3a8e03db225937d4
15 ledger 60845b4eb6020742
15 pitch refused "read.lines_need_unwindowed_lines"
15 formants 9b2dd5be4d834db8
15 stereo refused "type.width_mismatch"
15 bands 74755e4f730436ee
15 crest d917bbf168ed5585
15 loudness 312a0676bb131c82
15 onsets 2ee490475a8f7109
15 alias e2b7b32bc2dd05d1
15 bindings e05e77fedf4af597
15 arguments bcd81530c99ec21a
16 stdout 6d9beba6f3f07c0c
16 wav 9007d5d55c305e67
16 store:26ad75243b2f79c4c406c6f5bf4ec486 c418462b254b42d5
16 store:3c3e02ce6997bbbc519549daeed604f4 4ba0d4a350140222
16 store:721873409668da2d147b4a20e9a0d6e3 2a704778eabb22b0
16 store:7d2a34d66251dbf649d696726b2f90f8 85083b1a5bdc5bf4
16 store:7febd1dbb83c13997bf57fbce8e72f50 0d28f9abd7177441
16 lines refused "type.samples_in_closed_form"
16 spectrum 0ee76e47533a10a7
16 envelope d4fda186064032c6
16 derivative 96021b1e5f15d28f
16 ledger efb8c509ef569717
16 pitch 135c20238a75ceb5
16 formants 48462a1fe9e06f13
16 stereo refused "type.width_mismatch"
16 bands 4061eaaf25b936bc
16 crest 282fdcfbc9451b50
16 loudness 3d95c949a7ccae46
16 onsets 90baed5b283bf6bb
16 alias refused "engine.alias_needs_a_closed_form"
16 bindings 5f52bae6ae309701
16 arguments 978232e3e49fac6c
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
