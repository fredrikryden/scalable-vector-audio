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
    "score",
    "alias",
    "bindings",
    "arguments",
];

/// One line per case and item, `<case> <item> <hash, or the codes it refused>`.
const PINNED: &str = r#"
0 stdout e17e3c4311440757
0 wav 665924a7a274d695
0 store:3620e0612b5d84925fa1bd15f1b7716e ff884fd063aa9943
0 lines refused "read.lines_need_unwindowed_lines"
0 spectrum refused "read.lines_need_unwindowed_lines"
0 envelope refused "cast.left_algebra"
0 derivative 1bc92e818f2bdc1f
0 ledger bab16b78532317dc
0 pitch refused "read.lines_need_unwindowed_lines"
0 formants 29f1d5ddc6d00369
0 stereo refused "type.width_mismatch"
0 bands f04d3b16247e5f19
0 crest a2c8c2e78d8f9cf1
0 loudness 8582303e957a8e75
0 score d96960107fecd4f7
0 alias 3bb4e45ef131405d
0 bindings cca84563df793651
0 arguments ac479335c0844369
1 stdout e17e3c4311440757
1 wav 26291e77cc78e829
1 store:90af903a105917870539e1943a4b72c7 e486b7fa427b50f1
1 lines refused "read.lines_need_unwindowed_lines"
1 spectrum refused "read.lines_need_unwindowed_lines"
1 envelope refused "cast.left_algebra"
1 derivative 2f6bf49684a384e3
1 ledger da2a64d875d1c69d
1 pitch refused "read.lines_need_unwindowed_lines"
1 formants fbaa5c1382ef6c79
1 stereo refused "type.width_mismatch"
1 bands b161e6ee73e55b4c
1 crest 7ca68c4a6796ad38
1 loudness 7071e3c0a828fe27
1 score f5178d99e46f5db1
1 alias 348141409e4cefb5
1 bindings cca84563df793651
1 arguments ac479335c0844369
2 stdout e17e3c4311440757
2 wav 968e2e9858e90ed9
2 store:29c7e17ebaef85e9ff784832c3108192 3cd3a5b9c477837b
2 lines refused "read.no_spectral_sum"
2 spectrum refused "read.no_spectral_sum"
2 envelope refused "read.no_spectral_sum"
2 derivative refused "read.no_spectral_sum"
2 ledger 874e05dc33fe4563
2 pitch refused "read.no_spectral_sum"
2 formants 936baae8b988c87f
2 stereo refused "type.width_mismatch"
2 bands 4e72baf77a1443f2
2 crest caf3127749b59aa3
2 loudness 23625b1c39ba3033
2 score 37921df57891b38a
2 alias 873578962fb721c3
2 bindings cca84563df793651
2 arguments 639edc93953a6808
3 stdout 71f4f9a9b3cd9d30
3 wav 64318e943f03cc25
3 store:273d32d1fbc24d73b322a79899449566 f18cd7bc7d2b5fcc
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
3 score 371994ee5977f165
3 alias 57b3de39acaa9baf
3 bindings f15293e1080c5049
3 arguments eb3ed92e88f435e1
4 stdout 5b546f69552c55f0
4 wav d8d646db74dc9cb2
4 store:7d17584ff33a1d5714e10c35ba698e01 d615f99e44169e0c
4 lines fba1b60d66f0321f
4 spectrum dba0d0713ae772d8
4 envelope c554c0bf3802fe1a
4 derivative b27c2b62c8a14697
4 ledger 27d7d5b8a6163962
4 pitch 23d8f7fe797d1fdd
4 formants fde0d1ee9f046299
4 stereo refused "type.width_mismatch"
4 bands f410abcf9fab400b
4 crest 1e7b0f8cf99beaf1
4 loudness 16d1e9220d30ef01
4 score d2376bfc2e54aff5
4 alias 36e530d2d79e46b1
4 bindings 5a6d3cadc8d1b9e7
4 arguments f74b8084fd1b914d
5 stdout e17e3c4311440757
5 wav 177aed0c28183ef5
5 store:52690610e792cbd9ec87410effce6ac0 0d3844afd44cffac
5 lines 65ef389a409cc7a3
5 spectrum 6b1a5e9ec974b0d4
5 envelope 8c1c2bc6da6a2f6d
5 derivative 09ebc5558a19d4d5
5 ledger 727b0eafa3c05099
5 pitch f5ce71dc646bdde3
5 formants 4b04788e05e26c65
5 stereo refused "type.width_mismatch"
5 bands 71dc94f150aac73e
5 crest c637ebb5a774e8ce
5 loudness b5fec56adb3ab3c8
5 score 397e504c8b14def6
5 alias 33a97127b994e8ca
5 bindings cca84563df793651
5 arguments ac479335c0844369
6 stdout 9108e37d0b7327c6
6 wav 72bce77b8424e0c6
6 store:0aa625a325ad009e42736fb0b55da15e 7f22ded8ab77b43e
6 store:11c136c343d5887a356999a4fc86e8f7 2055e96176a337a1
6 store:14f6414df6fcdd24e3ba1efa6b601e02 a10c53bea5fa6fb4
6 store:2a812a178319a3d66188660d033eee56 71120361e4bc1ebd
6 store:2b1dfb572460ba3c3e8a08891ad3feda ad4fe9b5b4e89580
6 store:2cc1e056c0cd5c6013c986cb46ea8680 c711cde7c6a2c601
6 store:57a6b91a9d6ee92c9f220a471c761060 1b8588905109966f
6 store:6e22a9b54e73a673f2f7252c439f5d3c 9737e0b409c99287
6 store:6f9bcd032cb9677f874eba226b581950 0e5750eea070fb4c
6 store:d18c8f806552ff35a1cabe95f566fe2e c73410db5fa71ee6
6 store:d4b554f1adbd1e2708edf6cef39dd919 39b950300dd0ea43
6 store:dfbe3e234fa8c68c2717f78df327e90f 6873865c1459d01f
6 store:e41ecc22b1d8d907ef9ba75bd804e1e7 d407455edbe2c9de
6 store:e51a27e86bcbd36a2c884364cb358a1e 58d09cfcf4f76f06
6 lines refused "type.samples_in_closed_form"
6 spectrum 82e214032e137ddc
6 envelope 87ffd87017994a9c
6 derivative 4a59aeeefeec93cb
6 ledger 47c19538102a198c
6 pitch 1b6bd302bb109bc9
6 formants e5756e914a7859ca
6 stereo refused "type.width_mismatch"
6 bands 7fa7653c5922631c
6 crest 392023cf91d0234e
6 loudness 1aa3d241079c6bbc
6 score c2ad9e8059f8f5c9
6 alias refused "engine.alias_needs_a_closed_form"
6 bindings f836b683b117cd32
6 arguments db19184c92016629
7 stdout 71454a132807dddb
7 wav 3372a403bff9f10a
7 store:48cc333eda6f48499d525bb87ae7a03c 0c4bda04efdbfa06
7 lines refused "read.lines_need_unwindowed_lines"
7 spectrum refused "read.lines_need_unwindowed_lines"
7 envelope refused "cast.left_algebra"
7 derivative d910c16d47459556
7 ledger 685c6261e2752b66
7 pitch refused "read.lines_need_unwindowed_lines"
7 formants 24f3497fa8f7c4bb
7 stereo f860dadfe8a098db
7 bands dd275113e108efb4
7 crest 80a2c9984aeceed9
7 loudness 5dfa0194c31f6138
7 score 169f67cc4fe87c77
7 alias da4a988d67554a8b
7 bindings b63bdcd55385e2fe
7 arguments 7c1d152f8c0e3714
8 stdout bb69bdf6de9adeb8
8 wav 083e3fdfa6fd2b2d
8 store:0435c2987fec5f6fa5f7bc60ed42587e 81921a3b9bd9659e
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
8 score 58e4936c1f4acb9c
8 alias refused "engine.alias_needs_a_closed_form"
8 bindings 5ce608c3a8b79a86
8 arguments b642f09606176b94
9 stdout e17e3c4311440757
9 wav 3a9ca1a53a5fa42a
9 store:3180ee7f5ba55875b73514cbc66c3107 153c0b83c44a5e90
9 lines refused "read.lines_need_unwindowed_lines"
9 spectrum refused "read.lines_need_unwindowed_lines"
9 envelope refused "cast.left_algebra"
9 derivative d877ae4e9f8f299f
9 ledger 10db306f97c70e03
9 pitch refused "read.lines_need_unwindowed_lines"
9 formants b1f285b9dc70a44c
9 stereo refused "type.width_mismatch"
9 bands d14f7a29e1719f86
9 crest 228a7120752264ca
9 loudness 6598bb4502ec4b88
9 score f09a3d9a2d219487
9 alias ed3d46ef4bf5c3c4
9 bindings cca84563df793651
9 arguments ac479335c0844369
10 stdout 5687cc05a27c2ec3
10 wav a9866247325c001f
10 store:6655845ef1b852c2dfba6667b7a17cb7 871a5e655a7a07d5
10 lines refused "read.lines_need_unwindowed_lines"
10 spectrum refused "read.lines_need_unwindowed_lines"
10 envelope refused "cast.left_algebra"
10 derivative 89dc769fdd11eaeb
10 ledger 80da51e67a8e4884
10 pitch refused "read.lines_need_unwindowed_lines"
10 formants d73ea9de84db8418
10 stereo refused "type.width_mismatch"
10 bands 9b12def01a17b6f2
10 crest 69d62d0f89507f8a
10 loudness 58c6c8ee21be16b7
10 score 5e776e1fafc013db
10 alias 975f208e3e3b0e06
10 bindings 978c10e7996dc9a1
10 arguments b5457effabd50fe5
11 stdout a05558226a6cba5a
11 wav c187b7408a55ebf7
11 store:49f4a6817fb359e1ca2669494fe63af2 4855692bd01ab8c4
11 store:d974ac29049427884f6c2e4f1edbb778 d7ac41564a09891a
11 store:dc788d72ef609cbfbd531912e37ffe4d fca8bffdc5b295a6
11 lines refused "type.samples_in_closed_form"
11 spectrum ded0d98afdabe1c2
11 envelope 246ad5cbcea17abc
11 derivative ebab06da808292f7
11 ledger 8646598869f22200
11 pitch 1e07dbc2872bd280
11 formants df041b146a7886be
11 stereo refused "type.width_mismatch"
11 bands c1c37f4949700ea9
11 crest 1123d4796a36e1ac
11 loudness 4be0c964d84a5b9b
11 score 3d381733c132a8aa
11 alias refused "engine.alias_needs_a_closed_form"
11 bindings 193483886e1b97eb
11 arguments fbff5288f7442249
12 stdout 24a0009133ff612c
12 wav 3fc9f322aaf6fc5f
12 store:a754efa011f428d52010e3099bc14cb1 5c08d4bdb6ba6e78
12 store:b75918a010c072bb16f03119a7dcbf64 06105b935e44745a
12 lines refused "type.samples_in_closed_form"
12 spectrum 825801a3e3836779
12 envelope 76ebff8097b673e5
12 derivative 637a4ec822194906
12 ledger 018f32977283f40c
12 pitch 503bb56214bbac0c
12 formants 6ffd998758cc10c3
12 stereo refused "type.width_mismatch"
12 bands 73903e5f0dfb4ddc
12 crest 06d266c4cdbaab1e
12 loudness 9aa069ac80d29f9d
12 score 21a115f2866a09e5
12 alias refused "engine.alias_needs_a_closed_form"
12 bindings 4a6686d8e212fcf1
12 arguments 58316c250a5fe29f
13 stdout 189201d106182ae6
13 wav 256b6dfecab93743
13 store:995a04465e3289abd658d5254fb949f3 18fda3844d31e332
13 store:e8cfef4afb2349bb0fe8c2e3f1cea74f 39d5abbd007a308a
13 lines refused "type.samples_in_closed_form"
13 spectrum 2a931325439f7f57
13 envelope 4fcbb7b86395ee78
13 derivative ae081a30d1a80ad2
13 ledger 73f0393695329d20
13 pitch af249d3fd8672241
13 formants c07d264d28652a7b
13 stereo refused "type.width_mismatch"
13 bands 53a04ea4eebf8119
13 crest 2e9ae18a2107f3d7
13 loudness 84ce4794d96a463c
13 score 4e3db6497d892243
13 alias refused "engine.alias_needs_a_closed_form"
13 bindings 0255ef5c70e74e5e
13 arguments 0adbeca7c937da37
14 stdout b6ad884dff66b63b
14 wav 0c1d8e0ad4b6a019
14 store:194d57ecf1b72fa62c9b17ec02129d13 d2e61eae8a8edf6e
14 store:32547f9d11c1e39ed00836d8ff4dbff2 2cd5ff02f727fb11
14 store:4d307b3a5d05232779527d608b4de3ad acf6806834f89cfb
14 store:52d1c7bcf24fd65556b3a64eddbd2e59 6b6aa19fe26712a7
14 lines refused "type.samples_in_closed_form"
14 spectrum 7cc8578303345247
14 envelope acf0c3401443618d
14 derivative 17553d9009052742
14 ledger 8627764b071b832a
14 pitch 77a89f607aa5db32
14 formants 1c54adf534c1860c
14 stereo refused "type.width_mismatch"
14 bands eda5bf90ef7aaad4
14 crest 9372ca6cc136731c
14 loudness 66e292a929985ca2
14 score 3143cdf2ac834110
14 alias refused "engine.alias_needs_a_closed_form"
14 bindings 72871833b56a7987
14 arguments 566a07fc7cfce2ce
15 stdout ea658b80403f05b8
15 wav 82773cac15ff1cca
15 store:b5146c74a37c137f36a90b33af43d7d6 2ec0bda95398a573
15 lines refused "read.lines_need_unwindowed_lines"
15 spectrum refused "read.lines_need_unwindowed_lines"
15 envelope refused "cast.left_algebra"
15 derivative da1daf2367fd7eb8
15 ledger ce425ec6d31bd0d2
15 pitch refused "read.lines_need_unwindowed_lines"
15 formants 183f5368c9619dee
15 stereo refused "type.width_mismatch"
15 bands 8ef3c118720f80ed
15 crest 9e817c422209e973
15 loudness 4adf8f9ad435e99c
15 score 0673880605028994
15 alias 02c73a3ebc5c38db
15 bindings 1b8e53e12cdbbcab
15 arguments b6603b7216e51daa
16 stdout 6d9beba6f3f07c0c
16 wav 9007d5d55c305e67
16 store:043708e3e666563747bf4fa06e8134c9 ed2ea1c576b8d6b8
16 store:1523d3727e3560f4c087cc26bcdb92c1 16ca052268dcaca0
16 store:56d9a9664634a8707c8206366f7ace03 da64376dd1b62d6e
16 store:b00e0698a765f996175369b4040463f1 bd377884dd05697b
16 store:e98c8e1e8b33053c9a7f3b9e64d00f51 e19be9607323929d
16 lines refused "type.samples_in_closed_form"
16 spectrum 5e8fcac4dca3d486
16 envelope 3754f67ead52869a
16 derivative b558f9f1c5c02c8f
16 ledger 64275943fe91acbc
16 pitch 99685dbfee49705d
16 formants 2227659ae84a9f3d
16 stereo refused "type.width_mismatch"
16 bands 76999a3d81fc8d3f
16 crest 1a8af9f9c594ab7b
16 loudness 32ccab095a454402
16 score 8e8806aa474f19cb
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
