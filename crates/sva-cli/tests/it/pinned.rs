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
0 store:3620e0612b5d84925fa1bd15f1b7716e f4ebc2177cc230d2
0 lines refused "read.lines_need_unwindowed_lines"
0 spectrum refused "read.lines_need_unwindowed_lines"
0 envelope refused "cast.left_algebra"
0 derivative 1bc92e818f2bdc1f
0 ledger bab16b78532317dc
0 pitch refused "read.lines_need_unwindowed_lines"
0 formants 38f4a37c497b6d65
0 stereo refused "type.width_mismatch"
0 bands d296ccdef14df607
0 crest ae95a217fe01c136
0 loudness e6a362bb961a8f90
0 score b67d3d3ccded8bee
0 alias ffd0b296eaeac57b
0 bindings cca84563df793651
0 arguments ac479335c0844369
1 stdout e17e3c4311440757
1 wav 26291e77cc78e829
1 store:90af903a105917870539e1943a4b72c7 4c6690b777e1520d
1 lines refused "read.lines_need_unwindowed_lines"
1 spectrum refused "read.lines_need_unwindowed_lines"
1 envelope refused "cast.left_algebra"
1 derivative 2f6bf49684a384e3
1 ledger da2a64d875d1c69d
1 pitch refused "read.lines_need_unwindowed_lines"
1 formants db049b90dc049f94
1 stereo refused "type.width_mismatch"
1 bands bc0d3ba2e34e221e
1 crest d4c9cab1b01a245b
1 loudness c24836bd41790e95
1 score 79f7c5ac79699710
1 alias 1cbd36f2ce620963
1 bindings cca84563df793651
1 arguments ac479335c0844369
2 stdout e17e3c4311440757
2 wav 968e2e9858e90ed9
2 store:29c7e17ebaef85e9ff784832c3108192 ef6a3d0cb8348f79
2 lines refused "read.no_spectral_sum"
2 spectrum refused "read.no_spectral_sum"
2 envelope refused "read.no_spectral_sum"
2 derivative refused "read.no_spectral_sum"
2 ledger 874e05dc33fe4563
2 pitch refused "read.no_spectral_sum"
2 formants b390ac97053f70bf
2 stereo refused "type.width_mismatch"
2 bands b294c5719c9fe82f
2 crest b1720ad620406d8f
2 loudness aa04962e07c4f5e2
2 score dacc45c5a24620ba
2 alias a367e6b444e75c9e
2 bindings cca84563df793651
2 arguments 639edc93953a6808
3 stdout 71f4f9a9b3cd9d30
3 wav 64318e943f03cc25
3 store:273d32d1fbc24d73b322a79899449566 f00765eeaa966470
3 lines ef35f29de2cc8837
3 spectrum d76d6d5aa41272bc
3 envelope 508733913bd67342
3 derivative 9685ab0be14bd4b4
3 ledger 1cdfc7f7172a3d74
3 pitch bc6a1749d2e897fa
3 formants 54dc893d1b7f3366
3 stereo refused "type.width_mismatch"
3 bands 2272ce539291a9b7
3 crest 9fae68afb06d5a49
3 loudness 9a9579bd56fa81e1
3 score 643f5fd4fdd83f3f
3 alias c0eeacde04390c3e
3 bindings f15293e1080c5049
3 arguments eb3ed92e88f435e1
4 stdout 5b546f69552c55f0
4 wav d8d646db74dc9cb2
4 store:7d17584ff33a1d5714e10c35ba698e01 43ea5446bc0dd2c9
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
5 store:52690610e792cbd9ec87410effce6ac0 abf31a2968447f9e
5 lines 65ef389a409cc7a3
5 spectrum 6b1a5e9ec974b0d4
5 envelope 8c1c2bc6da6a2f6d
5 derivative 09ebc5558a19d4d5
5 ledger 727b0eafa3c05099
5 pitch f5ce71dc646bdde3
5 formants 581049d0b3521d35
5 stereo refused "type.width_mismatch"
5 bands 0eca3fbcda9ca639
5 crest 27a91bd8023f6904
5 loudness f45e25ca544ef233
5 score ed15546ead69d8de
5 alias abbc574e635044c1
5 bindings cca84563df793651
5 arguments ac479335c0844369
6 stdout 9108e37d0b7327c6
6 wav 72bce77b8424e0c6
6 store:0aa625a325ad009e42736fb0b55da15e 853ab0424391199a
6 store:11c136c343d5887a356999a4fc86e8f7 e13517411beda609
6 store:14f6414df6fcdd24e3ba1efa6b601e02 bf0616d2c8c1fe85
6 store:2a812a178319a3d66188660d033eee56 cf94db766a7c0da8
6 store:2b1dfb572460ba3c3e8a08891ad3feda 41a7d92a5b32a53c
6 store:2cc1e056c0cd5c6013c986cb46ea8680 2063a05a83d4cdd7
6 store:57a6b91a9d6ee92c9f220a471c761060 f14836ef42f9b79f
6 store:6e22a9b54e73a673f2f7252c439f5d3c 27499a6683b85aec
6 store:6f9bcd032cb9677f874eba226b581950 3f73516e8e83df33
6 store:d18c8f806552ff35a1cabe95f566fe2e bd5a9310177d3735
6 store:d4b554f1adbd1e2708edf6cef39dd919 46837197ed8b21bd
6 store:dfbe3e234fa8c68c2717f78df327e90f 22818f14c9d4d831
6 store:e41ecc22b1d8d907ef9ba75bd804e1e7 da731480cd4b9701
6 store:e51a27e86bcbd36a2c884364cb358a1e 9d2713bec48731b1
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
7 store:48cc333eda6f48499d525bb87ae7a03c 4ec0b21457815098
7 lines refused "read.lines_need_unwindowed_lines"
7 spectrum refused "read.lines_need_unwindowed_lines"
7 envelope refused "cast.left_algebra"
7 derivative d910c16d47459556
7 ledger 685c6261e2752b66
7 pitch refused "read.lines_need_unwindowed_lines"
7 formants ec0b8751e0241093
7 stereo 863b17be723f9952
7 bands e0f3393595ce6203
7 crest f3b12fbafbb8cf9d
7 loudness 5dfa0194c31f6138
7 score a912872000d7358e
7 alias 3646019e5d05ee33
7 bindings b63bdcd55385e2fe
7 arguments 7c1d152f8c0e3714
8 stdout bb69bdf6de9adeb8
8 wav 083e3fdfa6fd2b2d
8 store:0435c2987fec5f6fa5f7bc60ed42587e a07f1b5887cb94ca
8 lines refused "type.samples_in_closed_form"
8 spectrum 1283b8abc3f8cd1b
8 envelope 318124550bf22b6d
8 derivative 7e5fe43cb629a9ad
8 ledger 544a33c60a492a40
8 pitch a1bb7d37afebe1e6
8 formants 59b0bcf9f4b47905
8 stereo be4b410523c86ec8
8 bands 1b92efe195aabde1
8 crest 511440016fbe6578
8 loudness 8a420d1668a8de02
8 score 95b4958e24cec744
8 alias refused "engine.alias_needs_a_closed_form"
8 bindings 5ce608c3a8b79a86
8 arguments b642f09606176b94
9 stdout e17e3c4311440757
9 wav 3a9ca1a53a5fa42a
9 store:3180ee7f5ba55875b73514cbc66c3107 8f365156f2e2869a
9 lines refused "read.lines_need_unwindowed_lines"
9 spectrum refused "read.lines_need_unwindowed_lines"
9 envelope refused "cast.left_algebra"
9 derivative d877ae4e9f8f299f
9 ledger 10db306f97c70e03
9 pitch refused "read.lines_need_unwindowed_lines"
9 formants fe7abeb30b3d2fc2
9 stereo refused "type.width_mismatch"
9 bands 10bf09fbaf30d5a6
9 crest 4f9afb40d9743f97
9 loudness f0cf9ce6ed9dea83
9 score ab15784764a304ba
9 alias 90dc4309d6c944cf
9 bindings cca84563df793651
9 arguments ac479335c0844369
10 stdout 5687cc05a27c2ec3
10 wav a9866247325c001f
10 store:6655845ef1b852c2dfba6667b7a17cb7 eb8effa2696aa495
10 lines refused "read.lines_need_unwindowed_lines"
10 spectrum refused "read.lines_need_unwindowed_lines"
10 envelope refused "cast.left_algebra"
10 derivative 89dc769fdd11eaeb
10 ledger 80da51e67a8e4884
10 pitch refused "read.lines_need_unwindowed_lines"
10 formants b14a0f0f0840f70f
10 stereo refused "type.width_mismatch"
10 bands b9d575d1d199b574
10 crest c917f5dbb97b8206
10 loudness a48211676dd3df72
10 score b12cce32f6f0074c
10 alias a0795ab7eaabaebd
10 bindings 978c10e7996dc9a1
10 arguments b5457effabd50fe5
11 stdout a05558226a6cba5a
11 wav c187b7408a55ebf7
11 store:49f4a6817fb359e1ca2669494fe63af2 77fa84e14953f438
11 store:d974ac29049427884f6c2e4f1edbb778 8e901f88f86afaca
11 store:dc788d72ef609cbfbd531912e37ffe4d 0fc72841c9b10311
11 lines refused "type.samples_in_closed_form"
11 spectrum cf56a78dc4f5c8f7
11 envelope ae401e82346d0e29
11 derivative 868c58ec33a7d8cb
11 ledger f5ee82dfce3fc38a
11 pitch e58bc2071e8f7a9c
11 formants f5143e22f5af09c2
11 stereo refused "type.width_mismatch"
11 bands 10dbdd593ce205f1
11 crest 83b03389a020426b
11 loudness cf9f60606fc30d24
11 score 7b5129f5b25e0c2e
11 alias refused "engine.alias_needs_a_closed_form"
11 bindings 193483886e1b97eb
11 arguments fbff5288f7442249
12 stdout 24a0009133ff612c
12 wav 3fc9f322aaf6fc5f
12 store:a754efa011f428d52010e3099bc14cb1 588228e59de6a430
12 store:b75918a010c072bb16f03119a7dcbf64 a3a994fd5b146879
12 lines refused "type.samples_in_closed_form"
12 spectrum e2e11d0d9027df24
12 envelope 881172fade4b59a6
12 derivative ae083f2d3b99ae73
12 ledger e455a51170bff138
12 pitch 44f1a64fd7d09886
12 formants 8471db8682ef34f8
12 stereo refused "type.width_mismatch"
12 bands b2dd2a0db715cda3
12 crest ccd71d9c7a289513
12 loudness 2c3149a83b17ba89
12 score a5f94c746a15d1bd
12 alias refused "engine.alias_needs_a_closed_form"
12 bindings 4a6686d8e212fcf1
12 arguments 58316c250a5fe29f
13 stdout 189201d106182ae6
13 wav 256b6dfecab93743
13 store:995a04465e3289abd658d5254fb949f3 630d607da44745f1
13 store:e8cfef4afb2349bb0fe8c2e3f1cea74f 04ac76af4e45c175
13 lines refused "type.samples_in_closed_form"
13 spectrum 9de94f4a71374cf2
13 envelope a97f21e9a7d4ff53
13 derivative 9e11c54b3bf1592d
13 ledger fe493576f5391745
13 pitch 17dc59f08adbd1af
13 formants ab92561b93be2c35
13 stereo refused "type.width_mismatch"
13 bands 293691741cf73df3
13 crest f6062011aec78547
13 loudness 4c404a9d18601ba5
13 score 5a6dbf8689c81d7f
13 alias refused "engine.alias_needs_a_closed_form"
13 bindings 0255ef5c70e74e5e
13 arguments 0adbeca7c937da37
14 stdout b6ad884dff66b63b
14 wav 0c1d8e0ad4b6a019
14 store:194d57ecf1b72fa62c9b17ec02129d13 e41da9892e20b152
14 store:32547f9d11c1e39ed00836d8ff4dbff2 bd3d222876fa3ca1
14 store:4d307b3a5d05232779527d608b4de3ad 73074bea782bd97f
14 store:52d1c7bcf24fd65556b3a64eddbd2e59 6c19462082538777
14 lines refused "type.samples_in_closed_form"
14 spectrum 8a02a5bbe6fb1896
14 envelope 8a05d06c94a4ebd2
14 derivative b295446119f7bca0
14 ledger 95b8ee72fa8ec4ca
14 pitch ae04a6c4fe518e61
14 formants c491402b7080017d
14 stereo refused "type.width_mismatch"
14 bands f41e10e1acc94e8d
14 crest a1fa1897a90b31eb
14 loudness 9b2bbea979a56dc7
14 score 81ba44e8bee61996
14 alias refused "engine.alias_needs_a_closed_form"
14 bindings 72871833b56a7987
14 arguments 566a07fc7cfce2ce
15 stdout ea658b80403f05b8
15 wav 82773cac15ff1cca
15 store:b5146c74a37c137f36a90b33af43d7d6 1c8b51ed3b3c11fe
15 lines refused "read.lines_need_unwindowed_lines"
15 spectrum refused "read.lines_need_unwindowed_lines"
15 envelope refused "cast.left_algebra"
15 derivative da1daf2367fd7eb8
15 ledger ce425ec6d31bd0d2
15 pitch refused "read.lines_need_unwindowed_lines"
15 formants a934d953276500d1
15 stereo refused "type.width_mismatch"
15 bands 453162b22108001c
15 crest 77a58410efb0e790
15 loudness 4adf8f9ad435e99c
15 score 0673880605028994
15 alias 6e0059b5f3196535
15 bindings 1b8e53e12cdbbcab
15 arguments b6603b7216e51daa
16 stdout 6d9beba6f3f07c0c
16 wav 9007d5d55c305e67
16 store:043708e3e666563747bf4fa06e8134c9 801b25cc3f955405
16 store:1523d3727e3560f4c087cc26bcdb92c1 0a87f5a9b424aba0
16 store:56d9a9664634a8707c8206366f7ace03 1a4ad1748dc22a2d
16 store:b00e0698a765f996175369b4040463f1 f8b074593263784a
16 store:e98c8e1e8b33053c9a7f3b9e64d00f51 cd58f46d7c4e3fcc
16 lines refused "type.samples_in_closed_form"
16 spectrum 4d4058d6097ddc90
16 envelope 43e9dc89486b18d8
16 derivative 3a46884beab051bb
16 ledger 6392e4ebc222e92f
16 pitch c165d69523e26d9f
16 formants 0abc90d4e0cf0088
16 stereo refused "type.width_mismatch"
16 bands 8f724102016e18cf
16 crest 765af78c04829b0a
16 loudness 75dfbb8d9266b72c
16 score cf37f0de964c37e2
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
