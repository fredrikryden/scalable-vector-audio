// Concern: the `--help` page, each flag's default printed from its own constant | Non-concern: parsing those flags (args/), the JSON a subcommand answers (output.rs) | IO: () -> the page

use sva_core::DEFAULT_STORE_BYTES;
use sva_engine::{
    DEFAULT_BUFFER_SECS, DEFAULT_CELL_SECS, DEFAULT_FLOOR_DB, DEFAULT_FRAME_SECS,
    DEFAULT_LEDGER_DEPTH, DEFAULT_MAX_PAIRS, DEFAULT_MAX_PEAKS, DEFAULT_MODULATION_WINDOW_SECS,
    DEFAULT_NOVELTY_WINDOW_SECS, DEFAULT_OVERSAMPLE, DEFAULT_PERIODICITY_WINDOW_SECS,
    DEFAULT_SAMPLE_RATE, DEFAULT_SMOOTH_SECS, PSYCHOACOUSTIC_V1, READINGS, Reads,
    self_similarity::MAX_CELLS, spectrum,
};

/// Read off the constants the parser itself defaults to, so a printed default cannot drift
/// from the one a render actually uses.
pub fn help_text() -> String {
    let bits = PSYCHOACOUSTIC_V1.precision_bits;
    let store_gb = DEFAULT_STORE_BYTES >> 30;
    let representations = representations();
    let (min_frame, max_frame) = (spectrum::MIN_FRAME, spectrum::MAX_FRAME);
    format!(
        r#"USAGE:
  sva-cli (render | analyze | lint | trace | builtins | outline | new) [arguments]

DESCRIPTION:
  A composition is a directory of node files, each one closed-form expression in
  `t` or `f` on one line, or a score placing notes in time. sva-cli reads that
  composition and prints what it is and what it sounds like, as JSON on stdout.
  `render`, `lint` and `trace` read the current directory; `new` writes beside it
  and `analyze` reads a file.

SCORES:
  A score is a `.jsonl` file: JSON Lines, one object per line, sorted by `at`;
  `@score/drop` reads `score/drop.jsonl`, and a ref never writes `.jsonl`. A
  file with no extension opening with `{{` refuses with the rename. Line 1 is
  its annotation, the doc comment's four fields, which `lint` holds it to:
    {{"annotation": "Models: ... | Neglects: ... | IO: (t) -> amplitude | Tags: ..."}}
  Every other line is one kind, told by its keys, and may add `"comment"`:
    event    "at", "node", "deg", "len" and the node's parameters: plays it
    mark     "at", "mark": a point `mark(name)` reads in this file's times
    comment  "at", "comment": a note to the reader at a time
    move     "at", "lane", "to", and "node", "len", "shape" where written
  A score lowers to equations, the sum of its events in written order, each
  its node shifted to its `at`; marks, comments and moves sound nothing.
    at     a time string, "0.25b" or "0.5s", or arithmetic over times, "2b +
           1b/3" for a triplet; a bare number has no unit and refuses. A file
           counts every time in bars or every one in seconds. "mark(verse) -
           2b" is a position, "mark(drop) - mark(build)" a duration.
    node   an `@ref` from the score's own directory: a sound, an equation or
           another score. On a line placing a score `len` crops it there and
           nothing else binds; a `node` list mixing scores and sounds refuses
           `len`.
    deg    a whole scale degree counted from 1, the tonic: sugar for "f0":
           "deg(n)". In a seven-row scale 8 is the tonic an octave up, 0 the
           seventh below and -6 the tonic an octave down. A chromatic note
           binds "f0": "deg(7)*1st"; a line writing both refuses.
    len    a time as `at` takes, bound as the node's `len`.
    other  binds the node's parameter of that name: a number, or a constant in
           a string, "0.03s", "-6db", "900hz", "A1", "0.8*0.9", "deg(5)/2".
  Any field but `comment` may be a list. A line listing N items is N events,
  moves or marks, item i of every list going to the i-th and a single value to
  all; lists of different lengths refuse. A chord is a `deg` list on one `at`:
    {{"at": "0b", "node": "@../instruments/keys", "deg": [1, 3, 5],
      "vel": [0.8, 0.5, 0.6]}}
  `deg(n)` is the tonic in `variables/key` times degree n's ratio in the scale
  `variables/scale` names, read where the line sits, in a score or in any
  equation. A scale is one the language names, as `minor` or `dorian` (`sva-cli
  builtins` lists each with its cents), or a file of one row per degree in cents
  above the tonic, ascending, the last the period, named by a ref: a file
  `scales/custom` holding 200 400 500 700 900 1100 1200 is the major scale. A
  value reading `t` or a `@ref`, a field the node does not read, a parameter it
  reads that nothing binds, and a `deg` with no scale refuse before any audio,
  each naming the score's file, line and column.

  A lane is one value moving through the song: a node's parameter, a move
  naming "node" and "lane": "cutoff", or a variable under `variables/`, a move
  naming "lane" alone: "energy", "key", "scale", "bpm" (never "meter"). A move
  with no `len` sets the lane at `at`; with "len" and "shape", `lin` or `exp`,
  it ramps there from the value it holds:
    {{"at": "24b", "node": "@../instruments/saw", "lane": "cutoff",
      "to": "9khz", "len": "8b", "shape": "exp"}}
    {{"at": "72b", "lane": "key", "to": "key*2st", "comment": "a tone up"}}
    {{"at": "92b", "lane": "bpm", "to": "bpm*8/9", "len": "4b", "shape": "lin"}}
  A `to` naming the lane means its home value, the node's default or the
  variable's file, so it follows `--set`. A lane reaches every instance of its
  node, from a score or an equation, inside a wrapper too, unless the line or
  an invoking equation passes that parameter, which wins. Where it holds one
  value over an instance's span it binds that number; where it moves there the
  instance reads it at its song time, and where the engine refuses a moving
  value, such as `sat`'s `drive`, the move refuses as `bad-lane`: pass a
  constant, move the lane between notes, or close the sound's tail with a crop.
  A score's times ride the tempo curve the `bpm` lane draws; `b` in an equation
  stays a bar of `variables/bpm`. A song places its sections at marks:
    {{"at": "0b", "mark": "intro"}}
    {{"at": "mark(intro)", "node": "@intro"}}
    {{"at": "mark(intro) + 8b", "mark": "verse"}}
    {{"at": "mark(verse)", "node": "@verse"}}
  `concat` and `repeat` place parts on the curve too: `repeat(@score/riff-1b, 4)`.
  An equation reading a score at `t - d`, an echo or a send, hears the song as
  it was d earlier: its lines keep their song time, and a move however many
  reads reach it is one move.
  A control score is one no sounding score places, read by an equation as a
  signal: `@score/song(t) * (1 - @score/pump(t))` ducks the song under it.
  `sva-cli outline @score/song` prints a song's marks, tempo, lanes, events and
  size. Lint and render warn of a line passing a parameter whose lane ramps
  under it (`line-overrides-lane`), and of a lane no instance reads
  (`lane-reaches-nothing`).

RESERVED VARIABLES:
  Four nodes under `variables/` are reserved: a ref reads one like any node,
  and a bar or a degree reads one by name:
    variables/bpm     a number, the beats per minute: 120
    variables/meter   beats per bar over the note: 4/4
    variables/key     the tonic `deg` counts from, a note or hertz: C3
    variables/scale   the scale `deg` counts in, a built-in's name: major, or a
                      ref to a scale file: @../scales/custom
  A `b` literal or a file's `-2b` span reads bpm and meter; a `deg(n)` reads key
  and scale. Before anything renders, each read is replaced by its value, by one
  pass for all four: a bar becomes `n*(beats*60/bpm)` seconds and `deg(n)`
  becomes `@variables/key` times degree n's ratio, the tonic still a ref. A
  read with no value to replace it refuses as `bad-tempo` or `bad-scale`, as
  does a variable holding no such value, such as bpm without meter, or a
  scale name no built-in has, which refuses listing every built-in. A stream's
  edit to a variable, or to the scale file it names, replaces every read of it
  again. A node elsewhere named `bpm` is an ordinary node. Any other file under
  `variables/`, as `variables/energy` holding 0.5, is a variable of the
  composition's own: a ref reads it, `--set` sets it, and a score's moves move
  it, as they move the reserved ones but `meter` (SCORES).

RENDER:
  sva-cli render '<expression>' --representation <list> [--until '<condition>']
                 [--bits <n>] [--rate <hz>] [--threads <n>] [--cache <path|none>]
                 [--set <name>=<text>]... [--confirm]

  Renders one expression, in the grammar a node file's body uses, and prints one
  reading per representation under `data.representations`. `@path` reads a node
  from the current directory and `@/abs/path` one anywhere; there is no default
  target. Every reading states its `source` (exact or measured), the
  conformance profile it ran under, and its rate.

  The target's own ref may read an interval: `@piano4([0, 2b], f0=C4,
  vel=0.5, release=1s)`, where `release` is the instrument's own parameter. Its ends are the language's literals (`s`, `ms`, `b`,
  `sp`, or a bare `0`); `[a, inf)` and `[a,)` leave the end open; the other
  arguments are the node's own named ones. The start trims the output alone:
  history before it is still computed, so a loop or a filter carries the state
  it had there. With no interval the render starts at 0, earlier only where a
  crop reaches before it, and its end is open. Inside an expression a window is
  a `crop`. A target that samples its one ref read, `sample(@piano4([0, 2b],
  f0=C4))`, reads that ref's interval and binds and the same node values.

  Every node is computed over its support met with what reads it, and nowhere
  else: outside its support a node is exactly zero. A closed interval renders
  exactly its length, nothing cut. An open one ends where the root's support
  does: a crop's end, the exact underflow of an `exp`, a clamp holding a factor
  at zero; or earlier, where a proven upper bound on the root's magnitude stays
  under the profile's silence threshold (-120 dBFS) from a sample on. A root
  with no such bound (a held sine, a loop) is never cut. The label's `pruned`
  states the threshold as `db` and the root's cut, `from` the sample the render
  is treated as silent from. A stream's term leaves `@notes` where its support
  ends. An open interval over a root whose support never ends and is never cut
  (a held sine, a physical solver) refuses as `render.no_end`. A short-time
  transform reads its input whole, and refuses one with no end as
  `engine.unbounded_extent`. A form in `f` with no dual in `t` has no samples
  and refuses as its dual does, `cast.left_algebra`.

  A render is a stream pulled to its end: a closed form, a short-time
  transform and what either reads are computed over their whole extent first,
  and every other node block by block. `--until '<condition>'` stops the render
  at the first sample the condition holds at, or at the interval's end,
  whichever is first. It is checked as each block is pulled, so no node driven
  block by block runs past the block it holds in. A condition compares (`<`,
  `<=`, `>`, `>=`) `t`, `envelope(t)` (the RMS over every channel of the
  `envelope` representation's frame holding `t`, framed by its own `frame`
  where one is asked) and literals, joined by `and`/`or`.

  `--representation <list>` takes a comma list of readings and may repeat. Each
  is a call in the language's own syntax, its options its named arguments:
  `spectrum(peaks=8, frame=50ms)`, `ledger(depth=3, brief=1)`. A `frame`,
  `window`, `cell` or `smooth` is a time as a score writes one, in seconds,
  `50ms`, or in bars, `1b/8` (`1/8b` is one over eight bars), a bar lasting
  what the render's tempo says; where none is known, as in `analyze`, bars
  refuse as `tempo.invalid_bpm_meter`. An entry
  followed by `=<path>` writes a file instead, listed under `written`:
  `samples=<path>.wav` writes audio, encoded as `--bits` says; any other path
  takes the reading's JSON, uncapped. A path that already holds a file refuses
  unless `--confirm` is written. `samples=alsa:<device>` plays the samples on
  that ALSA device (`alsa:default`, `alsa:hw:1`, `alsa:plughw:1,0`) as each
  block is pulled, in real time, through `aplay` from alsa-utils; Linux only,
  anywhere else it refuses as `destination.unsupported_platform`. It sends
  32-bit floats, or integer PCM where `--bits` is 16 or under. A device that
  cannot take the format refuses with aplay's reason, and one aplay says plays
  another rate than the render's stops there as `destination.rate_inexact`.
  `samples(buffer=<t>)` is how much the device holds ahead of what it plays.
  Once it has played out, `data.played` states the device, its seconds, rate,
  channels, format, the buffer asked and the one granted, and every underrun
  aplay reported, with what it cannot see.
  `lines`, `atoms`, `derivative`, `bindings`, `arguments`, and a closed form's
  `spectrum`, `envelope` and `pitch`, read the expression and ignore the range.

  A closed form's `spectrum` and `pitch` are its exact lines, so a term under a
  crop or an envelope, which has a width and is no line, refuses them; so do
  `spectrum(frame=)` and `envelope(frame=)`, as a closed form has no frames. To
  measure any of them frame by frame, a reader samples the target's ref,
  `sample(@x([0, 1s]))`, and an author writes `sample(...)` inside the node.
  A closed form with no exact envelope, such as a crop of a decay, refuses
  `envelope` with the reason rather than measuring it unasked; `sample(...)`
  asks for the measured one. A measured `spectrum` is one spectrum: whole
  windows across the range, averaged, the first at its first sample and the last
  ending at its last, evenly at most half a window apart; a window is `frame`
  rounded up to a power of two samples, at least {min_frame}, or without `frame`
  the range's length so rounded, at most {max_frame}. `pitch` is framed, one entry
  per `frame`; for the spectrum at one instant, read a range one frame long.
  A measured `envelope` is framed too, and reads every channel: a frame's
  `rms` is the root mean square of all its channels' samples together, its
  `peak` the largest magnitude in any channel. `onsets` reads every channel
  as well: a frame's flux is the sum of each channel's, and a strike's energy
  that of all channels' samples together, so opposite phases cancel nothing.
  Every other measured reading but `stereo` reads every channel in power, as
  `envelope` reads a level: a spectrum's power per bin (`spectrum`, `pitch`,
  the partials `dissonance`, `voice_leading`, `tonal_distance`,
  `inharmonicity` and `self_similarity` pick), a formant frame's
  autocorrelation and each auditory band's energy (`bands`, `periodicity`,
  `modulation`, `self_similarity`) are each channel's averaged, so a channel
  copied to two reads as itself; `crest` sets any channel's peak against the
  RMS of all of them together. A closed form's exact readings read every
  channel in power too: a partial's squared amplitude (`pitch`, `dissonance`,
  `voice_leading`, `tonal_distance`, `inharmonicity`) is summed over the
  channels at its frequency and divided by their count, and a spectrum of more
  than one channel lists each frequency once, its amplitude the root of that
  mean power, real, as channels share no phase.

  `--bits <n>` is the precision every sample is written to, from 2 to 52: the
  point where a series is truncated, and the encoding of a `.wav`, integer PCM
  at n bits up to 16 and 32-bit float above. `--rate <hz>` (default
  {DEFAULT_SAMPLE_RATE}) is the one rate a render samples at: the target is read
  at its instants, and every node at the instants its reader asks for. A
  closed form is exact at any instant, and so is a continuous loop, one
  constant delay at a gain under 1 over a closed form such as
  `x + 0.5*self(t - 17ms)`, which is its series. A filter, a discrete loop, a
  solver and `rand` step at the rate asked for, where `1sp` is one step, so an
  `sp` count or `self[idx(t) - 1]` means one sample at whatever rate is asked.
  `rand` draws once per step, keyed by that step's index; a key between steps
  reads the step nearest it, ties to even.

  `--set <name>=<text>` renders as if the variable `variables/<name>` held
  `<text>`, and writes no file: `--set key=G3` transposes every degree to G3,
  `--set scale=dorian` changes mode (a scale file is a ref written from
  `variables/`, as the file holds it: `--set scale=@../scales/custom`), `--set
  bpm=140` moves every bar, `--set meter=3/4` regroups them and `--set
  energy=0.8` sets a variable of the composition's own. It sets a lane's home
  value: a move whose `to` names it follows. It may repeat, once per variable: a
  variable set twice refuses as `render.set_twice`, a name with no file under
  `variables/` that is no reserved one as `render.set_unreserved`, listing the
  variables there, and the file's path, `variables/key=G3`, as
  `render.set_path`, naming the short form.

  `--threads <n>` (default: every core the machine reports) is the most
  threads a render computes on; `1` computes every value in turn on one.
  Values none of the others reads, and the blocks of a value that holds no
  state, compute apart, to the same bits on any number of threads.

  A discrete loop reads its own past only by index; `self(t - d)` in one
  refuses as `type.discrete_self_at_time`, naming what made it discrete. A
  read `@x(k*t - d)` steps `x` at `k` times the step, every input it reads,
  `sp` and `idx` with it, and rounds `d` to the nearest sample of that step,
  ties to even: shifts are rounded to the nearest sample so placements share
  one cached value; timing is exact to half a sample, and the label's
  `moved_s` states the most any read moved. A node is computed once
  per step however many reads ask for it. A node that holds state read at a
  time that moves refuses as `type.stateful_warp`, naming what
  holds its state; `@x[idx(...)]` reads its nearest step instead, as `p[idx(...)]`
  does a signal passed in as parameter `p`, and any `idx`, as
  `idx(t - 5ms - 2ms*sin(2*pi*t))`, is read sample by sample. An edit to a
  stream plays from the next sample on.

  A render keeps in a store on disk its target and, under it, each node of
  samples that holds state (a filter, a discrete loop or a solver in it) or
  that two or more nodes read. Nothing else is kept. A closed form, `sample`
  of one and `rand` have a value at any instant and are computed where read.
  A node without state that one node alone reads, however many times, is
  computed again by that reader from what the node itself reads: no other
  reader could ask for it, and without state it restarts at any sample, where
  a filter or a loop must run from its start. So a stage worth keeping is a
  node of its own that ends in a filter or a loop, or one two nodes read. A
  kept value is named by what it computes: never its file's name, directory,
  comments, spacing, or the order its named arguments are written in, so a
  renamed or moved copy, a constant lifted into a default, an expression split
  into a file of its own or written back inline, and two addends swapped all
  read one entry. The next render of anything that reads the same value reads
  it back, bit for bit, and computes nothing under it. The store is on by
  default, at `$XDG_CACHE_HOME/sva`, else `~/.cache/sva`; `--cache <path>`
  moves it (a `/dev/shm` path keeps it in memory) and `--cache none` turns it
  off. It holds at most {store_gb} GB, the least recently used values going
  first, and a store another build of sva-cli wrote is emptied when opened. A
  render stages what it computes beside the store as it goes, and the store
  itself is written once, when the render ends, fails, or is stopped by SIGINT
  or SIGTERM.

  `ledger` prints one row per node under the target. A row's `share` is the
  part of its reader's own energy that row accounts for, so one reader's refs
  sum to 1; a ref no addend isolates, such as one factor of a product, and
  every row under a silent reader, which has no energy to share out, print
  `null`. A ref carrying a share reads as the addend it gave its reader, its
  `rms`, `peak` and `clipped` too: in `2*@a`, `a`'s row is `a` doubled, and in
  `@a - @a` it is silence. Each ref carrying a share is collapsed once on its
  own, so a ledger costs one collapse per attributed ref beyond the render,
  and `depth` bounds how many, counted in refs from one file to another. Every
  row is computed whole over the range, however late a reader reads it.
  `brief=1` keeps only the rows that clipped, `skim=1` drops the wider fields.

  `arguments` renders nothing: for every instance under the target it prints
  each builtin call's named arguments as the numbers the call was lowered
  with, a solver's whole parameter set with `written: false` on each default
  it filled in, and the operand each `min`/`max` `chosen` where a call folds
  one to a number: in named arguments and a solver's, modal bank's or
  `noise`'s positionals. One inside a filter's or cast's positional or
  `rand`'s key or seed is not listed. `at` spans are bytes of the instance's
  own body, as `sva-cli outline` counts them.

REPRESENTATIONS:
  Each one a `--representation` call, its arguments in parentheses, every
  default under DEFAULTS. `exact` is read off the written math, `measured` off
  samples.

{representations}

ANALYZE:
  sva-cli analyze <file.wav> --representation <list> [--confirm]

  Runs the same readings over a whole external `.wav` at its own rate, never
  resampled. Only the readings a buffer answers alone apply, each listed under
  REPRESENTATIONS as `measured` or `exact off a closed form, measured off
  samples`; the rest need the graph behind it.

LINT:
  sva-cli lint ['<expression>'] [--format <json|text>]

  Checks the current directory without rendering a sample. With no target it
  checks every file under its own rules, and types every entry point, which
  types every file it reaches. With a target, in render's grammar, it checks
  the files that target reaches, and prints the interval a render of it reads.
  Either refuses a type error as `render` does.

  A score's annotation is its doc comment, under the same codes, and its lines
  no body, so the length checks pass it by; a file of two or more lines that is
  neither a score nor a scale refuses as `multiline-body` before any check
  runs. Each song lint types prints its warnings beside the findings.

  Every check prints one `data.diagnostics` item. `warning` exits 0, `error`
  exits non-zero, so branch on the verdict and never on whether the array is
  empty. No flag downgrades an error.

  error    missing-comment       no `;` comment line
           multiline-comment     more than one
           malformed-comment     not four ` | ` fields, `Models:` `Neglects:`
                                 `IO: <in> -> <out>` `Tags: <tag>[, <tag>...]`
           long-comment-block    a `;` block over 1000 characters, the line-1
                                 doc comment's own run exempted
           long-expression-body  a body over 10000 characters, a backstop rather
                                 than a complexity budget
           literal-sample-rate   a written rate where `sp` belongs
           arity                 a builtin called without an argument it reads,
                                 or with one it does not, as `rand(seed=k)`
  warning  key-is-not-a-pitch    `variables/key` holds neither a note name nor
                                 a number of hertz
           parameter-has-no-default
                                 a node reads a parameter no `name = value` line
                                 binds, so a bare `@node` cannot play
           support-never-ends    a node's support, cut where its bound falls
                                 under the silence threshold, has no end for a
                                 bare render to stop at

TRACE:
  sva-cli trace <node|expression>

  Prints one node's position without rendering audio: what it reads (`down`, one
  hop), everything that reads it (`up`, transitively to an entry point), each
  beside the expression doing the reading, the node that made it discrete, and
  the feedback loop it sits in, if any. A score is shown as its line count, its
  first and last `at` and the nodes it plays, never the sum it lowers to; every
  entry point's song is lowered first, each at song time 0. An interval the
  target reads is ignored.

BUILTINS:
  sva-cli builtins

  Prints the whole callable and syntactic vocabulary: every builtin with its
  arity and named arguments, each argument's `meaning`, `unit`, the model
  `part` it sets and whether it `moves` with `t` (a filter's cutoff, q and gain;
  every other named argument is one number, refused when it names none), each
  positional's meaning where the model states one, unit suffixes, the note-name
  grammar, the scales `variables/scale` may name with their cents, reserved
  identifiers, special call shapes, and what has no operator at all.

OUTLINE:
  sva-cli outline <expression>

  Prints the parse tree the engine builds from one expression, each node with
  the byte `span` it was written in: calls by `name` with positional and named
  `args`, operators by `op`, refs by `path` with their `binds`, refs and `self`
  by how they `read`, `time` or `index`, literals by `value` and `unit`, names
  by `name`. A node the parser supplies itself, the `0` of a prefix minus or the
  `t` of a bare `@ref`, has `written: false`. Reads no composition. Score text,
  JSON opening with `{{`, prints `score` instead of `outline`: one item per item
  a line lists, with its `kind`, `line`, `span`, `at` with marks resolved, and
  its `node`, `deg`, `args`, `mark` or `lane`, `to`, `len` and `shape`.

  sva-cli outline @score/song

  Run in a composition, outlines that score as the song a render of it is:
  `marks` in bars and seconds, the `tempo` curve's pieces, each lane with its
  home and every move resolved with its placement chain, how many instances it
  reaches and how many see it move, and each placed event once, however many
  reads reach it: its song time `at` (where it reads lanes, key and scale), when
  it first `sounds` (later by the least, over the reads reaching it, of the
  delays of the `t - d` reads of a score above it), its `len` in song time where
  its line writes one, node, pitch in hertz and where each parameter comes from
  (`line`, `lane` or `default`), and the `size`: placed events and the engine's
  instances beside its limit.

NEW:
  sva-cli new <name> [--idempotency-key <key>]

  Writes a starter composition at ./<name>, and refuses if that directory
  exists. `--idempotency-key <key>` records the key beside the composition, so a
  retry under the same key succeeds identically while the tree still holds what
  was written. Any other key, or an edited tree, refuses.

EXAMPLES:
  sva-cli new song1 && cd song1
  sva-cli render '@master' --representation samples=/tmp/song1.wav
  sva-cli render '@master([0, 8b])' --representation 'ledger(depth=2),loudness'
  sva-cli render '@voice/note([0, inf), f0=C4, len=2s)' \
    --until 'envelope(t) < -60db and t > 1s' --representation samples=/tmp/note.wav
  sva-cli render 'sample(@chord/home)' --representation 'spectrum(peaks=8)' --rate 48000
  sva-cli render 'sample(@voice/note([0, 2s], f0=C4))' --representation pitch
  sva-cli render '@master' --set key=G3 --set bpm=140 --set scale=dorian \
    --representation samples=/tmp/song1-g.wav
  sva-cli render '@master' \
    --representation 'samples(buffer=100ms)=alsa:default,loudness'
  sva-cli lint
  sva-cli lint '@master'
  sva-cli trace score/phrase
  sva-cli outline @score/phrase
  sva-cli builtins

OUTPUT:
  {{"status": "success", "data": {{"target": "@master([0, 8b])", "sample_rate":
  {DEFAULT_SAMPLE_RATE}, "bits": {bits}, "profile": "psychoacoustic-v1",
  "interval": {{"start_secs":
  0, "end_secs": 16}}, "label": {{...}}, "written": {{"items": [...]}},
  "representations": {{"ledger": {{...}}}}, "diagnostics": {{"items": []}}}},
  "meta": {{"request_id": "req_...", "timestamp": 1700000000}}}}

  `interval` is null where no reading read samples. An error adds "error":
  {{"code", "message", "details": {{"count", "codes"}}}}. Success or error,
  every response carries every finding in full at "data": {{"diagnostics":
  {{"items": [{{"code", "severity", "message", "location", "help"}}],
  "pagination": {{"count", "has_more", "next_cursor"}}}}}}, empty where it found
  none.

  Every collection carries that same {{items, pagination}} pair. `count` is the
  whole reading's, `has_more` says `items` holds less than that, and
  `next_cursor` is the interval start, as `0.5s`, the rest is read from. A
  reading written to a file is written whole, uncapped. This page is an
  envelope of its own, at "data": {{"help"}}.

DEFAULTS:
  --rate <hz>          the one rate a render samples at, and `1sp`.
                       Default {DEFAULT_SAMPLE_RATE}.
  --bits <n>           the precision every sample is written to. Default {bits},
                       the `psychoacoustic-v1` profile's own.
  ledger(depth=<n>)    how deep below its target a `ledger` walks.
                       Default {DEFAULT_LEDGER_DEPTH}.
  (peaks=<n>)          peaks a `spectrum` keeps, notes a `pitch`, formants a
                       `formants`, stripes and novelty peaks a `self_similarity`
                       lists per view. Default {DEFAULT_MAX_PEAKS}.
  alias(oversample=<n>) the multiple `alias` re-renders at to hear what folded.
                       Default {DEFAULT_OVERSAMPLE}.
  (frame=<t>)          the step a framed reading advances by, in seconds.
                       Default {DEFAULT_FRAME_SECS}, except `spectrum`, which sizes
                       its own transform to the range unless it is set.
  (floor=<db>)         the amplitude, as `-40db`, under which a harmonic reading
                       or `self_similarity` drops a partial, and under which
                       `self_similarity` counts a band level as zero.
                       Default {DEFAULT_FLOOR_DB}db.
  voice_leading(norm=<l1|l2>) how a voice leading's moves add up. Default l1.
  dissonance(pairs=<n>) the most partial pairs one frame sums before it refuses.
                       Default {DEFAULT_MAX_PAIRS}.
  periodicity(window=<t>) the span of onset strength one window autocorrelates,
                       in seconds, at least twice the shortest period it reads;
                       it reads periods up to half its span.
                       Default {DEFAULT_PERIODICITY_WINDOW_SECS}.
  periodicity(curve=<0|1>) `1` adds every window's height at every lag.
                       Default 0.
  modulation(window=<t>) the span of band envelope one window transforms, in
                       seconds, at least twice the fastest rate's period; it
                       reads rates down to two periods per span.
                       Default {DEFAULT_MODULATION_WINDOW_SECS}.
  modulation(bpm=<n>)  a tempo, as `140`, adding each rate's period in beats.
                       Default none.
  self_similarity(cell=<t>) the side of one matrix cell, in seconds, a whole
                       number of frames holding a sample of the band
                       envelopes; the target may span at most
                       {MAX_CELLS} cells. Default {DEFAULT_CELL_SECS}.
  self_similarity(smooth=<t>) the length each diagonal is averaged over
                       before stripes are picked, in seconds.
                       Default {DEFAULT_SMOOTH_SECS}.
  self_similarity(window=<t>) the whole width of the novelty kernel, in
                       seconds, at most {MAX_CELLS} cells a side.
                       Default {DEFAULT_NOVELTY_WINDOW_SECS}.
  self_similarity(curve=<0|1>) `1` adds every cell's chroma and band levels and
                       each view's matrix. Default 0.
  bindings(node=@<path>) the instance whose bindings are read; required. An
                       instance is named exactly as the render spells it; any
                       other spelling refuses, listing its file's instances.
  samples(buffer=<t>)  how much an `alsa:` device holds ahead of what it plays,
                       in seconds. Default {DEFAULT_BUFFER_SECS}; only a device
                       has one.
  ledger(brief, skim)  `0` unless set `1`.
  --format <json|text> how `lint` prints its findings: the envelope, or one
                       terminal line each, colored where stdout is a terminal.
                       The same objects either way. Default json.
  --cache <path|none>  where the store lives. Default `$XDG_CACHE_HOME/sva`, else
                       `~/.cache/sva`, holding at most {store_gb} GB; `none` keeps
                       no store. A store that fails fails no render: it answers
                       as without one, warned why in `diagnostics`.
  --set <name>=<text>  a variable read as `<text>` for this render.
                       Default none: each is read off its file.
  --confirm            replaces a destination that already holds a file. Without
                       it a path already taken refuses as `conflict` and nothing
                       is written.

ENVIRONMENT:
  SVA_LOG=debug        `render` logs, once it ends, what it asked of each value
                       to stderr; stdout and the samples are unchanged. Any
                       other value, or none, logs nothing. One line per second
                       of output, then one per node and a total:
                         sva-cache pass hit=0 miss=190 ... reused=0 cum-hit=0.0%
                         sva-cache t=1.022s hit=1 miss=2 ... cum-hit=0.5%
                         sva-cache node hit=1 miss=0 ... reused=3 <node>
                         sva-cache total hit=1 miss=198 ... hit-rate=0.5% ...
                       `pass` is what was looked up before the first block.
                       Each value is looked up once: a `hit` was answered by
                       the store, `prefix` found a stored run up to a switch,
                       `miss` computed it, `new` held it in memory, where the
                       store takes what it keeps from. Each read of a value
                       past its first is `reused`.

EXIT CODES:
  0  success (error.code absent)
  1  internal_error (a destination could not be written)
  3  validation_error (bad arguments, a composition that failed to parse, or a
     render the engine refused)
  4  conflict (a name `new` would overwrite, or a destination already holding a
     file, without `--confirm`)
  24 not_found (a `.wav` file, or a node this composition does not define)

VERB ALIASES:
  validate = lint, list = builtins, create = new, show = trace, from the `cli`
  standard's own verb list. `render` and `analyze` take a reading, which that
  list has no word for, so they keep their own names.

SEE ALSO:
  sva-cli --version    Show version information"#
    )
}

/// One entry per reading, off the one table: its call, what it answers off, and its line.
fn representations() -> String {
    READINGS
        .iter()
        .map(|row| {
            let keys: Vec<&str> = row.takes.iter().map(|arg| arg.key()).collect();
            let call = match keys[..] {
                [] => row.name.to_string(),
                _ => format!("{}({})", row.name, keys.join(", ")),
            };
            let source = match row.reads {
                Reads::ClosedForm | Reads::Structure => "exact",
                Reads::Either => "exact off a closed form, measured off samples",
                Reads::Samples => "measured",
                Reads::Graph => "measured, and the graph behind the samples",
            };
            format!("  {call:<26} {source}\n      {}", row.help)
        })
        .collect::<Vec<_>>()
        .join("\n")
}
