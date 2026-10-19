// Concern: opens a recording, navigates, plays and crops it, reads the crop with sva-wasm in a worker | Non-concern: the layout (bench.html), the readings (sva-wasm) | IO: (an audio file) -> a .txt

const RATE = 48000;
const WASM = "./node_modules/@scalable-vector-audio/sva-wasm/sva_wasm.js";
/** Their answer is the samples themselves, which JSON caps at 4096 values. */
const RAW = new Set(["samples", "derivative"]);

if (typeof document === "undefined") worker();
else page();

/** The same file run as a module worker: loads sva-wasm, names the readings, answers each call asked. */
async function worker() {
  const { default: init, analyze, readings } = await import(WASM);
  await init();
  onmessage = ({ data: { run, target, channels, calls } }) => {
    for (const call of calls) {
      const started = performance.now();
      try {
        const data = analyze(target, RATE, channels, [call]);
        postMessage({ run, call, data, ms: performance.now() - started });
      } catch (error) {
        postMessage({ run, call, refusal: error.refusal ?? String(error), message: error.refusal ? `${error.name}: ${error.message}` : String(error), ms: performance.now() - started });
      }
    }
    postMessage({ run, done: true });
  };
  const sampled = readings().filter((r) => (r.reads === "samples" || r.reads === "either") && !RAW.has(r.name));
  postMessage({ readings: sampled.map((r) => ({ name: r.name, window: r.takes.includes("window") })) });
}

function page() {
  const $ = (id) => document.getElementById(id);
  const canvas = $("wave");
  const state = { run: 0, audio: null, file: "", start: 0, end: 0, peaks: null, readings: null, results: new Map(), asked: null, crop: null, playing: null, anchor: null, scrub: null, hover: null, view: { from: 0, span: 1 } };
  let context = null;

  function failed(error) {
    console.error(error);
    $("failure").textContent = String(error?.message ?? error);
    document.body.dataset.state = "failed";
  }

  window.addEventListener("error", (event) => failed(event.error ?? event.message));
  window.addEventListener("unhandledrejection", (event) => failed(event.reason));

  const analyst = new Worker(new URL(import.meta.url), { type: "module" });
  analyst.onerror = (event) => failed(event.message ?? "the worker failed to load sva-wasm");
  analyst.onmessage = ({ data }) => {
    if (data.run !== undefined && data.run !== state.run) return;
    if (data.readings) {
      state.readings = data.readings;
      document.body.dataset.state = "ready";
      enabled();
    } else if (data.done) {
      $("download").disabled = false;
      document.body.dataset.state = "done";
      enabled();
    } else {
      state.results.set(data.call, data);
      shown();
    }
  };

  function enabled() {
    const running = document.body.dataset.state === "running";
    $("run").disabled = !(state.readings && state.audio && state.end > state.start) || running;
    for (const id of ["play", "start", "length"]) $(id).disabled = !state.audio;
  }

  const secs = (samples) => samples / RATE;
  const fixed = (s) => s.toFixed(3);
  const clamp = (v, low, high) => Math.min(high, Math.max(low, v));
  const RULER = 24;
  /** A scrub grain's length and the shortest gap between grains, in seconds. */
  const GRAIN = 0.08;
  const GRAIN_EVERY = 0.06;

  function selectionShown() {
    $("start").value = fixed(secs(state.start));
    $("length").value = fixed(secs(state.end - state.start));
    $("readings").classList.toggle("stale", Boolean(state.crop) && (state.crop.start !== state.start || state.crop.end !== state.end));
    enabled();
  }

  // Waveform: one min/max pair per canvas column over every channel, for the samples in view.
  function measured() {
    const { from, span } = state.view;
    const width = canvas.width;
    const planes = Array.from({ length: state.audio.numberOfChannels }, (_, c) => state.audio.getChannelData(c));
    const per = span / width;
    const peaks = new Float32Array(width * 2);
    for (let x = 0; x < width; x++) {
      let low = 0;
      let high = 0;
      for (const plane of planes) {
        for (let i = Math.floor(from + x * per), end = Math.min(plane.length, Math.floor(from + (x + 1) * per) + 1); i < end; i++) {
          if (plane[i] < low) low = plane[i];
          if (plane[i] > high) high = plane[i];
        }
      }
      peaks[2 * x] = low;
      peaks[2 * x + 1] = high;
    }
    state.peaks = peaks;
  }

  const xOf = (sample) => ((sample - state.view.from) / state.view.span) * canvas.width;

  function viewed(from, span) {
    state.view = { from: clamp(from, 0, state.audio.length - span), span };
    state.peaks = null;
    drawn();
  }

  function ruled(g, height, color) {
    const perPixel = secs(state.view.span) / canvas.clientWidth;
    const step = [1, 2, 5].flatMap((m) => [-3, -2, -1, 0, 1, 2].map((e) => m * 10 ** e)).sort((a, b) => a - b).find((s) => s >= 70 * perPixel) ?? 100;
    const digits = clamp(-Math.floor(Math.log10(step) + 1e-9), 0, 3);
    g.fillStyle = color("--muted");
    g.font = `${11 * devicePixelRatio}px system-ui, sans-serif`;
    g.textBaseline = "top";
    for (let t = Math.ceil(secs(state.view.from) / step) * step; t * RATE <= state.view.from + state.view.span; t += step) {
      const x = Math.round(xOf(t * RATE));
      g.fillRect(x, height - 5 * devicePixelRatio, devicePixelRatio, 5 * devicePixelRatio);
      g.fillText(t.toFixed(digits), x + 3 * devicePixelRatio, 2 * devicePixelRatio);
    }
    g.fillRect(0, height - devicePixelRatio, canvas.width, devicePixelRatio);
  }

  function drawn() {
    if (!state.audio) return;
    if (!state.peaks) measured();
    const g = canvas.getContext("2d");
    const style = getComputedStyle(document.documentElement);
    const color = (name) => style.getPropertyValue(name).trim();
    const { width, height } = canvas;
    const top = RULER * devicePixelRatio;
    const wave = height - top;
    g.clearRect(0, 0, width, height);
    ruled(g, top, color);
    g.fillStyle = color("--line");
    g.fillRect(xOf(state.start), top, Math.max(0, xOf(state.end) - xOf(state.start)), wave);
    const [played, upTo] = state.playing ? [xOf(state.playing.at), xOf(state.playing.now)] : [0, 0];
    const ink = color("--ink");
    const mark = color("--mark");
    for (let x = 0; x < width; x++) {
      const high = top + ((1 - state.peaks[2 * x + 1]) / 2) * wave;
      const low = top + ((1 - state.peaks[2 * x]) / 2) * wave;
      g.fillStyle = x >= played && x < upTo ? mark : ink;
      g.fillRect(x, high, 1, Math.max(1, low - high));
    }
    g.fillStyle = color("--mark");
    for (const at of state.end > state.start ? [state.start, state.end] : [state.start]) {
      const x = Math.min(width - 2 * devicePixelRatio, xOf(at));
      g.fillRect(x, top, 2 * devicePixelRatio, wave);
    }
    const head = state.scrub?.at ?? state.playing?.now;
    if (head !== undefined) g.fillRect(xOf(head), 0, devicePixelRatio, height);
    if (state.hover !== null) hovered(g, top, color);
  }

  /** The pointer's time over the canvas, and on the ruler the span a click would seek across. */
  function hovered(g, top, color) {
    const x = Math.round(xOf(state.hover.at));
    if (state.hover.ruler && state.playing && !state.scrub) {
      g.fillStyle = color("--mark");
      g.globalAlpha = 0.25;
      g.fillRect(Math.min(x, xOf(state.playing.now)), 0, Math.abs(x - xOf(state.playing.now)), top);
      g.globalAlpha = 1;
    }
    g.fillStyle = color("--muted");
    g.fillRect(x, 0, devicePixelRatio, canvas.height);
    const text = fixed(secs(state.hover.at));
    const w = g.measureText(text).width + 8 * devicePixelRatio;
    const left = x + w + 2 * devicePixelRatio > canvas.width ? x - w - devicePixelRatio : x + devicePixelRatio;
    g.fillStyle = color("--ink");
    g.fillRect(left, top, w, 16 * devicePixelRatio);
    g.fillStyle = color("--surface");
    g.fillText(text, left + 4 * devicePixelRatio, top + 2 * devicePixelRatio);
  }

  /** Moves the playhead each frame; when it runs off the right of the view, the view pages after it. */
  function ticked() {
    if (!state.playing) return;
    const was = state.playing.now;
    const end = state.view.from + state.view.span;
    state.playing.now = Math.min(state.playing.to, state.playing.at + (context.currentTime - state.playing.from) * RATE);
    if (was <= end && state.playing.now > end) viewed(state.playing.now, state.view.span);
    else drawn();
    requestAnimationFrame(ticked);
  }

  new ResizeObserver(() => {
    canvas.width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
    canvas.height = Math.round(canvas.clientHeight * devicePixelRatio);
    state.peaks = null;
    drawn();
  }).observe(canvas);

  const sampleAt = (event) => {
    const box = canvas.getBoundingClientRect();
    return Math.round(state.view.from + clamp((event.clientX - box.left) / box.width, 0, 1) * state.view.span);
  };

  /** The selection edge within a few pixels of the pointer, if the selection is not empty. */
  const edgeAt = (event) => {
    if (state.end === state.start) return null;
    const x = (event.clientX - canvas.getBoundingClientRect().left) * devicePixelRatio;
    const near = (sample) => Math.abs(xOf(sample) - x) <= 6 * devicePixelRatio;
    return near(state.start) ? "start" : near(state.end) ? "end" : null;
  };

  const onRuler = (event) => event.clientY - canvas.getBoundingClientRect().top < RULER;

  /** One short faded slice of the recording at the scrub position, so a drag on the ruler is heard where it is. */
  function grained() {
    if (context.currentTime < state.scrub.next) return;
    context.resume();
    const t = context.currentTime;
    const gain = new GainNode(context, { gain: 0 });
    gain.gain.linearRampToValueAtTime(1, t + 0.005);
    gain.gain.setValueAtTime(1, t + GRAIN - 0.01);
    gain.gain.linearRampToValueAtTime(0, t + GRAIN);
    const source = new AudioBufferSourceNode(context, { buffer: state.audio });
    source.connect(gain).connect(context.destination);
    source.start(t, secs(state.scrub.at), GRAIN);
    state.scrub.next = t + GRAIN_EVERY;
  }

  // The ruler scrubs and seeks; below it a click places the cursor, a drag selects, a drag at an edge moves that edge.
  canvas.addEventListener("pointerdown", (event) => {
    if (!state.audio || event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    if (onRuler(event)) {
      stopped();
      state.scrub = { at: sampleAt(event), next: 0 };
      grained();
      return drawn();
    }
    const edge = edgeAt(event);
    state.anchor = edge === "start" ? state.end : edge === "end" ? state.start : sampleAt(event);
    moved(event);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.audio) return;
    state.hover = { at: sampleAt(event), ruler: onRuler(event) };
    if (state.scrub) {
      state.scrub.at = state.hover.at;
      grained();
    } else if (state.anchor !== null) return moved(event);
    else canvas.style.cursor = state.hover.ruler ? "pointer" : edgeAt(event) ? "ew-resize" : "text";
    drawn();
  });
  canvas.addEventListener("pointerleave", () => {
    state.hover = null;
    drawn();
  });
  for (const type of ["pointerup", "pointercancel"]) {
    canvas.addEventListener(type, () => {
      state.anchor = null;
      if (!state.scrub) return;
      const { at } = state.scrub;
      state.scrub = null;
      if (type === "pointerup") playing(at, state.audio.length);
      else drawn();
    });
  }

  function moved(event) {
    const at = sampleAt(event);
    state.start = Math.min(at, state.anchor);
    state.end = Math.max(at, state.anchor);
    stopped();
    selectionShown();
    drawn();
  }

  // Ctrl+scroll (or a pinch) zooms about the pointer; Shift+scroll or a sideways swipe scrolls.
  canvas.addEventListener("wheel", (event) => {
    if (!state.audio) return;
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientWidth : 1;
    const { from, span } = state.view;
    if (event.ctrlKey || event.metaKey) {
      event.preventDefault();
      const box = canvas.getBoundingClientRect();
      const at = from + clamp((event.clientX - box.left) / box.width, 0, 1) * span;
      const next = clamp(span * Math.exp(event.deltaY * unit * 0.002), Math.min(canvas.width, state.audio.length), state.audio.length);
      viewed(at - ((at - from) * next) / span, next);
    } else if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      event.preventDefault();
      viewed(from + ((event.deltaX || event.deltaY) * unit * span) / canvas.clientWidth, span);
    }
  }, { passive: false });

  /** Typed times in seconds: a new start keeps the length, as Audacity's Start and Length fields do. */
  function typed(start, length) {
    const samples = Math.min(Math.round(length * RATE), state.audio.length);
    state.start = clamp(Math.round(start * RATE), 0, state.audio.length - samples);
    state.end = state.start + samples;
    stopped();
    selectionShown();
    const { from, span } = state.view;
    if (state.start < from || state.end > from + span) viewed(state.start - Math.max(0, span - samples) / 2, span);
    else drawn();
  }
  const read = (id, held) => (Number.isFinite(parseFloat($(id).value)) ? Math.max(0, parseFloat($(id).value)) : held);
  $("start").addEventListener("change", () => typed(read("start", secs(state.start)), secs(state.end - state.start)));
  $("length").addEventListener("change", () => typed(secs(state.start), read("length", secs(state.end - state.start))));

  function stopped() {
    if (!state.playing) return;
    state.playing.source.onended = null;
    state.playing.source.stop();
    state.playing = null;
    $("play").textContent = "Play";
  }

  // Started before the context resumes: its clock waits with the source, so the playhead stays in step.
  function playing(at, to) {
    stopped();
    context.resume();
    const source = new AudioBufferSourceNode(context, { buffer: state.audio });
    source.connect(context.destination);
    source.onended = () => {
      state.playing = null;
      $("play").textContent = "Play";
      drawn();
    };
    state.playing = { source, from: context.currentTime, at, to, now: at };
    source.start(0, secs(at), secs(to - at));
    $("play").textContent = "Stop";
    ticked();
  }

  /** Plays the selection, or from the cursor to the end when nothing is selected. */
  const played = () => (state.playing ? stopped() : playing(state.start, state.end > state.start ? state.end : state.audio.length));

  $("play").addEventListener("click", played);
  const typing = (event) => event.target.closest?.("input, textarea, select");
  document.addEventListener("keydown", (event) => {
    if (event.key !== " " || !state.audio || typing(event)) return;
    event.preventDefault();
    if (!event.repeat) played();
  });
  document.addEventListener("keyup", (event) => event.key === " " && !typing(event) && event.preventDefault());

  const drop = $("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    if (!event.dataTransfer.files[0]) return;
    $("file").files = event.dataTransfer.files;
    opened(event.dataTransfer.files[0]);
  });
  $("file").addEventListener("change", () => $("file").files[0] && opened($("file").files[0]));

  // A file that does not decode leaves no recording open and says why; a failure of the page itself stays shown.
  async function opened(file) {
    stopped();
    const fatal = document.body.dataset.state === "failed";
    if (!fatal) $("failure").textContent = "";
    context ??= new AudioContext({ sampleRate: RATE });
    let audio = null;
    try {
      audio = await context.decodeAudioData(await file.arrayBuffer());
    } catch (error) {
      if (!fatal) $("failure").textContent = `${file.name} does not decode in this browser: ${error.message}`;
    }
    state.run += 1;
    Object.assign(state, { audio, file: file.name, crop: null, start: 0, end: 0, view: { from: 0, span: audio?.length ?? 1 }, peaks: null, results: new Map(), asked: null });
    $("download").disabled = true;
    shown();
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    if (audio) selectionShown();
    drawn();
    if (!fatal) document.body.dataset.state = state.readings ? "ready" : "loading";
    enabled();
  }

  $("run").addEventListener("click", () => {
    stopped();
    const length = secs(state.end - state.start);
    state.asked = state.readings.map((r) => (r.window ? `${r.name}(window=${length}s)` : r.name));
    state.results = new Map();
    state.run += 1;
    state.crop = { start: state.start, end: state.end };
    const channels = Array.from({ length: state.audio.numberOfChannels }, (_, c) => state.audio.getChannelData(c).slice(state.start, state.end));
    document.body.dataset.state = "running";
    $("download").disabled = true;
    enabled();
    shown();
    selectionShown();
    analyst.postMessage({ run: state.run, target: targetOf(), channels, calls: state.asked }, channels.map((c) => c.buffer));
  });

  const targetOf = () => `${state.file}([${secs(state.crop.start)}s, ${secs(state.crop.end)}s])`;

  function shown() {
    const table = $("readings");
    table.replaceChildren();
    if (!state.asked) return;
    const top = table.insertRow();
    for (const [name, numeric] of [["asked", false], ["ms", true], ["answer", false]]) {
      const th = document.createElement("th");
      th.textContent = name;
      if (numeric) th.className = "n";
      top.append(th);
    }
    for (const call of state.asked) {
      const held = state.results.get(call);
      const row = table.insertRow();
      row.insertCell().textContent = call;
      const ms = row.insertCell();
      ms.className = "n";
      ms.textContent = held ? Math.round(held.ms) : "";
      const answer = row.insertCell();
      if (!held) answer.textContent = "";
      else if (held.refusal) {
        answer.className = "bad";
        answer.textContent = held.message;
      } else answer.textContent = `${Math.round(JSON.stringify(held.data).length / 1024)} KB`;
    }
  }

  $("download").addEventListener("click", () => {
    const readings = {};
    for (const call of state.asked) {
      const held = state.results.get(call);
      readings[call] = held.refusal ? { refusal: held.refusal } : held.data;
    }
    const output = {
      file: state.file,
      decoded: { sample_rate: RATE, channels: state.audio.numberOfChannels, by: "decodeAudioData, which resamples to the AudioContext's rate; the file's own rate is not read" },
      crop: { start_secs: secs(state.crop.start), end_secs: secs(state.crop.end), samples: state.crop.end - state.crop.start },
      readings,
    };
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify(output)], { type: "text/plain" }));
    link.download = `${state.file.replace(/\.[^.]*$/, "")}-${fixed(secs(state.crop.start))}-${fixed(secs(state.crop.end))}s.txt`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 0);
  });
}
