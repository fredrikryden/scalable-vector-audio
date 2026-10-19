// Concern: decodes a dropped recording, crops and plays it, reads the crop with sva-wasm in a worker | Non-concern: the layout (bench.html), the readings (sva-wasm) | IO: (a dropped file) -> a .txt

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
  const state = { run: 0, audio: null, file: "", start: 0, end: 0, peaks: null, readings: null, results: new Map(), asked: null, playing: null, grabbed: null };
  let context = null;

  function said(text, bad = false) {
    $("status").textContent = text;
    $("status").className = bad ? "bad" : "";
  }

  function failed(error) {
    console.error(error);
    said(String(error?.message ?? error), true);
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
      said(state.audio ? "Ready." : "Ready: drop an audio file.");
      document.body.dataset.state = "ready";
      enabled();
    } else if (data.done) {
      const refused = [...state.results.values()].filter((r) => r.refusal).length;
      said(`Read ${state.results.size} readings of the crop${refused ? `, ${refused} refused` : ""}.`);
      $("download").disabled = false;
      document.body.dataset.state = "done";
      enabled();
    } else {
      state.results.set(data.call, data);
      shown();
      said(`Read ${state.results.size} of ${state.asked.length}: ${data.call}.`);
    }
  };

  function enabled() {
    const running = document.body.dataset.state === "running";
    $("run").disabled = !(state.readings && state.audio) || running;
    $("play").disabled = !state.audio;
  }

  const secs = (samples) => samples / RATE;
  const fixed = (s) => s.toFixed(3);

  function cropShown() {
    $("crop").textContent = `${fixed(secs(state.start))} s to ${fixed(secs(state.end))} s, ${fixed(secs(state.end - state.start))} s`;
  }

  // Waveform: one min/max pair per canvas column over every channel.
  function measured() {
    const width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
    canvas.width = width;
    canvas.height = Math.round(canvas.clientHeight * devicePixelRatio);
    const planes = Array.from({ length: state.audio.numberOfChannels }, (_, c) => state.audio.getChannelData(c));
    const per = state.audio.length / width;
    const peaks = new Float32Array(width * 2);
    for (let x = 0; x < width; x++) {
      let low = 0;
      let high = 0;
      for (const plane of planes) {
        for (let i = Math.floor(x * per), end = Math.min(plane.length, Math.floor((x + 1) * per) + 1); i < end; i++) {
          if (plane[i] < low) low = plane[i];
          if (plane[i] > high) high = plane[i];
        }
      }
      peaks[2 * x] = low;
      peaks[2 * x + 1] = high;
    }
    state.peaks = peaks;
  }

  const xOf = (sample) => (sample / state.audio.length) * canvas.width;

  function drawn() {
    if (!state.audio) return;
    const g = canvas.getContext("2d");
    const style = getComputedStyle(document.documentElement);
    const color = (name) => style.getPropertyValue(name).trim();
    const { width, height } = canvas;
    g.clearRect(0, 0, width, height);
    g.fillStyle = color("--line");
    g.fillRect(0, 0, xOf(state.start), height);
    g.fillRect(xOf(state.end), 0, width - xOf(state.end), height);
    g.fillStyle = color("--ink");
    for (let x = 0; x < width; x++) {
      const top = ((1 - state.peaks[2 * x + 1]) / 2) * height;
      const bottom = ((1 - state.peaks[2 * x]) / 2) * height;
      g.fillRect(x, top, 1, Math.max(1, bottom - top));
    }
    g.fillStyle = color("--mark");
    for (const at of [state.start, state.end]) {
      const x = Math.min(width - 2 * devicePixelRatio, xOf(at));
      g.fillRect(x, 0, 2 * devicePixelRatio, height);
      g.fillRect(x - 4 * devicePixelRatio, 0, 10 * devicePixelRatio, 10 * devicePixelRatio);
    }
    if (state.playing) {
      const at = state.start + (context.currentTime - state.playing.from) * RATE;
      if (at < state.end) g.fillRect(xOf(at), 0, devicePixelRatio, height);
      requestAnimationFrame(drawn);
    }
  }

  new ResizeObserver(() => {
    if (!state.audio) return;
    measured();
    drawn();
  }).observe(canvas);

  const sampleAt = (event) => {
    const box = canvas.getBoundingClientRect();
    return Math.round(Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)) * state.audio.length);
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.audio) return;
    const at = sampleAt(event);
    state.grabbed = Math.abs(at - state.start) <= Math.abs(at - state.end) ? "start" : "end";
    canvas.setPointerCapture(event.pointerId);
    moved(event);
  });
  canvas.addEventListener("pointermove", (event) => state.grabbed && moved(event));
  canvas.addEventListener("pointerup", () => (state.grabbed = null));

  function moved(event) {
    const at = sampleAt(event);
    if (state.grabbed === "start") state.start = Math.min(at, state.end - 1);
    else state.end = Math.max(at, state.start + 1);
    stopped();
    cropShown();
    drawn();
  }

  function stopped() {
    if (!state.playing) return;
    state.playing.source.onended = null;
    state.playing.source.stop();
    state.playing = null;
    $("play").textContent = "Play crop";
  }

  $("play").addEventListener("click", async () => {
    if (state.playing) return stopped();
    await context.resume();
    const source = new AudioBufferSourceNode(context, { buffer: state.audio });
    source.connect(context.destination);
    source.onended = () => {
      state.playing = null;
      $("play").textContent = "Play crop";
      drawn();
    };
    state.playing = { source, from: context.currentTime };
    source.start(0, secs(state.start), secs(state.end - state.start));
    $("play").textContent = "Stop";
    drawn();
  });

  const drop = $("drop");
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", async (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    const file = event.dataTransfer.files[0];
    if (!file) return;
    stopped();
    context ??= new AudioContext({ sampleRate: RATE });
    said(`Decoding ${file.name}.`);
    try {
      state.audio = await context.decodeAudioData(await file.arrayBuffer());
    } catch (error) {
      failed(`${file.name} does not decode in this browser: ${error.message}`);
      return;
    }
    state.run += 1;
    Object.assign(state, { file: file.name, start: 0, end: state.audio.length, results: new Map(), asked: null });
    drop.textContent = `${file.name}: ${state.audio.numberOfChannels} channels, ${fixed(state.audio.duration)} s. Drop another to replace it.`;
    $("download").disabled = true;
    shown();
    measured();
    cropShown();
    drawn();
    said(state.readings ? "Ready." : "Loading sva-wasm.");
    document.body.dataset.state = state.readings ? "ready" : "loading";
    enabled();
  });

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
    said(`Reading ${state.asked.length} readings of the crop.`);
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
