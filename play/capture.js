// Concern: records the audio of a shared screen, window or tab as PCM until stopped | Non-concern: what is done with the recording (bench.js) | IO: (AudioContext) -> Float32Array planes

// An input-less block records as silence; "end" answers "stop" after every posted block, so no tail is lost.
const PROCESSOR = `registerProcessor("pcm", class extends AudioWorkletProcessor {
  constructor() {
    super();
    this.on = true;
    this.port.onmessage = () => {
      this.on = false;
      this.port.postMessage("end");
    };
  }
  process([input]) {
    if (this.on) this.port.postMessage(input.length ? input.map((plane) => plane.slice()) : [new Float32Array(128)]);
    return this.on;
  }
});`;
const loaded = new WeakSet();

/** Resolves to { stop, done }; `done` gives one plane per channel once stopped or once the share ends. */
export async function capture(context) {
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    systemAudio: "include",
  });
  for (const track of stream.getVideoTracks()) {
    track.stop();
    stream.removeTrack(track);
  }
  const [track] = stream.getAudioTracks();
  if (!track) throw new Error("No audio shared: the chosen screen, window or tab carried no audio track.");
  try {
    if (!loaded.has(context)) {
      const url = URL.createObjectURL(new Blob([PROCESSOR], { type: "text/javascript" }));
      await context.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      loaded.add(context);
    }
    await context.resume();
  } catch (error) {
    track.stop();
    throw new Error(`Recording could not start: ${error.message}`);
  }
  const source = new MediaStreamAudioSourceNode(context, { mediaStream: stream });
  const node = new AudioWorkletNode(context, "pcm", { numberOfOutputs: 0 });
  const blocks = [];
  let stop;
  const done = new Promise((resolve) => {
    node.port.onmessage = ({ data }) => {
      if (data !== "end") return blocks.push(data);
      source.disconnect();
      const channels = blocks.reduce((most, block) => Math.max(most, block.length), 0);
      const length = blocks.reduce((sum, block) => sum + block[0].length, 0);
      const planes = Array.from({ length: channels }, () => new Float32Array(length));
      let at = 0;
      for (const block of blocks) {
        planes.forEach((plane, c) => plane.set(block[c] ?? block[0], at));
        at += block[0].length;
      }
      resolve(planes);
    };
    stop = () => {
      track.removeEventListener("ended", stop);
      track.stop();
      node.port.postMessage("stop");
    };
  });
  source.connect(node);
  track.addEventListener("ended", stop);
  return { stop, done };
}
