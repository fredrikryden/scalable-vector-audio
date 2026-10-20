// Concern: writes an AudioBuffer's planes as a 32-bit float WAV, every sample as held | Non-concern: reading one, choosing its name | IO: (AudioBuffer) -> Blob

/** Planes as a 32-bit float WAV, every sample as held. */
export function wav(audio) {
  const [channels, frames] = [audio.numberOfChannels, audio.length];
  const bytes = new DataView(new ArrayBuffer(44 + 4 * channels * frames));
  const text = (at, word) => [...word].forEach((c, i) => bytes.setUint8(at + i, c.charCodeAt(0)));
  text(0, "RIFF");
  bytes.setUint32(4, 36 + 4 * channels * frames, true);
  text(8, "WAVEfmt ");
  bytes.setUint32(16, 16, true);
  bytes.setUint16(20, 3, true);
  bytes.setUint16(22, channels, true);
  bytes.setUint32(24, audio.sampleRate, true);
  bytes.setUint32(28, 4 * channels * audio.sampleRate, true);
  bytes.setUint16(32, 4 * channels, true);
  bytes.setUint16(34, 32, true);
  text(36, "data");
  bytes.setUint32(40, 4 * channels * frames, true);
  const planes = Array.from({ length: channels }, (_, c) => audio.getChannelData(c));
  for (let i = 0; i < frames; i++) planes.forEach((plane, c) => bytes.setFloat32(44 + 4 * (i * channels + c), plane[i], true));
  return new Blob([bytes], { type: "audio/wav" });
}
