// Concern: the waveform view's math (zoom, pinch, pan, the span shown), kept inside the recording | Non-concern: events and drawing (bench.js) | IO: (view, gesture) -> view

// A view is { from, span } in samples.
export function clamped({ from, span }, length, least) {
  const fit = Math.min(Math.max(span, Math.min(least, length)), length);
  return { from: Math.min(Math.max(from, 0), length - fit), span: fit };
}

export function zoomed({ from, span }, fraction, factor) {
  const at = from + fraction * span;
  return { from: at - fraction * span * factor, span: span * factor };
}

// The sample under the first centre follows the centre; apart 0 is one finger.
export function pinched(start, apartThen, apartNow, centreThen, centreNow) {
  const at = start.from + centreThen * start.span;
  const span = apartThen > 0 && apartNow > 0 ? (start.span * apartThen) / apartNow : start.span;
  return { from: at - centreNow * span, span };
}

export const panned = ({ from, span }, fraction) => ({ from: from + fraction * span, span });

export const visible = ({ from, span }) => [Math.round(from), Math.round(from + span)];

export const showing = (start, end) => ({ from: start, span: end - start });

export const revealed = (view, start, end) =>
  start >= view.from && end <= view.from + view.span ? view : { from: start - Math.max(0, view.span - (end - start)) / 2, span: view.span };
