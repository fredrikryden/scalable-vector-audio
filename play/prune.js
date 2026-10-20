// Concern: fits each band's track of a `bands` reading with knots within a stated dB bound, stating the error reached | Non-concern: taking the reading (sva-wasm) | IO: (bands value) -> pruned value

/** How far a pruned band's knots stay from its track, in dB: about the least level change a listener hears. */
const PRUNE_EPS_DB = 1;
/** Against each band's own peak, the level under which a pruned track is cut. */
const PRUNE_FLOOR_DB = -80;
const PRUNED_MODEL = "each band's track as sva-wasm's `bands` answered it, one level per frame at `rate_hz` from `start_secs`, in dB, 20 log10 of its rms, any level under the band's `floor_db` (its peak plus the file's `floor_db`) cut to it. Its knots are a swing filter's (Elmeleegy, H. et al. (2009), Online piece-wise linear approximation of numerical streams with precision guarantees, Proc. VLDB Endow. 2(1) 145-156): from each knot the next is the last frame some line from it stays within eps_db - g/2 of every frame since, its value that line's, rounded to a grid of g dB, g the largest power of ten, at most 1, not over eps_db / 5, so the polyline through the knots, joined linearly in dB, is within eps_db of the cut track at every frame. A knot is [frame, dB]. `max_error_db` is the largest distance between the two, evaluated at every frame of the track as answered, before printing: per band, and over every band beside `eps_db`. Peak, time to peak, rise and the impulse floor are the track's own";

/** As `PRUNED_MODEL` states: each run's line within `within` of every level, its end moved at most half a grid step. */
function swing(levels, eps) {
  const per = 10 ** Math.max(0, Math.ceil(Math.log10(5 / eps)));
  const onGrid = (db) => Math.round(db * per) / per;
  const within = (eps - 0.5 / per) * (1 - 1e-9);
  if (!levels.length) return [];
  let anchor = [0, onGrid(levels[0])];
  const knots = [anchor];
  let [last, lo, hi] = [[0, levels[0]], -Infinity, Infinity];
  const close = (from, [at, db]) => {
    const run = at - from[0];
    return [at, onGrid(from[1] + clamp((db - from[1]) / run, lo, hi) * run)];
  };
  for (let at = 1; at < levels.length; at++) {
    const db = levels[at];
    const run = at - anchor[0];
    const [low, high] = [(db - within - anchor[1]) / run, (db + within - anchor[1]) / run];
    if (Math.max(low, lo) > Math.min(high, hi)) {
      anchor = close(anchor, last);
      knots.push(anchor);
      const after = at - anchor[0];
      [lo, hi] = [(db - within - anchor[1]) / after, (db + within - anchor[1]) / after];
    } else [lo, hi] = [Math.max(low, lo), Math.min(high, hi)];
    last = [at, db];
  }
  if (last[0] > anchor[0]) knots.push(close(anchor, last));
  return knots;
}

/** The polyline through `knots`, linear in dB, against every level it stands for. */
function worst(levels, knots) {
  let most = 0;
  let k = 0;
  levels.forEach((db, at) => {
    while (k + 1 < knots.length && knots[k + 1][0] < at) k++;
    const [a, b] = [knots[k], knots[k + 1]];
    const line = b ? a[1] + ((b[1] - a[1]) * (at - a[0])) / (b[0] - a[0]) : a[1];
    most = Math.max(most, Math.abs(line - db));
  });
  return most;
}

const clamp = (v, low, high) => Math.min(high, Math.max(low, v));
const listed = (items) => ({ items, pagination: { count: items.length, has_more: false, next_cursor: null } });

/** A `bands` value with each band's `rms` track replaced by its knots within `PRUNE_EPS_DB`. */
export function pruned(value) {
  const bands = value.bands.items.map(({ rms, ...band }) => {
    const track = rms.items;
    const floor = band.peak > 0 ? 20 * Math.log10(band.peak) + PRUNE_FLOOR_DB : null;
    const levels = floor === null ? [] : track.map((v) => Math.max(20 * Math.log10(v), floor));
    const knots = swing(levels, PRUNE_EPS_DB);
    return { ...band, floor_db: floor, max_error_db: knots.length ? worst(levels, knots) : 0, knots: listed(knots) };
  });
  const max = bands.reduce((most, band) => Math.max(most, band.max_error_db), 0);
  return { model: PRUNED_MODEL, rate_hz: value.rate_hz, start_secs: value.start_secs, eps_db: PRUNE_EPS_DB, floor_db: PRUNE_FLOOR_DB, max_error_db: max, bands: listed(bands) };
}
