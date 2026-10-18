// Concern: the tempo curve: a song's bars against its seconds, from the home tempo and the bpm lane's moves | Non-concern: sweeping those moves (lanes.rs) | IO: (bpm, beats, moves) -> seconds <-> bars

use crate::score::{Ratio, Shape};

#[derive(Clone, Copy, Debug)]
pub(crate) struct TempoMove {
    pub start: Ratio,
    pub len: Ratio,
    pub shape: Shape,
    pub from: f64,
    pub to: f64,
}

#[derive(Clone, Debug)]
struct Piece {
    start: f64,
    end: f64,
    ra: f64,
    rb: f64,
    shape: Shape,
    duration: f64,
}

impl Piece {
    fn within(&self, b: f64) -> f64 {
        let (ra, rb, d) = (self.ra, self.rb, self.duration);
        if ra == rb {
            return b / ra;
        }
        match self.shape {
            Shape::Lin => {
                let a = (rb - ra) / (2.0 * d);
                (-ra + (ra * ra + 4.0 * a * b).sqrt()) / (2.0 * a)
            }
            Shape::Exp => {
                let k = rb / ra;
                d * (1.0 + b * k.ln() / (ra * d)).ln() / k.ln()
            }
        }
    }

    fn bars(&self, s: f64) -> f64 {
        let (ra, rb, d) = (self.ra, self.rb, self.duration);
        if ra == rb {
            return s * ra;
        }
        match self.shape {
            Shape::Lin => ra * s + (rb - ra) * s * s / (2.0 * d),
            Shape::Exp => {
                let k = rb / ra;
                ra * d / k.ln() * (k.powf(s / d) - 1.0)
            }
        }
    }
}

/// Bar 0 is second 0. Past a ramp's start seconds are irrational: `ln`, `sqrt` and `powf` in
/// f64 err a few ulps, under 1e-9 s in a song under a day, far under the half sample each
/// shift rounds to, which a render reports as `moved_s`.
#[derive(Clone, Debug)]
pub struct Curve {
    per_bar: f64,
    beats: f64,
    first: Option<Ratio>,
    pieces: Vec<Piece>,
}

impl Curve {
    pub(crate) fn new(bpm: f64, beats: f64, moves: &[TempoMove]) -> Curve {
        let rate = |bpm: f64| bpm / (60.0 * beats);
        let mut held = rate(bpm);
        let mut pieces = Vec::with_capacity(moves.len());
        for m in moves {
            let (ra, rb) = (rate(m.from).max(0.0), rate(m.to));
            let len = m.len.to_f64();
            let (ra, duration) = match len > 0.0 {
                true => (ra, duration(len, ra, rb, m.shape)),
                false => (held, 0.0),
            };
            pieces.push(Piece {
                start: m.start.to_f64(),
                end: m.start.to_f64() + len,
                ra,
                rb,
                shape: m.shape,
                duration,
            });
            held = rb;
        }
        Curve {
            per_bar: beats * 60.0 / bpm,
            beats,
            first: moves.first().map(|m| m.start),
            pieces,
        }
    }

    pub fn moves(&self) -> bool {
        !self.pieces.is_empty()
    }

    pub fn seconds_per_bar(&self) -> f64 {
        self.per_bar
    }

    pub fn steady_until(&self, bar: Ratio) -> bool {
        self.first.is_none_or(|first| bar <= first)
    }

    pub fn seconds(&self, bar: Ratio) -> f64 {
        if self.steady_until(bar) {
            return bar.to_f64() * self.per_bar;
        }
        self.seconds_f(bar.to_f64())
    }

    fn seconds_f(&self, bar: f64) -> f64 {
        let first = self.pieces[0].start;
        let (mut t, mut from, mut rate) = (first * self.per_bar, first, 1.0 / self.per_bar);
        for piece in &self.pieces {
            if bar <= piece.start {
                return t + (bar - from) / rate;
            }
            t += (piece.start - from) / rate;
            if bar < piece.end {
                return t + piece.within(bar - piece.start);
            }
            t += piece.duration;
            from = piece.end;
            rate = piece.rb;
        }
        t + (bar - from) / rate
    }

    /// From bar `from` to `from + len`: n bars are `n * (60 * beats / bpm)` seconds while the
    /// tempo holds its home.
    pub fn duration(&self, from: Ratio, len: Ratio) -> f64 {
        let to = from.plus(len).expect("a placed bar fits");
        match self.steady_until(from.max(to)) {
            true => len.to_f64() * self.per_bar,
            false => self.seconds(to) - self.seconds(from),
        }
    }

    /// The seconds `len` bars last from song second `s`, where any of them leaves the home
    /// tempo; `None` where all of them keep it, so they last `len` home bars.
    pub fn after(&self, s: f64, len: Ratio) -> Option<f64> {
        let from = self.bars(s);
        let to = from + len.to_f64();
        let first = self.first?.to_f64();
        (from.max(to) > first).then(|| self.seconds_f(to) - s)
    }

    pub fn bars(&self, seconds: f64) -> f64 {
        let steady = seconds / self.per_bar;
        let Some(first) = self.pieces.first() else {
            return steady;
        };
        if steady <= first.start {
            return steady;
        }
        let (mut t, mut from, mut rate) =
            (first.start * self.per_bar, first.start, 1.0 / self.per_bar);
        for piece in &self.pieces {
            let reach = t + (piece.start - from) / rate;
            if seconds < reach {
                return from + (seconds - t) * rate;
            }
            if seconds < reach + piece.duration {
                return piece.start + piece.bars(seconds - reach);
            }
            t = reach + piece.duration;
            from = piece.end;
            rate = piece.rb;
        }
        from + (seconds - t) * rate
    }

    pub fn pieces(&self) -> Vec<(f64, f64, f64, f64, f64, f64)> {
        let bpm = |rate: f64| rate * 60.0 * self.beats;
        self.pieces
            .iter()
            .map(|p| {
                let start = self.seconds_f(p.start);
                (
                    p.start,
                    p.end,
                    start,
                    start + p.duration,
                    bpm(p.ra),
                    bpm(p.rb),
                )
            })
            .collect()
    }
}

/// How long a ramp of `len` bars lasts with its bar rate going `ra` to `rb`, the shape read in
/// seconds.
fn duration(len: f64, ra: f64, rb: f64, shape: Shape) -> f64 {
    if ra == rb {
        return len / ra;
    }
    match shape {
        Shape::Lin => len / ((ra + rb) / 2.0),
        Shape::Exp => len * (rb / ra).ln() / (rb - ra),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bars(n: i64) -> Ratio {
        Ratio::int(n)
    }

    #[test]
    fn a_steady_bar_is_the_home_product_and_a_ritardando_lengthens_its_bars() {
        let steady = Curve::new(144.0, 4.0, &[]);
        assert_eq!(steady.seconds(bars(3)), 3.0 * (4.0 * 60.0 / 144.0));
        let rit = TempoMove {
            start: bars(92),
            len: bars(4),
            shape: Shape::Lin,
            from: 144.0,
            to: 128.0,
        };
        let curve = Curve::new(144.0, 4.0, &[rit]);
        let ramp = curve.seconds(bars(96)) - curve.seconds(bars(92));
        let expected = 4.0 / ((144.0 / 240.0 + 128.0 / 240.0) / 2.0);
        assert!((ramp - expected).abs() < 1e-9, "{ramp} against {expected}");
        assert!((ramp - 7.0588).abs() < 1e-3);
        let after = curve.seconds(bars(97)) - curve.seconds(bars(96));
        assert!((after - 240.0 / 128.0).abs() < 1e-9);
        for b in [10.0, 93.5, 95.9, 100.0] {
            let s = curve.seconds(Ratio::of_f64(b).unwrap());
            assert!((curve.bars(s) - b).abs() < 1e-9, "{b}");
        }
        let exp = Curve::new(
            144.0,
            4.0,
            &[TempoMove {
                shape: Shape::Exp,
                ..rit
            }],
        );
        let (ra, rb) = (144.0 / 240.0, 128.0 / 240.0);
        let lasts = exp.seconds(bars(96)) - exp.seconds(bars(92));
        assert!((lasts - 4.0 * f64::ln(rb / ra) / (rb - ra)).abs() < 1e-9);
        let s = exp.seconds(Ratio::of_f64(94.25).unwrap());
        assert!((exp.bars(s) - 94.25).abs() < 1e-9);
    }
}
