// Concern: a knot table: the straight lines through its knots, or their running integral | Non-concern: parsing one, evaluating it over a grid (sva-samples) | IO: (times, values) -> Knots; (t) -> f64

use crate::num::Num;

/// Zero before the first knot. The curve is zero again from the last knot on; its integral
/// holds its total there. Two knots at one time are a jump, read from its right. The bounds
/// (`rounding`, `span`) are of the values alone.
#[derive(Clone, Debug)]
pub struct Knots<N: Num = f64> {
    times: Vec<N>,
    values: Vec<N>,
    integral: bool,
    prefix: Vec<N>,
    after: Vec<(f64, f64)>,
    rounding: f64,
}

impl<N: Num> PartialEq for Knots<N> {
    fn eq(&self, other: &Knots<N>) -> bool {
        let bits = |v: &[N]| v.iter().map(|x| x.key()).collect::<Vec<_>>();
        self.integral == other.integral
            && bits(&self.times) == bits(&other.times)
            && bits(&self.values) == bits(&other.values)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Unknotted {
    Odd,
    Short,
    NotFinite(usize),
    Backwards(usize),
    NoSpan,
}

impl<N: Num> Knots<N> {
    pub fn new(times: Vec<N>, values: Vec<N>, integral: bool) -> Result<Knots<N>, Unknotted> {
        if times.len() != values.len() {
            return Err(Unknotted::Odd);
        }
        if times.len() < 2 {
            return Err(Unknotted::Short);
        }
        let unheld = times
            .iter()
            .zip(&values)
            .position(|(t, v)| !t.v().is_finite() || !v.v().is_finite());
        if let Some(k) = unheld {
            return Err(Unknotted::NotFinite(k));
        }
        if let Some(k) = times.windows(2).position(|w| w[1].v() < w[0].v()) {
            return Err(Unknotted::Backwards(k + 1));
        }
        if times[0].v() == times[times.len() - 1].v() {
            return Err(Unknotted::NoSpan);
        }
        let n = times.len();
        let mut prefix = Vec::with_capacity(n);
        let (mut held, mut weight) = (N::ZERO, 0.0f64);
        prefix.push(N::ZERO);
        for k in 1..n {
            let dt = times[k] - times[k - 1];
            held = held + dt * (values[k - 1] + values[k]) * 0.5;
            weight += dt.v() * (values[k - 1].v().abs() + values[k].v().abs()) * 0.5;
            prefix.push(held);
        }
        let most = values.iter().fold(0.0f64, |m, v| m.max(v.v().abs()));
        let steep = times
            .windows(2)
            .zip(values.windows(2))
            .filter(|(t, _)| t[1].v() > t[0].v())
            .fold(0.0f64, |m, (t, v)| {
                m.max((v[1].v() - v[0].v()).abs() / (t[1].v() - t[0].v()))
            });
        let late = 4.0 * f64::EPSILON * times[0].v().abs().max(times[n - 1].v().abs());
        let rounding = match integral {
            true => (n as f64 + 16.0) * f64::EPSILON * weight + 2.0 * most * late,
            false => 16.0 * f64::EPSILON * most + 2.0 * steep * late,
        };
        let mut knots = Knots {
            times,
            values,
            integral,
            prefix,
            after: Vec::new(),
            rounding,
        };
        knots.after = knots.suffix();
        Ok(knots)
    }

    pub fn times(&self) -> &[N] {
        &self.times
    }

    pub fn values(&self) -> &[N] {
        &self.values
    }

    pub fn integral(&self) -> bool {
        self.integral
    }

    pub fn first(&self) -> f64 {
        self.times[0].v()
    }

    pub fn last(&self) -> f64 {
        self.times[self.times.len() - 1].v()
    }

    pub fn support(&self) -> (f64, f64) {
        match self.held().v() == 0.0 {
            true => (self.first(), self.last()),
            false => (self.first(), f64::INFINITY),
        }
    }

    pub fn held(&self) -> N {
        match self.integral {
            true => self.prefix[self.prefix.len() - 1],
            false => N::ZERO,
        }
    }

    /// A bound on how far an evaluated value sits from the exact one at the instant it is read
    /// at, a read snapped to a knot it lands on included.
    pub fn rounding(&self) -> f64 {
        self.rounding
    }

    pub fn segment(&self, t: f64) -> usize {
        let k = self.times.partition_point(|x| x.v() <= t).saturating_sub(1);
        k.min(self.times.len() - 2)
    }

    /// A column of rising instants walks forward from the last segment instead of bisecting.
    pub fn segment_from(&self, t: f64, from: usize) -> usize {
        let n = self.times.len();
        let mut k = from.min(n - 2);
        if self.times[k].v() <= t {
            for _ in 0..4 {
                if k + 1 < n - 1 && self.times[k + 1].v() <= t {
                    k += 1;
                } else {
                    return k;
                }
            }
        }
        self.segment(t)
    }

    pub fn inside(&self, t: N, k: usize) -> N {
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let (v0, v1) = (self.values[k], self.values[k + 1]);
        let into = t - t0;
        match self.integral {
            false => v0 + (v1 - v0) * into / (t1 - t0),
            true => self.prefix[k] + into * (v0 + N::of(0.5) * into * (v1 - v0) / (t1 - t0)),
        }
    }

    pub fn at(&self, t: N) -> N {
        self.read(t, false, &mut 0)
    }

    /// Its value at an exact time `u` names, `below` where that time is under `u` itself: a
    /// time on a knot reads the segment ending there. `hint` holds the segment last read.
    pub fn read(&self, u: N, below: bool, hint: &mut usize) -> N {
        let (first, last) = (self.first(), self.last());
        let at = u.v();
        if at < first || (at == first && below) {
            return N::ZERO;
        }
        if at > last || (at == last && !below) {
            return self.held();
        }
        let mut k = self.segment_from(at, *hint);
        if below && self.times[k].v() == at {
            k = self.times.partition_point(|x| x.v() < at) - 1;
        }
        *hint = k;
        self.inside(u, k)
    }

    /// The same table, each number written through `num`; its bounds are the values'.
    pub fn mapped<M: Num>(&self, num: &dyn Fn(N) -> M) -> Knots<M> {
        let of = |xs: &[N]| xs.iter().map(|x| num(*x)).collect();
        Knots {
            times: of(&self.times),
            values: of(&self.values),
            integral: self.integral,
            prefix: of(&self.prefix),
            after: self.after.clone(),
            rounding: self.rounding,
        }
    }

    pub fn span(&self, a: f64, b: f64) -> (f64, f64) {
        let at = |t: f64| self.at(N::of(t)).v();
        let mut held = (at(a), at(a));
        let mut take = |v: f64| held = (held.0.min(v), held.1.max(v));
        if b.is_finite() {
            take(at(b));
        } else {
            take(self.held().v());
        }
        if b >= self.last() && !self.integral {
            take(0.0);
        }
        let from = self.times.partition_point(|x| x.v() < a);
        if let Some(turn) = from.checked_sub(1).and_then(|k| self.turn(k))
            && a <= turn.0
            && turn.0 <= b
        {
            take(turn.1);
        }
        if b == f64::INFINITY {
            if let Some((lo, hi)) = self.after.get(from) {
                take(*lo);
                take(*hi);
            }
            return held;
        }
        for k in from..self.times.len() {
            if self.times[k].v() > b {
                break;
            }
            take(self.knot(k));
            if let Some(turn) = self.turn(k).filter(|turn| turn.0 <= b) {
                take(turn.1);
            }
        }
        held
    }

    fn knot(&self, k: usize) -> f64 {
        match self.integral {
            true => self.prefix[k].v(),
            false => self.values[k].v(),
        }
    }

    /// An integral turns where its curve crosses zero.
    fn turn(&self, k: usize) -> Option<(f64, f64)> {
        if !self.integral || k + 1 >= self.times.len() {
            return None;
        }
        let (v0, v1) = (self.values[k].v(), self.values[k + 1].v());
        if v0 == v1 || v0.signum() == v1.signum() || v0 == 0.0 {
            return None;
        }
        let (t0, t1) = (self.times[k].v(), self.times[k + 1].v());
        let at = t0 + (t1 - t0) * v0 / (v0 - v1);
        Some((at, self.inside(N::of(at.clamp(t0, t1)), k).v()))
    }

    fn suffix(&self) -> Vec<(f64, f64)> {
        let n = self.times.len();
        let mut out = vec![(0.0, 0.0); n];
        let mut held = (self.held().v(), self.held().v());
        let take = |held: &mut (f64, f64), v: f64| *held = (held.0.min(v), held.1.max(v));
        for k in (0..n).rev() {
            take(&mut held, self.knot(k));
            if let Some((_, v)) = self.turn(k) {
                take(&mut held, v);
            }
            if !self.integral {
                take(&mut held, 0.0);
            }
            out[k] = held;
        }
        out
    }
}
