// Concern: Fletcher's stiff-string law and its least-squares B | Non-concern: grouping series (pitch.rs), the fit's bounds (inharmonicity.rs) | IO: (n, B) -> stretch; (n, Hz) -> B

pub fn stretch(n: f64, b: f64) -> f64 {
    (1.0 + b * n * n).sqrt()
}

pub struct Weights {
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
}

pub fn weights(ns: &[u32]) -> Option<Weights> {
    let x: Vec<f64> = ns.iter().map(|n| f64::from(*n).powi(2)).collect();
    let m = x.len() as f64;
    let mean = x.iter().sum::<f64>() / m;
    let sxx: f64 = x.iter().map(|x| (x - mean).powi(2)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let slope: Vec<f64> = x.iter().map(|x| (x - mean) / sxx).collect();
    let intercept: Vec<f64> = slope.iter().map(|s| 1.0 / m - mean * s).collect();
    Some(Weights { slope, intercept })
}

pub fn dot(w: &[f64], y: &[f64]) -> f64 {
    w.iter().zip(y).map(|(w, y)| w * y).sum::<f64>()
}

pub fn fit(points: &[(u32, f64)]) -> Option<f64> {
    let ns: Vec<u32> = points.iter().map(|(n, _)| *n).collect();
    let w = weights(&ns)?;
    let y: Vec<f64> = points
        .iter()
        .map(|(n, hz)| (hz / f64::from(*n)).powi(2))
        .collect();
    let a = dot(&w.intercept, &y);
    (a > 0.0).then(|| dot(&w.slope, &y) / a)
}
