// Concern: the radix-2 complex transform, its inverse from half the bins | Non-concern: what a bin means (stft.rs, measure/) | IO: (&mut [f64], &mut [f64]) -> ()

use std::f64::consts::TAU;

/// One twiddle table serves every stage.
pub fn fft(re: &mut [f64], im: &mut [f64]) {
    fft_with(re, im, &twiddles(re.len()));
}

/// e^(-2 pi i k / n), k under n / 2: computed once for many transforms of one length.
pub fn twiddles(n: usize) -> Vec<(f64, f64)> {
    (0..n / 2)
        .map(|k| {
            let angle = -TAU * k as f64 / n as f64;
            (libm::cos(angle), libm::sin(angle))
        })
        .collect()
}

pub fn fft_with(re: &mut [f64], im: &mut [f64], twiddle: &[(f64, f64)]) {
    let n = re.len();
    assert_eq!(n, im.len(), "one real and one imaginary plane");
    assert!(n.is_power_of_two(), "radix-2 needs a power-of-two length");
    assert_eq!(twiddle.len(), n / 2, "the twiddles of this length");
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let stride = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = twiddle[k * stride];
                let (i0, i1) = (start + k, start + k + len / 2);
                let tr = re[i1] * wr - im[i1] * wi;
                let ti = re[i1] * wi + im[i1] * wr;
                re[i1] = re[i0] - tr;
                im[i1] = im[i0] - ti;
                re[i0] += tr;
                im[i0] += ti;
            }
        }
        len <<= 1;
    }
}

pub fn ifft(re: &mut [f64], im: &mut [f64]) {
    ifft_with(re, im, &twiddles(re.len()));
}

pub fn ifft_with(re: &mut [f64], im: &mut [f64], twiddle: &[(f64, f64)]) {
    for x in im.iter_mut() {
        *x = -*x;
    }
    fft_with(re, im, twiddle);
    let scale = 1.0 / re.len() as f64;
    for x in re.iter_mut() {
        *x *= scale;
    }
    for x in im.iter_mut() {
        *x *= -scale;
    }
}

/// The caller fills `0..=n/2`; this mirrors the rest.
pub fn irfft(bins_re: &[f64], bins_im: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(bins_re.len(), n / 2 + 1);
    assert_eq!(bins_im.len(), n / 2 + 1);
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for k in 0..=n / 2 {
        re[k] = bins_re[k];
        im[k] = bins_im[k];
        if k > 0 && k < n - k {
            re[n - k] = bins_re[k];
            im[n - k] = -bins_im[k];
        }
    }
    ifft(&mut re, &mut im);
    re
}

/// The DFT of any length, by Bluestein's chirp z-transform over a power-of-two convolution.
pub fn dft(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = re.len();
    assert_eq!(n, im.len(), "one real and one imaginary plane");
    if n.is_power_of_two() {
        let (mut r, mut i) = (re.to_vec(), im.to_vec());
        fft(&mut r, &mut i);
        return (r, i);
    }
    let chirp: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let angle = std::f64::consts::PI * ((k * k) % (2 * n)) as f64 / n as f64;
            (libm::cos(angle), -libm::sin(angle))
        })
        .collect();
    let m = (2 * n - 1).next_power_of_two();
    let (mut ar, mut ai) = (vec![0.0; m], vec![0.0; m]);
    let (mut br, mut bi) = (vec![0.0; m], vec![0.0; m]);
    for k in 0..n {
        let (wr, wi) = chirp[k];
        ar[k] = re[k] * wr - im[k] * wi;
        ai[k] = re[k] * wi + im[k] * wr;
        br[k] = wr;
        bi[k] = -wi;
        if k > 0 {
            br[m - k] = wr;
            bi[m - k] = -wi;
        }
    }
    let twiddle = twiddles(m);
    fft_with(&mut ar, &mut ai, &twiddle);
    fft_with(&mut br, &mut bi, &twiddle);
    for k in 0..m {
        let (r, i) = (ar[k] * br[k] - ai[k] * bi[k], ar[k] * bi[k] + ai[k] * br[k]);
        ar[k] = r;
        ai[k] = i;
    }
    ifft_with(&mut ar, &mut ai, &twiddle);
    (0..n)
        .map(|k| {
            let (wr, wi) = chirp[k];
            (ar[k] * wr - ai[k] * wi, ar[k] * wi + ai[k] * wr)
        })
        .unzip()
}

/// The inverse of `dft`, scaled by 1 / n.
pub fn idft(re: &[f64], im: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let neg: Vec<f64> = im.iter().map(|v| -v).collect();
    let (r, i) = dft(re, &neg);
    let scale = 1.0 / re.len() as f64;
    (
        r.iter().map(|v| v * scale).collect(),
        i.iter().map(|v| -v * scale).collect(),
    )
}
