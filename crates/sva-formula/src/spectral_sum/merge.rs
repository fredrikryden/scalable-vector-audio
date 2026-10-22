// Concern: sorts a lane, merges like atoms and folds exact zeros | Non-concern: what makes two atoms alike (atom.rs) | IO: (&mut Lane) -> ()

use crate::closed_form::Edge;
use crate::complex::Cx;
use crate::num::Num;
use crate::spectral_sum::Lane;
use crate::spectral_sum::atom::{Factors, Indicator, SpectralAtom};

/// Exact bits only. An epsilon would make equality intransitive, which breaks both the sort
/// and the hash, and the hash is the cache key.
pub fn simplify_over<N: Num>(lane: &mut Lane<N>) {
    collect(lane);
    let folded = fold_windows(std::mem::take(&mut lane.atoms));
    lane.atoms = folded;
    collect(lane);
}

fn collect<N: Num>(lane: &mut Lane<N>) {
    lane.atoms.sort_by_key(SpectralAtom::key);
    let mut merged: Vec<SpectralAtom<N>> = Vec::with_capacity(lane.atoms.len());
    for atom in lane.atoms.drain(..) {
        match merged.last_mut() {
            Some(last) if last.key() == atom.key() => last.c = last.c + atom.c,
            _ => merged.push(atom),
        }
    }
    merged.retain(|a| !a.c.is_zero());
    lane.atoms = merged;
}

/// Atoms alike but for their window are one piecewise-constant amplitude over the line.
/// Summing it per interval is what turns the two halves of an `sgn` back into one window.
fn fold_windows<N: Num>(atoms: Vec<SpectralAtom<N>>) -> Vec<SpectralAtom<N>> {
    let mut ordered = atoms;
    ordered.sort_by_key(SpectralAtom::key_without_window);
    let mut out = Vec::with_capacity(ordered.len());
    let mut run: Vec<SpectralAtom<N>> = Vec::new();
    for atom in ordered {
        let breaks = run
            .first()
            .is_some_and(|f| f.key_without_window() != atom.key_without_window());
        if breaks {
            out.extend(sweep(std::mem::take(&mut run)));
        }
        run.push(atom);
    }
    out.extend(sweep(run));
    out
}

/// Two edges at one instant the varied numbers move apart bound an interval of their own.
fn sweep<N: Num>(run: Vec<SpectralAtom<N>>) -> Vec<SpectralAtom<N>> {
    if run.len() < 2 || run.iter().any(SpectralAtom::is_delta) {
        return run;
    }
    let mut cuts: Vec<Edge<N>> = vec![Edge::NegInf, Edge::PosInf];
    for atom in &run {
        let window = atom.ind.unwrap_or(Indicator::ALL);
        cuts.push(window.l);
        cuts.push(window.r);
    }
    cuts.sort();
    cuts.dedup();

    let template = run[0];
    let factors = Factors {
        ind: None,
        ..template.factors()
    };
    let mut out: Vec<SpectralAtom<N>> = Vec::new();
    let mut open: Option<(Edge<N>, Cx<N>)> = None;
    for pair in cuts.windows(2) {
        let (from, to) = (pair[0], pair[1]);
        let height = run
            .iter()
            .filter(|a| {
                let w = a.ind.unwrap_or(Indicator::ALL);
                w.l <= from && to <= w.r
            })
            .fold(Cx::ZERO, |acc, a| acc + a.c);
        match open {
            Some((_, standing)) if standing == height => continue,
            Some((start, standing)) => out.push(piece(&template, factors, standing, start, from)),
            None => {}
        }
        open = Some((from, height));
    }
    if let Some((start, standing)) = open {
        out.push(piece(&template, factors, standing, start, Edge::PosInf));
    }
    out.retain(|a| !a.c.is_zero());
    out
}

fn piece<N: Num>(
    template: &SpectralAtom<N>,
    factors: Factors<N>,
    height: Cx<N>,
    from: Edge<N>,
    to: Edge<N>,
) -> SpectralAtom<N> {
    SpectralAtom::new(
        height,
        Factors {
            ind: Some(Indicator { l: from, r: to }),
            ..factors
        },
        template.sing,
        template.origin,
    )
}

pub fn simplify(lane: &mut Lane) {
    simplify_over(lane)
}
