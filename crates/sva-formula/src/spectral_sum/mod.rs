// Concern: declares SpectralSum and Lane, the canonical form of a closed form | Non-concern: producing one (build.rs) | IO: none

pub mod atom;
pub mod build;
pub mod image;
pub(crate) mod kink;
pub mod merge;
pub mod product;
pub mod spread;
pub mod sup;

use crate::closed_form::{Body, ModalBank, Part, Series, Var, map_children};
use crate::num::Num;
use crate::origin::Origin;
use atom::SpectralAtom;

/// The infinite two sit beside the atoms because they carry their own truncation.
#[derive(Clone, Debug, PartialEq)]
pub struct Lane<N: Num = f64> {
    pub atoms: Vec<SpectralAtom<N>>,
    pub series: Vec<Series<N>>,
    pub modal: Vec<ModalBank>,
}

impl<N: Num> Default for Lane<N> {
    fn default() -> Lane<N> {
        Lane {
            atoms: Vec::new(),
            series: Vec::new(),
            modal: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralSum<N: Num = f64> {
    pub var: Var,
    pub lanes: Vec<Lane<N>>,
}

impl<N: Num> Lane<N> {
    pub fn of(atoms: Vec<SpectralAtom<N>>) -> Lane<N> {
        Lane {
            atoms,
            ..Lane::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty() && self.series.is_empty() && self.modal.is_empty()
    }

    /// Finitely many atoms is what every product rule is written over.
    pub fn is_finite_sum(&self) -> bool {
        self.series.is_empty() && self.modal.is_empty()
    }

    pub fn expanded(mut self) -> Lane<N> {
        for bank in std::mem::take(&mut self.modal) {
            let atoms = crate::modal::atoms(&bank, crate::origin::Origin::UNKNOWN);
            self.atoms
                .extend(atoms.iter().map(SpectralAtom::lifted::<N>));
        }
        self
    }
}

impl<N: Num> SpectralSum<N> {
    pub fn of(var: Var, lanes: Vec<Lane<N>>) -> SpectralSum<N> {
        SpectralSum { var, lanes }
    }

    pub fn mono(var: Var, atoms: Vec<SpectralAtom<N>>) -> SpectralSum<N> {
        SpectralSum {
            var,
            lanes: vec![Lane::of(atoms)],
        }
    }

    pub fn width(&self) -> usize {
        self.lanes.len()
    }

    pub fn atoms(&self) -> impl Iterator<Item = &SpectralAtom<N>> {
        self.lanes.iter().flat_map(|l| l.atoms.iter())
    }

    /// The same value, every atom and term naming `origin` as where it was written.
    pub fn located(mut self, origin: Origin) -> SpectralSum<N> {
        for lane in &mut self.lanes {
            for atom in &mut lane.atoms {
                atom.origin = origin;
            }
            for series in &mut lane.series {
                series.term = Part::new(origin, at(&series.term.body, origin));
            }
        }
        self
    }
}

fn at<N: Num>(f: &Body<N>, origin: Origin) -> Body<N> {
    map_children(f, |p| Part::new(origin, at(&p.body, origin)))
}
