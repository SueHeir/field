//! [`Vector`] and [`LinearOperator`] — the matrix-free algebra a Krylov solver
//! (CG, GMRES, BiCGStab) is written against, plus [`CellVector`], the concrete
//! [`FieldData`]-backed vector that carries a solve's DOFs on the FIELD substrate.
//!
//! # Why this lives in the substrate (and stays equation-agnostic)
//!
//! A Krylov method needs exactly two things from the problem it solves: a way to
//! form linear combinations and inner products of solution-shaped vectors
//! (`Vector`), and a way to apply the operator to a vector *without ever building
//! the matrix* (`LinearOperator::apply`). Neither knows what the operator *is* —
//! CG converges the same whether `A` is a pressure-Poisson Laplacian, an implicit
//! diffusion operator, or a Helmholtz shift. So the interface is pure substrate:
//! FIELD supplies the vector space over its cells; a solver tier on top supplies
//! the iteration, and a physics tier supplies the concrete `A`.
//!
//! This is the deliberate mirror of what [`crate::hierarchy`] does for geometric
//! multigrid — one exposes the grid *hierarchy*, this one exposes the *vector
//! space and operator seam* — so a global/implicit solver can be built on FIELD
//! for **any** mesh equation without the substrate learning a single flux, EOS, or
//! boundary condition.
//!
//! # `CellVector`: a `FieldData` vector
//!
//! [`CellVector`] stores one `f64` per cell in the mesh's full, ghost-inclusive
//! layout and implements [`FieldData`], so a **matrix-free** operator can halo-
//! exchange it (`owner → ghost`) before reading a stencil across a rank boundary,
//! exactly like any other registered field. Its reductions ([`dot`](CellVector::dot),
//! [`norm`](CellVector::norm)) run over **owned** cells only, so once each rank's
//! partial is summed with an MPI all-reduce they give the correct global inner
//! product; the elementwise ops ([`axpy`](CellVector::axpy),
//! [`scale`](CellVector::scale), …) run over the whole store (touching ghosts is
//! harmless and avoids a branch). The all-reduce itself is a solver-tier concern —
//! the substrate exposes the owned-only local reduction and the halo plan, and a
//! parallel Krylov layer wires in the collective.

use std::any::Any;
use std::sync::Arc;

use crate::field_data::FieldData;
use crate::FvMesh;

/// The abstract vector operations a Krylov iteration uses. Written as a trait so a
/// solver tier can be generic over the vector type (a mesh [`CellVector`] today, a
/// coupled particle+mesh vector later) while staying blind to the substrate.
///
/// Reductions ([`dot`](Vector::dot), [`norm`](Vector::norm)) are the *local* (this
/// rank, owned-DOF) contribution; a parallel solver sums them across ranks. All
/// operations assume the two vectors share a layout (same mesh).
pub trait Vector {
    /// Local inner product `⟨self, other⟩` over owned DOFs.
    fn dot(&self, other: &Self) -> f64;

    /// Local Euclidean norm `√⟨self, self⟩` over owned DOFs. For a parallel global
    /// norm, reduce [`dot`](Vector::dot) across ranks and take the root there.
    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// `self ← self + alpha · x` (AXPY).
    fn axpy(&mut self, alpha: f64, x: &Self);

    /// `self ← alpha · self`.
    fn scale(&mut self, alpha: f64);

    /// `self ← x` (copy contents).
    fn copy_from(&mut self, x: &Self);

    /// Set every entry to `value`.
    fn fill(&mut self, value: f64);
}

/// A matrix-free linear operator `A`: given `x`, produce `y = A·x` **without
/// assembling `A`**. The whole point of the substrate seam — an implementor reads
/// `x` (halo-exchanging first if it needs off-rank neighbours) and writes `y`,
/// evaluating the stencil on the fly.
pub trait LinearOperator {
    /// The vector type this operator maps between.
    type Vector;

    /// Compute `y ← A · x`. `x` and `y` must not alias.
    fn apply(&self, x: &Self::Vector, y: &mut Self::Vector);
}

/// A vector of one `f64` DOF per mesh cell, stored in the mesh's full
/// ghost-inclusive layout and exchangeable as [`FieldData`].
///
/// Construct it from a mesh with [`from_mesh`](CellVector::from_mesh); the owned-cell
/// index list (shared cheaply via [`Arc`] so cloning a vector does not re-scan the
/// mesh) drives the owned-only reductions.
pub struct CellVector {
    /// One entry per cell (interior + ghost), indexed exactly as the mesh indexes.
    values: Vec<f64>,
    /// Flat indices of the owned (local, non-ghost) cells — the reduction set.
    owned: Arc<[usize]>,
}

impl CellVector {
    /// Allocate a zeroed vector shaped to `mesh`: length `mesh.n_cells_total()`,
    /// with the owned-cell reduction set taken from [`FvMesh::is_local_cell`].
    pub fn from_mesh(mesh: &dyn FvMesh) -> Self {
        let n = mesh.n_cells_total();
        let owned: Vec<usize> = (0..n).filter(|&c| mesh.is_local_cell(c)).collect();
        Self {
            values: vec![0.0; n],
            owned: owned.into(),
        }
    }

    /// A zeroed vector sharing another's shape and owned set (no mesh re-scan).
    pub fn zeros_like(other: &CellVector) -> Self {
        Self {
            values: vec![0.0; other.values.len()],
            owned: Arc::clone(&other.owned),
        }
    }

    /// Number of cells stored (interior + ghost) — the addressable length.
    pub fn n_total(&self) -> usize {
        self.values.len()
    }

    /// Number of owned DOFs the reductions run over.
    pub fn n_owned(&self) -> usize {
        self.owned.len()
    }

    /// Read cell `c`.
    #[inline]
    pub fn get(&self, c: usize) -> f64 {
        self.values[c]
    }

    /// Write cell `c`.
    #[inline]
    pub fn set(&mut self, c: usize, v: f64) {
        self.values[c] = v;
    }

    /// The full backing slice (interior + ghost), for an operator's stencil loop.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Mutable backing slice.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// The owned-cell index set (the reduction domain).
    pub fn owned(&self) -> &[usize] {
        &self.owned
    }
}

impl Vector for CellVector {
    fn dot(&self, other: &Self) -> f64 {
        debug_assert_eq!(
            self.values.len(),
            other.values.len(),
            "dot: length mismatch"
        );
        self.owned
            .iter()
            .map(|&c| self.values[c] * other.values[c])
            .sum()
    }

    fn axpy(&mut self, alpha: f64, x: &Self) {
        debug_assert_eq!(self.values.len(), x.values.len(), "axpy: length mismatch");
        for (y, &xi) in self.values.iter_mut().zip(x.values.iter()) {
            *y += alpha * xi;
        }
    }

    fn scale(&mut self, alpha: f64) {
        for y in &mut self.values {
            *y *= alpha;
        }
    }

    fn copy_from(&mut self, x: &Self) {
        debug_assert_eq!(
            self.values.len(),
            x.values.len(),
            "copy_from: length mismatch"
        );
        self.values.copy_from_slice(&x.values);
    }

    fn fill(&mut self, value: f64) {
        for y in &mut self.values {
            *y = value;
        }
    }
}

/// A `CellVector` rides the halo like any field: a single forward-exchanged `f64`
/// column (`owner → ghost`, overwrite), which is what a matrix-free operator needs
/// to read a cross-rank stencil neighbour. It carries no reverse/zero columns.
impl FieldData for CellVector {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn len(&self) -> usize {
        self.values.len()
    }
    fn resize(&mut self, n: usize) {
        self.values.resize(n, 0.0);
    }
    fn forward_size(&self) -> usize {
        1
    }
    fn pack_forward(&self, c: usize, buf: &mut Vec<f64>) {
        buf.push(self.values[c]);
    }
    fn unpack_forward(&mut self, c: usize, buf: &[f64]) -> usize {
        self.values[c] = buf[0];
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uniform::{UniformMesh, UniformMeshConfig};
    use crate::StructuredMesh;

    fn mesh(nx: usize, ny: usize, nz: usize, ng: usize) -> UniformMesh {
        UniformMesh::from_config(&UniformMeshConfig {
            nx,
            ny,
            nz,
            ng,
            bounds_lo: [0.0; 3],
            bounds_hi: [nx as f64, ny as f64, nz as f64], // unit spacing on every axis
            y_edges: None,
            z_edges: None,
        })
    }

    #[test]
    fn reductions_run_over_owned_cells_only() {
        let m = mesh(2, 1, 1, 1);
        let mut a = CellVector::from_mesh(&m);
        let mut b = CellVector::from_mesh(&m);
        assert_eq!(a.n_owned(), 2);

        // Set the two owned cells; poison a ghost to prove it is excluded.
        a.set(m.idx(0, 0, 0), 3.0);
        a.set(m.idx(1, 0, 0), 4.0);
        b.set(m.idx(0, 0, 0), 3.0);
        b.set(m.idx(1, 0, 0), 4.0);
        a.set(m.idx_raw(0, 0, 0), 1e9); // a ghost cell — must not enter dot/norm

        assert_eq!(a.dot(&b), 3.0 * 3.0 + 4.0 * 4.0); // 25, ghost ignored
        assert!((a.norm() - 5.0).abs() < 1e-14); // √25, ghost ignored
    }

    #[test]
    fn axpy_scale_copy_fill() {
        let m = mesh(3, 1, 1, 1);
        let mut x = CellVector::from_mesh(&m);
        x.fill(2.0);
        let mut y = CellVector::zeros_like(&x);
        y.fill(1.0);
        y.axpy(3.0, &x); // 1 + 3*2 = 7 everywhere
        for &c in y.owned() {
            assert_eq!(y.get(c), 7.0);
        }
        y.scale(0.5); // 3.5
        for &c in y.owned() {
            assert_eq!(y.get(c), 3.5);
        }
        let mut z = CellVector::zeros_like(&x);
        z.copy_from(&y);
        assert_eq!(z.dot(&z), y.dot(&y));
    }
}
