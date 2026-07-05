//! # `field_transfer` — the particle ⇄ mesh transfer primitive
//!
//! Every particle-in-cell family — PIC, MPM, CFD-DEM, IBM — needs the *same* two
//! operations at the Lagrangian/Eulerian boundary:
//!
//! - **scatter** (particle → cell): deposit a per-particle quantity onto the
//!   mesh (momentum onto the PIC grid, void fraction / drag reaction onto the
//!   CFD-DEM cells, mass onto the MPM grid);
//! - **gather** (cell → particle): sample a cell field back at the particle
//!   positions (grid velocity in PIC/MPM, fluid velocity for the drag law in
//!   CFD-DEM, image-point interpolation in IBM).
//!
//! Historically each hybrid re-rolls these, gets the conservation bookkeeping
//! subtly wrong, and cannot be reused. This crate is that primitive, once, so no
//! hybrid re-rolls it.
//!
//! ## Why it is solver-agnostic
//!
//! The **particle side is plain slices** — `positions: &[[f64; 3]]` and a
//! `values: &[f64]` column — so *any* Lagrangian method supplies them; this crate
//! has no dependency on SOIL or on any particle method. The **mesh side is a
//! [`TransferMesh`]** — the minimal geometry the operators need (cell count, cell
//! volume, and a shape-function stencil for a point). Every FIELD
//! [`StructuredMesh`](field_core::StructuredMesh) satisfies it through a blanket
//! impl (uniform grids today, octree-AMR forests when they implement the trait
//! family), and a non-FIELD grid can implement [`TransferMesh`] directly.
//!
//! ## The conservation contract
//!
//! Both operators use **one shape function**: cloud-in-cell (CIC), i.e. the
//! trilinear kernel over the eight cells bracketing a point. Its defining
//! property is a **partition of unity** — a particle's eight weights sum to one.
//! That single fact gives both guarantees a hybrid depends on:
//!
//! - [`scatter`] is **conservative**: the total extensive quantity is preserved,
//!   `Σ_c cells[c]` gains exactly `Σ_p values[p]`, because each particle spreads
//!   its value across cells whose weights sum to one (mass in == mass out).
//! - [`gather`] is **constant-preserving**: a uniform cell field is sampled back
//!   exactly, so a hybrid does not manufacture spurious gradients at the coupling.
//!
//! `scatter` and `gather` are also a matched **adjoint pair** (same weights both
//! ways): `⟨scatter(q), f⟩_cells == ⟨q, gather(f)⟩_particles`. Using the same
//! kernel in both directions is what keeps hybrid schemes stable.
#![warn(missing_docs)]

use field_core::{FvMesh, StructuredMesh};

/// Number of cells one particle couples to under the cloud-in-cell (trilinear)
/// stencil: the eight corners of the cell bracket around the point.
pub const CIC_STENCIL: usize = 8;

/// A particle's shape-function stencil: the cells it couples to and the weight on
/// each. The weights are a **partition of unity** (`Σ w == 1`) — the property
/// that makes [`scatter`] conservative and [`gather`] constant-preserving.
///
/// Produced by [`TransferMesh::stencil`]; consumed by [`scatter`] / [`gather`].
/// Fixed-capacity (`CIC_STENCIL`) so no allocation happens per particle.
#[derive(Clone, Copy, Debug)]
pub struct Stencil {
    cells: [usize; CIC_STENCIL],
    weights: [f64; CIC_STENCIL],
    len: usize,
}

impl Stencil {
    /// An empty stencil (used by mesh implementations that build incrementally).
    pub const fn new() -> Self {
        Self { cells: [0; CIC_STENCIL], weights: [0.0; CIC_STENCIL], len: 0 }
    }

    /// Append one `(cell, weight)` term. Panics if more than [`CIC_STENCIL`]
    /// terms are pushed — a CIC stencil never exceeds eight cells.
    pub fn push(&mut self, cell: usize, weight: f64) {
        assert!(self.len < CIC_STENCIL, "CIC stencil holds at most {CIC_STENCIL} cells");
        self.cells[self.len] = cell;
        self.weights[self.len] = weight;
        self.len += 1;
    }

    /// Number of `(cell, weight)` terms in this stencil.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the stencil has no terms.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate the `(cell_index, weight)` terms.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        (0..self.len).map(move |i| (self.cells[i], self.weights[i]))
    }

    /// Sum of the weights — `1.0` for any interior point under a partition-of-
    /// unity kernel. Exposed for tests and diagnostics.
    pub fn weight_sum(&self) -> f64 {
        self.weights[..self.len].iter().sum()
    }
}

impl Default for Stencil {
    fn default() -> Self {
        Self::new()
    }
}

/// The mesh capability the transfer operators need — and nothing more.
///
/// Implement this and both [`scatter`] and [`gather`] work against your mesh.
/// FIELD's [`StructuredMesh`](field_core::StructuredMesh) already does, via the
/// blanket impl below, so uniform grids (and any future forest/AMR mesh that
/// implements the FIELD trait family) are supported out of the box. A grid from
/// outside the FIELD ecosystem can implement this directly — that is what
/// "solver-agnostic" means in practice.
pub trait TransferMesh {
    /// Number of cells the field arrays are indexed by. `cells` / `values`
    /// slices passed to [`scatter`] / [`gather`] must have this length.
    fn n_cells(&self) -> usize;

    /// Control volume of cell `c`. Used only by the density helpers
    /// ([`scatter_density`]); the bare [`scatter`]/[`gather`] are volume-free.
    fn cell_volume(&self, c: usize) -> f64;

    /// The cloud-in-cell (trilinear) shape-function stencil for `point`, or
    /// `None` if the point lies outside the mesh's addressable extent (including
    /// its ghost layers). The returned weights partition unity.
    fn stencil(&self, point: [f64; 3]) -> Option<Stencil>;
}

/// Every FIELD structured mesh is a transfer mesh: its `locate` gives the cell
/// bracket + trilinear fractions of a point, which *is* the CIC stencil.
impl<M: StructuredMesh + ?Sized> TransferMesh for M {
    fn n_cells(&self) -> usize {
        FvMesh::n_cells_total(self)
    }

    fn cell_volume(&self, c: usize) -> f64 {
        FvMesh::cell_volume(self, c)
    }

    fn stencil(&self, point: [f64; 3]) -> Option<Stencil> {
        // `locate` returns the raw (ghost-inclusive) lower-corner cell `lo` and
        // the trilinear fractions `t ∈ [0,1]³` of the eight-cell bracket. The CIC
        // weight of corner `(di,dj,dk)` is the product of the per-axis linear
        // hat: `t` on the far side, `1 - t` on the near side. The eight products
        // sum to `Π_axis ((1-t)+t) = 1`, i.e. a partition of unity.
        let (lo, t) = self.locate(point)?;
        let mut stencil = Stencil::new();
        for di in 0..2 {
            let wx = if di == 1 { t[0] } else { 1.0 - t[0] };
            for dj in 0..2 {
                let wy = if dj == 1 { t[1] } else { 1.0 - t[1] };
                for dk in 0..2 {
                    let wz = if dk == 1 { t[2] } else { 1.0 - t[2] };
                    let cell = self.idx_raw(lo[0] + di, lo[1] + dj, lo[2] + dk);
                    stencil.push(cell, wx * wy * wz);
                }
            }
        }
        Some(stencil)
    }
}

/// How many particles a transfer touched, split by whether they landed on the
/// mesh. `skipped` particles fell outside the mesh extent and were ignored by
/// [`scatter`] (contributing nothing) or written with the fallback by [`gather`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransferStats {
    /// Particles whose stencil was found and applied.
    pub applied: usize,
    /// Particles outside the mesh extent.
    pub skipped: usize,
}

/// Conservative **scatter** (particle → cell): accumulate each particle's
/// extensive quantity onto the cells of its CIC stencil,
/// `cells[c] += Σ_p w(p, c) · values[p]`.
///
/// Because each particle's stencil weights sum to one, the operation is
/// **conservative**: `Σ_c cells[c]` increases by exactly the sum of `values` over
/// the particles that landed on the mesh (see [`TransferStats::applied`]). This
/// is the mass-conservation contract — deposit `1.0` per particle and the cell
/// total equals the particle count.
///
/// Deposit is **additive** and does not zero `cells` first, so callers control
/// accumulation (e.g. summing several species). Zero the field yourself for a
/// fresh deposit.
///
/// # Panics
/// If `positions.len() != values.len()` or `cells.len() != mesh.n_cells()`.
pub fn scatter<M: TransferMesh + ?Sized>(
    mesh: &M,
    positions: &[[f64; 3]],
    values: &[f64],
    cells: &mut [f64],
) -> TransferStats {
    assert_eq!(positions.len(), values.len(), "positions and values must be parallel");
    assert_eq!(cells.len(), mesh.n_cells(), "cells length must equal mesh.n_cells()");

    let mut stats = TransferStats::default();
    for (p, &pos) in positions.iter().enumerate() {
        match mesh.stencil(pos) {
            Some(s) => {
                let v = values[p];
                for (c, w) in s.iter() {
                    cells[c] += w * v;
                }
                stats.applied += 1;
            }
            None => stats.skipped += 1,
        }
    }
    stats
}

/// Conservative scatter of a **density**: like [`scatter`], but the deposited
/// contribution to each cell is divided by that cell's volume, so `cells` carries
/// a per-volume field (void fraction, mass density, source strength) rather than
/// an extensive total. The *volume-integrated* quantity is still conserved —
/// `Σ_c cells[c] · V_c == Σ_p values[p]` — which is the CFD-DEM/PIC form of the
/// same contract. See [`scatter`] for accumulation and panic semantics.
pub fn scatter_density<M: TransferMesh + ?Sized>(
    mesh: &M,
    positions: &[[f64; 3]],
    values: &[f64],
    cells: &mut [f64],
) -> TransferStats {
    assert_eq!(positions.len(), values.len(), "positions and values must be parallel");
    assert_eq!(cells.len(), mesh.n_cells(), "cells length must equal mesh.n_cells()");

    let mut stats = TransferStats::default();
    for (p, &pos) in positions.iter().enumerate() {
        match mesh.stencil(pos) {
            Some(s) => {
                let v = values[p];
                for (c, w) in s.iter() {
                    let vol = mesh.cell_volume(c);
                    debug_assert!(vol > 0.0, "cell {c} has non-positive volume {vol}");
                    cells[c] += w * v / vol;
                }
                stats.applied += 1;
            }
            None => stats.skipped += 1,
        }
    }
    stats
}

/// Consistent **gather** (cell → particle): sample the cell field at each
/// particle with the same CIC shape function,
/// `values[p] = Σ_c w(p, c) · cells[c]`.
///
/// Partition-of-unity weights make gather **constant-preserving**: a uniform
/// field is reproduced exactly. `gather` is the adjoint of [`scatter`] under the
/// same stencil. Particles outside the mesh are written `fallback` (commonly
/// `0.0`, or `f64::NAN` to flag them) and counted in [`TransferStats::skipped`].
///
/// # Panics
/// If `positions.len() != values.len()` or `cells.len() != mesh.n_cells()`.
pub fn gather<M: TransferMesh + ?Sized>(
    mesh: &M,
    positions: &[[f64; 3]],
    cells: &[f64],
    values: &mut [f64],
    fallback: f64,
) -> TransferStats {
    assert_eq!(positions.len(), values.len(), "positions and values must be parallel");
    assert_eq!(cells.len(), mesh.n_cells(), "cells length must equal mesh.n_cells()");

    let mut stats = TransferStats::default();
    for (p, &pos) in positions.iter().enumerate() {
        match mesh.stencil(pos) {
            Some(s) => {
                let mut acc = 0.0;
                for (c, w) in s.iter() {
                    acc += w * cells[c];
                }
                values[p] = acc;
                stats.applied += 1;
            }
            None => {
                values[p] = fallback;
                stats.skipped += 1;
            }
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use field_core::{UniformMesh, UniformMeshConfig};

    /// Tiny deterministic LCG so the tests are reproducible without an `rand`
    /// dependency and without relying on wall-clock/entropy.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Top 53 bits → [0,1).
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// An 8³ uniform grid over [0,1]³ with two ghost layers.
    fn test_mesh() -> UniformMesh {
        UniformMesh::from_config(&UniformMeshConfig {
            nx: 8,
            ny: 8,
            nz: 8,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [1.0, 1.0, 1.0],
            y_edges: None,
            z_edges: None,
        })
    }

    /// Particles sprinkled through the interior [0.2,0.8]³ (well inside the
    /// bracketable extent, so every one deposits its full stencil).
    fn interior_particles(n: usize, seed: u64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let mut rng = Lcg(seed);
        let mut pos = Vec::with_capacity(n);
        let mut val = Vec::with_capacity(n);
        for _ in 0..n {
            let x = 0.2 + 0.6 * rng.next_f64();
            let y = 0.2 + 0.6 * rng.next_f64();
            let z = 0.2 + 0.6 * rng.next_f64();
            pos.push([x, y, z]);
            val.push(0.5 + rng.next_f64()); // strictly positive masses
        }
        (pos, val)
    }

    #[test]
    fn stencil_partitions_unity() {
        let mesh = test_mesh();
        let (pos, _) = interior_particles(64, 1);
        for p in &pos {
            let s = mesh.stencil(*p).expect("interior point must locate");
            assert_eq!(s.len(), CIC_STENCIL);
            assert!((s.weight_sum() - 1.0).abs() < 1e-14, "weights must sum to 1");
            for (_, w) in s.iter() {
                assert!((0.0..=1.0).contains(&w), "each CIC weight in [0,1]");
            }
        }
    }

    /// THE acceptance test: sum over cells == sum over particles after scatter.
    #[test]
    fn scatter_conserves_mass() {
        let mesh = test_mesh();
        let (pos, val) = interior_particles(500, 42);

        let mut cells = vec![0.0f64; mesh.n_cells()];
        let stats = scatter(&mesh, &pos, &val, &mut cells);
        assert_eq!(stats.skipped, 0, "all interior particles must deposit");
        assert_eq!(stats.applied, pos.len());

        let sum_particles: f64 = val.iter().sum();
        let sum_cells: f64 = cells.iter().sum();
        let rel_err = (sum_cells - sum_particles).abs() / sum_particles;
        assert!(
            rel_err < 1e-12,
            "mass not conserved: cells={sum_cells}, particles={sum_particles}, rel_err={rel_err:e}"
        );
    }

    /// Depositing `1.0` per particle: the cell total is exactly the particle
    /// count — the sharpest statement of the conservation contract.
    #[test]
    fn scatter_unit_deposit_counts_particles() {
        let mesh = test_mesh();
        let (pos, _) = interior_particles(313, 7);
        let ones = vec![1.0f64; pos.len()];

        let mut cells = vec![0.0f64; mesh.n_cells()];
        scatter(&mesh, &pos, &ones, &mut cells);

        let total: f64 = cells.iter().sum();
        assert!((total - pos.len() as f64).abs() < 1e-9, "total {total} != {}", pos.len());
    }

    /// Volume-weighted conservation form used by CFD-DEM / PIC density deposits.
    #[test]
    fn scatter_density_conserves_volume_integral() {
        let mesh = test_mesh();
        let (pos, val) = interior_particles(400, 99);

        let mut dens = vec![0.0f64; mesh.n_cells()];
        scatter_density(&mesh, &pos, &val, &mut dens);

        // Σ_c density_c · V_c == Σ_p value_p.
        let integral: f64 =
            (0..mesh.n_cells()).map(|c| dens[c] * FvMesh::cell_volume(&mesh, c)).sum();
        let sum_particles: f64 = val.iter().sum();
        let rel_err = (integral - sum_particles).abs() / sum_particles;
        assert!(rel_err < 1e-12, "volume integral {integral} != particles {sum_particles}");
    }

    /// Gather reproduces a spatially-uniform cell field exactly.
    #[test]
    fn gather_preserves_constant() {
        let mesh = test_mesh();
        let (pos, _) = interior_particles(256, 5);
        let cells = vec![3.75f64; mesh.n_cells()];

        let mut out = vec![0.0f64; pos.len()];
        let stats = gather(&mesh, &pos, &cells, &mut out, f64::NAN);
        assert_eq!(stats.skipped, 0);
        for v in out {
            assert!((v - 3.75).abs() < 1e-12, "constant field must gather exactly, got {v}");
        }
    }

    /// Gather reproduces a linear field exactly — trilinear interpolation is
    /// first-order exact, an independent check the weights are the right kernel.
    #[test]
    fn gather_is_linearly_exact() {
        let mesh = test_mesh();
        // Cell field f(x) = 2x + 3y - z sampled at cell centroids.
        let mut cells = vec![0.0f64; mesh.n_cells()];
        for (c, cell) in cells.iter_mut().enumerate() {
            let ctr = FvMesh::cell_centroid(&mesh, c);
            *cell = 2.0 * ctr[0] + 3.0 * ctr[1] - ctr[2];
        }
        let (pos, _) = interior_particles(200, 2024);
        let mut out = vec![0.0f64; pos.len()];
        gather(&mesh, &pos, &cells, &mut out, f64::NAN);

        for (p, v) in pos.iter().zip(out.iter()) {
            let exact = 2.0 * p[0] + 3.0 * p[1] - p[2];
            assert!((v - exact).abs() < 1e-12, "linear gather off: got {v}, want {exact}");
        }
    }

    /// scatter and gather share one kernel: `⟨scatter(q), f⟩ == ⟨q, gather(f)⟩`.
    #[test]
    fn scatter_gather_are_adjoint() {
        let mesh = test_mesh();
        let (pos, q) = interior_particles(150, 314);

        // A per-cell test field f.
        let mut rng = Lcg(271828);
        let f: Vec<f64> = (0..mesh.n_cells()).map(|_| rng.next_f64()).collect();

        // lhs = Σ_c scatter(q)_c · f_c
        let mut scattered = vec![0.0f64; mesh.n_cells()];
        scatter(&mesh, &pos, &q, &mut scattered);
        let lhs: f64 = scattered.iter().zip(f.iter()).map(|(a, b)| a * b).sum();

        // rhs = Σ_p q_p · gather(f)_p
        let mut gathered = vec![0.0f64; pos.len()];
        gather(&mesh, &pos, &f, &mut gathered, 0.0);
        let rhs: f64 = q.iter().zip(gathered.iter()).map(|(a, b)| a * b).sum();

        assert!((lhs - rhs).abs() <= 1e-10 * lhs.abs().max(1.0), "adjoint broken: {lhs} vs {rhs}");
    }

    /// Particles outside the mesh are skipped by scatter (no phantom mass) and
    /// filled with the fallback by gather.
    #[test]
    fn out_of_domain_particles_are_handled() {
        let mesh = test_mesh();
        let pos = vec![[0.5, 0.5, 0.5], [100.0, 0.0, 0.0], [-5.0, 0.5, 0.5]];
        let val = vec![1.0, 1.0, 1.0];

        let mut cells = vec![0.0f64; mesh.n_cells()];
        let s = scatter(&mesh, &pos, &val, &mut cells);
        assert_eq!(s.applied, 1);
        assert_eq!(s.skipped, 2);
        let total: f64 = cells.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "only the in-domain particle deposits");

        let mut out = vec![0.0f64; pos.len()];
        let g = gather(&mesh, &pos, &cells, &mut out, f64::NAN);
        assert_eq!(g.skipped, 2);
        assert!(out[1].is_nan() && out[2].is_nan(), "outside particles get fallback");
        assert!(!out[0].is_nan());
    }
}
