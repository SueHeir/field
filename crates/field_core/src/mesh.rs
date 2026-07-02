//! [`FvMesh`] — FIELD's **finite-volume** mesh view, and its structured / adaptive
//! extensions.
//!
//! # What this trait is (and isn't)
//!
//! `FvMesh` is deliberately scoped to *cell-centered finite volume*. Its
//! primitives are the FV data model and nothing else: control volumes
//! ([`cell_volume`](FvMesh::cell_volume)) and faces connecting an `owner` cell to
//! an `other` cell with an area-weighted normal ([`for_each_face`](FvMesh::for_each_face)).
//! Any FV physics written against those is automatically *mesh-shape*-agnostic —
//! structured Cartesian, stretched, octree-AMR, and unstructured polyhedral
//! meshes all satisfy the same trait (the OpenFOAM face-reduction insight).
//!
//! It is **not** a general discretization substrate:
//! - **FDM / lattice-Boltzmann** run on the same grids but via [`StructuredMesh`]
//!   (index + spacing); they simply don't use the face/volume methods.
//! - **FEM / DG / spectral-element** are element/DOF-centric, not
//!   control-volume/face-centric, and are out of scope here. If they are ever
//!   wanted they belong in a *sibling* topology trait (e.g. a future `FemMesh`),
//!   layered on the same method-agnostic substrate — [`crate::FieldData`], the
//!   [`crate::HaloPlan`], and the partition, none of which depend on `FvMesh`.
//!
//! So the only finite-volume-specific surface is this trait's three geometry/
//! topology methods plus its extensions. Field storage, halo exchange, and
//! decomposition sit below it and would be reused unchanged by a non-FV view.
//!
//! # The extension traits
//!
//! Capabilities that resist the universal face view live in extensions required
//! only by the solvers that need them:
//! - [`StructuredMesh`] — directional `(i,j,k)` indexing for wide/structured
//!   stencils (central-difference viscous terms, dimensional splitting, structured
//!   multigrid). The face list can't express "the cell two to the left"; this can.
//! - [`AdaptiveMesh`] — refinement levels + coarse/fine face pairs for AMR reflux.
//!
//! [`crate::UniformMesh`] implements `FvMesh + StructuredMesh`; a future
//! `ForestMesh` would implement `FvMesh + AdaptiveMesh`; an unstructured mesh
//! would implement `FvMesh` alone.

use crate::halo::HaloPlan;

/// A 3-component spatial vector (or point), in physical units.
pub type Vec3 = [f64; 3];

/// Identifies a named boundary patch. For [`StructuredMesh`] the convention is
/// the six [`BoundarySide`] values `0..=5`; unstructured meshes assign their own.
pub type PatchId = u32;

/// The six axis-aligned boundary sides of a structured subdomain, used as
/// [`PatchId`] values by [`StructuredMesh`] implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BoundarySide {
    XLo = 0,
    XHi = 1,
    YLo = 2,
    YHi = 3,
    ZLo = 4,
    ZHi = 5,
}

impl BoundarySide {
    /// The axis (0=x, 1=y, 2=z) this side is normal to.
    pub fn axis(self) -> usize {
        (self as u32 / 2) as usize
    }
    /// `false` for the low side, `true` for the high side.
    pub fn is_high(self) -> bool {
        (self as u32 % 2) == 1
    }
    /// Recover a side from its [`PatchId`] index (the inverse of `as u32`), for
    /// physics that reads [`Face::patch`] and dispatches on the side.
    pub fn from_index(i: u32) -> Option<BoundarySide> {
        Some(match i {
            0 => BoundarySide::XLo,
            1 => BoundarySide::XHi,
            2 => BoundarySide::YLo,
            3 => BoundarySide::YHi,
            4 => BoundarySide::ZLo,
            5 => BoundarySide::ZHi,
            _ => return None,
        })
    }
}

/// One face of the mesh, presented to the generic (shape-agnostic) flux loop.
///
/// A face always separates two cells. `area_normal = A_f * n̂`, with `n̂`
/// pointing from `owner` toward `other` (so a flux `F` computed for this face is
/// added to `owner` and subtracted from `other`, scaled by `1/volume`).
///
/// - **Interior face:** `patch == None`; `other` is a real neighbor cell (which
///   may be a halo/ghost cell whose value is supplied by [`crate::halo`]).
/// - **Physical-boundary face:** `patch == Some(side)`; `other` is the ghost cell
///   on that side, whose value a boundary-condition plugin fills. A subdomain
///   face that has an MPI neighbor is *not* a boundary face — it is interior, and
///   its ghost is filled by halo exchange.
pub struct Face {
    /// Owning cell index (always a local interior cell).
    pub owner: usize,
    /// The other cell index — interior, halo, or boundary ghost.
    pub other: usize,
    /// Area-weighted outward normal, pointing `owner` → `other`.
    pub area_normal: Vec3,
    /// Face centroid in physical space.
    pub centroid: Vec3,
    /// `Some(side)` if this face lies on a physical domain boundary.
    pub patch: Option<PatchId>,
}

/// The cell-centered finite-volume mesh interface.
///
/// Object-safe on purpose: tooling and I/O can hold an `&dyn FvMesh`. Performance-
/// critical physics should instead be generic `<M: FvMesh>` (and bound on
/// [`StructuredMesh`] when it needs structured indexing) so the per-face closure
/// inlines and the structured fast path avoids dynamic dispatch entirely.
pub trait FvMesh: Send + Sync + 'static {
    /// Number of owned (local, non-ghost) cells.
    fn n_local_cells(&self) -> usize;

    /// Total number of cells including ghost/halo cells. This is the length every
    /// [`crate::FieldData`] store is resized to.
    fn n_cells_total(&self) -> usize;

    /// Whether cell `c` is a local *owned* cell, as opposed to a ghost/halo cell.
    ///
    /// Owned cells are not necessarily a contiguous prefix of the storage (on a
    /// structured grid they are interleaved with ghost layers in the flat
    /// layout), so `c < n_local_cells()` is **not** a valid test — use this.
    /// Reductions over owned quantities (drag, kinetic energy, mass) must filter
    /// on this so halo copies are never double-counted, and the flux loop uses it
    /// to avoid scattering into cells this rank does not own.
    fn is_local_cell(&self, c: usize) -> bool;

    /// Control volume of cell `c`.
    fn cell_volume(&self, c: usize) -> f64;

    /// Centroid of cell `c` in physical space.
    fn cell_centroid(&self, c: usize) -> Vec3;

    /// Visit every face exactly once. Interior faces carry `patch == None`;
    /// physical-boundary faces carry `patch == Some(side)`.
    ///
    /// Implementors may *compute* faces on the fly (structured grids) or iterate
    /// a stored face list (unstructured); callers cannot tell the difference.
    fn for_each_face(&self, f: &mut dyn FnMut(&Face));

    /// The halo exchange plan for this rank's subdomain. Empty (no links) for a
    /// serial single-rank run; see [`crate::halo`]. (The plan and the exchange
    /// routines are themselves method-agnostic — only this accessor lives on the
    /// FV trait.)
    fn halo_plan(&self) -> &HaloPlan;
}

/// Extension for meshes with structured `(i,j,k)` topology.
///
/// Required by solvers that need directional, wide-stencil access — the thing the
/// universal [`FvMesh::for_each_face`] view deliberately cannot provide. Indices
/// are ghost-offset: interior cell `(0,0,0)` lives at flat index `idx(0,0,0)`,
/// which is *not* `0` when there are ghost layers. (This is also the path an FDM
/// or lattice-Boltzmann solver would ride.)
pub trait StructuredMesh: FvMesh {
    /// Interior cell counts `[ni, nj, nk]` (this rank's subdomain).
    fn dims(&self) -> [usize; 3];

    /// Number of ghost cell layers on each side.
    fn n_ghost(&self) -> usize;

    /// Total cell counts including ghost layers, `[ni_total, nj_total, nk_total]`.
    fn total_dims(&self) -> [usize; 3];

    /// Flat storage index of interior cell `(i, j, k)` (ghost-offset applied).
    fn idx(&self, i: usize, j: usize, k: usize) -> usize;

    /// Flat storage index from **raw, ghost-inclusive** per-axis indices, each in
    /// `0..total_dims[axis]`.
    ///
    /// Where [`idx`](StructuredMesh::idx) takes interior indices and hides the
    /// ghost offset, this addresses the ghost layers directly — the access wide
    /// stencils need. The neighbor of interior cell `i` one step along `+x` is
    /// `idx_raw(i + n_ghost() + 1, j + n_ghost(), k + n_ghost())`, which stays
    /// valid into the ghost band at the subdomain edge; a centered difference at
    /// the low edge reads `idx_raw(n_ghost() - 1, …)`. (`idx(i,j,k)` is exactly
    /// `idx_raw(i + n_ghost(), j + n_ghost(), k + n_ghost())`.)
    fn idx_raw(&self, ir: usize, jr: usize, kr: usize) -> usize;

    /// Cell width along `axis` (0=x, 1=y, 2=z) at *interior* index `n` along that
    /// axis. Supports per-index stretching in y and z.
    fn spacing(&self, axis: usize, n: usize) -> f64;

    /// Locate an arbitrary physical `point`: return the **raw** (ghost-inclusive)
    /// lower-corner cell indices `lo` and the trilinear fractions `t ∈ [0,1]³` of
    /// the 8-cell bracket around it, or `None` if the point is outside this rank's
    /// full extent (interior + ghost). The bracketing cells are
    /// `idx_raw(lo[0]+di, lo[1]+dj, lo[2]+dk)` for `di,dj,dk ∈ {0,1}`.
    ///
    /// This is the geometry query IBM image-point sampling needs; it encapsulates
    /// the grid's coordinate system (uniform or stretched) so physics never
    /// reaches into per-axis coordinate arrays.
    fn locate(&self, point: Vec3) -> Option<([usize; 3], [f64; 3])>;
}

/// Extension for **axis-aligned (Cartesian)** meshes — uniform grids and
/// Cartesian octree forests — exposing per-cell face neighbours grouped by the
/// six axis directions.
///
/// This is what *directional reconstruction stencils* need (MUSCL slopes,
/// central-difference gradients): "the neighbour(s) across my +x face", which
/// the unordered [`FvMesh::for_each_face`] view deliberately cannot give. A
/// coarse cell may abut up to four finer neighbours on one side, so the query
/// returns a slice. Physics generic over this trait runs unchanged on a uniform
/// grid and on a p4est forest.
pub trait CartesianMesh: FvMesh {
    /// Neighbour cell indices across cell `c`'s face on `axis` (0=x, 1=y, 2=z),
    /// the high side if `hi` else the low side.
    ///
    /// Returns one index for a same- or coarser-level neighbour (or a boundary
    /// ghost cell, whose value a BC has filled), up to four for finer
    /// neighbours, and an **empty slice** when the neighbour is across a
    /// non-local (cross-rank) boundary — the caller then falls back to first
    /// order there.
    fn axis_neighbors(&self, c: usize, axis: usize, hi: bool) -> &[usize];
}

/// A coarse/fine face pair across an AMR refinement boundary, for flux refluxing.
///
/// After both levels have computed their fluxes independently, the coarse face's
/// flux must be replaced by the sum of the overlapping fine-face fluxes to stay
/// conservative — a *reverse accumulate* in [`crate::FieldData`] terms.
pub struct CoarseFineFace {
    pub coarse: usize,
    pub fine: usize,
    pub area_normal: Vec3,
    pub centroid: Vec3,
}

/// Extension for adaptively-refined meshes (octree/quadtree forests, block AMR).
///
/// Deliberately thin for now — the refinement/coarsening *operations* and the
/// p4est-backed implementation will land in the `field_amr` crate (milestone M2).
/// This trait fixes the read-side contract a conservative solver needs.
pub trait AdaptiveMesh: FvMesh {
    /// Refinement level of cell `c` (0 = coarsest).
    fn cell_level(&self, c: usize) -> u8;

    /// Visit each coarse/fine face pair that needs flux correction (refluxing).
    fn for_each_coarse_fine_face(&self, f: &mut dyn FnMut(&CoarseFineFace));
}
