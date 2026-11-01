//! p4est-backed forest-of-octrees grid.
//!
//! This crate is Phase 1 of the AMR migration: it exposes a Rust-friendly
//! `ForestGrid` type wrapping a small C shim around p4est's `p8est_*` API.
//! Downstream code that wants to refine a Cartesian domain near an immersed
//! body uses `ForestGrid::refine` with a Rust closure as the criterion.
//!
//! At Phase 1 there is **no integration with `cfd_state` / `cfd_solver`** —
//! that arrives in Phase 2 (FlowField generic over the grid backend). For
//! now this is the same machinery `crates/amr_demo` uses.

use std::ffi::{CString, NulError, c_int, c_void};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

#[repr(C)]
struct AmrForest {
    _private: [u8; 0],
}

type AmrRefineFn =
    unsafe extern "C" fn(cx: f64, cy: f64, cz: f64, h: f64, ctx: *mut c_void) -> c_int;

type AmrCoarsenFn =
    unsafe extern "C" fn(cx: f64, cy: f64, cz: f64, h: f64, ctx: *mut c_void) -> c_int;

/// Mirrors the C `AmrLeafInfo` struct. Layout must match exactly — keep
/// the field order, types, and the explicit padding aligned with shim.h.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LeafInfo {
    pub tree_id: i32,
    pub level: i8,
    _pad: [i8; 7],
    pub center: [f64; 3],
    pub size: [f64; 3],
}

/// Identifies one of the six axis-aligned faces of a leaf. Matches
/// p4est's face numbering: 0=x-, 1=x+, 2=y-, 3=y+, 4=z-, 5=z+.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    XLo = 0,
    XHi = 1,
    YLo = 2,
    YHi = 3,
    ZLo = 4,
    ZHi = 5,
}

impl Face {
    pub fn all() -> [Face; 6] {
        [Face::XLo, Face::XHi, Face::YLo, Face::YHi, Face::ZLo, Face::ZHi]
    }
}

/// Result of a face-neighbor query for a single leaf.
///
/// `Same` and `Coarser` carry one neighbor (`idx[0]`); `Finer` carries
/// four neighbors covering this leaf's face. `Boundary` means the face
/// is on the physical domain boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborSet {
    Boundary,
    Same(usize),
    Coarser(usize),
    /// Four finer neighbors covering this leaf's face. Each slot's index is a
    /// local leaf index, unless the corresponding `ghost_mask` bit is set, in
    /// which case it is a **ghost-layer index** (a finer neighbor on another
    /// rank). See [`FinerSet`].
    Finer(FinerSet),
    /// A single same/coarser neighbor that lives on another MPI rank, carrying
    /// its **ghost-layer index** (see [`ForestGrid::ghosts`]).
    Ghost(usize),
}

/// The four finer neighbors across one face, with a per-slot local/ghost split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FinerSet {
    /// Per-slot index: a local leaf index, or a ghost-layer index when the
    /// matching `ghost_mask` bit is set.
    pub idx: [usize; 4],
    /// Bit `h` set ⇒ `idx[h]` is a ghost-layer index (a cross-rank finer neighbor).
    pub ghost_mask: u8,
}

impl FinerSet {
    /// Whether slot `h`'s neighbor lives on another rank (a ghost).
    pub fn is_ghost(&self, h: usize) -> bool {
        self.ghost_mask & (1 << h) != 0
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AmrFaceNeighbors {
    kind:  i32,
    count: i32,
    idx:   [i64; 4],
    ghost_mask: i32,
}

const AMR_NB_BOUNDARY: i32 = 0;
const AMR_NB_SAME:     i32 = 1;
const AMR_NB_COARSER:  i32 = 2;
const AMR_NB_FINER:    i32 = 3;
const AMR_NB_GHOST:    i32 = 4;

extern "C" {
    fn amr_init() -> c_int;
    fn amr_finalize();
    fn amr_forest_new(
        trees_x: c_int,
        trees_y: c_int,
        trees_z: c_int,
        xmin: f64, xmax: f64,
        ymin: f64, ymax: f64,
        zmin: f64, zmax: f64,
        min_level: c_int,
    ) -> *mut AmrForest;
    fn amr_forest_destroy(f: *mut AmrForest);
    fn amr_forest_refine(
        f: *mut AmrForest,
        max_level: c_int,
        cb: AmrRefineFn,
        ctx: *mut c_void,
    );
    fn amr_forest_coarsen(
        f: *mut AmrForest,
        min_level: c_int,
        cb: AmrCoarsenFn,
        ctx: *mut c_void,
    );
    fn amr_forest_write_vtk(f: *mut AmrForest, filename: *const c_char);
    fn amr_forest_leaf_count(f: *const AmrForest) -> i64;
    fn amr_forest_n_local_leaves(f: *const AmrForest) -> i64;
    fn amr_forest_cell_size_at_level(f: *const AmrForest, level: c_int) -> f64;
    fn amr_forest_fill_leaves(f: *const AmrForest, out: *mut LeafInfo);
    fn amr_forest_face_neighbors(
        f: *const AmrForest,
        leaf_idx: i64,
        face: c_int,
        out: *mut AmrFaceNeighbors,
    );
    fn amr_forest_search_point(f: *const AmrForest, x: f64, y: f64, z: f64) -> i64;
    fn amr_forest_owner_rank(f: *const AmrForest, x: f64, y: f64, z: f64) -> c_int;
    fn amr_forest_overlapping_ranks(
        f: *const AmrForest,
        lo: *const f64,
        hi: *const f64,
        out: *mut i32,
    ) -> c_int;
    fn amr_forest_mpisize(f: *const AmrForest) -> c_int;
    fn amr_forest_mpirank(f: *const AmrForest) -> c_int;
    fn amr_forest_ghost_count(f: *const AmrForest) -> i64;
    fn amr_forest_fill_ghosts(f: *const AmrForest, out: *mut LeafInfo);
    fn amr_forest_ghost_proc_offsets(f: *const AmrForest, out: *mut i32);
    fn amr_forest_mirror_send_count(f: *const AmrForest) -> i64;
    fn amr_forest_mirror_proc_offsets(f: *const AmrForest, out: *mut i32);
    fn amr_forest_mirror_locals(f: *const AmrForest, out: *mut i32);
}

/// Initialize the p4est runtime (`MPI_Init`, `sc_init`, `p4est_init`).
///
/// Safe to call from any thread, any number of times — guarded by
/// `std::sync::Once` so the underlying `MPI_Init` only fires once.
/// Multiple parallel `cargo test` threads all call this safely.
/// Phase 7 will replace this with an integration that defers to
/// `grass_mpi`'s existing `MPI_Init` so the two MPI worlds share a
/// communicator.
pub fn init() -> Result<(), Error> {
    static INIT_ONCE: Once = Once::new();
    static INIT_OK: AtomicBool = AtomicBool::new(false);

    INIT_ONCE.call_once(|| {
        let rc = unsafe { amr_init() };
        INIT_OK.store(rc == 0, Ordering::SeqCst);
    });

    if INIT_OK.load(Ordering::SeqCst) {
        Ok(())
    } else {
        Err(Error::InitFailed)
    }
}

/// Tear down the p4est runtime (`sc_finalize`, `MPI_Finalize`).
pub fn finalize() {
    unsafe { amr_finalize() };
}

/// Errors from the p4est wrapper.
#[derive(Debug)]
pub enum Error {
    InitFailed,
    InvalidPath(NulError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InitFailed => write!(f, "p4est init failed"),
            Error::InvalidPath(e) => write!(f, "invalid VTK path: {}", e),
        }
    }
}

impl std::error::Error for Error {}

/// Layout of a forest of octrees over a 3D Cartesian box. The `trees`
/// fields define a brick of unit-cube root trees (each tree is one cube
/// in the forest's logical space); the affine map onto physical space is
/// `[xmin..xmax] × [ymin..ymax] × [zmin..zmax]`.
///
/// For cells to stay isotropic at every refinement level, choose
/// `trees_{x,y,z}` so that `(xmax-xmin)/trees_x == (ymax-ymin)/trees_y ==
/// (zmax-zmin)/trees_z`. The `validate` method enforces this.
#[derive(Debug, Clone, Copy)]
pub struct ForestLayout {
    pub trees_x: i32,
    pub trees_y: i32,
    pub trees_z: i32,
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
    pub zmin: f64,
    pub zmax: f64,
    pub min_level: i32,
}

impl ForestLayout {
    /// Per-tree edge length in each axis. For cells to be cubic in physical
    /// space, all three should be equal.
    pub fn tree_extent(&self) -> [f64; 3] {
        [
            (self.xmax - self.xmin) / self.trees_x as f64,
            (self.ymax - self.ymin) / self.trees_y as f64,
            (self.zmax - self.zmin) / self.trees_z as f64,
        ]
    }
}

/// Forest-of-octrees grid wrapping a `*mut AmrForest` from the C shim.
///
/// Owns the underlying p4est forest; calling `Drop` releases it. It is `Send`
/// (sole ownership) and `Sync`, but the `Sync` guarantee is narrow: only the
/// **read-only** queries (`leaves`, `face_neighbors`, `ghosts`/`mirror_sends`,
/// `leaf_count`, …) may be called through a shared `&ForestGrid`. The routines
/// that transiently write the forest's `user_pointer` — `refine`/`coarsen` and
/// `find_leaf_at` — all take `&mut self`, so the borrow checker forbids calling
/// them concurrently with anything else. (Within this codebase the App
/// scheduler drives systems single-threaded regardless.)
///
/// Maintains a `Vec<LeafInfo>` cache of leaf metadata in Morton order
/// (rank-local). The cache is the canonical leaf indexing scheme on the
/// Rust side: `leaves()[i]` is the leaf with flat-index `i`, and any
/// `Vec<T>` of per-leaf data should be sized to `leaves().len()` and
/// indexed by the same `i`. The cache is rebuilt automatically after
/// `refine()`; consumers that mutate the forest by other means must call
/// `rebuild_leaf_cache()` to keep it consistent.
pub struct ForestGrid {
    handle: *mut AmrForest,
    layout: ForestLayout,
    leaf_cache: Vec<LeafInfo>,
}

// SAFETY: the `&self` query operations (face_neighbors, leaves, leaf_count,
// fill_leaves/fill_ghosts, the ghost/mirror accessors) only READ the forest, so
// sharing `&ForestGrid` across threads is sound. The operations that mutate
// shared state — refine/balance/partition and `find_leaf_at` (which stashes a
// callback ctx in p8est's `user_pointer`) — all take `&mut self`, so the borrow
// checker forbids racing them against the reads. The leaf_cache is a plain
// `Vec<LeafInfo>`. (Per-cell rayon loops must read through `&ForestGrid` and use
// the read-only queries only; `find_leaf_at` needs exclusive access.)
unsafe impl Sync for ForestGrid {}
unsafe impl Send for ForestGrid {}

impl ForestGrid {
    /// Build a forest from a layout. Calls `amr_init` if not yet called.
    pub fn new(layout: ForestLayout) -> Result<Self, Error> {
        // amr_init is idempotent in the shim, so calling it on every
        // ForestGrid::new is safe and gives single-binary use cases a
        // sensible default. Long-running hosts (toy-cfd) should call
        // `init()` once explicitly.
        init()?;
        let handle = unsafe {
            amr_forest_new(
                layout.trees_x,
                layout.trees_y,
                layout.trees_z,
                layout.xmin,
                layout.xmax,
                layout.ymin,
                layout.ymax,
                layout.zmin,
                layout.zmax,
                layout.min_level,
            )
        };
        if handle.is_null() {
            return Err(Error::InitFailed);
        }
        let mut grid = Self { handle, layout, leaf_cache: Vec::new() };
        grid.rebuild_leaf_cache();
        Ok(grid)
    }

    /// Refine recursively up to `max_level`, then 2:1-balance and partition.
    /// `criterion(cx, cy, cz, h)` returns `true` to subdivide a leaf with
    /// center `(cx, cy, cz)` and edge length `h` (all in physical units).
    /// Rebuilds the leaf cache before returning.
    pub fn refine<F>(&mut self, max_level: i32, mut criterion: F)
    where
        F: FnMut(f64, f64, f64, f64) -> bool,
    {
        // Trampoline pattern: the C-side function pointer can only carry
        // a `void*` context, so we pass a pointer to the closure and let
        // the trampoline invoke it.
        unsafe extern "C" fn trampoline<F>(
            cx: f64,
            cy: f64,
            cz: f64,
            h: f64,
            ctx: *mut c_void,
        ) -> c_int
        where
            F: FnMut(f64, f64, f64, f64) -> bool,
        {
            let f = &mut *(ctx as *mut F);
            if f(cx, cy, cz, h) { 1 } else { 0 }
        }

        unsafe {
            amr_forest_refine(
                self.handle,
                max_level,
                trampoline::<F>,
                &mut criterion as *mut F as *mut c_void,
            );
        }
        self.rebuild_leaf_cache();
    }

    /// Coarsen recursively, stopping at `min_level` (cells stay at least
    /// that coarse). `criterion(cx, cy, cz, h)` returns `true` when the
    /// 8 children of the parent at `(cx, cy, cz)` with edge length `h`
    /// should be merged. The 2:1-balance + partition + leaf-cache rebuild
    /// run automatically before returning.
    ///
    /// Used by Phase 6 dynamic regrid: cells outside the body's
    /// refinement zone get coarsened back to the base level when the
    /// body moves away.
    pub fn coarsen<F>(&mut self, min_level: i32, mut criterion: F)
    where
        F: FnMut(f64, f64, f64, f64) -> bool,
    {
        unsafe extern "C" fn trampoline<F>(
            cx: f64,
            cy: f64,
            cz: f64,
            h: f64,
            ctx: *mut c_void,
        ) -> c_int
        where
            F: FnMut(f64, f64, f64, f64) -> bool,
        {
            let f = &mut *(ctx as *mut F);
            if f(cx, cy, cz, h) { 1 } else { 0 }
        }

        unsafe {
            amr_forest_coarsen(
                self.handle,
                min_level,
                trampoline::<F>,
                &mut criterion as *mut F as *mut c_void,
            );
        }
        self.rebuild_leaf_cache();
    }

    /// Combined adapt: coarsen first (to free memory where the criterion
    /// no longer holds), then refine (to add detail where the criterion
    /// now holds). The two passes use independent closures because what
    /// "should be coarsened" and "should be refined" are typically
    /// different — e.g., coarsen far from a moving body, refine in the
    /// body's new wake region.
    ///
    /// After both passes, the leaf cache is rebuilt; the caller is
    /// responsible for transferring solution state to the new mesh
    /// (see `cfd_p4est_solver` solution-transfer helpers).
    pub fn adapt<C, R>(
        &mut self,
        min_level: i32,
        max_level: i32,
        coarsen_criterion: C,
        refine_criterion: R,
    )
    where
        C: FnMut(f64, f64, f64, f64) -> bool,
        R: FnMut(f64, f64, f64, f64) -> bool,
    {
        self.coarsen(min_level, coarsen_criterion);
        self.refine(max_level, refine_criterion);
    }

    /// Re-populate the rank-local leaf cache by querying the C shim.
    /// Called automatically after `refine`; expose for callers that may
    /// trigger forest changes through other paths (Phase 6's regrid loop).
    pub fn rebuild_leaf_cache(&mut self) {
        let n = unsafe { amr_forest_n_local_leaves(self.handle) } as usize;
        // Zero-initialize, then let the C shim overwrite all `n` entries in
        // Morton order (avoids exposing uninitialized memory to safe code).
        self.leaf_cache = vec![LeafInfo::default(); n];
        // SAFETY: amr_forest_fill_leaves writes exactly `n` LeafInfo.
        unsafe {
            amr_forest_fill_leaves(self.handle, self.leaf_cache.as_mut_ptr());
        }
    }

    /// Borrow the cached leaf metadata. Length equals `n_local_leaves`.
    pub fn leaves(&self) -> &[LeafInfo] {
        &self.leaf_cache
    }

    /// Number of leaves owned by this rank.
    pub fn n_local_leaves(&self) -> usize {
        self.leaf_cache.len()
    }

    /// Total leaf count across all ranks.
    pub fn leaf_count(&self) -> i64 {
        unsafe { amr_forest_leaf_count(self.handle) }
    }

    /// Number of MPI ranks the forest is partitioned across (1 when serial).
    pub fn mpisize(&self) -> usize {
        (unsafe { amr_forest_mpisize(self.handle) }).max(0) as usize
    }

    /// This rank's index.
    pub fn mpirank(&self) -> usize {
        (unsafe { amr_forest_mpirank(self.handle) }).max(0) as usize
    }

    /// Geometry of every ghost (off-rank) quadrant, in ghost-layer order
    /// (grouped by owner rank). Empty on a single rank.
    pub fn ghosts(&self) -> Vec<LeafInfo> {
        let n = unsafe { amr_forest_ghost_count(self.handle) }.max(0) as usize;
        let mut out = vec![LeafInfo::default(); n];
        if n > 0 {
            // SAFETY: amr_forest_fill_ghosts writes exactly `n` LeafInfo.
            unsafe { amr_forest_fill_ghosts(self.handle, out.as_mut_ptr()) };
        }
        out
    }

    /// Ghost recv ranges by owner rank: a `mpisize+1` prefix-sum array, so the
    /// ghosts owned by rank `r` are the layer indices `[out[r], out[r+1])`.
    pub fn ghost_proc_offsets(&self) -> Vec<i32> {
        let mut out = vec![0i32; self.mpisize() + 1];
        unsafe { amr_forest_ghost_proc_offsets(self.handle, out.as_mut_ptr()) };
        out
    }

    /// `(proc_offsets, local_indices)` for the **mirror** (send) side: local
    /// cells that are ghosts on other ranks. `proc_offsets` is a `mpisize+1`
    /// prefix sum; the cells to send to rank `r` are
    /// `local_indices[proc_offsets[r]..proc_offsets[r+1]]`, ordered to match the
    /// receiver's ghost order. Both empty on a single rank.
    pub fn mirror_sends(&self) -> (Vec<i32>, Vec<i32>) {
        let mut offsets = vec![0i32; self.mpisize() + 1];
        unsafe { amr_forest_mirror_proc_offsets(self.handle, offsets.as_mut_ptr()) };
        let total = unsafe { amr_forest_mirror_send_count(self.handle) }.max(0) as usize;
        let mut locals = vec![0i32; total];
        if total > 0 {
            unsafe { amr_forest_mirror_locals(self.handle, locals.as_mut_ptr()) };
        }
        (offsets, locals)
    }

    /// Cell edge length in physical units at refinement level L. Assumes
    /// isotropic per-tree extent (otherwise returns the smallest of the
    /// three axes).
    pub fn cell_size_at_level(&self, level: i32) -> f64 {
        unsafe { amr_forest_cell_size_at_level(self.handle, level) }
    }

    /// Look up the face neighbors of leaf `leaf_idx` across `face`.
    /// Returns the kind + indices needed by an FV stencil. Indices are
    /// flat leaf indices into `leaves()`.
    pub fn face_neighbors(&self, leaf_idx: usize, face: Face) -> NeighborSet {
        let mut raw = AmrFaceNeighbors {
            kind: AMR_NB_BOUNDARY,
            count: 0,
            idx: [-1; 4],
            ghost_mask: 0,
        };
        unsafe {
            amr_forest_face_neighbors(
                self.handle,
                leaf_idx as i64,
                face as c_int,
                &mut raw as *mut _,
            );
        }
        match raw.kind {
            AMR_NB_BOUNDARY => NeighborSet::Boundary,
            AMR_NB_SAME => NeighborSet::Same(raw.idx[0] as usize),
            AMR_NB_COARSER => NeighborSet::Coarser(raw.idx[0] as usize),
            AMR_NB_FINER => NeighborSet::Finer(FinerSet {
                idx: [
                    raw.idx[0] as usize,
                    raw.idx[1] as usize,
                    raw.idx[2] as usize,
                    raw.idx[3] as usize,
                ],
                ghost_mask: raw.ghost_mask as u8,
            }),
            AMR_NB_GHOST => NeighborSet::Ghost(raw.idx[0] as usize),
            _ => NeighborSet::Boundary,
        }
    }

    /// Find the local leaf containing the physical point `(x, y, z)`.
    /// Returns `None` if the point is outside this rank's subdomain
    /// (or outside the global domain).
    ///
    /// Takes `&mut self` even though it is logically a query: the underlying
    /// `p8est_search_local` stashes its callback context in the forest's
    /// `user_pointer`, so it transiently mutates shared state. The exclusive
    /// borrow is what keeps the type's [`Sync`] impl sound — concurrent
    /// `&ForestGrid` access is then limited to the genuinely read-only queries.
    pub fn find_leaf_at(&mut self, x: f64, y: f64, z: f64) -> Option<usize> {
        let r = unsafe { amr_forest_search_point(self.handle, x, y, z) };
        if r < 0 { None } else { Some(r as usize) }
    }

    /// Owner rank of a physical point according to p8est's current global
    /// partition markers. This local query stays valid after repartitioning.
    pub fn owner_rank_at(&self, point: [f64; 3]) -> Option<i32> {
        let rank = unsafe { amr_forest_owner_rank(self.handle, point[0], point[1], point[2]) };
        (rank >= 0).then_some(rank)
    }

    /// Sorted owner ranks whose current partition has positive-volume overlap
    /// with an axis-aligned support. Degenerate supports are point queries.
    /// The query uses p8est's compact global markers and does not communicate.
    pub fn overlapping_ranks(&self, lo: [f64; 3], hi: [f64; 3]) -> Vec<i32> {
        let mut marked = vec![0i32; self.mpisize()];
        unsafe {
            amr_forest_overlapping_ranks(
                self.handle,
                lo.as_ptr(),
                hi.as_ptr(),
                marked.as_mut_ptr(),
            );
        }
        marked
            .into_iter()
            .enumerate()
            .filter_map(|(rank, present)| (present != 0).then_some(rank as i32))
            .collect()
    }

    /// Number of root trees in the brick: `trees_x * trees_y * trees_z`.
    pub fn n_trees(&self) -> i64 {
        (self.layout.trees_x as i64)
            * (self.layout.trees_y as i64)
            * (self.layout.trees_z as i64)
    }

    /// Equivalent uniform-grid cell count if the entire forest were
    /// refined to `level` (without the actual refinement criterion).
    pub fn uniform_count_at_level(&self, level: i32) -> u64 {
        let cells_per_tree = 1u64 << (3 * level as u64);
        cells_per_tree * (self.n_trees() as u64)
    }

    /// Borrow the layout the forest was built from.
    pub fn layout(&self) -> &ForestLayout {
        &self.layout
    }

    /// Write a VTK file: `{stem}.vtu` (and `.pvtu` in parallel runs).
    pub fn write_vtk(&self, stem: impl AsRef<Path>) -> Result<(), Error> {
        let s = stem.as_ref().to_string_lossy().into_owned();
        let cstr = CString::new(s).map_err(Error::InvalidPath)?;
        unsafe { amr_forest_write_vtk(self.handle, cstr.as_ptr()) };
        Ok(())
    }
}

impl Drop for ForestGrid {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { amr_forest_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forest_layout_tree_extent() {
        let layout = ForestLayout {
            trees_x: 5, trees_y: 8, trees_z: 1,
            xmin: 0.0, xmax: 2.5e-3,
            ymin: 0.0, ymax: 4.0e-3,
            zmin: 0.0, zmax: 0.5e-3,
            min_level: 3,
        };
        let e = layout.tree_extent();
        assert!((e[0] - 0.5e-3).abs() < 1e-12);
        assert!((e[1] - 0.5e-3).abs() < 1e-12);
        assert!((e[2] - 0.5e-3).abs() < 1e-12);
    }

    // Note: tests that actually create a ForestGrid call MPI_Init, which
    // can't be called twice — so they live in integration tests, not unit
    // tests, to keep each MPI_Init in its own process.
}
