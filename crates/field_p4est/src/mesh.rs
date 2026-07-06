//! [`ForestMesh`] — a [`field_core::FvMesh`] + [`field_core::AdaptiveMesh`] backed
//! by a real p4est forest (via [`ForestGrid`]).
//!
//! The face list is built from p4est's 2:1-balanced face-neighbor queries:
//! - `Same`     → one face at the shared (full) resolution, emitted once.
//! - `Coarser`  → this (fine) leaf abuts a coarser leaf; emit a **fine-face**
//!   between them. p4est reports this from all four fine leaves on the coarse
//!   face, so four fine faces cover the coarse face — **conservative by
//!   construction** (the solver's two-sided scatter sums them into the coarse
//!   owner; no separate refluxing pass).
//! - `Finer`    → skipped: the four fine neighbours emit the faces via `Coarser`.
//! - `Boundary` → a ghost cell mirrored across the face, for the BC to fill.
//! - `Ghost`    → a cross-rank ghost cell that is filled by the mesh `HaloPlan`.
//!
//! The forest is *dynamic*: [`ForestMesh::refine`] / [`coarsen`](ForestMesh::coarsen)
//! adapt the underlying forest and rebuild the FV topology in place. Pair them
//! with [`locate_in`] to transfer a per-cell field onto the regridded mesh.

use crate::forest::{Face, ForestGrid, NeighborSet};
use field_core::{
    AdaptiveMesh, CartesianMesh, CoarseFineFace, Face as MeshFace, FvMesh, HaloLink, HaloPlan, Vec3,
};

#[derive(Clone, Copy)]
struct Cell {
    center: Vec3,
    size: Vec3,
    level: u8,
    is_ghost: bool,
}

/// Neighbour cell indices across one of a cell's six axis faces (up to four on
/// the finer side). Backs [`CartesianMesh::axis_neighbors`] for directional
/// reconstruction stencils.
#[derive(Clone, Copy, Default)]
struct DirNeighbors {
    idx: [usize; 4],
    n: u8,
}

impl DirNeighbors {
    fn one(i: usize) -> Self {
        DirNeighbors { idx: [i, 0, 0, 0], n: 1 }
    }
    fn four(v: [usize; 4]) -> Self {
        DirNeighbors { idx: v, n: 4 }
    }
    fn as_slice(&self) -> &[usize] {
        &self.idx[..self.n as usize]
    }
}

/// A p4est-backed adaptive mesh. Holds the owning [`ForestGrid`] plus a flat
/// cell list (leaves + boundary ghosts) and a prebuilt face list.
pub struct ForestMesh {
    grid: ForestGrid,
    cells: Vec<Cell>,
    n_local: usize,
    faces: Vec<MeshFace>,
    coarse_fine: Vec<CoarseFineFace>,
    /// Per local cell, the neighbour set across each of its six axis faces (in
    /// `Face` order XLo,XHi,YLo,YHi,ZLo,ZHi). Backs [`CartesianMesh`].
    axis_nbr: Vec<[DirNeighbors; 6]>,
    halo: HaloPlan,
}

fn make_face(
    owner: usize,
    other: usize,
    axis: usize,
    sign: f64,
    size: Vec3,
    center: Vec3,
    patch: Option<u32>,
) -> MeshFace {
    let o1 = (axis + 1) % 3;
    let o2 = (axis + 2) % 3;
    let area = size[o1] * size[o2];
    let mut area_normal = [0.0; 3];
    area_normal[axis] = sign * area;
    let mut centroid = center;
    centroid[axis] += sign * 0.5 * size[axis];
    MeshFace { owner, other, area_normal, centroid, patch }
}

type Topology =
    (Vec<Cell>, usize, Vec<MeshFace>, Vec<CoarseFineFace>, Vec<[DirNeighbors; 6]>, HaloPlan);

/// Build the flat cell list + face list (+ per-cell axis neighbours + halo plan)
/// from a forest's current leaves and ghost layer.
fn build_topology(grid: &ForestGrid) -> Topology {
    let n = grid.n_local_leaves();
    let mut cells: Vec<Cell> = grid
        .leaves()
        .iter()
        .map(|l| Cell { center: l.center, size: l.size, level: l.level.max(0) as u8, is_ghost: false })
        .collect();

    // INVARIANT: cross-rank ghost cells occupy a CONTIGUOUS block at indices
    // [n_local, n_local + n_ghost), in ghost-layer order. Everything downstream
    // depends on this: a `Ghost(g)`/finer-ghost slot maps to mesh cell
    // `ghost_base + g`, and `build_halo_plan` maps p4est `proc_offsets` ranges to
    // these same indices. Boundary ghosts are appended *after* this block during
    // the face loop — do NOT interleave them here, or the halo recv mapping and
    // the ghost-index translation both break (silently, and only under MPI).
    let ghosts = grid.ghosts();
    let n_ghost = ghosts.len();
    let ghost_base = n;
    for g in &ghosts {
        cells.push(Cell {
            center: g.center,
            size: g.size,
            level: g.level.max(0) as u8,
            is_ghost: true,
        });
    }
    debug_assert_eq!(cells.len(), ghost_base + n_ghost, "ghost block must be contiguous");

    let mut faces = Vec::new();
    let mut coarse_fine = Vec::new();
    let mut axis_nbr = vec![[DirNeighbors::default(); 6]; n];
    for li in 0..n {
        for face in Face::all() {
            let dir = face as usize;
            let axis = dir / 2;
            let hi = dir % 2 == 1;
            let sign = if hi { 1.0 } else { -1.0 };
            match grid.face_neighbors(li, face) {
                NeighborSet::Boundary => {
                    let owner = cells[li];
                    let mut center = owner.center;
                    center[axis] += sign * owner.size[axis];
                    let gi = cells.len();
                    cells.push(Cell { center, size: owner.size, level: owner.level, is_ghost: true });
                    faces.push(make_face(li, gi, axis, sign, owner.size, owner.center, Some(face as u32)));
                    axis_nbr[li][dir] = DirNeighbors::one(gi);
                }
                NeighborSet::Same(nj) => {
                    if li < nj {
                        let owner = cells[li];
                        faces.push(make_face(li, nj, axis, sign, owner.size, owner.center, None));
                    }
                    axis_nbr[li][dir] = DirNeighbors::one(nj);
                }
                NeighborSet::Coarser(nj) => {
                    let owner = cells[li];
                    let f = make_face(li, nj, axis, sign, owner.size, owner.center, None);
                    coarse_fine.push(CoarseFineFace {
                        coarse: nj,
                        fine: li,
                        area_normal: f.area_normal,
                        centroid: f.centroid,
                    });
                    faces.push(f);
                    axis_nbr[li][dir] = DirNeighbors::one(nj);
                }
                NeighborSet::Finer(fs) => {
                    // Four finer neighbours. Local slots emit their own faces
                    // from the fine side (as Coarser), so here we only emit the
                    // **cross-rank** coarse/fine sub-faces: for a finer neighbour
                    // on another rank, this coarse cell owns a sub-face to the
                    // fine ghost cell (¼ of the coarse face, using the fine cell's
                    // geometry). The two-sided scatter then telescopes across the
                    // partition exactly as it does within a rank — the remote
                    // rank's fine cell emits the matching face and computes the
                    // identical flux, so conservation holds by construction.
                    let mut mesh_idx = [li; 4];
                    for (h, slot) in mesh_idx.iter_mut().enumerate() {
                        if fs.is_ghost(h) {
                            let g = fs.idx[h];
                            if g < n_ghost {
                                let gmesh = ghost_base + g;
                                let fine = cells[gmesh];
                                faces.push(make_face(li, gmesh, axis, sign, fine.size, fine.center, None));
                                *slot = gmesh;
                            }
                        } else {
                            *slot = fs.idx[h];
                        }
                    }
                    axis_nbr[li][dir] = DirNeighbors::four(mesh_idx);
                }
                NeighborSet::Ghost(g) if g < n_ghost => {
                    // Same/coarser neighbour on another rank: emit an interior
                    // face to its ghost cell (whose value the halo fills). The
                    // owner scatters its outgoing flux; the remote rank scatters
                    // its own side.
                    let owner = cells[li];
                    let nj = ghost_base + g;
                    faces.push(make_face(li, nj, axis, sign, owner.size, owner.center, None));
                    axis_nbr[li][dir] = DirNeighbors::one(nj);
                }
                NeighborSet::Ghost(_) => {} // out-of-range ghost index: skip (shouldn't occur)
            }
        }
    }

    let halo = build_halo_plan(grid, ghost_base);
    (cells, n, faces, coarse_fine, axis_nbr, halo)
}

/// Translate p4est's ghost (recv) + mirror (send) ranges into a [`HaloPlan`].
/// p4est emits this rank's mirrors to rank `r` in the same order rank `r`
/// receives them as ghosts, so `send_cells[k] ↔ recv_cells[k]` lines up by
/// construction. Empty (serial no-op) on a single rank.
fn build_halo_plan(grid: &ForestGrid, ghost_base: usize) -> HaloPlan {
    let mpisize = grid.mpisize();
    if mpisize <= 1 {
        return HaloPlan::empty();
    }
    let myrank = grid.mpirank();
    let recv_off = grid.ghost_proc_offsets();
    let (send_off, send_locals) = grid.mirror_sends();
    // The proc-offset ranges must cover exactly the ghost block sized at
    // `build_topology` (recv indices are `ghost_base + gi`), else the recv→cell
    // mapping is inconsistent with the appended ghost cells.
    debug_assert_eq!(
        recv_off[mpisize] as usize,
        grid.ghosts().len(),
        "ghost_proc_offsets total must equal the ghost count",
    );

    let mut links = Vec::new();
    for r in 0..mpisize {
        if r == myrank {
            continue;
        }
        let recv: Vec<usize> =
            (recv_off[r] as usize..recv_off[r + 1] as usize).map(|gi| ghost_base + gi).collect();
        let send: Vec<usize> = send_locals[send_off[r] as usize..send_off[r + 1] as usize]
            .iter()
            .map(|&l| l as usize)
            .collect();
        if recv.is_empty() && send.is_empty() {
            continue;
        }
        links.push(HaloLink { rank: r as i32, send_cells: send, recv_cells: recv });
    }
    HaloPlan { links }
}

impl ForestMesh {
    /// Build the FV mesh from a forest. The grid must already have its p4est
    /// `mesh` available — `ForestGrid::refine`/`coarsen` must have run once.
    pub fn new(grid: ForestGrid) -> Self {
        let (cells, n_local, faces, coarse_fine, axis_nbr, halo) = build_topology(&grid);
        ForestMesh { grid, cells, n_local, faces, coarse_fine, axis_nbr, halo }
    }

    /// Refine the underlying forest (then 2:1-balance) and rebuild the FV
    /// topology. The `criterion(cx, cy, cz, h)` may read a previously-snapshotted
    /// solution to drive solution-adaptive refinement.
    pub fn refine<F: FnMut(f64, f64, f64, f64) -> bool>(&mut self, max_level: i32, criterion: F) {
        self.grid.refine(max_level, criterion);
        self.rebuild();
    }

    /// Coarsen the underlying forest (then 2:1-balance) and rebuild the topology.
    pub fn coarsen<F: FnMut(f64, f64, f64, f64) -> bool>(&mut self, min_level: i32, criterion: F) {
        self.grid.coarsen(min_level, criterion);
        self.rebuild();
    }

    /// Recompute the cell + face lists from the (possibly regridded) forest.
    pub fn rebuild(&mut self) {
        let (cells, n_local, faces, coarse_fine, axis_nbr, halo) = build_topology(&self.grid);
        self.cells = cells;
        self.n_local = n_local;
        self.faces = faces;
        self.coarse_fine = coarse_fine;
        self.axis_nbr = axis_nbr;
        self.halo = halo;
    }

    /// The underlying forest (for VTK output, level queries, …).
    pub fn grid(&self) -> &ForestGrid {
        &self.grid
    }

    /// Full axis-aligned extent `[dx, dy, dz]` of cell `c` (for solution transfer).
    pub fn cell_size_at(&self, c: usize) -> Vec3 {
        self.cells[c].size
    }

    /// `(center, size)` of each interior leaf, in cell-index order — the geometry
    /// needed to transfer a per-cell field onto a regridded mesh (see [`locate_in`]).
    pub fn interior_geometry(&self) -> Vec<(Vec3, Vec3)> {
        self.cells[..self.n_local].iter().map(|c| (c.center, c.size)).collect()
    }

    /// Number of interior faces whose `other` cell is a cross-rank ghost at a
    /// *different* refinement level — i.e. cross-rank coarse/fine interfaces. A
    /// diagnostic for multi-rank tests to confirm a refinement seam actually
    /// straddles the partition (otherwise such a test is vacuous).
    pub fn cross_rank_refinement_faces(&self) -> usize {
        // An interior face (patch None) whose `other` is a ghost is a cross-rank
        // p4est ghost (boundary ghosts carry patch = Some). Different level ⇒ a
        // coarse/fine interface straddling the partition.
        self.faces
            .iter()
            .filter(|f| {
                f.patch.is_none()
                    && self.cells[f.other].is_ghost
                    && self.cells[f.owner].level != self.cells[f.other].level
            })
            .count()
    }

    /// (coarse-leaf count, fine-leaf count) over interior leaves, for diagnostics.
    pub fn level_counts(&self) -> (usize, usize) {
        let min = self.cells[..self.n_local].iter().map(|c| c.level).min().unwrap_or(0);
        let coarse = self.cells[..self.n_local].iter().filter(|c| c.level == min).count();
        (coarse, self.n_local - coarse)
    }
}

/// Index of the old cell whose axis-aligned box contains `point`, from a
/// `(center, size)` geometry list (e.g. [`ForestMesh::interior_geometry`] taken
/// *before* a regrid). The point-sample basis of solution transfer: a refined
/// child inherits its parent's value; a coarsened cell takes one child's value.
pub fn locate_in(old_geom: &[(Vec3, Vec3)], point: Vec3) -> Option<usize> {
    old_geom.iter().position(|(c, s)| {
        // Scale-relative slop so on-face points resolve consistently regardless
        // of the absolute cell size (a fixed 1e-12 is meaningless for sub-micron
        // or astronomical domains).
        (0..3).all(|d| (point[d] - c[d]).abs() <= 0.5 * s[d] * (1.0 + 1e-9))
    })
}

/// **Conservative** solution transfer of a single new cell's value by recursive
/// sub-octant sampling (toy-cfd's `transfer_field_conservative`). For a new cell
/// `(center, size)` over the old `(geom, values)`:
/// - if the old cell at `center` is at least as large (refine / unchanged), copy
///   its value — exact;
/// - if smaller (the new cell coarsened several old cells), split into 8 equal
///   sub-octants and average via `avg` — the box nests perfectly under
///   2:1 balance, so the result is the volume-weighted mean, conserving
///   mass/momentum/energy to roundoff.
///
/// Cells need not be cubic, but they must share an aspect ratio (true for any
/// p4est forest: every leaf is the per-tree box scaled by a power of two), so
/// the termination test below checks **all three axes**, not just one.
///
/// `default` is returned for points outside the old domain (boundary ghosts). A
/// query box that straddles the domain edge will mix `default` into the average;
/// in practice the transfer is over interior cells whose sub-octants stay inside.
pub fn transfer_recursive<T: Copy>(
    old_geom: &[(Vec3, Vec3)],
    old_val: &[T],
    default: T,
    center: Vec3,
    size: Vec3,
    avg: &dyn Fn(&[T]) -> T,
) -> T {
    let Some(i) = locate_in(old_geom, center) else {
        return default;
    };
    // Old cell ⊇ query box (in every axis) ⇒ copy. All-axis test so anisotropic
    // (non-cubic but uniform-aspect) cells transfer conservatively too.
    let os = old_geom[i].1;
    if (0..3).all(|d| os[d] >= size[d] * (1.0 - 1e-9)) {
        return old_val[i];
    }
    // Query cube spans multiple old cells → recurse into 8 equal sub-octants.
    let half = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
    let qtr = [size[0] * 0.25, size[1] * 0.25, size[2] * 0.25];
    let mut vals = [default; 8];
    let mut k = 0;
    for &sz in &[-1.0f64, 1.0] {
        for &sy in &[-1.0f64, 1.0] {
            for &sx in &[-1.0f64, 1.0] {
                let sub = [center[0] + sx * qtr[0], center[1] + sy * qtr[1], center[2] + sz * qtr[2]];
                vals[k] = transfer_recursive(old_geom, old_val, default, sub, half, avg);
                k += 1;
            }
        }
    }
    avg(&vals)
}

impl FvMesh for ForestMesh {
    fn n_local_cells(&self) -> usize {
        self.n_local
    }
    fn n_cells_total(&self) -> usize {
        self.cells.len()
    }
    fn is_local_cell(&self, c: usize) -> bool {
        !self.cells[c].is_ghost
    }
    fn cell_volume(&self, c: usize) -> f64 {
        let s = self.cells[c].size;
        s[0] * s[1] * s[2]
    }
    fn cell_centroid(&self, c: usize) -> Vec3 {
        self.cells[c].center
    }
    fn for_each_face(&self, f: &mut dyn FnMut(&MeshFace)) {
        for face in &self.faces {
            f(face);
        }
    }
    fn halo_plan(&self) -> &HaloPlan {
        &self.halo
    }
}

impl CartesianMesh for ForestMesh {
    fn axis_neighbors(&self, c: usize, axis: usize, hi: bool) -> &[usize] {
        if c >= self.n_local {
            return &[]; // ghost cells carry no stencil
        }
        let dir = axis * 2 + hi as usize;
        self.axis_nbr[c][dir].as_slice()
    }
}

impl AdaptiveMesh for ForestMesh {
    fn cell_level(&self, c: usize) -> u8 {
        self.cells[c].level
    }
    fn for_each_coarse_fine_face(&self, f: &mut dyn FnMut(&CoarseFineFace)) {
        for cf in &self.coarse_fine {
            f(cf);
        }
    }
}
