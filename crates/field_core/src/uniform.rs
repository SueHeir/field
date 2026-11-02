//! [`UniformMesh`] — structured Cartesian grid with ghost layers and pencil MPI
//! decomposition. The physics-free descendant of toy-cfd's `cfd_grid::UniformGrid`.
//!
//! Two things changed in the move to a substrate:
//!
//! 1. **No physics.** `UniformGrid::ghost_exchange` hardcoded `ConsVar` and five
//!    `f64`s per cell. Here the grid only produces a [`HaloPlan`] (index lists);
//!    the generic [`crate::halo`] routines move whatever fields are registered.
//! 2. **It implements the trait family.** `UniformMesh: FvMesh + StructuredMesh`,
//!    so generic FVM physics runs on it, and structured-stencil physics can take
//!    the `(i,j,k)` fast path.
//!
//! X is uniform (spacing `dx`); Y and Z support per-index spacing arrays for
//! stretching. Indices are ghost-offset: interior cell `(0,0,0)` is at flat index
//! `idx(0,0,0) = (ng, ng, ng)` raw.

use crate::halo::{HaloLink, HaloPlan};
use crate::mesh::{BoundarySide, CartesianMesh, Face, FvMesh, StructuredMesh, Vec3};
use std::fmt;

/// Configuration for building a [`UniformMesh`] (the substrate-level grid knobs;
/// a physics crate wraps this in its own TOML section). Deserializable so a
/// `[grid]` config section maps straight onto it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct UniformMeshConfig {
    /// Interior cell count along x.
    pub nx: usize,
    /// Interior cell count along y.
    pub ny: usize,
    /// Interior cell count along z.
    pub nz: usize,
    /// Number of ghost-cell layers on each side.
    pub ng: usize,
    /// Lower corner of the domain bounding box.
    pub bounds_lo: Vec3,
    /// Upper corner of the domain bounding box.
    pub bounds_hi: Vec3,
    /// Optional explicit cell edges (length `ny+1` / `nz+1`) for stretching.
    pub y_edges: Option<Vec<f64>>,
    /// Optional explicit z cell edges (length `nz+1`) for stretching.
    pub z_edges: Option<Vec<f64>>,
}

impl Default for UniformMeshConfig {
    fn default() -> Self {
        Self {
            nx: 1,
            ny: 1,
            nz: 1,
            ng: 2,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0; 3],
            y_edges: None,
            z_edges: None,
        }
    }
}

/// Per-axis periodicity for a [`UniformMesh`], as a standalone resource.
///
/// Kept separate from [`UniformMeshConfig`] so adding it did not break the many
/// struct-literal configs downstream. Insert it before setup runs, e.g.
/// `app.add_resource(MeshPeriodicity([true, false, false]))`, and the mesh
/// plugin's `PreSetup` decomposition wraps those axes: the low and high faces
/// become halo links (to another rank, or to this rank itself when the axis is
/// not split), so the generic halo exchange fills periodic ghosts serial or MPI.
/// Faces on a periodic side carry no boundary patch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MeshPeriodicity(pub [bool; 3]);

/// Structured Cartesian grid resource.
pub struct UniformMesh {
    /// Interior cell counts (local to this rank).
    ni: usize,
    nj: usize,
    nk: usize,
    /// Ghost layers per side.
    ng: usize,
    /// Uniform x spacing.
    dx: f64,
    /// Per-raw-index spacings (length `*_total`), ghosts extrapolated.
    dy_arr: Vec<f64>,
    dz_arr: Vec<f64>,
    /// Per-raw-index cell-center coords for y, z (length `*_total`).
    y_arr: Vec<f64>,
    z_arr: Vec<f64>,
    /// Global domain lower bound.
    origin: Vec3,
    /// Local sub-domain cell offset for this rank.
    offset: [usize; 3],
    /// Total cells per axis including ghosts.
    ni_total: usize,
    nj_total: usize,
    nk_total: usize,
    /// Neighbor rank per `[axis][lo=0/hi=1]`; `-1` = physical boundary. On a
    /// periodic axis this wraps, and may be this rank itself.
    neighbor: [[i32; 2]; 3],
    /// Which axes wrap periodically.
    periodic: [bool; 3],
    /// Precomputed static halo pattern.
    halo: HaloPlan,
    /// Per-axis halo plans that also fill edge and corner ghosts when run in
    /// order (see [`UniformMesh::corner_halo_plans`]).
    corner_halo: [HaloPlan; 3],
    /// Precomputed per-cell axis neighbours (flat, `6·n_cells_total`), indexed
    /// `6·c + (axis·2 + hi)` — the [`CartesianMesh`] stencil for MUSCL/gradients.
    /// Each local cell's six entries are the raw flat indices of the adjacent
    /// cells (interior cell or BC-filled boundary ghost). Non-local cells are
    /// never queried (the trait method returns an empty slice for them).
    axis_nbr: Vec<usize>,
}

impl UniformMesh {
    /// Build a single-process mesh (no decomposition).
    pub fn from_config(cfg: &UniformMeshConfig) -> Self {
        Self::from_config_decomposed(cfg, [1, 1, 1], [0, 0, 0])
    }

    /// Build this rank's local partition for the given process-grid layout.
    pub fn from_config_decomposed(
        cfg: &UniformMeshConfig,
        decomp: [i32; 3],
        pos: [i32; 3],
    ) -> Self {
        Self::from_config_periodic(cfg, decomp, pos, [false; 3])
    }

    /// Build this rank's local partition with the given axes wrapped
    /// periodically. A periodic axis that is not split across ranks gets a
    /// self-link, so the halo exchange fills its ghosts with a local copy.
    ///
    /// # Panics
    /// If a halo link would need more ghost layers than this rank has interior
    /// cells along that axis (`ng > local_n`): the send slab would then include
    /// ghost cells and deliver stale data.
    pub fn from_config_periodic(
        cfg: &UniformMeshConfig,
        decomp: [i32; 3],
        pos: [i32; 3],
        periodic: [bool; 3],
    ) -> Self {
        let global_n = [cfg.nx, cfg.ny, cfg.nz];
        let lo = cfg.bounds_lo;
        let hi = cfg.bounds_hi;

        let mut local_n = [0usize; 3];
        let mut offset = [0usize; 3];
        let dx = (hi[0] - lo[0]) / global_n[0] as f64;

        for dim in 0..3 {
            let base = global_n[dim] / decomp[dim] as usize;
            let remainder = global_n[dim] % decomp[dim] as usize;
            let p = pos[dim] as usize;
            if p < remainder {
                local_n[dim] = base + 1;
                offset[dim] = p * (base + 1);
            } else {
                local_n[dim] = base;
                offset[dim] = remainder * (base + 1) + (p - remainder) * base;
            }
        }

        let ng = cfg.ng;
        let np = decomp;

        let build_global_edges =
            |n: usize, emin: f64, emax: f64, user: &Option<Vec<f64>>| -> Vec<f64> {
                if let Some(e) = user {
                    assert_eq!(e.len(), n + 1, "edges must have length n+1 = {}", n + 1);
                    e.clone()
                } else {
                    (0..=n)
                        .map(|i| emin + (i as f64) * (emax - emin) / (n as f64))
                        .collect()
                }
            };
        let y_edges_global = build_global_edges(cfg.ny, lo[1], hi[1], &cfg.y_edges);
        let z_edges_global = build_global_edges(cfg.nz, lo[2], hi[2], &cfg.z_edges);

        let build_local_arrays =
            |global_edges: &[f64], local_n: usize, offset: usize, ng: usize| -> (Vec<f64>, Vec<f64>) {
                let interior_edges = &global_edges[offset..offset + local_n + 1];
                let mut full_edges: Vec<f64> = Vec::with_capacity(local_n + 2 * ng + 1);
                let d_left = interior_edges[1] - interior_edges[0];
                for k in 0..ng {
                    full_edges.push(interior_edges[0] - ((ng - k) as f64) * d_left);
                }
                full_edges.extend_from_slice(interior_edges);
                let d_right = interior_edges[local_n] - interior_edges[local_n - 1];
                for k in 1..=ng {
                    full_edges.push(interior_edges[local_n] + (k as f64) * d_right);
                }
                let n_total = local_n + 2 * ng;
                let mut d_arr = Vec::with_capacity(n_total);
                let mut c_arr = Vec::with_capacity(n_total);
                for j in 0..n_total {
                    d_arr.push(full_edges[j + 1] - full_edges[j]);
                    c_arr.push(0.5 * (full_edges[j] + full_edges[j + 1]));
                }
                (d_arr, c_arr)
            };
        let (dy_arr, y_arr) = build_local_arrays(&y_edges_global, local_n[1], offset[1], ng);
        let (dz_arr, z_arr) = build_local_arrays(&z_edges_global, local_n[2], offset[2], ng);

        let rank_of =
            |px: i32, py: i32, pz: i32| -> i32 { px * (np[1] * np[2]) + py * np[2] + pz };
        let mut neighbor = [[-1i32; 2]; 3];
        for dim in 0..3 {
            for (side, step) in [(0usize, -1i32), (1, 1)] {
                let mut p = pos;
                p[dim] += step;
                if p[dim] < 0 || p[dim] >= np[dim] {
                    if !periodic[dim] {
                        continue;
                    }
                    p[dim] = p[dim].rem_euclid(np[dim]);
                }
                neighbor[dim][side] = rank_of(p[0], p[1], p[2]);
            }
            // A periodic axis on one rank links to itself; its ghosts can wrap
            // round the interior more than once, so it may have fewer cells
            // than ghost layers (a 1-cell-thick "2-D" axis).
            let self_wrap = periodic[dim] && np[dim] == 1;
            if neighbor[dim] != [-1, -1] && !self_wrap {
                assert!(
                    local_n[dim] >= ng,
                    "UniformMesh: axis {dim} has {} local cells but ng = {ng} ghost layers; \
                     a halo link needs at least ng interior layers",
                    local_n[dim]
                );
            }
        }

        let mut mesh = UniformMesh {
            ni: local_n[0],
            nj: local_n[1],
            nk: local_n[2],
            ng,
            dx,
            dy_arr,
            dz_arr,
            y_arr,
            z_arr,
            origin: lo,
            offset,
            ni_total: local_n[0] + 2 * ng,
            nj_total: local_n[1] + 2 * ng,
            nk_total: local_n[2] + 2 * ng,
            neighbor,
            periodic,
            halo: HaloPlan::empty(),
            corner_halo: [HaloPlan::empty(), HaloPlan::empty(), HaloPlan::empty()],
            axis_nbr: Vec::new(),
        };
        mesh.halo = mesh.build_halo_plan();
        mesh.corner_halo = [0, 1, 2].map(|dim| mesh.build_axis_halo_plan(dim, true));
        mesh.axis_nbr = mesh.build_axis_nbr();
        mesh
    }

    /// Raw flat index from raw (ghost-inclusive) per-axis indices.
    #[inline]
    fn raw_idx(&self, ir: usize, jr: usize, kr: usize) -> usize {
        ir * self.nj_total * self.nk_total + jr * self.nk_total + kr
    }

    /// Decompose a flat index back into raw per-axis indices.
    #[inline]
    fn raw_ijk(&self, c: usize) -> (usize, usize, usize) {
        let kr = c % self.nk_total;
        let jr = (c / self.nk_total) % self.nj_total;
        let ir = c / (self.nj_total * self.nk_total);
        (ir, jr, kr)
    }

    /// Which axes wrap periodically.
    pub fn periodic(&self) -> [bool; 3] {
        self.periodic
    }

    /// Neighbor rank across the low (`hi = false`) or high face of `axis`, or
    /// `None` at a physical (non-periodic) boundary. May be this rank itself on a
    /// periodic axis that is not split.
    pub fn neighbor_rank(&self, axis: usize, hi: bool) -> Option<i32> {
        let r = self.neighbor[axis][hi as usize];
        (r >= 0).then_some(r)
    }

    /// Translate toy-cfd's slab packing into explicit per-neighbor index lists.
    /// `send_cells` are the `ng` interior layers adjacent to a face; `recv_cells`
    /// are the `ng` ghost layers on that face. Both ends iterate `(g, i1, i2)` in
    /// the same order, so the flat buffers line up cell-for-cell.
    ///
    /// When the low and high neighbours along an axis are the *same* rank (a
    /// periodic axis split in two, or not split at all), the two sides are merged
    /// into one link: send `[lo slab, hi slab]`, receive into `[hi ghosts, lo
    /// ghosts]`. That is symmetric — the peer's lo slab lands in our hi ghosts
    /// and vice versa — so both ends agree without message tags, and the plan
    /// keeps at most one link per neighbour rank per axis (the ordering contract
    /// of [`HaloPlan`]).
    fn build_halo_plan(&self) -> HaloPlan {
        let links = (0..3)
            .flat_map(|dim| self.build_axis_halo_plan(dim, false).links)
            .collect();
        HaloPlan { links }
    }

    /// The links for one axis. With `corners`, the slab also spans the ghost
    /// layers of every *earlier* axis that has a halo link on that side, so
    /// running the three per-axis plans in x, y, z order carries values into
    /// edge and corner ghosts (and the reverse order folds them back). Both
    /// ends of a link sit at the same position along the earlier axes, so they
    /// agree on which ghost layers are included and the orderings match.
    fn build_axis_halo_plan(&self, dim: usize, corners: bool) -> HaloPlan {
        let ng = self.ng;
        let n = [self.ni, self.nj, self.nk];
        let mut links = Vec::new();
        let dim1 = (dim + 1) % 3;
        let dim2 = (dim + 2) % 3;
        // Transverse raw-index range: interior, widened by the linked ghost
        // layers of an earlier axis when `corners` is set.
        let range = |a: usize| -> std::ops::Range<usize> {
            let widen = corners && a < dim;
            let lo = if widen && self.neighbor[a][0] >= 0 { 0 } else { ng };
            let hi = if widen && self.neighbor[a][1] >= 0 { n[a] + 2 * ng } else { ng + n[a] };
            lo..hi
        };
        let (r1, r2) = (range(dim1), range(dim2));

        let raw = |d: usize, d1: usize, d2: usize| -> usize {
            let mut ijk = [0usize; 3];
            ijk[dim] = d;
            ijk[dim1] = d1;
            ijk[dim2] = d2;
            self.raw_idx(ijk[0], ijk[1], ijk[2])
        };
        // `ng` layers along `dim` starting at raw index `start`.
        let slab = |start: usize| -> Vec<usize> {
            let mut v = Vec::with_capacity(ng * r1.len() * r2.len());
            for g in 0..ng {
                for i1 in r1.clone() {
                    for i2 in r2.clone() {
                        v.push(raw(start + g, i1, i2));
                    }
                }
            }
            v
        };
        // One layer along `dim` at raw index `d`.
        let slab_at = |d: usize| -> Vec<usize> {
            let mut v = Vec::with_capacity(r1.len() * r2.len());
            for i1 in r1.clone() {
                for i2 in r2.clone() {
                    v.push(raw(d, i1, i2));
                }
            }
            v
        };
        // lo: send the first ng interior layers, recv into the lo ghosts.
        // hi: send the last ng interior layers, recv into the hi ghosts.
        let [lo, hi] = self.neighbor[dim];
        if lo >= 0 && lo == hi && n[dim] < ng {
            // Only a periodic axis on one rank gets here (checked at build):
            // each ghost layer copies the interior layer it wraps onto, which
            // can be the same layer for several ghosts.
            let wrap = |layer: isize| ng + layer.rem_euclid(n[dim] as isize) as usize;
            let mut send_cells = Vec::new();
            let mut recv_cells = Vec::new();
            for g in 0..ng {
                // lo ghost layer g sits at logical index g - ng; hi ghost layer
                // g at n + g.
                send_cells.extend(slab_at(wrap(g as isize - ng as isize)));
                recv_cells.extend(slab_at(g));
                send_cells.extend(slab_at(wrap((n[dim] + g) as isize)));
                recv_cells.extend(slab_at(ng + n[dim] + g));
            }
            links.push(HaloLink { rank: lo, send_cells, recv_cells });
            return HaloPlan { links };
        }
        if lo >= 0 && lo == hi {
            let mut send_cells = slab(ng);
            send_cells.extend(slab(n[dim]));
            let mut recv_cells = slab(ng + n[dim]);
            recv_cells.extend(slab(0));
            links.push(HaloLink { rank: lo, send_cells, recv_cells });
            return HaloPlan { links };
        }
        if lo >= 0 {
            links.push(HaloLink { rank: lo, send_cells: slab(ng), recv_cells: slab(0) });
        }
        if hi >= 0 {
            links.push(HaloLink {
                rank: hi,
                send_cells: slab(n[dim]),
                recv_cells: slab(ng + n[dim]),
            });
        }
        HaloPlan { links }
    }

    /// Three per-axis halo plans (x, y, z) that, run **in order**, also fill
    /// edge and corner (diagonal) ghosts, which [`FvMesh::halo_plan`] leaves
    /// alone. Run them in reverse order for a reverse (ghost → owner) exchange.
    /// Use them with [`crate::halo_exchange_forward_all_with_corners`] and
    /// [`crate::halo_exchange_reverse_all_with_corners`].
    ///
    /// Needed by anything that reads or deposits through a trilinear stencil
    /// near a subdomain corner: particle-in-cell charge deposit and field
    /// gather, MPM grid transfers, IBM interpolation. Costs three dependent
    /// rounds of messages instead of one, so face-only FVM keeps using
    /// [`FvMesh::halo_plan`].
    pub fn corner_halo_plans(&self) -> &[HaloPlan; 3] {
        &self.corner_halo
    }

    /// Precompute the six axis neighbours of every cell as raw flat indices, in
    /// the [`CartesianMesh::axis_neighbors`] direction order (`axis·2 + hi`). Only
    /// the entries of *local* cells are ever read; on a uniform grid every local
    /// cell has one neighbour on each side (an interior cell, or a boundary ghost
    /// the BC fills), so the table is exact and never empty for a local cell.
    /// Ghost-cell rows are filled with the cell's own index as a harmless
    /// placeholder — `axis_neighbors` returns an empty slice for non-local cells,
    /// so those rows are never sliced.
    fn build_axis_nbr(&self) -> Vec<usize> {
        let total = self.ni_total * self.nj_total * self.nk_total;
        // Flat strides: x is outermost, z innermost (see `raw_idx`).
        let stride = [self.nj_total * self.nk_total, self.nk_total, 1usize];
        let extent = [self.ni_total, self.nj_total, self.nk_total];
        let mut nbr = vec![0usize; total * 6];
        for c in 0..total {
            let ijk = {
                let (ir, jr, kr) = self.raw_ijk(c);
                [ir, jr, kr]
            };
            for axis in 0..3 {
                for hi in [false, true] {
                    let dir = axis * 2 + hi as usize;
                    let idx = self.raw_ijk_step(c, ijk, axis, hi, stride[axis], extent[axis]);
                    nbr[c * 6 + dir] = idx;
                }
            }
        }
        nbr
    }

    /// Raw flat index of the neighbour of `c` (per-axis indices `ijk`) one cell
    /// along `axis` toward the high side if `hi`. Returns `c` itself when the step
    /// would leave the padded grid — only happens on the outermost ghost ring,
    /// which is never queried.
    #[inline]
    fn raw_ijk_step(
        &self,
        c: usize,
        ijk: [usize; 3],
        axis: usize,
        hi: bool,
        stride: usize,
        extent: usize,
    ) -> usize {
        if hi {
            if ijk[axis] + 1 < extent {
                c + stride
            } else {
                c
            }
        } else if ijk[axis] >= 1 {
            c - stride
        } else {
            c
        }
    }

    /// x cell-center coordinate for a raw i index (handles ghosts via signed offset).
    #[inline]
    fn x_center_raw(&self, ir: usize) -> f64 {
        let i_int = ir as i64 - self.ng as i64; // may be negative in the ghost band
        self.origin[0] + (self.offset[0] as i64 + i_int) as f64 * self.dx + 0.5 * self.dx
    }
}

impl FvMesh for UniformMesh {
    fn n_local_cells(&self) -> usize {
        self.ni * self.nj * self.nk
    }

    fn n_cells_total(&self) -> usize {
        self.ni_total * self.nj_total * self.nk_total
    }

    fn is_local_cell(&self, c: usize) -> bool {
        let (ir, jr, kr) = self.raw_ijk(c);
        let ng = self.ng;
        ir >= ng
            && ir < ng + self.ni
            && jr >= ng
            && jr < ng + self.nj
            && kr >= ng
            && kr < ng + self.nk
    }

    fn cell_volume(&self, c: usize) -> f64 {
        let (_, jr, kr) = self.raw_ijk(c);
        self.dx * self.dy_arr[jr] * self.dz_arr[kr]
    }

    fn cell_centroid(&self, c: usize) -> Vec3 {
        let (ir, jr, kr) = self.raw_ijk(c);
        [self.x_center_raw(ir), self.y_arr[jr], self.z_arr[kr]]
    }

    fn for_each_face(&self, f: &mut dyn FnMut(&Face)) {
        let ng = self.ng;
        // Emit, for each interior cell, its three positive-axis faces, plus the
        // low-side faces of the cells sitting on the low boundary. A positive face
        // at the high boundary, and a low face at the low boundary, are physical
        // boundary faces only when there is no MPI neighbor on that side.
        for i in 0..self.ni {
            for j in 0..self.nj {
                for k in 0..self.nk {
                    let owner = self.raw_idx(i + ng, j + ng, k + ng);
                    let dy = self.dy_arr[j + ng];
                    let dz = self.dz_arr[k + ng];
                    let dx = self.dx;
                    let cc = self.cell_centroid(owner);

                    // +x
                    {
                        let other = self.raw_idx(i + ng + 1, j + ng, k + ng);
                        let area = dy * dz;
                        let patch = if i + 1 == self.ni && self.neighbor[0][1] < 0 {
                            Some(BoundarySide::XHi as u32)
                        } else {
                            None
                        };
                        f(&Face {
                            owner,
                            other,
                            area_normal: [area, 0.0, 0.0],
                            centroid: [cc[0] + 0.5 * dx, cc[1], cc[2]],
                            patch,
                        });
                    }
                    // -x boundary face only at the low edge
                    if i == 0 {
                        let other = self.raw_idx(i + ng - 1, j + ng, k + ng);
                        let area = dy * dz;
                        let patch = if self.neighbor[0][0] < 0 {
                            Some(BoundarySide::XLo as u32)
                        } else {
                            None
                        };
                        f(&Face {
                            owner,
                            other,
                            area_normal: [-area, 0.0, 0.0],
                            centroid: [cc[0] - 0.5 * dx, cc[1], cc[2]],
                            patch,
                        });
                    }

                    // +y
                    {
                        let other = self.raw_idx(i + ng, j + ng + 1, k + ng);
                        let area = dx * dz;
                        let patch = if j + 1 == self.nj && self.neighbor[1][1] < 0 {
                            Some(BoundarySide::YHi as u32)
                        } else {
                            None
                        };
                        f(&Face {
                            owner,
                            other,
                            area_normal: [0.0, area, 0.0],
                            centroid: [cc[0], cc[1] + 0.5 * dy, cc[2]],
                            patch,
                        });
                    }
                    if j == 0 {
                        let other = self.raw_idx(i + ng, j + ng - 1, k + ng);
                        let area = dx * dz;
                        let patch = if self.neighbor[1][0] < 0 {
                            Some(BoundarySide::YLo as u32)
                        } else {
                            None
                        };
                        f(&Face {
                            owner,
                            other,
                            area_normal: [0.0, -area, 0.0],
                            centroid: [cc[0], cc[1] - 0.5 * dy, cc[2]],
                            patch,
                        });
                    }

                    // +z
                    {
                        let other = self.raw_idx(i + ng, j + ng, k + ng + 1);
                        let area = dx * dy;
                        let patch = if k + 1 == self.nk && self.neighbor[2][1] < 0 {
                            Some(BoundarySide::ZHi as u32)
                        } else {
                            None
                        };
                        f(&Face {
                            owner,
                            other,
                            area_normal: [0.0, 0.0, area],
                            centroid: [cc[0], cc[1], cc[2] + 0.5 * dz],
                            patch,
                        });
                    }
                    if k == 0 {
                        let other = self.raw_idx(i + ng, j + ng, k + ng - 1);
                        let area = dx * dy;
                        let patch = if self.neighbor[2][0] < 0 {
                            Some(BoundarySide::ZLo as u32)
                        } else {
                            None
                        };
                        f(&Face {
                            owner,
                            other,
                            area_normal: [0.0, 0.0, -area],
                            centroid: [cc[0], cc[1], cc[2] - 0.5 * dz],
                            patch,
                        });
                    }
                }
            }
        }
    }

    fn halo_plan(&self) -> &HaloPlan {
        &self.halo
    }
}

impl StructuredMesh for UniformMesh {
    fn dims(&self) -> [usize; 3] {
        [self.ni, self.nj, self.nk]
    }

    fn n_ghost(&self) -> usize {
        self.ng
    }

    fn total_dims(&self) -> [usize; 3] {
        [self.ni_total, self.nj_total, self.nk_total]
    }

    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        self.raw_idx(i + self.ng, j + self.ng, k + self.ng)
    }

    fn idx_raw(&self, ir: usize, jr: usize, kr: usize) -> usize {
        self.raw_idx(ir, jr, kr)
    }

    fn spacing(&self, axis: usize, n: usize) -> f64 {
        match axis {
            0 => self.dx,
            1 => self.dy_arr[n + self.ng],
            2 => self.dz_arr[n + self.ng],
            _ => panic!("axis must be 0, 1, or 2"),
        }
    }

    fn locate(&self, point: Vec3) -> Option<([usize; 3], [f64; 3])> {
        // Uniform x: raw cell ir covers interior i = ir-ng, center at
        // origin + (offset + i + 0.5)*dx.
        let pf = (point[0] - self.origin[0]) / self.dx - 0.5 - self.offset[0] as f64
            + self.ng as f64;
        let lo_i = pf.floor();
        if lo_i < 0.0 || lo_i >= (self.ni_total as f64) - 1.0 {
            return None;
        }
        let i_lo = lo_i as usize;
        let tx = (pf - lo_i).clamp(0.0, 1.0);

        // Stretched y / z: bracket against the raw cell-center arrays.
        let (j_lo, ty) = bracket_centers(point[1], &self.y_arr)?;
        let (k_lo, tz) = bracket_centers(point[2], &self.z_arr)?;
        Some(([i_lo, j_lo, k_lo], [tx, ty, tz]))
    }
}

impl CartesianMesh for UniformMesh {
    fn axis_neighbors(&self, c: usize, axis: usize, hi: bool) -> &[usize] {
        // Only local cells carry a reconstruction stencil (mirrors ForestMesh);
        // a boundary-adjacent local cell's outward neighbour is the BC-filled
        // ghost, exactly what the MUSCL slope pass wants.
        if !self.is_local_cell(c) {
            return &[];
        }
        let dir = axis * 2 + hi as usize;
        &self.axis_nbr[c * 6 + dir..c * 6 + dir + 1]
    }
}

/// Bracket `p` against a raw cell-center array: find `lo` with
/// `arr[lo] <= p < arr[lo+1]` and the fraction within. `None` if outside.
fn bracket_centers(p: f64, arr: &[f64]) -> Option<(usize, f64)> {
    if arr.len() < 2 || p < arr[0] || p > arr[arr.len() - 1] {
        return None;
    }
    let mut lo = arr.len() - 2;
    for j in 0..arr.len() - 1 {
        if arr[j] <= p && p < arr[j + 1] {
            lo = j;
            break;
        }
    }
    let denom = arr[lo + 1] - arr[lo];
    let t = if denom.abs() < 1e-30 { 0.0 } else { ((p - arr[lo]) / denom).clamp(0.0, 1.0) };
    Some((lo, t))
}

/// Error returned by the fallible mesh decomposition entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionError {
    /// MPI reported a non-positive rank count.
    InvalidRankCount(i32),
    /// More ranks were requested than can be assigned without zero-cell ranks.
    TooManyRanks {
        /// Requested communicator size.
        size: i32,
        /// Global cell counts used to constrain the process grid.
        global: [usize; 3],
    },
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompositionError::InvalidRankCount(size) => {
                write!(f, "rank count must be >= 1, got {size}")
            }
            DecompositionError::TooManyRanks { size, global } => write!(
                f,
                "cannot decompose {size} ranks over grid {global:?}: too many ranks for the cell count"
            ),
        }
    }
}

impl std::error::Error for DecompositionError {}

/// Factor `size` MPI ranks into a balanced 3D process grid `[npx, npy, npz]`,
/// MPI_Dims_create-style: distribute the prime factors of `size` to whichever axis
/// currently has the most cells per rank, never splitting an axis finer than its
/// global cell count. The product of the result always equals `size` on success.
///
/// Returns [`DecompositionError::TooManyRanks`] if `size` cannot be placed
/// without giving some rank zero cells.
pub fn factor_decomposition(
    size: i32,
    global: [usize; 3],
) -> Result<[i32; 3], DecompositionError> {
    if size < 1 {
        return Err(DecompositionError::InvalidRankCount(size));
    }
    let mut decomp = [1i32; 3];
    if size == 1 {
        return Ok(decomp);
    }

    // Prime factorization of `size`, largest factor first (greedy balances better).
    let mut factors = Vec::new();
    let mut n = size;
    let mut f = 2;
    while f * f <= n {
        while n % f == 0 {
            factors.push(f);
            n /= f;
        }
        f += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors.sort_unstable_by(|a, b| b.cmp(a));

    for fac in factors {
        let mut best: Option<usize> = None;
        let mut best_load = -1.0f64;
        for d in 0..3 {
            let next = decomp[d] as usize * fac as usize;
            if next <= global[d].max(1) {
                let load = global[d] as f64 / decomp[d] as f64;
                if load > best_load {
                    best_load = load;
                    best = Some(d);
                }
            }
        }
        let d = best.ok_or(DecompositionError::TooManyRanks { size, global })?;
        decomp[d] *= fac;
    }
    Ok(decomp)
}

/// This rank's `[px, py, pz]` position in the process grid, inverting the rank
/// ordering `rank = px*(npy*npz) + py*npz + pz` that [`UniformMesh`] uses for
/// neighbor lookup.
pub fn rank_position(rank: i32, decomp: [i32; 3]) -> [i32; 3] {
    let ny = decomp[1];
    let nz = decomp[2];
    [rank / (ny * nz), (rank / nz) % ny, rank % nz]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> UniformMeshConfig {
        UniformMeshConfig {
            nx: 10,
            ny: 10,
            nz: 10,
            ng: 2,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0; 3],
            y_edges: None,
            z_edges: None,
        }
    }

    #[test]
    fn cell_counts() {
        let m = UniformMesh::from_config(&cfg());
        assert_eq!(m.n_cells_total(), 14 * 14 * 14);
        assert_eq!(m.n_local_cells(), 1000);
    }

    #[test]
    fn idx_and_centroid() {
        let m = UniformMesh::from_config(&cfg());
        assert_eq!(m.idx(0, 0, 0), 422); // (2)*196 + (2)*14 + 2
        let cc = m.cell_centroid(m.idx(0, 0, 0));
        assert!((cc[0] - 0.05).abs() < 1e-12);
        assert!((cc[1] - 0.05).abs() < 1e-12);
        assert!((cc[2] - 0.05).abs() < 1e-12);
    }

    #[test]
    fn idx_raw_addresses_ghosts_and_matches_interior() {
        let m = UniformMesh::from_config(&cfg()); // ng = 2
        assert_eq!(m.idx(3, 4, 5), m.idx_raw(3 + 2, 4 + 2, 5 + 2));
        // a raw index inside the ghost band is not a local cell.
        assert!(!m.is_local_cell(m.idx_raw(1, 2, 2)));
    }

    #[test]
    fn local_cell_predicate_counts_interior() {
        let m = UniformMesh::from_config(&cfg());
        assert!(m.is_local_cell(m.idx(0, 0, 0)));
        assert!(!m.is_local_cell(m.idx_raw(0, 0, 0))); // a corner ghost
        let local = (0..m.n_cells_total()).filter(|&c| m.is_local_cell(c)).count();
        assert_eq!(local, m.n_local_cells());
    }

    #[test]
    fn axis_neighbors_step_one_cell_each_direction() {
        let m = UniformMesh::from_config(&cfg()); // 10³, ng=2
        // Interior cell (3,4,5): each axis neighbour is exactly the adjacent
        // interior cell, matching the structured index step.
        let c = m.idx(3, 4, 5);
        assert_eq!(m.axis_neighbors(c, 0, true), &[m.idx(4, 4, 5)]);
        assert_eq!(m.axis_neighbors(c, 0, false), &[m.idx(2, 4, 5)]);
        assert_eq!(m.axis_neighbors(c, 1, true), &[m.idx(3, 5, 5)]);
        assert_eq!(m.axis_neighbors(c, 1, false), &[m.idx(3, 3, 5)]);
        assert_eq!(m.axis_neighbors(c, 2, true), &[m.idx(3, 4, 6)]);
        assert_eq!(m.axis_neighbors(c, 2, false), &[m.idx(3, 4, 4)]);
    }

    #[test]
    fn axis_neighbors_boundary_cell_returns_ghost_and_ghosts_return_empty() {
        let m = UniformMesh::from_config(&cfg()); // ng = 2
        // The low-x boundary interior cell's outward (-x) neighbour is the raw
        // ghost just outside the domain (index one step below in raw x), which a
        // BC fills — a single valid index, never empty.
        let edge = m.idx(0, 4, 5);
        let lo = m.axis_neighbors(edge, 0, false);
        assert_eq!(lo.len(), 1);
        assert!(!m.is_local_cell(lo[0]), "outward neighbour of a boundary cell is a ghost");
        assert_eq!(lo[0], m.idx_raw(1, 4 + 2, 5 + 2));
        // Non-local (ghost) cells carry no reconstruction stencil.
        assert!(m.axis_neighbors(m.idx_raw(0, 0, 0), 0, true).is_empty());
    }

    #[test]
    fn locate_brackets_a_point() {
        let m = UniformMesh::from_config(&cfg()); // 10³, ng=2, [0,1]³, dx=0.1
        // 0.1 sits halfway between raw cell 2 (center 0.05) and 3 (center 0.15).
        let (lo, t) = m.locate([0.1, 0.1, 0.1]).unwrap();
        assert_eq!(lo, [2, 2, 2]);
        assert!((0..3).all(|a| (t[a] - 0.5).abs() < 1e-12));
        // The 8 bracketing cells are addressable and include the interior cell.
        assert_eq!(m.idx_raw(lo[0], lo[1], lo[2]), m.idx(0, 0, 0));
        // Outside the domain → None.
        assert!(m.locate([5.0, 0.5, 0.5]).is_none());
    }

    #[test]
    fn uniform_cell_volume() {
        let m = UniformMesh::from_config(&cfg());
        let v = m.cell_volume(m.idx(3, 4, 5));
        assert!((v - 0.1 * 0.1 * 0.1).abs() < 1e-15);
    }

    #[test]
    fn serial_has_no_halo() {
        let m = UniformMesh::from_config(&cfg());
        assert!(m.halo_plan().is_serial());
    }

    #[test]
    fn decomposition_factors_balance_and_multiply() {
        assert_eq!(factor_decomposition(1, [10, 10, 10]).unwrap(), [1, 1, 1]);
        assert_eq!(factor_decomposition(8, [16, 16, 16]).unwrap(), [2, 2, 2]);
        // z is a single cell → all splitting goes to x and y.
        assert_eq!(factor_decomposition(4, [16, 16, 1]).unwrap(), [2, 2, 1]);
        assert_eq!(factor_decomposition(7, [14, 1, 1]).unwrap(), [7, 1, 1]);
        for &(size, g) in &[(6i32, [12usize, 12, 12]), (12, [24, 24, 24]), (5, [10, 5, 5])] {
            let d = factor_decomposition(size, g).unwrap();
            assert_eq!(d[0] * d[1] * d[2], size, "product must equal rank count");
            for axis in 0..3 {
                assert!(d[axis] as usize <= g[axis], "never finer than the cell count");
            }
        }
    }

    #[test]
    fn decomposition_rejects_over_subscribed_ranks_without_panicking() {
        let err = factor_decomposition(9, [2, 2, 2]).unwrap_err();
        assert_eq!(err, DecompositionError::TooManyRanks { size: 9, global: [2, 2, 2] });
        assert_eq!(
            err.to_string(),
            "cannot decompose 9 ranks over grid [2, 2, 2]: too many ranks for the cell count"
        );
    }

    #[test]
    fn rank_position_inverts_neighbor_ordering() {
        let decomp = [2, 2, 2];
        // rank = px*(ny*nz) + py*nz + pz = 1*4 + 0*2 + 1 = 5
        assert_eq!(rank_position(5, decomp), [1, 0, 1]);
        // every rank maps to a distinct, in-range position.
        let n = decomp[0] * decomp[1] * decomp[2];
        let mut seen = std::collections::HashSet::new();
        for r in 0..n {
            let p = rank_position(r, decomp);
            assert!((0..3).all(|a| p[a] >= 0 && p[a] < decomp[a]));
            assert!(seen.insert(p), "positions must be unique");
        }
    }

    #[test]
    fn decomposed_halo_links_match_interface_size() {
        // Split x into 2; rank 0 has one +x neighbor, so exactly one link.
        let m = UniformMesh::from_config_decomposed(&cfg(), [2, 1, 1], [0, 0, 0]);
        let plan = m.halo_plan();
        assert_eq!(plan.links.len(), 1);
        let link = &plan.links[0];
        // ng layers * nj * nk interior cells on the shared face.
        assert_eq!(link.send_cells.len(), 2 * 10 * 10);
        assert_eq!(link.send_cells.len(), link.recv_cells.len());
    }

    #[test]
    fn every_interior_cell_owns_faces() {
        let m = UniformMesh::from_config(&cfg());
        let mut count = 0usize;
        let mut boundary = 0usize;
        m.for_each_face(&mut |face| {
            count += 1;
            if face.patch.is_some() {
                boundary += 1;
            }
        });
        // 10^3 cells * 3 positive faces + 3*10^2 low-boundary faces.
        assert_eq!(count, 1000 * 3 + 3 * 100);
        // 6 sides * 100 faces are physical boundaries (serial: all sides physical).
        assert_eq!(boundary, 6 * 100);
    }
}
