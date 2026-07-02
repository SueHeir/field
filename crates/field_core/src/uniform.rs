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
use crate::mesh::{BoundarySide, Face, FvMesh, StructuredMesh, Vec3};

/// Configuration for building a [`UniformMesh`] (the substrate-level grid knobs;
/// a physics crate wraps this in its own TOML section). Deserializable so a
/// `[grid]` config section maps straight onto it.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct UniformMeshConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub ng: usize,
    pub bounds_lo: Vec3,
    pub bounds_hi: Vec3,
    /// Optional explicit cell edges (length `ny+1` / `nz+1`) for stretching.
    pub y_edges: Option<Vec<f64>>,
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
    /// Neighbor rank per `[axis][lo=0/hi=1]`; `-1` = physical boundary.
    neighbor: [[i32; 2]; 3],
    /// Precomputed static halo pattern.
    halo: HaloPlan,
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
        if pos[0] > 0 {
            neighbor[0][0] = rank_of(pos[0] - 1, pos[1], pos[2]);
        }
        if pos[0] < np[0] - 1 {
            neighbor[0][1] = rank_of(pos[0] + 1, pos[1], pos[2]);
        }
        if pos[1] > 0 {
            neighbor[1][0] = rank_of(pos[0], pos[1] - 1, pos[2]);
        }
        if pos[1] < np[1] - 1 {
            neighbor[1][1] = rank_of(pos[0], pos[1] + 1, pos[2]);
        }
        if pos[2] > 0 {
            neighbor[2][0] = rank_of(pos[0], pos[1], pos[2] - 1);
        }
        if pos[2] < np[2] - 1 {
            neighbor[2][1] = rank_of(pos[0], pos[1], pos[2] + 1);
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
            halo: HaloPlan::empty(),
        };
        mesh.halo = mesh.build_halo_plan();
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

    /// Translate toy-cfd's slab packing into explicit per-neighbor index lists.
    /// `send_cells` are the `ng` interior layers adjacent to a face; `recv_cells`
    /// are the `ng` ghost layers on that face. Both ends iterate `(g, i1, i2)` in
    /// the same order, so the flat buffers line up cell-for-cell.
    fn build_halo_plan(&self) -> HaloPlan {
        let ng = self.ng;
        let n = [self.ni, self.nj, self.nk];
        let mut links = Vec::new();

        for dim in 0..3usize {
            let dim1 = (dim + 1) % 3;
            let dim2 = (dim + 2) % 3;
            let n1 = n[dim1];
            let n2 = n[dim2];

            let raw = |d: usize, d1: usize, d2: usize| -> usize {
                let mut ijk = [0usize; 3];
                ijk[dim] = d;
                ijk[dim1] = d1;
                ijk[dim2] = d2;
                self.raw_idx(ijk[0], ijk[1], ijk[2])
            };

            // lo side: send the first ng interior layers, recv into lo ghosts.
            if self.neighbor[dim][0] >= 0 {
                let mut send = Vec::with_capacity(ng * n1 * n2);
                let mut recv = Vec::with_capacity(ng * n1 * n2);
                for g in 0..ng {
                    for i1 in 0..n1 {
                        for i2 in 0..n2 {
                            send.push(raw(ng + g, ng + i1, ng + i2));
                            recv.push(raw(g, ng + i1, ng + i2));
                        }
                    }
                }
                links.push(HaloLink { rank: self.neighbor[dim][0], send_cells: send, recv_cells: recv });
            }

            // hi side: send the last ng interior layers, recv into hi ghosts.
            if self.neighbor[dim][1] >= 0 {
                let mut send = Vec::with_capacity(ng * n1 * n2);
                let mut recv = Vec::with_capacity(ng * n1 * n2);
                for g in 0..ng {
                    for i1 in 0..n1 {
                        for i2 in 0..n2 {
                            send.push(raw(n[dim] + g, ng + i1, ng + i2));
                            recv.push(raw(ng + n[dim] + g, ng + i1, ng + i2));
                        }
                    }
                }
                links.push(HaloLink { rank: self.neighbor[dim][1], send_cells: send, recv_cells: recv });
            }
        }

        HaloPlan { links }
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

/// Factor `size` MPI ranks into a balanced 3D process grid `[npx, npy, npz]`,
/// MPI_Dims_create-style: distribute the prime factors of `size` to whichever axis
/// currently has the most cells per rank, never splitting an axis finer than its
/// global cell count. The product of the result always equals `size`.
///
/// Panics if `size` cannot be placed without giving some rank zero cells (more
/// ranks than the grid can be split into).
pub fn factor_decomposition(size: i32, global: [usize; 3]) -> [i32; 3] {
    assert!(size >= 1, "rank count must be >= 1");
    let mut decomp = [1i32; 3];
    if size == 1 {
        return decomp;
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
        let d = best.unwrap_or_else(|| {
            panic!("cannot decompose {size} ranks over grid {global:?}: too many ranks for the cell count")
        });
        decomp[d] *= fac;
    }
    decomp
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
        assert_eq!(factor_decomposition(1, [10, 10, 10]), [1, 1, 1]);
        assert_eq!(factor_decomposition(8, [16, 16, 16]), [2, 2, 2]);
        // z is a single cell → all splitting goes to x and y.
        assert_eq!(factor_decomposition(4, [16, 16, 1]), [2, 2, 1]);
        assert_eq!(factor_decomposition(7, [14, 1, 1]), [7, 1, 1]);
        for &(size, g) in &[(6i32, [12usize, 12, 12]), (12, [24, 24, 24]), (5, [10, 5, 5])] {
            let d = factor_decomposition(size, g);
            assert_eq!(d[0] * d[1] * d[2], size, "product must equal rank count");
            for axis in 0..3 {
                assert!(d[axis] as usize <= g[axis], "never finer than the cell count");
            }
        }
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
