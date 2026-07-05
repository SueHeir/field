//! `field_amr` — a block-adaptive [`ForestMesh`] for the FIELD substrate.
//!
//! This is the M2 AMR branch FIELD was designed for: `FvMesh` made the physics
//! mesh-shape-agnostic, and [`field_core::AdaptiveMesh`] reserved the read-side
//! contract for refinement. `ForestMesh` is a two-level, block-refined Cartesian
//! mesh — a coarse base grid where selected interior cells are split into a
//! `2×2×2` block of fine leaves — that implements `FvMesh + AdaptiveMesh`, so the
//! existing inviscid solver runs on it unchanged.
//!
//! ## Conservation by construction
//!
//! The subtle part of AMR is the coarse/fine flux interface: a coarse cell's face
//! abuts several smaller fine faces, and naive independent fluxes are
//! non-conservative (the classic "refluxing" problem). `ForestMesh` sidesteps the
//! separate reflux pass: [`for_each_face`](field_core::FvMesh::for_each_face)
//! emits **one face per fine sub-interface** (at fine-face resolution) everywhere.
//! The solver's two-sided scatter then sums the fine-face fluxes into the coarse
//! owner automatically — conservative by construction. Same-level coarse faces
//! are emitted as their (2 or 4) fine sub-faces too: redundant flux evaluations,
//! but exactly conservative.
//!
//! This is a pure-Rust, single-rank demonstration of the AMR architecture (no
//! p4est, no MPI, no dynamic regridding yet) — enough to prove the substrate
//! contract carries adaptive meshes and that the unchanged physics conserves
//! across refinement boundaries.

#![warn(missing_docs)]

// `ForestMesh` is intentionally NOT a `StructuredMesh` (no global i,j,k):
// structured-only physics (viscous, IBM) stays on `UniformMesh`; the inviscid
// solver runs here through the generic `FvMesh` path.
use field_core::{AdaptiveMesh, CoarseFineFace, Face, FvMesh, HaloPlan, Vec3};

/// Build spec for a [`ForestMesh`]: a coarse base grid plus the interior coarse
/// cells (0-based interior indices) to refine `2×2×2`.
#[derive(Clone, Debug)]
pub struct ForestConfig {
    /// Coarse base-grid cell count along x.
    pub ncx: usize,
    /// Coarse base-grid cell count along y.
    pub ncy: usize,
    /// Coarse base-grid cell count along z.
    pub ncz: usize,
    /// Lower corner of the domain bounding box.
    pub bounds_lo: Vec3,
    /// Upper corner of the domain bounding box.
    pub bounds_hi: Vec3,
    /// Interior coarse cells `[i, j, k]` (each in `0..nc`) to refine. Cells on the
    /// boundary cannot be refined (keeps boundary faces coarse-coarse).
    pub refine: Vec<[usize; 3]>,
}

#[derive(Clone, Copy)]
struct Leaf {
    level: u8,
    center: Vec3,
    size: Vec3,
    is_ghost: bool,
}

/// A two-level block-adaptive Cartesian mesh.
pub struct ForestMesh {
    leaves: Vec<Leaf>,
    n_local: usize,
    faces: Vec<Face>,
    coarse_fine: Vec<CoarseFineFace>,
    halo: HaloPlan,
}

impl ForestMesh {
    /// Builds a [`ForestMesh`] from a [`ForestConfig`] (coarse base grid plus refinement list).
    pub fn from_config(cfg: &ForestConfig) -> Self {
        let nc = [cfg.ncx, cfg.ncy, cfg.ncz];
        let ng = 1usize; // one coarse ghost layer
        let dc = [
            (cfg.bounds_hi[0] - cfg.bounds_lo[0]) / nc[0] as f64,
            (cfg.bounds_hi[1] - cfg.bounds_lo[1]) / nc[1] as f64,
            (cfg.bounds_hi[2] - cfg.bounds_lo[2]) / nc[2] as f64,
        ];
        let nct = [nc[0] + 2 * ng, nc[1] + 2 * ng, nc[2] + 2 * ng]; // coarse incl ghost
        let nf = [2 * nct[0], 2 * nct[1], 2 * nct[2]]; // fine slot grid

        // Refined coarse cells, by raw (ghost-inclusive) coarse index.
        let mut refined = std::collections::HashSet::new();
        for r in &cfg.refine {
            assert!(r[0] < nc[0] && r[1] < nc[1] && r[2] < nc[2], "refine index out of range");
            refined.insert([r[0] + ng, r[1] + ng, r[2] + ng]);
        }

        // Lower corner of a raw coarse cell along an axis.
        let coarse_lo = |ci: usize, axis: usize| cfg.bounds_lo[axis] + (ci as f64 - ng as f64) * dc[axis];
        let is_ghost_coarse = |c: [usize; 3]| {
            (0..3).any(|a| c[a] < ng || c[a] >= ng + nc[a])
        };

        let mut leaves: Vec<Leaf> = Vec::new();
        // slot_to_leaf[fi*nf1*nf2 + fj*nf2 + fk] = leaf index
        let mut slot_to_leaf = vec![usize::MAX; nf[0] * nf[1] * nf[2]];
        let slot_idx = |fi: usize, fj: usize, fk: usize| (fi * nf[1] + fj) * nf[2] + fk;

        // Build leaves coarse cell by coarse cell.
        for ci in 0..nct[0] {
            for cj in 0..nct[1] {
                for ck in 0..nct[2] {
                    let cc = [ci, cj, ck];
                    let ghost = is_ghost_coarse(cc);
                    let lo = [coarse_lo(ci, 0), coarse_lo(cj, 1), coarse_lo(ck, 2)];
                    if refined.contains(&cc) {
                        // 8 fine leaves.
                        let hf = [dc[0] / 2.0, dc[1] / 2.0, dc[2] / 2.0];
                        for sx in 0..2 {
                            for sy in 0..2 {
                                for sz in 0..2 {
                                    let center = [
                                        lo[0] + (sx as f64 + 0.5) * hf[0],
                                        lo[1] + (sy as f64 + 0.5) * hf[1],
                                        lo[2] + (sz as f64 + 0.5) * hf[2],
                                    ];
                                    let li = leaves.len();
                                    leaves.push(Leaf { level: 1, center, size: hf, is_ghost: ghost });
                                    slot_to_leaf[slot_idx(2 * ci + sx, 2 * cj + sy, 2 * ck + sz)] = li;
                                }
                            }
                        }
                    } else {
                        let center = [lo[0] + 0.5 * dc[0], lo[1] + 0.5 * dc[1], lo[2] + 0.5 * dc[2]];
                        let li = leaves.len();
                        leaves.push(Leaf { level: 0, center, size: dc, is_ghost: ghost });
                        for sx in 0..2 {
                            for sy in 0..2 {
                                for sz in 0..2 {
                                    slot_to_leaf[slot_idx(2 * ci + sx, 2 * cj + sy, 2 * ck + sz)] = li;
                                }
                            }
                        }
                    }
                }
            }
        }
        let n_local = leaves.iter().filter(|l| !l.is_ghost).count();

        // Build faces by walking fine-slot interfaces. Each axis: the +direction
        // face of slot s connects to slot s+ê; emit once when the leaves differ.
        let mut faces = Vec::new();
        let mut coarse_fine = Vec::new();
        for fi in 0..nf[0] {
            for fj in 0..nf[1] {
                for fk in 0..nf[2] {
                    let a = slot_to_leaf[slot_idx(fi, fj, fk)];
                    let here = [fi, fj, fk];
                    for axis in 0..3 {
                        let mut nb = here;
                        nb[axis] += 1;
                        if nb[axis] >= nf[axis] {
                            continue;
                        }
                        let b = slot_to_leaf[slot_idx(nb[0], nb[1], nb[2])];
                        if a == b {
                            continue;
                        }
                        let (ag, bg) = (leaves[a].is_ghost, leaves[b].is_ghost);
                        if ag && bg {
                            continue; // corner/edge ghost-ghost: irrelevant
                        }
                        // Fine sub-face area = product of the half-spacings on the
                        // two orthogonal axes.
                        let o1 = (axis + 1) % 3;
                        let o2 = (axis + 2) % 3;
                        let area = (dc[o1] / 2.0) * (dc[o2] / 2.0);

                        // Orient so `owner` is the local (interior) leaf.
                        let (owner, other, sign, patch) = if !ag {
                            // a (−side) is interior: outward normal points +axis.
                            let p = if bg { Some((2 * axis + 1) as u32) } else { None };
                            (a, b, 1.0, p)
                        } else {
                            // a is ghost ⇒ b (+side) is interior; outward normal −axis.
                            (b, a, -1.0, Some((2 * axis) as u32))
                        };
                        let mut area_normal = [0.0; 3];
                        area_normal[axis] = sign * area;
                        let centroid = [
                            0.5 * (leaves[owner].center[0] + leaves[other].center[0]),
                            0.5 * (leaves[owner].center[1] + leaves[other].center[1]),
                            0.5 * (leaves[owner].center[2] + leaves[other].center[2]),
                        ];
                        if leaves[owner].level != leaves[other].level {
                            coarse_fine.push(CoarseFineFace {
                                coarse: if leaves[owner].level == 0 { owner } else { other },
                                fine: if leaves[owner].level == 1 { owner } else { other },
                                area_normal,
                                centroid,
                            });
                        }
                        faces.push(Face { owner, other, area_normal, centroid, patch });
                    }
                }
            }
        }

        ForestMesh { leaves, n_local, faces, coarse_fine, halo: HaloPlan::empty() }
    }

    /// Number of leaves at each refinement level (for diagnostics/tests).
    pub fn level_counts(&self) -> (usize, usize) {
        let fine = self.leaves.iter().filter(|l| l.level == 1).count();
        (self.leaves.len() - fine, fine)
    }
}

impl FvMesh for ForestMesh {
    fn n_local_cells(&self) -> usize {
        self.n_local
    }
    fn n_cells_total(&self) -> usize {
        self.leaves.len()
    }
    fn is_local_cell(&self, c: usize) -> bool {
        !self.leaves[c].is_ghost
    }
    fn cell_volume(&self, c: usize) -> f64 {
        let s = self.leaves[c].size;
        s[0] * s[1] * s[2]
    }
    fn cell_centroid(&self, c: usize) -> Vec3 {
        self.leaves[c].center
    }
    fn for_each_face(&self, f: &mut dyn FnMut(&Face)) {
        for face in &self.faces {
            f(face);
        }
    }
    fn halo_plan(&self) -> &HaloPlan {
        &self.halo
    }
}

impl AdaptiveMesh for ForestMesh {
    fn cell_level(&self, c: usize) -> u8 {
        self.leaves[c].level
    }
    fn for_each_coarse_fine_face(&self, f: &mut dyn FnMut(&CoarseFineFace)) {
        for cf in &self.coarse_fine {
            f(cf);
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // explicit axis indexing reads clearest
mod tests {
    use super::*;

    fn cfg() -> ForestConfig {
        ForestConfig {
            ncx: 4,
            ncy: 4,
            ncz: 1,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0, 1.0, 0.25],
            refine: vec![[1, 1, 0], [2, 1, 0], [1, 2, 0], [2, 2, 0]], // a 2×2 refined block
        }
    }

    #[test]
    fn leaf_counts_and_volumes() {
        let m = ForestMesh::from_config(&cfg());
        // 6*6*3 coarse cells = 108; 4 refined → each +7 leaves (8 fine − 1 coarse).
        assert_eq!(m.n_cells_total(), 108 + 4 * 7);
        // Total volume of interior leaves = domain volume (1*1*0.25).
        let vol: f64 = (0..m.n_cells_total())
            .filter(|&c| m.is_local_cell(c))
            .map(|c| m.cell_volume(c))
            .sum();
        assert!((vol - 0.25).abs() < 1e-12, "interior volume {vol}");
        let (coarse, fine) = m.level_counts();
        assert_eq!(fine, 32); // 4 cells × 8
        assert!(coarse > 0);
    }

    #[test]
    fn faces_are_area_conservative_per_cell() {
        // For every interior leaf, the signed face areas sum to ~zero (closed
        // control volume) — a strong check that the face list is geometrically
        // consistent across coarse/fine interfaces.
        let m = ForestMesh::from_config(&cfg());
        let n = m.n_cells_total();
        let mut net = vec![[0.0f64; 3]; n];
        m.for_each_face(&mut |f| {
            for d in 0..3 {
                net[f.owner][d] -= f.area_normal[d];
                if f.patch.is_none() {
                    net[f.other][d] += f.area_normal[d];
                }
            }
        });
        for c in 0..n {
            if m.is_local_cell(c) {
                for d in 0..3 {
                    assert!(net[c][d].abs() < 1e-12, "cell {c} axis {d} net area {}", net[c][d]);
                }
            }
        }
    }

    #[test]
    fn flux_scatter_is_conservative() {
        // Mimic the solver's scatter with a unit flux: for every interior face the
        // owner −= area/Vo and other += area/Vother must telescope to zero, so the
        // total `Σ Vc·rhs` over interior cells equals only the boundary (one-sided)
        // contribution `−Σ_boundary area`. Any mismatch = a non-conservative face.
        let m = ForestMesh::from_config(&cfg());
        let n = m.n_cells_total();
        let area = |an: [f64; 3]| (an[0] * an[0] + an[1] * an[1] + an[2] * an[2]).sqrt();
        let mut rhs = vec![0.0f64; n];
        m.for_each_face(&mut |f| {
            let a = area(f.area_normal);
            rhs[f.owner] -= a / m.cell_volume(f.owner);
            if f.patch.is_none() && m.is_local_cell(f.other) {
                rhs[f.other] += a / m.cell_volume(f.other);
            }
        });
        let total: f64 = (0..n)
            .filter(|&c| m.is_local_cell(c))
            .map(|c| rhs[c] * m.cell_volume(c))
            .sum();
        let mut boundary = 0.0;
        m.for_each_face(&mut |f| {
            if f.patch.is_some() {
                boundary += area(f.area_normal);
            }
        });
        assert!(
            (total + boundary).abs() < 1e-9,
            "non-conservative face list: Σ Vc·rhs = {total}, −boundary = {}",
            -boundary
        );
    }

    #[test]
    fn coarse_fine_faces_detected() {
        let m = ForestMesh::from_config(&cfg());
        let mut n_cf = 0;
        m.for_each_coarse_fine_face(&mut |_| n_cf += 1);
        assert!(n_cf > 0, "expected coarse/fine interface faces");
    }
}
