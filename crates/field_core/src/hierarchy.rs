//! [`MeshHierarchy`] — a coarsen/refine ladder of structured meshes plus the
//! **restriction** and **prolongation** grid-transfer operators a geometric
//! multigrid solve rides on.
//!
//! # Why this lives in the substrate (and stays equation-agnostic)
//!
//! Geometric multigrid needs one thing from the mesh that a single grid cannot
//! give: a *sequence* of nested grids and the two operators that move a field
//! between adjacent levels. Those operators are **purely geometric** — they are
//! defined by which fine cells sit inside which coarse cell, and by nothing about
//! what the field *means*. A density, a pressure, a temperature, or a multigrid
//! residual all restrict and prolong identically. So the hierarchy belongs here,
//! next to [`crate::UniformMesh`], exactly as the halo plan does: it is structure,
//! not physics. FIELD exposes the structure; a solver tier written on top supplies
//! the smoother, the operator, and the cycle.
//!
//! # The two operators (cell-centered, 2:1 coarsening)
//!
//! A coarse cell `(I,J,K)` agglomerates the `ratio_x·ratio_y·ratio_z` fine cells
//! it geometrically contains, where `ratio_a = 2` on every axis with more than one
//! cell and `1` on a singleton axis (so a 2-D `nz = 1` problem coarsens only in x
//! and y). With that agglomeration:
//!
//! - **Restriction** `R` (fine → coarse): each coarse value is the *average* of
//!   its fine children. This is the finite-volume / cell-centered full-weighting
//!   restriction (a volume-weighted mean; with equal child volumes it is the plain
//!   mean).
//! - **Prolongation** `P` (coarse → fine): each fine child copies its coarse
//!   parent's value (piecewise-constant injection).
//!
//! These are transposes up to a diagonal scaling, and they satisfy the exact
//! algebraic identity **`R(P(x)) = x`** for any coarse field `x`: prolonging fills
//! every child of a coarse cell with the same value, and averaging equal values
//! returns it unchanged. That identity is the correctness contract a multigrid
//! transfer pair must meet, and it is what [`crate::hierarchy`]'s test checks on a
//! manufactured field.
//!
//! # Scope
//!
//! Like [`crate::field_amr`]'s first cut, the hierarchy is built on each rank's
//! **local** structured subdomain (single-rank coarsening of the interior). Each
//! level still carries its own [`crate::HaloPlan`] via its [`UniformMesh`], so a
//! solver exchanges ghosts per level; genuinely cross-rank *coarsening* (agglomer-
//! ating cells that live on different ranks near the coarse limit) is a later
//! addition and is out of scope for this substrate primitive.

use crate::uniform::{UniformMesh, UniformMeshConfig};
use crate::{FvMesh, StructuredMesh};

/// A ladder of structured [`UniformMesh`] levels related by 2:1 cell-centered
/// coarsening, with restriction (fine → coarse) and prolongation (coarse → fine)
/// operators between adjacent levels.
///
/// Level `0` is the **finest**; increasing the index coarsens. A field at level
/// `l` is a flat `f64` slice of length `self.level(l).n_cells_total()` — the same
/// full, ghost-inclusive layout every other FIELD store uses — so the transfer
/// operators drop straight in beside the halo exchange and touch only interior
/// cells (ghosts are refilled by a per-level halo exchange, not by transfer).
pub struct MeshHierarchy {
    levels: Vec<UniformMesh>,
}

impl MeshHierarchy {
    /// Build `n_levels` levels from a finest-level config by successive 2:1
    /// coarsening. Every axis with more than one cell must stay divisible by two
    /// through the whole ladder (i.e. each finer level's interior count is even on
    /// that axis); a singleton axis (`n = 1`) is carried through unchanged.
    ///
    /// Panics with a clear message if a level cannot be halved, so a misconfigured
    /// ladder fails loudly at construction rather than silently transferring onto a
    /// mismatched grid.
    pub fn from_base(cfg: &UniformMeshConfig, n_levels: usize) -> Self {
        assert!(n_levels >= 1, "a hierarchy needs at least one level");
        let mut levels = Vec::with_capacity(n_levels);
        let mut cur = cfg.clone();
        for l in 0..n_levels {
            levels.push(UniformMesh::from_config(&cur));
            if l + 1 < n_levels {
                cur = coarsen_config(&cur);
            }
        }
        Self { levels }
    }

    /// Number of levels in the ladder.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// The mesh at level `l` (`0` = finest).
    pub fn level(&self, l: usize) -> &UniformMesh {
        &self.levels[l]
    }

    /// The finest level (`0`).
    pub fn finest(&self) -> &UniformMesh {
        &self.levels[0]
    }

    /// The coarsest level (`n_levels - 1`).
    pub fn coarsest(&self) -> &UniformMesh {
        &self.levels[self.levels.len() - 1]
    }

    /// Per-axis coarsening ratio between fine level `l` and coarse level `l+1`
    /// (`2` on refined axes, `1` on singleton axes). Panics if the two levels are
    /// not related by an integer 2:1 (or 1:1) ratio on every axis.
    fn ratio(&self, fine_level: usize) -> [usize; 3] {
        let fd = self.levels[fine_level].dims();
        let cd = self.levels[fine_level + 1].dims();
        let mut r = [1usize; 3];
        for a in 0..3 {
            assert!(
                cd[a] >= 1 && fd[a] == cd[a] * (fd[a] / cd[a]) && (fd[a] / cd[a]) >= 1,
                "levels {fine_level}/{} are not 2:1 on axis {a}: fine {} vs coarse {}",
                fine_level + 1,
                fd[a],
                cd[a]
            );
            r[a] = fd[a] / cd[a];
        }
        r
    }

    /// **Restriction** — average each coarse cell from its fine children, mapping a
    /// field on fine level `fine_level` down onto coarse level `fine_level + 1`.
    ///
    /// `fine` must be sized to `self.level(fine_level).n_cells_total()` and
    /// `coarse` to `self.level(fine_level + 1).n_cells_total()`; only interior
    /// coarse cells are written.
    pub fn restrict(&self, fine_level: usize, fine: &[f64], coarse: &mut [f64]) {
        let r = self.ratio(fine_level);
        let fmesh = &self.levels[fine_level];
        let cmesh = &self.levels[fine_level + 1];
        assert_eq!(
            fine.len(),
            fmesh.n_cells_total(),
            "fine slice size mismatch"
        );
        assert_eq!(
            coarse.len(),
            cmesh.n_cells_total(),
            "coarse slice size mismatch"
        );

        let cd = cmesh.dims();
        let count = (r[0] * r[1] * r[2]) as f64;
        for ci in 0..cd[0] {
            for cj in 0..cd[1] {
                for ck in 0..cd[2] {
                    let mut acc = 0.0;
                    for di in 0..r[0] {
                        for dj in 0..r[1] {
                            for dk in 0..r[2] {
                                let f = fmesh.idx(r[0] * ci + di, r[1] * cj + dj, r[2] * ck + dk);
                                acc += fine[f];
                            }
                        }
                    }
                    coarse[cmesh.idx(ci, cj, ck)] = acc / count;
                }
            }
        }
    }

    /// **Prolongation** — inject each coarse value into all of its fine children,
    /// mapping a field on coarse level `fine_level + 1` up onto fine level
    /// `fine_level`.
    ///
    /// `coarse` must be sized to `self.level(fine_level + 1).n_cells_total()` and
    /// `fine` to `self.level(fine_level).n_cells_total()`; only interior fine cells
    /// are written.
    pub fn prolong(&self, fine_level: usize, coarse: &[f64], fine: &mut [f64]) {
        let r = self.ratio(fine_level);
        let fmesh = &self.levels[fine_level];
        let cmesh = &self.levels[fine_level + 1];
        assert_eq!(
            coarse.len(),
            cmesh.n_cells_total(),
            "coarse slice size mismatch"
        );
        assert_eq!(
            fine.len(),
            fmesh.n_cells_total(),
            "fine slice size mismatch"
        );

        let cd = cmesh.dims();
        for ci in 0..cd[0] {
            for cj in 0..cd[1] {
                for ck in 0..cd[2] {
                    let v = coarse[cmesh.idx(ci, cj, ck)];
                    for di in 0..r[0] {
                        for dj in 0..r[1] {
                            for dk in 0..r[2] {
                                let f = fmesh.idx(r[0] * ci + di, r[1] * cj + dj, r[2] * ck + dk);
                                fine[f] = v;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Halve a config's interior cell counts (axes with `n = 1` are left alone), and
/// decimate any explicit stretching edges so the coarse cell edges are a subset of
/// the fine ones. Panics if a multi-cell axis is not even.
fn coarsen_config(cfg: &UniformMeshConfig) -> UniformMeshConfig {
    let halve = |n: usize, axis: char| -> usize {
        if n <= 1 {
            1
        } else {
            assert!(
                n.is_multiple_of(2),
                "cannot coarsen a {n}-cell axis ({axis}) by two: not even"
            );
            n / 2
        }
    };
    // Coarse edges keep every other fine edge: node 0, 2, 4, … 2·nc = n.
    let decimate = |edges: &Option<Vec<f64>>, nc: usize| -> Option<Vec<f64>> {
        edges.as_ref().map(|e| (0..=nc).map(|i| e[2 * i]).collect())
    };
    let ny = halve(cfg.ny, 'y');
    let nz = halve(cfg.nz, 'z');
    UniformMeshConfig {
        nx: halve(cfg.nx, 'x'),
        ny,
        nz,
        ng: cfg.ng,
        bounds_lo: cfg.bounds_lo,
        bounds_hi: cfg.bounds_hi,
        y_edges: decimate(&cfg.y_edges, ny),
        z_edges: decimate(&cfg.z_edges, nz),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FvMesh;

    fn base(nx: usize, ny: usize, nz: usize) -> UniformMeshConfig {
        UniformMeshConfig {
            nx,
            ny,
            nz,
            ng: 2,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0, 1.0, 1.0],
            y_edges: None,
            z_edges: None,
        }
    }

    #[test]
    fn ladder_dims_halve_each_level() {
        let h = MeshHierarchy::from_base(&base(8, 8, 1), 3);
        assert_eq!(h.n_levels(), 3);
        assert_eq!(h.level(0).dims(), [8, 8, 1]);
        assert_eq!(h.level(1).dims(), [4, 4, 1]); // z stays 1
        assert_eq!(h.level(2).dims(), [2, 2, 1]);
    }

    #[test]
    fn restriction_of_prolongation_is_identity_on_a_manufactured_field() {
        // Acceptance test #1: R∘P = I. Build a two-level ladder, lay down a smooth
        // manufactured field on the COARSE level, prolong it to the fine level,
        // restrict it back, and require the round trip to reproduce the original
        // coarse field to machine precision.
        let h = MeshHierarchy::from_base(&base(8, 8, 4), 2);
        let fine = h.level(0);
        let coarse = h.level(1);

        // Manufactured coarse field: f(x,y,z) = sin(2πx)·cos(3πy) + z², sampled at
        // each coarse cell centroid. Nothing about R∘P=I depends on the choice — it
        // holds for ANY field — which is exactly the point: the identity is a
        // property of the operators, not of the data.
        use std::f64::consts::PI;
        let manufactured = |c: usize, m: &UniformMesh| -> f64 {
            let p = m.cell_centroid(c);
            (2.0 * PI * p[0]).sin() * (3.0 * PI * p[1]).cos() + p[2] * p[2]
        };
        let mut x_coarse = vec![0.0; coarse.n_cells_total()];
        for ci in 0..coarse.dims()[0] {
            for cj in 0..coarse.dims()[1] {
                for ck in 0..coarse.dims()[2] {
                    let idx = coarse.idx(ci, cj, ck);
                    x_coarse[idx] = manufactured(idx, coarse);
                }
            }
        }

        let mut x_fine = vec![0.0; fine.n_cells_total()];
        h.prolong(0, &x_coarse, &mut x_fine);

        let mut x_back = vec![0.0; coarse.n_cells_total()];
        h.restrict(0, &x_fine, &mut x_back);

        let mut max_err = 0.0f64;
        for ci in 0..coarse.dims()[0] {
            for cj in 0..coarse.dims()[1] {
                for ck in 0..coarse.dims()[2] {
                    let idx = coarse.idx(ci, cj, ck);
                    max_err = max_err.max((x_back[idx] - x_coarse[idx]).abs());
                }
            }
        }
        assert!(max_err < 1e-14, "R(P(x)) deviated from x by {max_err:e}");
    }

    #[test]
    fn prolongation_fills_every_child_with_its_parent() {
        // A direct check of the injection semantics behind the identity: set one
        // coarse cell to a distinctive value and confirm all 2×2×2 fine children
        // pick it up, with a coarse average that returns it.
        let h = MeshHierarchy::from_base(&base(4, 4, 4), 2);
        let coarse = h.level(1);
        let fine = h.level(0);
        let mut xc = vec![0.0; coarse.n_cells_total()];
        xc[coarse.idx(1, 1, 1)] = 7.5;
        let mut xf = vec![0.0; fine.n_cells_total()];
        h.prolong(0, &xc, &mut xf);
        for di in 0..2 {
            for dj in 0..2 {
                for dk in 0..2 {
                    assert_eq!(xf[fine.idx(2 + di, 2 + dj, 2 + dk)], 7.5);
                }
            }
        }
        let mut back = vec![0.0; coarse.n_cells_total()];
        h.restrict(0, &xf, &mut back);
        assert!((back[coarse.idx(1, 1, 1)] - 7.5).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "not even")]
    fn odd_axis_is_rejected() {
        // A 6→3→? ladder can't reach three levels: 3 is odd.
        MeshHierarchy::from_base(&base(6, 6, 1), 3);
    }
}
