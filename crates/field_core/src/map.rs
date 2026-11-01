//! [`StructuredMapView`] — a FIELD structured mesh seen as a
//! [`grass_map::Discretization`], so GRASS can map data between it and any
//! other discretization (particles, triangle surfaces, another grid, or a
//! preCICE participant through `grass_precice`) without copying the mesh into
//! one of `grass_map`'s built-in types.
//!
//! The view covers this rank's **owned** cells only, with piecewise-constant
//! (finite-volume) data: entity `k` is owned cell `(i, j, k)` in x-fastest order,
//! the same ordering `grass_map::UniformGrid` uses, and
//! [`cells`](StructuredMapView::cells) gives the FIELD cell index of each
//! entity. A point outside this rank's owned box is outside the view's support,
//! so under MPI each rank maps its own subdomain. Stretched y/z spacing is
//! honored: the view is built from the mesh's own cell edges.
//!
//! ```rust
//! use field_core::{StructuredMapView, UniformMesh, UniformMeshConfig};
//! use grass_map::{Discretization, Field, Intensive, MapBuilder, UniformGrid};
//!
//! let mesh = UniformMesh::from_config(&UniformMeshConfig {
//!     nx: 4, ny: 4, nz: 1, ..Default::default()
//! });
//! let view = StructuredMapView::new(&mesh);
//! let other = UniformGrid::new([0.0; 3], [1.0; 3], [7, 5, 1]);
//!
//! // A field stored per mesh cell becomes a per-entity grass_map field…
//! let mut column = vec![0.0; mesh.n_cells_total()];
//! for &c in view.cells() { column[c] = 300.0; }
//! let t = Field::<Intensive>::scalar(view.gather(&column));
//!
//! // …and maps onto the other grid with constants preserved.
//! let mut t_other = Field::<Intensive>::zeros(other.len(), 1);
//! MapBuilder::new(&view, &other).consistent().apply(&t, &mut t_other);
//! assert!(t_other.values.iter().all(|v| (v - 300.0).abs() < 1e-9));
//! # use field_core::FvMesh;
//! ```

use crate::StructuredMesh;
use grass_map::{Aabb, Discretization, Point, SamplePoint, TopoDim};

/// An owned snapshot of a structured mesh's owned cells, implementing
/// [`grass_map::Discretization`]. See the [module docs](self).
#[derive(Debug, Clone)]
pub struct StructuredMapView {
    /// Cell edges along each axis for the owned cells (`dims[a] + 1` each).
    edges: [Vec<f64>; 3],
    /// Owned cell counts per axis.
    dims: [usize; 3],
    /// FIELD cell index of each entity, x fastest.
    cells: Vec<usize>,
    /// Gauss–Legendre points per axis per cell (1 to 3).
    quad_order: usize,
    /// Fraction of a boundary cell's width a point may lie outside and still
    /// count as inside.
    tolerance: f64,
}

impl StructuredMapView {
    /// Snapshot `mesh`'s owned cells, with 2-point Gauss quadrature per axis.
    ///
    /// # Panics
    /// If the mesh has no owned cells along some axis.
    pub fn new<M: StructuredMesh>(mesh: &M) -> Self {
        let dims = mesh.dims();
        assert!(
            dims.iter().all(|&n| n > 0),
            "StructuredMapView: mesh has no owned cells"
        );
        let c0 = mesh.cell_centroid(mesh.idx(0, 0, 0));
        let edges = [0, 1, 2].map(|a| {
            let mut e = Vec::with_capacity(dims[a] + 1);
            e.push(c0[a] - 0.5 * mesh.spacing(a, 0));
            for i in 0..dims[a] {
                let next = e[i] + mesh.spacing(a, i);
                e.push(next);
            }
            e
        });
        let mut cells = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    cells.push(mesh.idx(i, j, k));
                }
            }
        }
        Self {
            edges,
            dims,
            cells,
            quad_order: 2,
            tolerance: 1e-9,
        }
    }

    /// Gauss–Legendre points per axis per cell (1 to 3).
    ///
    /// # Panics
    /// If `order` is not 1, 2 or 3.
    pub fn with_quadrature_order(mut self, order: usize) -> Self {
        assert!(
            (1..=3).contains(&order),
            "quadrature order must be 1, 2 or 3"
        );
        self.quad_order = order;
        self
    }

    /// Points within `tol ×` the boundary cell's width outside the owned box
    /// still count as inside.
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// The FIELD cell index of each entity (entity `k` lives on `cells()[k]`).
    pub fn cells(&self) -> &[usize] {
        &self.cells
    }

    /// Owned cells per axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Read a per-cell column (indexed by FIELD cell) into per-entity order.
    pub fn gather(&self, column: &[f64]) -> Vec<f64> {
        self.cells.iter().map(|&c| column[c]).collect()
    }

    /// Write per-entity values back into a per-cell column. Ghost cells are
    /// left untouched.
    ///
    /// # Panics
    /// If `values` does not have one entry per entity.
    pub fn scatter(&self, values: &[f64], column: &mut [f64]) {
        assert_eq!(values.len(), self.cells.len(), "one value per entity");
        for (&c, &v) in self.cells.iter().zip(values) {
            column[c] = v;
        }
    }

    fn entity(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// Owned-cell index along `axis` containing `x`, or `None` outside.
    fn cell_along(&self, axis: usize, x: f64) -> Option<usize> {
        let e = &self.edges[axis];
        let n = self.dims[axis];
        let tol_lo = self.tolerance * (e[1] - e[0]);
        let tol_hi = self.tolerance * (e[n] - e[n - 1]);
        if x < e[0] - tol_lo || x > e[n] + tol_hi {
            return None;
        }
        // First edge strictly above x, minus one, clamped into the owned range.
        let above = e.partition_point(|&edge| edge <= x);
        Some(above.saturating_sub(1).min(n - 1))
    }
}

fn gauss_legendre(order: usize) -> &'static [(f64, f64)] {
    // Points on [0, 1] with weights summing to 1.
    const G1: [(f64, f64); 1] = [(0.5, 1.0)];
    const G2: [(f64, f64); 2] = [
        (0.211_324_865_405_187_1, 0.5),
        (0.788_675_134_594_812_9, 0.5),
    ];
    const G3: [(f64, f64); 3] = [
        (0.112_701_665_379_258_3, 5.0 / 18.0),
        (0.5, 8.0 / 18.0),
        (0.887_298_334_620_741_7, 5.0 / 18.0),
    ];
    match order {
        1 => &G1,
        2 => &G2,
        _ => &G3,
    }
}

impl Discretization for StructuredMapView {
    fn len(&self) -> usize {
        self.cells.len()
    }

    fn topo_dim(&self) -> TopoDim {
        TopoDim::Volume
    }

    fn centroid(&self, idx: usize) -> Point {
        let i = idx % self.dims[0];
        let j = (idx / self.dims[0]) % self.dims[1];
        let k = idx / (self.dims[0] * self.dims[1]);
        let mid = |a: usize, n: usize| 0.5 * (self.edges[a][n] + self.edges[a][n + 1]);
        [mid(0, i), mid(1, j), mid(2, k)]
    }

    fn bounds(&self) -> Aabb {
        Aabb {
            lo: [0, 1, 2].map(|a| self.edges[a][0]),
            hi: [0, 1, 2].map(|a| self.edges[a][self.dims[a]]),
        }
    }

    fn sample_points(&self, out: &mut Vec<SamplePoint>) {
        let g = gauss_legendre(self.quad_order);
        let [ex, ey, ez] = &self.edges;
        out.reserve(self.cells.len() * g.len().pow(3));
        for k in 0..self.dims[2] {
            let (z0, hz) = (ez[k], ez[k + 1] - ez[k]);
            for j in 0..self.dims[1] {
                let (y0, hy) = (ey[j], ey[j + 1] - ey[j]);
                for i in 0..self.dims[0] {
                    let (x0, hx) = (ex[i], ex[i + 1] - ex[i]);
                    let dof = self.entity(i, j, k);
                    let vol = hx * hy * hz;
                    for &(sz, wz) in g {
                        for &(sy, wy) in g {
                            for &(sx, wx) in g {
                                out.push(SamplePoint {
                                    dof,
                                    point: [x0 + sx * hx, y0 + sy * hy, z0 + sz * hz],
                                    weight: vol * wx * wy * wz,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    fn evaluate(&self, p: Point, out: &mut Vec<(usize, f64)>) -> bool {
        let (Some(i), Some(j), Some(k)) = (
            self.cell_along(0, p[0]),
            self.cell_along(1, p[1]),
            self.cell_along(2, p[2]),
        ) else {
            return false;
        };
        out.push((self.entity(i, j, k), 1.0));
        true
    }

    fn measures(&self) -> Vec<f64> {
        let w = |a: usize, n: usize| self.edges[a][n + 1] - self.edges[a][n];
        let mut m = Vec::with_capacity(self.cells.len());
        for k in 0..self.dims[2] {
            for j in 0..self.dims[1] {
                for i in 0..self.dims[0] {
                    m.push(w(0, i) * w(1, j) * w(2, k));
                }
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FvMesh, UniformMesh, UniformMeshConfig};
    use grass_map::{Extensive, Field, MapBuilder, UniformGrid};

    fn stretched() -> UniformMesh {
        UniformMesh::from_config(&UniformMeshConfig {
            nx: 4,
            ny: 3,
            nz: 2,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [2.0, 1.0, 1.0],
            y_edges: Some(vec![0.0, 0.1, 0.4, 1.0]),
            z_edges: None,
        })
    }

    #[test]
    fn measures_match_cell_volumes_and_sum_to_domain() {
        let mesh = stretched();
        let view = StructuredMapView::new(&mesh);
        assert_eq!(view.len(), mesh.n_local_cells());
        let m = view.measures();
        for (e, &c) in view.cells().iter().enumerate() {
            assert!((m[e] - mesh.cell_volume(c)).abs() < 1e-14);
            let cv = mesh.cell_centroid(c);
            let vc = view.centroid(e);
            assert!((0..3).all(|a| (cv[a] - vc[a]).abs() < 1e-14));
        }
        assert!((m.iter().sum::<f64>() - 2.0).abs() < 1e-12);
        // Default measures (from the sample points) agree with the override.
        let mut pts = Vec::new();
        view.sample_points(&mut pts);
        let total: f64 = pts.iter().map(|s| s.weight).sum();
        assert!((total - 2.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_finds_the_enclosing_stretched_cell() {
        let mesh = stretched();
        let view = StructuredMapView::new(&mesh);
        let mut out = Vec::new();
        assert!(view.evaluate([1.3, 0.35, 0.75], &mut out));
        assert_eq!(out.len(), 1);
        assert_eq!(view.cells()[out[0].0], mesh.idx(2, 1, 1));
        out.clear();
        assert!(!view.evaluate([1.0, 1.5, 0.5], &mut out));
        assert!(out.is_empty());
        // Domain walls are inside.
        assert!(view.evaluate([2.0, 1.0, 1.0], &mut out));
        assert_eq!(view.cells()[out[0].0], mesh.idx(3, 2, 1));
    }

    #[test]
    fn conservative_map_to_a_grass_grid_keeps_the_total() {
        let mesh = stretched();
        let view = StructuredMapView::new(&mesh);
        let grid = UniformGrid::new([0.0; 3], [2.0, 1.0, 1.0], [5, 4, 3]);
        let q: Vec<f64> = (0..view.len()).map(|e| 1.0 + e as f64).collect();
        let total: f64 = q.iter().sum();
        let mut q_grid = Field::<Extensive>::zeros(grid.len(), 1);
        MapBuilder::new(&view, &grid)
            .conservative()
            .apply(&Field::<Extensive>::scalar(q), &mut q_grid);
        assert!((q_grid.totals()[0] - total).abs() < 1e-9 * total);
    }

    #[test]
    fn decomposed_view_covers_only_owned_cells() {
        let cfg = UniformMeshConfig {
            nx: 8,
            ny: 2,
            nz: 2,
            ..Default::default()
        };
        let mesh = UniformMesh::from_config_decomposed(&cfg, [2, 1, 1], [1, 0, 0]);
        let view = StructuredMapView::new(&mesh);
        assert_eq!(view.dims(), [4, 2, 2]);
        let b = view.bounds();
        assert!((b.lo[0] - 0.5).abs() < 1e-14 && (b.hi[0] - 1.0).abs() < 1e-14);
        let mut out = Vec::new();
        assert!(!view.evaluate([0.25, 0.5, 0.5], &mut out));
        let mut col = vec![-1.0; mesh.n_cells_total()];
        view.scatter(&vec![2.0; view.len()], &mut col);
        assert_eq!(view.gather(&col), vec![2.0; view.len()]);
        assert_eq!(col.iter().filter(|&&v| v == 2.0).count(), view.len());
    }
}
