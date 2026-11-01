//! fem_poisson — steady-state Poisson by P1 finite elements on the FIELD mesh.
//!
//! # What this proves
//!
//! FIELD's `README`/`mesh` docs scope the `FvMesh` trait to *finite volume* and
//! note that FEM/implicit-global solvers are out of the `FvMesh` surface — they
//! would ride the **method-agnostic** part of the substrate (the mesh geometry,
//! `FieldData`, the halo plan, the partition) as a *sibling* topology, not a
//! change to `FvMesh`. The open question GRASS's own README raised was whether an
//! *implicit global solve* (assemble a matrix, factor it, back-substitute — no
//! time stepping) fits the GRASS scheduler at all.
//!
//! This example answers it by example. It solves
//!
//! ```text
//!   -∇²u = f          on Ω = (0,1)²
//!      u = 0          on ∂Ω
//! ```
//!
//! with continuous P1 (linear-triangle) finite elements, driving the whole thing
//! through a GRASS `App`:
//!
//!   * the mesh is FIELD's [`UniformMesh`] — the node coordinates and the
//!     element connectivity are read straight off the field substrate's
//!     structured geometry (`dims`, `spacing`, `cell_centroid`);
//!   * the work is three GRASS **systems** ordered by a custom implicit
//!     [`PoissonSolveSet`] schedule (`Assemble → Solve → Validate`), exactly the
//!     "assemble → linear-solve → converge" anatomy `field_core::schedule`
//!     predicted an implicit solver would define for itself;
//!   * `Solve` performs **one** global sparse linear solve `K u = b` — a sparse
//!     Cholesky factorization + back-substitution (`nalgebra-sparse`) — as a
//!     single system call. There is no timestep loop: the driver `prepare()`s the
//!     scheduler and `run()`s the update schedule exactly once.
//!
//! # Method of manufactured solutions
//!
//! Take `u(x,y) = sin(πx) sin(πy)`. Then `-∇²u = 2π² sin(πx) sin(πy) = f`, and
//! `u = 0` on the whole boundary of the unit square, so the homogeneous Dirichlet
//! data is exact. P1 elements are second-order accurate in `L²`, so the discrete
//! error should fall like `O(h²)` under mesh refinement — the convergence gate in
//! `sweep.py` checks the observed order is ≈ 2.
//!
//! # Run
//!
//! ```bash
//! cargo run --release -p fem_poisson -- examples/fem_poisson/config.toml
//! ```
//!
//! Prints one machine-readable `RESULT` line (grid, DOF count, matrix nnz, and the
//! L² / L∞ errors against the analytic solution). The `sweep.py` driver runs a
//! ladder of resolutions and PASS/FAIL-gates the observed convergence order.

use std::collections::HashMap;
use std::f64::consts::PI;

use field_core::prelude::*;
use field_core::UniformMesh;
use grass_app::prelude::*;
use grass_io::{Config, InputPlugin};
use grass_scheduler::{Res, ResMut};
use nalgebra::DVector;
use nalgebra_sparse::factorization::CscCholesky;
use nalgebra_sparse::{CooMatrix, CscMatrix};

/// The implicit-solve schedule: assemble the global system, solve it once, check
/// the result. This is the "assemble → linear-solve → converge" anatomy that
/// `field_core::schedule` says an implicit solver should define for itself rather
/// than reuse the explicit `MeshScheduleSet` (halo → boundary → flux → update).
#[derive(Debug, Clone, Copy, grass_derive::ScheduleSet)]
enum PoissonSolveSet {
    /// Build the sparse stiffness matrix `K` and load vector `b` over the mesh.
    Assemble,
    /// The single global linear solve `K u = b` (sparse Cholesky).
    Solve,
    /// Compare `u_h` against the analytic solution (L² / L∞ error).
    Validate,
}

/// The analytic (manufactured) solution `u = sin(πx) sin(πy)`.
fn exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// The corresponding source term `f = -∇²u = 2π² sin(πx) sin(πy)`.
fn source(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

/// The assembled FEM problem — a GRASS resource threaded through the schedule.
///
/// `K` is held as one hash-map per row during assembly (a compressed, genuinely
/// sparse representation — a P1 stencil touches only the ~7 neighbours of a node),
/// then converted to a `CscMatrix` for the factorization in `Solve`.
#[derive(Default)]
struct PoissonProblem {
    nx: usize,
    ny: usize,
    /// Node x/y coordinates (grid lines), lengths `nx+1` / `ny+1`.
    xnodes: Vec<f64>,
    ynodes: Vec<f64>,
    n_nodes: usize,
    /// Sparse stiffness rows: `rows[i][j] = K_ij`.
    rows: Vec<HashMap<usize, f64>>,
    /// Load vector.
    b: Vec<f64>,
    /// Solution DOFs (filled by `Solve`).
    u: Vec<f64>,
    /// Structural non-zeros of the assembled matrix.
    nnz: usize,
    l2_error: f64,
    linf_error: f64,
}

impl PoissonProblem {
    /// Global DOF index of grid node `(i, j)`, `i ∈ 0..=nx`, `j ∈ 0..=ny`.
    #[inline]
    fn node(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }
}

/// `Assemble`: read the node layout off the FIELD mesh, loop the mesh cells as
/// FEM elements (each Cartesian cell split into two P1 triangles), and scatter the
/// element stiffness + consistent load into the global sparse system. Homogeneous
/// Dirichlet BCs are then imposed symmetrically (row/column elimination) so `K`
/// stays symmetric positive-definite for the Cholesky solve.
fn assemble(mesh: Res<UniformMesh>, mut prob: ResMut<PoissonProblem>) {
    let dims = mesh.dims();
    let (nx, ny) = (dims[0], dims[1]);
    assert_eq!(
        dims[2], 1,
        "fem_poisson is a 2-D solver — set grid.nz = 1 in the config"
    );

    // Node grid-line coordinates, reconstructed from the substrate's own geometry
    // (cell 0 centroid minus half a cell, then walking cell spacings). This uses
    // only the method-agnostic structured-mesh view, so a stretched mesh would
    // work unchanged.
    let c0 = mesh.cell_centroid(mesh.idx(0, 0, 0));
    let mut xnodes = Vec::with_capacity(nx + 1);
    xnodes.push(c0[0] - 0.5 * mesh.spacing(0, 0));
    for i in 0..nx {
        let last = xnodes[i];
        xnodes.push(last + mesh.spacing(0, i));
    }
    let mut ynodes = Vec::with_capacity(ny + 1);
    ynodes.push(c0[1] - 0.5 * mesh.spacing(1, 0));
    for j in 0..ny {
        let last = ynodes[j];
        ynodes.push(last + mesh.spacing(1, j));
    }

    // The manufactured Dirichlet data (u = 0 on ∂Ω) is only exact on the unit
    // square, so refuse to silently produce garbage on a mis-configured domain.
    let tol = 1e-9;
    assert!(
        (xnodes[0]).abs() < tol
            && (xnodes[nx] - 1.0).abs() < tol
            && (ynodes[0]).abs() < tol
            && (ynodes[ny] - 1.0).abs() < tol,
        "fem_poisson expects the unit square: set grid.bounds_lo = [0,0,0], \
         grid.bounds_hi = [1,1,1]"
    );

    let n = (nx + 1) * (ny + 1);
    let mut rows: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n];
    let mut b = vec![0.0f64; n];

    prob.nx = nx;
    prob.ny = ny;
    prob.xnodes = xnodes;
    prob.ynodes = ynodes;
    prob.n_nodes = n;

    // Element assembly: each cell is a quad with CCW corners 0..3; split into
    // triangles (0,1,2) and (0,2,3).
    for cj in 0..ny {
        for ci in 0..nx {
            let corners = [(ci, cj), (ci + 1, cj), (ci + 1, cj + 1), (ci, cj + 1)];
            let gid: Vec<usize> = corners.iter().map(|&(i, j)| prob.node(i, j)).collect();
            let xy: Vec<[f64; 2]> = corners
                .iter()
                .map(|&(i, j)| [prob.xnodes[i], prob.ynodes[j]])
                .collect();
            for tri in [[0usize, 1, 2], [0, 2, 3]] {
                let p = [xy[tri[0]], xy[tri[1]], xy[tri[2]]];
                let g = [gid[tri[0]], gid[tri[1]], gid[tri[2]]];
                assemble_triangle(&p, &g, &mut rows, &mut b);
            }
        }
    }

    // --- Dirichlet BCs: pin every boundary node to u = 0, symmetrically. ---
    let mut is_boundary = vec![false; n];
    for i in 0..=nx {
        is_boundary[prob.node(i, 0)] = true;
        is_boundary[prob.node(i, ny)] = true;
    }
    for j in 0..=ny {
        is_boundary[prob.node(0, j)] = true;
        is_boundary[prob.node(nx, j)] = true;
    }
    // Boundary rows become the identity with rhs = g = 0.
    for d in 0..n {
        if is_boundary[d] {
            rows[d].clear();
            rows[d].insert(d, 1.0);
            b[d] = 0.0;
        }
    }
    // Drop boundary columns from interior rows to keep K symmetric. Since g = 0,
    // no interior rhs correction (b_i -= K_ij g_j) is needed.
    for i in 0..n {
        if is_boundary[i] {
            continue;
        }
        let bnd_cols: Vec<usize> = rows[i]
            .keys()
            .copied()
            .filter(|&j| is_boundary[j])
            .collect();
        for j in bnd_cols {
            rows[i].remove(&j);
        }
    }

    prob.nnz = rows.iter().map(|r| r.len()).sum();
    prob.rows = rows;
    prob.b = b;
}

/// Scatter one linear-triangle element's stiffness and consistent load.
///
/// For a P1 triangle the basis gradients are constant: with vertices
/// `(x_a, y_a)`, `b_a = y_{a+1} - y_{a+2}`, `c_a = x_{a+2} - x_{a+1}` (cyclic),
/// `∇φ_a = (b_a, c_a)/(2A)`, so `K_ad = (b_a b_d + c_a c_d)/(4A)`. The load uses
/// the consistent mass matrix `M = (A/12)·[[2,1,1],[1,2,1],[1,1,2]]` applied to
/// the nodal source values — second-order accurate, matching P1.
fn assemble_triangle(
    p: &[[f64; 2]; 3],
    g: &[usize; 3],
    rows: &mut [HashMap<usize, f64>],
    b: &mut [f64],
) {
    let (x0, y0) = (p[0][0], p[0][1]);
    let (x1, y1) = (p[1][0], p[1][1]);
    let (x2, y2) = (p[2][0], p[2][1]);

    let bb = [y1 - y2, y2 - y0, y0 - y1];
    let cc = [x2 - x1, x0 - x2, x1 - x0];
    // Signed area (CCW quads give a positive value).
    let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    assert!(area > 0.0, "degenerate or clockwise element");

    for a in 0..3 {
        for d in 0..3 {
            let k = (bb[a] * bb[d] + cc[a] * cc[d]) / (4.0 * area);
            *rows[g[a]].entry(g[d]).or_insert(0.0) += k;
        }
    }

    let fnodal = [source(x0, y0), source(x1, y1), source(x2, y2)];
    let mass = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]];
    for a in 0..3 {
        let mut s = 0.0;
        for d in 0..3 {
            s += mass[a][d] * fnodal[d];
        }
        b[g[a]] += area / 12.0 * s;
    }
}

/// `Solve`: the single implicit global operation. Convert the assembled sparse
/// rows to CSC, factor `K = L Lᵀ` (Cholesky, valid because the Dirichlet-reduced
/// stiffness is SPD), and back-substitute for `u`. One factorization, one solve —
/// no iteration over pseudo-time.
fn solve(mut prob: ResMut<PoissonProblem>) {
    let n = prob.n_nodes;
    let mut coo = CooMatrix::new(n, n);
    for (i, row) in prob.rows.iter().enumerate() {
        for (&j, &v) in row {
            coo.push(i, j, v);
        }
    }
    let csc = CscMatrix::from(&coo);
    let chol = CscCholesky::factor(&csc)
        .expect("stiffness matrix is not SPD — check BC application/orientation");
    let rhs = DVector::from_vec(prob.b.clone());
    let sol = chol.solve(&rhs);
    prob.u = sol.iter().copied().collect();
}

/// `Validate`: measure the discrete error against the analytic solution. L∞ over
/// nodes, and L² by a three-point (edge-midpoint) quadrature per triangle — exact
/// for quadratics, so it resolves the O(h²) error faithfully.
fn validate(mut prob: ResMut<PoissonProblem>) {
    let (nx, ny) = (prob.nx, prob.ny);

    // L∞ over all nodes.
    let mut linf = 0.0f64;
    for j in 0..=ny {
        for i in 0..=nx {
            let id = prob.node(i, j);
            let e = (prob.u[id] - exact(prob.xnodes[i], prob.ynodes[j])).abs();
            linf = linf.max(e);
        }
    }

    // L² via per-triangle edge-midpoint quadrature.
    let mut l2sq = 0.0f64;
    for cj in 0..ny {
        for ci in 0..nx {
            let corners = [(ci, cj), (ci + 1, cj), (ci + 1, cj + 1), (ci, cj + 1)];
            let gid: Vec<usize> = corners.iter().map(|&(i, j)| prob.node(i, j)).collect();
            let xy: Vec<[f64; 2]> = corners
                .iter()
                .map(|&(i, j)| [prob.xnodes[i], prob.ynodes[j]])
                .collect();
            for tri in [[0usize, 1, 2], [0, 2, 3]] {
                let p = [xy[tri[0]], xy[tri[1]], xy[tri[2]]];
                let uh = [prob.u[gid[tri[0]]], prob.u[gid[tri[1]]], prob.u[gid[tri[2]]]];
                let area = 0.5
                    * ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                        - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]));
                // Edge midpoints: value of a linear field is the vertex average.
                let edges = [(0usize, 1usize), (1, 2), (2, 0)];
                for &(a, d) in &edges {
                    let mx = 0.5 * (p[a][0] + p[d][0]);
                    let my = 0.5 * (p[a][1] + p[d][1]);
                    let uh_mid = 0.5 * (uh[a] + uh[d]);
                    let e = uh_mid - exact(mx, my);
                    l2sq += (area / 3.0) * e * e;
                }
            }
        }
    }

    prob.l2_error = l2sq.sqrt();
    prob.linf_error = linf;
}

fn main() {
    let mut app = App::new();
    // Read the declarative config file named on the CLI (argv[1]).
    app.add_plugins(InputPlugin);

    // Pull the [grid] section into the FIELD mesh config, then install the
    // substrate: comm backend + field registry + the structured mesh.
    let grid: UniformMeshConfig = {
        let cfg = app
            .get_resource_ref::<Config>()
            .expect("InputPlugin installs a Config resource");
        cfg.section::<UniformMeshConfig>("grid")
    };
    app.add_plugins(FieldDefaultPlugins { mesh: grid });

    // The FEM problem resource and the three implicit-solve systems.
    app.add_resource(PoissonProblem::default());
    app.add_update_system(assemble, PoissonSolveSet::Assemble);
    app.add_update_system(solve, PoissonSolveSet::Solve);
    app.add_update_system(validate, PoissonSolveSet::Validate);

    // A SINGLE global solve — NOT a timestep loop. `prepare()` runs the setup
    // phase (decompose the mesh, size the field registry) and arms the scheduler;
    // one `run()` executes Assemble → Solve → Validate exactly once.
    app.prepare();
    app.run();
    app.run_cleanup();

    let p = app
        .get_resource_ref::<PoissonProblem>()
        .expect("problem resource");
    let h = 1.0 / p.nx.max(1) as f64;
    println!(
        "RESULT nx={} ny={} h={:.6e} n_dof={} nnz={} l2_error={:.8e} linf_error={:.8e}",
        p.nx, p.ny, h, p.n_nodes, p.nnz, p.l2_error, p.linf_error
    );
}
