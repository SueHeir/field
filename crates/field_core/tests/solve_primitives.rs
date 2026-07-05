//! Integration tests for FIELD's global-solve substrate primitives, exercising
//! them *from the outside* exactly as a solver tier on top of FIELD would.
//!
//! The star of these tests — a discrete Laplacian — is deliberately defined **here
//! in the test**, not in `field_core`'s `src`. That placement is the whole point of
//! the agnosticism contract: the substrate ships the seams ([`LinearOperator`],
//! [`Vector`], [`CellVector`], [`MeshHierarchy`]) and knows nothing about the
//! Laplacian, Poisson, boundary conditions, or any equation. A concrete operator is
//! a *consumer* of the substrate, so it lives with the consumer.
//!
//! Two acceptance checks:
//!   1. `laplacian_apply_matches_hand_computed_case` — a matrix-free operator's
//!      `apply` reproduces a small case worked out by hand.
//!   2. `conjugate_gradient_solves_poisson_on_the_seam` — a textbook CG, written
//!      purely against `Vector` + `LinearOperator`, solves `-∇²u = f` and recovers a
//!      manufactured solution. This is the "a global/implicit solver can be built on
//!      top" claim, demonstrated.

use field_core::prelude::*;
use field_core::{CellVector, LinearOperator, Vector};

/// A matrix-free **negative** Laplacian `A = -∇²` (the SPD operator a Poisson /
/// implicit-diffusion solve actually uses), evaluated on a [`UniformMesh`] with a
/// standard 7-point stencil:
///
/// ```text
///   (A u)_c = Σ_axis ( 2u_c − u_{c-} − u_{c+} ) / h_axis²
/// ```
///
/// It reads only axis neighbours (a separable stencil the face-only halo covers),
/// via the structured `idx_raw` view, and writes owned cells. Off-domain ghosts are
/// whatever the caller left in the vector — zero for a homogeneous-Dirichlet solve,
/// or an explicitly filled value for the hand-computed apply check.
struct NegativeLaplacian<'m> {
    mesh: &'m UniformMesh,
}

impl<'m> LinearOperator for NegativeLaplacian<'m> {
    type Vector = CellVector;

    fn apply(&self, x: &CellVector, y: &mut CellVector) {
        let m = self.mesh;
        let ng = m.n_ghost();
        let d = m.dims();
        y.fill(0.0);
        for i in 0..d[0] {
            for j in 0..d[1] {
                for k in 0..d[2] {
                    let (ir, jr, kr) = (i + ng, j + ng, k + ng);
                    let c = m.idx_raw(ir, jr, kr);
                    let uc = x.get(c);
                    let mut a = 0.0;
                    // x axis
                    let hx = m.spacing(0, i);
                    a += (2.0 * uc
                        - x.get(m.idx_raw(ir - 1, jr, kr))
                        - x.get(m.idx_raw(ir + 1, jr, kr)))
                        / (hx * hx);
                    // y axis
                    let hy = m.spacing(1, j);
                    a += (2.0 * uc
                        - x.get(m.idx_raw(ir, jr - 1, kr))
                        - x.get(m.idx_raw(ir, jr + 1, kr)))
                        / (hy * hy);
                    // z axis
                    let hz = m.spacing(2, k);
                    a += (2.0 * uc
                        - x.get(m.idx_raw(ir, jr, kr - 1))
                        - x.get(m.idx_raw(ir, jr, kr + 1)))
                        / (hz * hz);
                    y.set(c, a);
                }
            }
        }
    }
}

fn unit_spacing_mesh(nx: usize, ny: usize, nz: usize, ng: usize) -> UniformMesh {
    // bounds_hi = n on each axis ⇒ dx = dy = dz = 1, so the stencil divides by 1.
    UniformMesh::from_config(&UniformMeshConfig {
        nx,
        ny,
        nz,
        ng,
        bounds_lo: [0.0; 3],
        bounds_hi: [nx as f64, ny as f64, nz as f64],
        y_edges: None,
        z_edges: None,
    })
}

#[test]
fn laplacian_apply_matches_hand_computed_case() {
    // Acceptance test #2. A 3×1×1 interior grid (unit spacing, one ghost layer).
    // Fill EVERY cell — interior and ghost — with u = (raw x-index)³, held constant
    // in y and z. Then:
    //   * the y- and z-second-differences vanish (equal neighbours), and
    //   * the x-second-difference of i³ is exactly (i-1)³ − 2i³ + (i+1)³ = 6i.
    // With A = −∇² and h = 1, the hand result at interior cells (raw i = 1,2,3) is
    //   (A u) = −6·i = [−6, −12, −18].
    let m = unit_spacing_mesh(3, 1, 1, 1);
    let total = m.total_dims(); // (5, 3, 3)
    let mut u = CellVector::from_mesh(&m);
    for ir in 0..total[0] {
        for jr in 0..total[1] {
            for kr in 0..total[2] {
                u.set(m.idx_raw(ir, jr, kr), (ir as f64).powi(3));
            }
        }
    }

    let a = NegativeLaplacian { mesh: &m };
    let mut y = CellVector::zeros_like(&u);
    a.apply(&u, &mut y);

    let got = [
        y.get(m.idx(0, 0, 0)),
        y.get(m.idx(1, 0, 0)),
        y.get(m.idx(2, 0, 0)),
    ];
    let want = [-6.0, -12.0, -18.0];
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(
            (g - w).abs() < 1e-12,
            "Laplacian apply {got:?} != hand-computed {want:?}"
        );
    }
}

/// Textbook conjugate gradient, written *only* through the [`Vector`] +
/// [`LinearOperator`] seam — no mesh, no stencil, no physics. This is the code a
/// solver tier would own; that it compiles and converges against `CellVector`/
/// `NegativeLaplacian` is the demonstration that FIELD exposes enough structure to
/// build a global implicit solver on top.
fn conjugate_gradient<A: LinearOperator<Vector = CellVector>>(
    a: &A,
    b: &CellVector,
    x: &mut CellVector,
    max_iters: usize,
    tol: f64,
) -> (usize, f64) {
    let mut ax = CellVector::zeros_like(x);
    a.apply(x, &mut ax);
    let mut r = CellVector::zeros_like(x);
    r.copy_from(b);
    r.axpy(-1.0, &ax); // r = b − A x
    let mut p = CellVector::zeros_like(x);
    p.copy_from(&r);
    let mut rs = r.dot(&r);

    for it in 0..max_iters {
        if rs.sqrt() <= tol {
            return (it, rs.sqrt());
        }
        a.apply(&p, &mut ax);
        let alpha = rs / p.dot(&ax);
        x.axpy(alpha, &p); // x += α p
        r.axpy(-alpha, &ax); // r −= α A p
        let rs_new = r.dot(&r);
        let beta = rs_new / rs;
        p.scale(beta); // p = r + β p
        p.axpy(1.0, &r);
        rs = rs_new;
    }
    (max_iters, rs.sqrt())
}

#[test]
fn conjugate_gradient_solves_poisson_on_the_seam() {
    // Solve A u = b with A = −∇² and homogeneous Dirichlet (u = 0 at the ghost
    // cells, which CellVector zeros on construction and CG never writes) on a 1-D
    // interior of N cells. Manufacture a smooth exact solution, form b = A u_exact,
    // run CG from zero, and require it to recover u_exact.
    use std::f64::consts::PI;
    let n = 15usize;
    let m = unit_spacing_mesh(n, 1, 1, 1);
    let a = NegativeLaplacian { mesh: &m };

    // Manufactured u_exact_i = sin(π (i+1)/(N+1)): smooth, zero at the boundaries so
    // the homogeneous-Dirichlet ghosts are exact.
    let mut u_exact = CellVector::from_mesh(&m);
    for i in 0..n {
        u_exact.set(
            m.idx(i, 0, 0),
            (PI * (i as f64 + 1.0) / (n as f64 + 1.0)).sin(),
        );
    }

    let mut b = CellVector::zeros_like(&u_exact);
    a.apply(&u_exact, &mut b); // b = A u_exact

    let mut x = CellVector::zeros_like(&u_exact);
    let (iters, resid) = conjugate_gradient(&a, &b, &mut x, 100, 1e-12);

    // Error against the manufactured solution.
    let mut err = CellVector::zeros_like(&x);
    err.copy_from(&x);
    err.axpy(-1.0, &u_exact);
    let e = err.norm();

    // CG on an SPD N-system converges in ≤ N iterations exactly (in exact
    // arithmetic); require both a tiny residual and a tiny solution error.
    assert!(iters <= n, "CG took {iters} iterations for N={n}");
    assert!(resid < 1e-9, "CG residual {resid:e} too large");
    assert!(e < 1e-9, "‖u_cg − u_exact‖ = {e:e} too large");
}
