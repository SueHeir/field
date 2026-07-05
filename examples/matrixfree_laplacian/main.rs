//! Worked example for FIELD's matrix-free [`LinearOperator`] seam.
//!
//! The concrete operator lives in this example, not in `field_core`, because a
//! Laplacian is an equation-level choice. The substrate provides the mesh-shaped
//! vector, halo-compatible storage, and `apply` interface; a consumer supplies the
//! stencil and any solver it wants above this seam.

use field_core::{
    CellVector, LinearOperator, StructuredMesh, UniformMesh, UniformMeshConfig, Vector,
};

struct NegativeLaplacian<'m> {
    mesh: &'m UniformMesh,
}

impl LinearOperator for NegativeLaplacian<'_> {
    type Vector = CellVector;

    fn apply(&self, x: &CellVector, y: &mut CellVector) {
        let mesh = self.mesh;
        let ng = mesh.n_ghost();
        let dims = mesh.dims();
        y.fill(0.0);

        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let ir = i + ng;
                    let jr = j + ng;
                    let kr = k + ng;
                    let c = mesh.idx_raw(ir, jr, kr);
                    let uc = x.get(c);

                    let hx = mesh.spacing(0, i);
                    let hy = mesh.spacing(1, j);
                    let hz = mesh.spacing(2, k);
                    let value = (2.0 * uc
                        - x.get(mesh.idx_raw(ir - 1, jr, kr))
                        - x.get(mesh.idx_raw(ir + 1, jr, kr)))
                        / (hx * hx)
                        + (2.0 * uc
                            - x.get(mesh.idx_raw(ir, jr - 1, kr))
                            - x.get(mesh.idx_raw(ir, jr + 1, kr)))
                            / (hy * hy)
                        + (2.0 * uc
                            - x.get(mesh.idx_raw(ir, jr, kr - 1))
                            - x.get(mesh.idx_raw(ir, jr, kr + 1)))
                            / (hz * hz);
                    y.set(c, value);
                }
            }
        }
    }
}

fn main() {
    let mesh = UniformMesh::from_config(&UniformMeshConfig {
        nx: 3,
        ny: 1,
        nz: 1,
        ng: 1,
        bounds_lo: [0.0; 3],
        bounds_hi: [3.0, 1.0, 1.0],
        y_edges: None,
        z_edges: None,
    });

    // Fill every cell, including ghosts, with u(raw_i) = raw_i^3. The y/z
    // differences vanish, and the negative x second difference is exactly
    // [-6, -12, -18] on the three owned cells.
    let mut u = CellVector::from_mesh(&mesh);
    let total = mesh.total_dims();
    for ir in 0..total[0] {
        for jr in 0..total[1] {
            for kr in 0..total[2] {
                u.set(mesh.idx_raw(ir, jr, kr), (ir as f64).powi(3));
            }
        }
    }

    let operator = NegativeLaplacian { mesh: &mesh };
    let mut y = CellVector::zeros_like(&u);
    operator.apply(&u, &mut y);

    let got = [
        y.get(mesh.idx(0, 0, 0)),
        y.get(mesh.idx(1, 0, 0)),
        y.get(mesh.idx(2, 0, 0)),
    ];
    let want = [-6.0, -12.0, -18.0];

    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
    }

    println!("matrix-free negative Laplacian apply: {got:?}");
}
