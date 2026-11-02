//! Assembled (not matrix-free) implicit solve over a partitioned unstructured
//! mesh: global row numbering, a distributed CSR matrix, and Jacobi-PCG.
//!
//! This is the shape an external solver binding (PETSc MPIAIJ, hypre IJ) needs:
//! contiguous owned rows per rank, columns in global numbering, and halo cells
//! that know their owner's row. The MPI test checks all of that on 2, 3, 4 and
//! 7 ranks against the serial matrix and solution.

use field_core::prelude::*;
use field_core::{CellVector, DistCsr, GlobalNumbering};
use grass_linalg::{preconditioned_conjugate_gradient, LinearOperator, Preconditioner};
use grass_mpi::CommBackend;
use std::cell::RefCell;

fn prisms() -> PolyMesh {
    PolyMesh::triangle_prisms(14, 12, [0.0, 0.0], [1.0, 1.0], 0.1, 0.3)
}

fn dist(a: Vec3, b: Vec3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Two-point Laplacian with Dirichlet zero on the side walls (folded into the
/// diagonal, so no column refers to a boundary ghost) and no flux in z.
fn assemble(mesh: &dyn FvMesh) -> DistCsr {
    let n = mesh.n_cells_total();
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    mesh.for_each_face(&mut |f| {
        let (o, c) = (f.owner, f.other);
        let an = f.area_normal;
        let k = (an[0] * an[0] + an[1] * an[1] + an[2] * an[2]).sqrt()
            / dist(mesh.cell_centroid(o), mesh.cell_centroid(c));
        match f.patch {
            Some(p) if p < 4 => rows[o].push((o, 2.0 * k)),
            Some(_) => {}
            None => {
                rows[o].extend([(o, k), (c, -k)]);
                if mesh.is_local_cell(c) {
                    rows[c].extend([(c, k), (o, -k)]);
                }
            }
        }
    });
    let owned: Vec<usize> = (0..n).filter(|&c| mesh.is_local_cell(c)).collect();
    let rows = owned
        .iter()
        .map(|&c| std::mem::take(&mut rows[c]))
        .collect();
    DistCsr::from_rows(owned, rows)
}

struct Op<'a> {
    a: &'a DistCsr,
    mesh: &'a dyn FvMesh,
    comm: &'a dyn CommBackend,
    work: RefCell<CellVector>,
}

impl LinearOperator for Op<'_> {
    fn dimension(&self) -> usize {
        self.mesh.n_cells_total()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let mut w = self.work.borrow_mut();
        w.as_mut_slice().copy_from_slice(x);
        self.a.apply(self.mesh, self.comm, &mut w, y);
    }
}

/// Inverse diagonal on owned cells, zero elsewhere (keeps ghost entries 0).
struct Jacobi(Vec<f64>);

impl Preconditioner for Jacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for ((z, r), d) in z.iter_mut().zip(r).zip(&self.0) {
            *z = d * r;
        }
    }
}

/// Owned `(PolyMesh cell, u)` of the assembled solve, and the iteration count.
fn solve(mesh: &UnstructuredMesh, comm: &dyn CommBackend) -> (Vec<(usize, f64)>, usize) {
    let a = assemble(mesh);
    let mut inv = vec![0.0; mesh.n_cells_total()];
    for (c, d) in a.diagonal() {
        inv[c] = 1.0 / d;
    }
    let mut rhs = CellVector::from_mesh(mesh);
    for c in 0..mesh.n_local_cells() {
        let x = mesh.cell_centroid(c);
        let pi = std::f64::consts::PI;
        rhs.set(
            c,
            2.0 * pi * pi * (pi * x[0]).sin() * (pi * x[1]).sin() * mesh.cell_volume(c),
        );
    }
    let op = Op {
        a: &a,
        mesh,
        comm,
        work: RefCell::new(CellVector::from_mesh(mesh)),
    };
    let mut x = vec![0.0; mesh.n_cells_total()];
    let report = preconditioned_conjugate_gradient(
        comm,
        &op,
        &Jacobi(inv),
        rhs.as_slice(),
        &mut x,
        2000,
        |s| s.relative_residual() < 1e-11,
    )
    .expect("PCG converged");
    let out = (0..mesh.n_local_cells())
        .map(|c| (mesh.global_id(c).expect("owned"), x[c]))
        .collect();
    (out, report.iterations)
}

#[test]
fn serial_numbering_is_identity_and_assembled_solve_converges() {
    let poly = prisms();
    let mesh = UnstructuredMesh::serial(&poly);
    let comm = grass_mpi::SingleProcessComm::new();
    let num = GlobalNumbering::new(&mesh, &comm);
    assert_eq!(num.first, 0);
    assert_eq!(num.n_global as usize, poly.cells.len());
    for c in 0..mesh.n_local_cells() {
        assert_eq!(num.row(c), Some(c as u64));
    }
    for c in mesh.n_local_cells()..mesh.n_cells_total() {
        assert_eq!(num.row(c), None, "boundary ghost {c} has no row");
    }
    // Symmetric matrix, every column owned.
    let a = assemble(&mesh);
    let t = a.global_triplets(&num);
    let mut map = std::collections::HashMap::new();
    for &(r, c, v) in &t {
        map.insert((r, c), v);
    }
    for (&(r, c), &v) in &map {
        assert!((map[&(c, r)] - v).abs() < 1e-14, "asymmetric at ({r},{c})");
    }
    let (u, iters) = solve(&mesh, &comm);
    assert!(iters > 0 && u.len() == poly.cells.len());
}

#[cfg(feature = "mpi_backend")]
mod mpi {
    use super::*;
    use std::process::Command;

    const CHILD_ENV: &str = "FIELD_DIST_MATRIX_MPI_CHILD";
    const RANK_COUNTS: [usize; 4] = [2, 3, 4, 7];

    fn run_child() {
        let poly = prisms();
        let n = poly.cells.len();
        let comm = grass_mpi::MpiCommBackend::new(grass_mpi::get_mpi_world());
        let (rank, size) = (comm.rank(), comm.size() as usize);
        let owner = poly.rcb_partition(size);
        let mesh = UnstructuredMesh::partition(&poly, &owner, rank);
        let num = GlobalNumbering::new(&mesh, &comm);

        // Rows are contiguous per rank and cover 0..n exactly once.
        assert_eq!(num.n_global as usize, n);
        let mut hits = vec![0.0; n];
        // Global row of every PolyMesh cell, as assigned by its owner.
        let mut row_of = vec![0.0; n];
        for c in 0..mesh.n_local_cells() {
            let r = num.row(c).expect("owned row");
            assert!((num.first..num.first + num.n_owned as u64).contains(&r));
            hits[r as usize] += 1.0;
            row_of[mesh.global_id(c).expect("owned")] = r as f64;
        }
        comm.all_reduce_sum_f64_slice(&mut hits);
        comm.all_reduce_sum_f64_slice(&mut row_of);
        assert!(hits.iter().all(|&h| h == 1.0), "rows are not a permutation");

        // Every halo cell knows its owner's row; boundary ghosts have none.
        let n_known = mesh.n_local_cells() + mesh.n_halo_cells();
        assert!(mesh.n_halo_cells() > 0);
        for c in mesh.n_local_cells()..n_known {
            let g = mesh.global_id(c).expect("halo cell");
            assert_eq!(
                num.row(c),
                Some(row_of[g] as u64),
                "halo cell {c} (gid {g})"
            );
        }
        for c in n_known..mesh.n_cells_total() {
            assert_eq!(num.row(c), None);
        }

        // The distributed matrix, mapped to PolyMesh numbering, equals the
        // serial one entry for entry.
        let serial = UnstructuredMesh::serial(&poly);
        let sa = assemble(&serial);
        let snum = GlobalNumbering::new(&serial, &grass_mpi::SingleProcessComm::new());
        let mut dense_ser = vec![0.0; n * n];
        for (r, c, v) in sa.global_triplets(&snum) {
            dense_ser[r as usize * n + c as usize] = v;
        }
        let mut gid_of_row = vec![0usize; n];
        for (g, &r) in row_of.iter().enumerate() {
            gid_of_row[r as usize] = g;
        }
        let mut dense_par = vec![0.0; n * n];
        for (r, c, v) in assemble(&mesh).global_triplets(&num) {
            dense_par[gid_of_row[r as usize] * n + gid_of_row[c as usize]] += v;
        }
        comm.all_reduce_sum_f64_slice(&mut dense_par);
        let worst = dense_par
            .iter()
            .zip(&dense_ser)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-12, "{size} ranks: matrix differs by {worst:e}");

        // Jacobi-PCG on the assembled matrix matches serial.
        let (par, _) = solve(&mesh, &comm);
        let mut got = vec![0.0; n];
        for (g, v) in par {
            got[g] = v;
        }
        comm.all_reduce_sum_f64_slice(&mut got);
        let (ser, _) = solve(&serial, &grass_mpi::SingleProcessComm::new());
        let worst = ser
            .iter()
            .map(|&(g, v)| (got[g] - v).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-8, "{size} ranks: solution differs by {worst:e}");

        drop(comm);
        grass_mpi::finalize_mpi();
    }

    #[test]
    fn mpi_distributed_matrix_matches_serial() {
        if std::env::var_os(CHILD_ENV).is_some() {
            run_child();
            return;
        }
        if Command::new("mpirun").arg("--version").output().is_err() {
            eprintln!("SKIP mpi_distributed_matrix_matches_serial: `mpirun` not on PATH");
            return;
        }
        let exe = std::env::current_exe().expect("locate test binary");
        for np in RANK_COUNTS {
            let status = Command::new("mpirun")
                .args(["--oversubscribe", "-np", &np.to_string()])
                .arg(&exe)
                .args([
                    "--exact",
                    "mpi::mpi_distributed_matrix_matches_serial",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, "1")
                .status()
                .expect("spawn mpirun");
            assert!(
                status.success(),
                "distributed matrix failed on {np} ranks: {status}"
            );
        }
    }
}
