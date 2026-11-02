//! Stress test: does FV physics written against [`FvMesh`] run unchanged on a
//! partitioned unstructured mesh?
//!
//! One generic advection–diffusion plugin (`ScalarPlugin<M>`) is written once
//! against `FvMesh` and composed into an App with either FIELD's
//! [`UniformMesh`] or the new [`UnstructuredMesh`]. The checks:
//!
//! 1. On the same hexahedra, the unstructured mesh reproduces the structured
//!    result to round-off (the face-list view is the same discretisation).
//! 2. On a jittered triangular-prism mesh, pure diffusion with zero-gradient
//!    walls conserves the total exactly and stays inside the initial bounds.
//! 3. Under `--features mpi_backend`, the same App on 2, 3, 4 and 7 ranks
//!    matches the serial run cell for cell (by global id), for the explicit
//!    scheme and for an implicit Poisson solve through `grass_linalg`'s CG.

use field_core::prelude::*;
use field_core::{halo_exchange_forward, halo_exchange_forward_all, CellVector, PatchId};
use grass_app::prelude::*;
use grass_linalg::{conjugate_gradient, LinearOperator};
use grass_mpi::{CommBackend, CommResource};
use grass_scheduler::Res;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

// ── Generic FV physics (knows only `FvMesh`) ────────────────────────────────

#[derive(FieldData, Default)]
struct Scalar {
    #[forward]
    u: Vec<f64>,
    #[zero]
    rhs: Vec<f64>,
}

#[derive(Clone, Copy)]
struct Physics {
    velocity: Vec3,
    diffusivity: f64,
    dt: f64,
    /// Dirichlet value on one patch; every other patch is zero-gradient.
    inflow: Option<(PatchId, f64)>,
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dist(a: Vec3, b: Vec3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    dot(d, d).sqrt()
}

fn initial(x: Vec3) -> f64 {
    let r2 = (x[0] - 0.4).powi(2) + (x[1] - 0.5).powi(2);
    (-r2 / 0.02).exp() + 0.1 * x[0]
}

fn fill_boundary<M: FvMesh>(mesh: &M, s: &mut Scalar, p: &Physics) {
    mesh.for_each_face(&mut |f| {
        if let Some(pid) = f.patch {
            s.u[f.other] = match p.inflow {
                // Mirror so the face value is exactly the Dirichlet value.
                Some((q, g)) if q == pid => 2.0 * g - s.u[f.owner],
                _ => s.u[f.owner],
            };
        }
    });
}

fn assemble_flux<M: FvMesh>(mesh: &M, s: &mut Scalar, p: &Physics) {
    mesh.for_each_face(&mut |f| {
        let (o, x) = (f.owner, f.other);
        let area = dot(f.area_normal, f.area_normal).sqrt();
        let vn = dot(p.velocity, f.area_normal);
        let advective = if vn >= 0.0 { vn * s.u[o] } else { vn * s.u[x] };
        let d = dist(mesh.cell_centroid(o), mesh.cell_centroid(x));
        let diffusive = -p.diffusivity * area * (s.u[x] - s.u[o]) / d;
        let flux = advective + diffusive;
        s.rhs[o] -= flux / mesh.cell_volume(o);
        if f.patch.is_none() && mesh.is_local_cell(x) {
            s.rhs[x] += flux / mesh.cell_volume(x);
        }
    });
}

fn sys_init<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>) {
    let mut s = reg.expect_mut::<Scalar>("Scalar");
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) {
            s.u[c] = initial(mesh.cell_centroid(c));
        }
    }
}

fn sys_halo<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>, comm: Res<CommResource>) {
    halo_exchange_forward_all(mesh.halo_plan(), &reg, &*comm.0);
}

fn sys_boundary<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>, p: Res<Physics>) {
    fill_boundary(&*mesh, &mut reg.expect_mut::<Scalar>("Scalar"), &p);
}

fn sys_zero<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>) {
    reg.zero_all(mesh.n_cells_total());
}

fn sys_flux<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>, p: Res<Physics>) {
    assemble_flux(&*mesh, &mut reg.expect_mut::<Scalar>("Scalar"), &p);
}

fn sys_update<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>, p: Res<Physics>) {
    let mut s = reg.expect_mut::<Scalar>("Scalar");
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) {
            s.u[c] += p.dt * s.rhs[c];
        }
    }
}

/// The physics plugin, written once, generic over the mesh type.
struct ScalarPlugin<M>(Physics, PhantomData<M>);

impl<M: FvMesh> Plugin for ScalarPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(self.0);
        register_field_data!(app, Scalar::default());
        app.add_setup_system(sys_init::<M>, ScheduleSetupSet::PostSetup)
            .add_update_system(sys_halo::<M>, MeshScheduleSet::Halo)
            .add_update_system(sys_boundary::<M>, MeshScheduleSet::Boundary)
            .add_update_system(sys_zero::<M>, MeshScheduleSet::PreFlux)
            .add_update_system(sys_flux::<M>, MeshScheduleSet::Flux)
            .add_update_system(sys_update::<M>, MeshScheduleSet::Update);
    }

    fn requires(&self) -> Vec<&str> {
        vec!["mesh", "field_registry"]
    }
}

/// Owned cells as `(centroid, value)` after `steps` steps.
fn run<M: FvMesh>(mesh_plugin: impl Plugin, physics: Physics, steps: usize) -> Vec<(Vec3, f64)> {
    let mut app = App::new();
    app.add_plugins(CommPlugin)
        .add_plugins(FieldRegistryPlugin)
        .add_plugins(mesh_plugin)
        .add_plugins(ScalarPlugin::<M>(physics, PhantomData));
    app.prepare();
    for _ in 0..steps {
        app.run();
    }
    let mesh = app.get_resource_ref::<M>().expect("mesh");
    let reg = app.get_resource_ref::<FieldRegistry>().expect("registry");
    let s = reg.expect::<Scalar>("Scalar");
    (0..mesh.n_cells_total())
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| (mesh.cell_centroid(c), s.u[c]))
        .collect()
}

#[cfg_attr(feature = "mpi_backend", allow(dead_code))]
fn cfg() -> UniformMeshConfig {
    UniformMeshConfig {
        nx: 12,
        ny: 10,
        nz: 1,
        ng: 1,
        bounds_lo: [0.0; 3],
        bounds_hi: [1.0, 1.0, 0.1],
        y_edges: None,
        z_edges: None,
    }
}

fn advect_diffuse() -> Physics {
    Physics {
        velocity: [1.0, 0.4, 0.0],
        diffusivity: 0.01,
        dt: 2e-3,
        inflow: Some((BoundarySide::XLo as u32, 1.0)),
    }
}

fn prisms() -> Arc<PolyMesh> {
    Arc::new(PolyMesh::triangle_prisms(
        14,
        12,
        [0.0, 0.0],
        [1.0, 1.0],
        0.1,
        0.3,
    ))
}

// These two build an App, which initialises MPI under `mpi_backend`; an
// MPI-initialised parent cannot then launch `mpirun` for the MPI test, so they
// run in the default (serial) build only, as in CI's plain `cargo test`.
#[cfg(not(feature = "mpi_backend"))]
#[test]
fn hex_box_matches_uniform_mesh() {
    let steps = 60;
    let mut a = run::<UniformMesh>(UniformMeshPlugin::new(cfg()), advect_diffuse(), steps);
    let mut b = run::<UnstructuredMesh>(
        UnstructuredMeshPlugin {
            mesh: Arc::new(PolyMesh::hex_box(&cfg())),
        },
        advect_diffuse(),
        steps,
    );
    assert_eq!(a.len(), b.len());
    let key = |c: &(Vec3, f64)| ((c.0[0] * 1e6) as i64, (c.0[1] * 1e6) as i64);
    a.sort_by_key(key);
    b.sort_by_key(key);
    let mut worst: f64 = 0.0;
    for (x, y) in a.iter().zip(&b) {
        assert!(dist(x.0, y.0) < 1e-12);
        worst = worst.max((x.1 - y.1).abs());
    }
    assert!(worst < 1e-12, "max difference {worst:e}");
}

#[cfg(not(feature = "mpi_backend"))]
#[test]
fn triangle_prisms_conserve_under_pure_diffusion() {
    let mesh = prisms();
    let physics = Physics {
        velocity: [0.0; 3],
        diffusivity: 0.01,
        dt: 1e-3,
        inflow: None,
    };
    let vol: f64 = mesh.cells.iter().map(|c| c.volume).sum();
    let total = |cells: &[(Vec3, f64)]| -> f64 {
        // Owned cells come out in global-id order on a single rank.
        cells
            .iter()
            .zip(&mesh.cells)
            .map(|(c, m)| c.1 * m.volume)
            .sum::<f64>()
    };
    let t0 = run::<UnstructuredMesh>(UnstructuredMeshPlugin { mesh: mesh.clone() }, physics, 0);
    let t1 = run::<UnstructuredMesh>(UnstructuredMeshPlugin { mesh: mesh.clone() }, physics, 200);
    let (m0, m1) = (total(&t0), total(&t1));
    assert!((m1 - m0).abs() < 1e-12 * vol, "mass drift {:e}", m1 - m0);
    let (lo, hi) = t0
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), c| (l.min(c.1), h.max(c.1)));
    assert!(t1.iter().all(|c| c.1 >= lo - 1e-12 && c.1 <= hi + 1e-12));
    // And it actually diffused.
    let spread = |v: &[(Vec3, f64)]| v.iter().map(|c| c.1).fold(f64::MIN, f64::max);
    assert!(spread(&t1) < 0.9 * spread(&t0));
}

// ── Implicit solve: -∇·∇u = f, u = 0 on the four side walls ─────────────────

/// Matrix-free two-point Laplacian over any `FvMesh`. Ghost entries of `y`
/// stay zero so CG's whole-slice dot products count owned cells only.
struct Laplacian<'m, M: FvMesh> {
    mesh: &'m M,
    comm: &'m dyn CommBackend,
    work: RefCell<CellVector>,
}

impl<M: FvMesh> LinearOperator for Laplacian<'_, M> {
    fn dimension(&self) -> usize {
        self.mesh.n_cells_total()
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        let mut w = self.work.borrow_mut();
        w.as_mut_slice().copy_from_slice(x);
        halo_exchange_forward(self.mesh.halo_plan(), &mut *w, self.comm);
        let u = w.as_slice();
        y.fill(0.0);
        self.mesh.for_each_face(&mut |f| {
            let (o, c) = (f.owner, f.other);
            let area = dot(f.area_normal, f.area_normal).sqrt();
            let d = dist(self.mesh.cell_centroid(o), self.mesh.cell_centroid(c));
            let k = area / d;
            match f.patch {
                // Dirichlet zero on the side walls (ghost = -owner), no flux in z.
                Some(p) if p < 4 => y[o] += 2.0 * k * u[o],
                Some(_) => {}
                None => {
                    y[o] += k * (u[o] - u[c]);
                    if self.mesh.is_local_cell(c) {
                        y[c] += k * (u[c] - u[o]);
                    }
                }
            }
        });
    }
}

/// Owned `(global id, u)` of the implicit solve, and the CG iteration count.
fn poisson(mesh: &UnstructuredMesh, comm: &dyn CommBackend) -> (Vec<(usize, f64)>, usize) {
    let mut rhs = CellVector::from_mesh(mesh);
    for c in 0..mesh.n_local_cells() {
        let x = mesh.cell_centroid(c);
        let f = 2.0
            * std::f64::consts::PI.powi(2)
            * (std::f64::consts::PI * x[0]).sin()
            * (std::f64::consts::PI * x[1]).sin();
        rhs.set(c, f * mesh.cell_volume(c));
    }
    let op = Laplacian {
        mesh,
        comm,
        work: RefCell::new(CellVector::from_mesh(mesh)),
    };
    let mut x = vec![0.0; mesh.n_cells_total()];
    let report = conjugate_gradient(comm, &op, rhs.as_slice(), &mut x, 2000, |s| {
        s.relative_residual() < 1e-11
    });
    let out = (0..mesh.n_local_cells())
        .map(|c| (mesh.global_id(c).expect("owned"), x[c]))
        .collect();
    (out, report.expect("CG converged").iterations)
}

#[test]
fn implicit_poisson_on_prisms_approximates_exact_solution() {
    let mesh = prisms();
    let local = UnstructuredMesh::serial(&mesh);
    let comm = grass_mpi::SingleProcessComm::new();
    let (u, iters) = poisson(&local, &comm);
    assert!(iters > 0);
    // Two-point fluxes on a skewed mesh are not consistent, so only ask for a
    // coarse match to sin(πx) sin(πy); the point is that the solve runs.
    let err = u
        .iter()
        .map(|&(g, v)| {
            let x = mesh.cells[g].centroid;
            (v - (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).sin()).abs()
        })
        .fold(0.0, f64::max);
    assert!(err < 0.1, "max error {err}");
}

#[cfg(feature = "mpi_backend")]
mod mpi {
    use super::*;
    use std::process::Command;

    const CHILD_ENV: &str = "FIELD_UNSTRUCTURED_MPI_CHILD";
    const RANK_COUNTS: [usize; 4] = [2, 3, 4, 7];

    /// Gather owned values by global id onto every rank.
    fn gather(comm: &dyn CommBackend, n: usize, owned: &[(usize, f64)]) -> Vec<f64> {
        let mut v = vec![0.0; n];
        for &(g, x) in owned {
            v[g] = x;
        }
        comm.all_reduce_sum_f64_slice(&mut v);
        v
    }

    fn run_child() {
        let mesh = prisms();
        let n = mesh.cells.len();

        // Explicit: the same App as the serial tests, now on several ranks.
        let parallel = run::<UnstructuredMesh>(
            UnstructuredMeshPlugin { mesh: mesh.clone() },
            advect_diffuse(),
            80,
        );
        let comm = grass_mpi::MpiCommBackend::new(grass_mpi::get_mpi_world());
        let size = comm.size() as usize;
        let owner = mesh.rcb_partition(size);
        let mine = UnstructuredMesh::partition(&mesh, &owner, comm.rank());
        assert!(mine.n_halo_cells() > 0, "a split mesh must have a halo");
        let owned: Vec<(usize, f64)> = parallel
            .iter()
            .enumerate()
            .map(|(c, &(_, v))| (mine.global_id(c).expect("owned"), v))
            .collect();
        let got = gather(&comm, n, &owned);

        // Serial reference, computed on every rank with no communication.
        let serial = UnstructuredMesh::serial(&mesh);
        let mut s = Scalar {
            u: vec![0.0; serial.n_cells_total()],
            rhs: vec![0.0; serial.n_cells_total()],
        };
        for c in 0..n {
            s.u[c] = initial(serial.cell_centroid(c));
        }
        for _ in 0..80 {
            fill_boundary(&serial, &mut s, &advect_diffuse());
            s.rhs.fill(0.0);
            assemble_flux(&serial, &mut s, &advect_diffuse());
            for c in 0..n {
                s.u[c] += advect_diffuse().dt * s.rhs[c];
            }
        }
        let worst = (0..n).map(|g| (got[g] - s.u[g]).abs()).fold(0.0, f64::max);
        assert!(
            worst < 1e-12,
            "{size} ranks: explicit differs from serial by {worst:e}"
        );

        // Implicit: distributed CG against the serial CG.
        let (par, _) = poisson(&mine, &comm);
        let got = gather(&comm, n, &par);
        let (ser, _) = poisson(&serial, &grass_mpi::SingleProcessComm::new());
        let worst = ser
            .iter()
            .map(|&(g, v)| (got[g] - v).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-8,
            "{size} ranks: implicit differs from serial by {worst:e}"
        );

        drop(comm);
        grass_mpi::finalize_mpi();
    }

    #[test]
    fn mpi_unstructured_matches_serial() {
        if std::env::var_os(CHILD_ENV).is_some() {
            run_child();
            return;
        }
        if Command::new("mpirun").arg("--version").output().is_err() {
            eprintln!("SKIP mpi_unstructured_matches_serial: `mpirun` not on PATH");
            return;
        }
        let exe = std::env::current_exe().expect("locate test binary");
        for np in RANK_COUNTS {
            let status = Command::new("mpirun")
                .args(["--oversubscribe", "-np", &np.to_string()])
                .arg(&exe)
                .args([
                    "--exact",
                    "mpi::mpi_unstructured_matches_serial",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, "1")
                .status()
                .expect("spawn mpirun");
            assert!(
                status.success(),
                "unstructured check failed on {np} ranks: {status}"
            );
        }
    }
}
