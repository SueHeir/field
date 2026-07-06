//! p4est_crossrank_halo — validate FIELD's p4est ghost cells through HaloPlan.
//!
//! The checked field is intentionally substrate-generic: a scalar cell column
//! with a value computed only from cell geometry. That lets the validation know
//! the exact value every received ghost must hold without depending on any CFD,
//! FEM, or LBM physics.

use std::any::Any;
use std::env;
use std::process;

use field_core::{halo_exchange_forward, FieldData, FvMesh};
use field_p4est::{finalize as finalize_p4est, ForestGrid, ForestLayout, ForestMesh};
use grass_io::{load_toml, Config};
use grass_mpi::{finalize_mpi, get_mpi_world, CommBackend, MpiCommBackend};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct ForestConfig {
    trees: [i32; 3],
    bounds_lo: [f64; 3],
    bounds_hi: [f64; 3],
    min_level: i32,
    max_level: i32,
    refine_center: [f64; 3],
    refine_radius: f64,
}

impl Default for ForestConfig {
    fn default() -> Self {
        Self {
            trees: [4, 4, 1],
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [1.0, 1.0, 0.25],
            min_level: 0,
            max_level: 1,
            refine_center: [0.5, 0.5, 0.125],
            refine_radius: 0.23,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct ValidationConfig {
    expected_ranks: i32,
    value_tol: f64,
    min_cross_rank_ghosts: usize,
    min_cross_rank_refinement_faces: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            expected_ranks: 2,
            value_tol: 1.0e-12,
            min_cross_rank_ghosts: 1,
            min_cross_rank_refinement_faces: 1,
        }
    }
}

#[derive(Default)]
struct ScalarField {
    value: Vec<f64>,
}

impl FieldData for ScalarField {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.value.len()
    }

    fn resize(&mut self, n: usize) {
        self.value.resize(n, 0.0);
    }

    fn forward_size(&self) -> usize {
        1
    }

    fn pack_forward(&self, c: usize, buf: &mut Vec<f64>) {
        buf.push(self.value[c]);
    }

    fn unpack_forward(&mut self, c: usize, buf: &[f64]) -> usize {
        self.value[c] = buf[0];
        1
    }
}

fn analytic_value(center: [f64; 3], volume: f64) -> f64 {
    1.0 + 2.0 * center[0] - 3.0 * center[1] + 5.0 * center[2] + 7.0 * volume
}

fn layout_from(cfg: &ForestConfig) -> ForestLayout {
    ForestLayout {
        trees_x: cfg.trees[0],
        trees_y: cfg.trees[1],
        trees_z: cfg.trees[2],
        xmin: cfg.bounds_lo[0],
        xmax: cfg.bounds_hi[0],
        ymin: cfg.bounds_lo[1],
        ymax: cfg.bounds_hi[1],
        zmin: cfg.bounds_lo[2],
        zmax: cfg.bounds_hi[2],
        min_level: cfg.min_level,
    }
}

fn fail(rank: i32, msg: &str) -> ! {
    eprintln!("rank {rank}: CHECKS FAILED: {msg}");
    process::exit(1);
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: p4est_crossrank_halo <config.toml>");
        process::exit(2);
    });

    let cfg = Config::from_table(load_toml(&path));
    let forest: ForestConfig = cfg.section("forest");
    let validation: ValidationConfig = cfg.section("validation");

    let world = get_mpi_world();
    let comm = MpiCommBackend::new(world);
    let rank = comm.rank();
    let size = comm.size();
    if size != validation.expected_ranks {
        fail(
            rank,
            &format!(
                "expected {} MPI ranks from config, got {size}",
                validation.expected_ranks
            ),
        );
    }

    let mut grid = ForestGrid::new(layout_from(&forest)).expect("p4est forest creation");
    let rc = forest.refine_center;
    let radius = forest.refine_radius;
    grid.refine(forest.max_level, |cx, cy, cz, _h| {
        let r2 = (cx - rc[0]).powi(2) + (cy - rc[1]).powi(2) + (cz - rc[2]).powi(2);
        r2.sqrt() < radius
    });

    let mesh = ForestMesh::new(grid);
    let plan = mesh.halo_plan();
    let local_recv_ghosts: usize = plan.links.iter().map(|link| link.recv_cells.len()).sum();
    let local_send_mirrors: usize = plan.links.iter().map(|link| link.send_cells.len()).sum();
    let local_refinement_faces = mesh.cross_rank_refinement_faces();

    let mut field = ScalarField::default();
    field.resize(mesh.n_cells_total());
    for c in 0..mesh.n_cells_total() {
        field.value[c] = if mesh.is_local_cell(c) {
            analytic_value(mesh.cell_centroid(c), mesh.cell_volume(c))
        } else {
            -9.87654321e30
        };
    }

    halo_exchange_forward(plan, &mut field, &comm);

    let mut changed = 0usize;
    let mut max_abs_err = 0.0f64;
    for link in &plan.links {
        for &c in &link.recv_cells {
            let got = field.value[c];
            if got != -9.87654321e30 {
                changed += 1;
            }
            let expect = analytic_value(mesh.cell_centroid(c), mesh.cell_volume(c));
            max_abs_err = max_abs_err.max((got - expect).abs());
        }
    }

    let global_recv = comm.all_reduce_sum_f64(local_recv_ghosts as f64) as usize;
    let global_send = comm.all_reduce_sum_f64(local_send_mirrors as f64) as usize;
    let global_changed = comm.all_reduce_sum_f64(changed as f64) as usize;
    let global_refinement_faces = comm.all_reduce_sum_f64(local_refinement_faces as f64) as usize;
    let global_max_err = -comm.all_reduce_min_f64(-max_abs_err);
    let global_links = comm.all_reduce_sum_f64(plan.links.len() as f64) as usize;

    let mut ok = true;
    if global_recv < validation.min_cross_rank_ghosts {
        ok = false;
    }
    if global_send < validation.min_cross_rank_ghosts {
        ok = false;
    }
    if global_changed != global_recv {
        ok = false;
    }
    if global_refinement_faces < validation.min_cross_rank_refinement_faces {
        ok = false;
    }
    if global_max_err > validation.value_tol {
        ok = false;
    }

    if rank == 0 {
        println!(
            "RESULT ranks={} links={} recv_ghosts={} send_mirrors={} changed_ghosts={} \
             cross_rank_refinement_faces={} max_abs_err={:.16e} tol={:.16e} status={}",
            size,
            global_links,
            global_recv,
            global_send,
            global_changed,
            global_refinement_faces,
            global_max_err,
            validation.value_tol,
            if ok { "PASS" } else { "FAIL" },
        );
    }

    comm.barrier();
    drop(field);
    drop(mesh);
    drop(comm);
    finalize_p4est();
    finalize_mpi();
    if !ok {
        process::exit(1);
    }
}
