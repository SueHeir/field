//! Integration test: build a real p4est forest, refine it, wrap it as a
//! `ForestMesh`, and verify the FV mesh is geometrically closed and the flux
//! scatter is conservative across the (arbitrary-level) coarse/fine interfaces.
//!
//! Lives in an integration test (its own process) because constructing a forest
//! calls `MPI_Init`, which a process may do only once.
#![allow(clippy::needless_range_loop)] // explicit axis indexing reads clearest here

use field_core::{AdaptiveMesh, FvMesh};
use field_p4est::{ForestGrid, ForestLayout, ForestMesh};

#[test]
fn p4est_forest_mesh_is_closed_and_conservative() {
    // 4×4×1 brick over [0,1]×[0,1]×[0,0.25]; per-tree extent 0.25 in x/y (isotropic
    // enough for the test), uniform base level 0.
    let layout = ForestLayout {
        trees_x: 4,
        trees_y: 4,
        trees_z: 1,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 0.25,
        min_level: 0,
    };
    let mut grid = ForestGrid::new(layout).expect("forest creation");

    // Refine leaves near the domain centre up to level 1 → real coarse/fine faces.
    grid.refine(1, |cx, cy, _cz, _h| {
        ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt() < 0.22
    });
    let mesh = ForestMesh::new(grid);

    // Mixed refinement levels.
    let (coarse, fine) = mesh.level_counts();
    assert!(fine > 0 && coarse > 0, "expected mixed levels: {coarse} coarse + {fine} fine");

    // Interior leaf volumes tile the domain exactly.
    let vol: f64 = (0..mesh.n_cells_total())
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| mesh.cell_volume(c))
        .sum();
    assert!((vol - 0.25).abs() < 1e-9, "interior volume {vol} != domain 0.25");

    // Per-cell geometric closure (Σ area_normal = 0) AND structural flux
    // conservation (unit flux scatter telescopes to the boundary contribution).
    let n = mesh.n_cells_total();
    let area = |an: [f64; 3]| (an[0] * an[0] + an[1] * an[1] + an[2] * an[2]).sqrt();
    let mut net = vec![[0.0f64; 3]; n];
    let mut rhs = vec![0.0f64; n];
    mesh.for_each_face(&mut |f| {
        let a = area(f.area_normal);
        for d in 0..3 {
            net[f.owner][d] -= f.area_normal[d];
        }
        rhs[f.owner] -= a / mesh.cell_volume(f.owner);
        if f.patch.is_none() && mesh.is_local_cell(f.other) {
            for d in 0..3 {
                net[f.other][d] += f.area_normal[d];
            }
            rhs[f.other] += a / mesh.cell_volume(f.other);
        }
    });
    for c in 0..n {
        if mesh.is_local_cell(c) {
            for d in 0..3 {
                assert!(net[c][d].abs() < 1e-9, "cell {c} not closed on axis {d}: {}", net[c][d]);
            }
        }
    }
    let total: f64 = (0..n)
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| rhs[c] * mesh.cell_volume(c))
        .sum();
    let mut boundary = 0.0;
    mesh.for_each_face(&mut |f| {
        if f.patch.is_some() {
            boundary += area(f.area_normal);
        }
    });
    assert!(
        (total + boundary).abs() < 1e-9,
        "non-conservative p4est face list: Σ Vc·rhs = {total}, −boundary = {}",
        -boundary
    );

    // The AdaptiveMesh contract surfaces the coarse/fine interface faces.
    let mut n_cf = 0;
    mesh.for_each_coarse_fine_face(&mut |_| n_cf += 1);
    assert!(n_cf > 0, "expected coarse/fine faces from the refined forest");
}

#[test]
fn p4est_ghost_layer_plumbing_is_serial_noop_on_one_rank() {
    // The cross-rank ghost layer + halo plan must be a correct *no-op* on a
    // single rank (and the FFI round-trips must return sane values), so the
    // multi-rank wiring never perturbs the serial path.
    let layout = ForestLayout {
        trees_x: 4,
        trees_y: 4,
        trees_z: 1,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 0.25,
        min_level: 0,
    };
    let mut grid = ForestGrid::new(layout).expect("forest");
    grid.refine(1, |cx, cy, _cz, _h| ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt() < 0.22);

    // Single-rank FFI round-trips.
    assert_eq!(grid.mpisize(), 1);
    assert_eq!(grid.mpirank(), 0);
    assert_eq!(grid.ghosts().len(), 0, "no off-rank ghosts on a single rank");
    assert_eq!(grid.ghost_proc_offsets(), vec![0, 0], "proc_offsets is [0,0] for mpisize 1");
    let (send_off, send_locals) = grid.mirror_sends();
    assert_eq!(send_off, vec![0, 0]);
    assert!(send_locals.is_empty());

    // The mesh's halo plan is serial (empty), so halo exchange is a no-op and the
    // cell list holds only local leaves + boundary ghosts (no off-rank cells).
    let n_local = grid.n_local_leaves();
    let mesh = ForestMesh::new(grid);
    assert!(mesh.halo_plan().is_serial(), "single-rank halo plan must be empty");
    assert_eq!(mesh.n_local_cells(), n_local);
}
