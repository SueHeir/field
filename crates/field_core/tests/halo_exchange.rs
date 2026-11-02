//! End-to-end checks of the `UniformMesh` halo plan + generic exchange, for
//! every combination of periodic axes.
//!
//! Every owned cell holds a value that encodes its *global* `(i, j, k)`. After a
//! forward exchange each face ghost must hold the value of the global cell it
//! images (wrapped on a periodic axis), and a ghost on a physical boundary must
//! be untouched. A reverse exchange of ones must add exactly one contribution per
//! halo link that reads an owned cell.
//!
//! The serial test covers periodic self-links. Under `--features mpi_backend`
//! the MPI test re-launches this binary under `mpirun` for several rank counts
//! (the grass_mpi pattern), covering split axes, two-rank periodic axes (both
//! sides talk to the same rank), and non-adjacent wraps.

use field_core::prelude::*;
use field_core::uniform::{factor_decomposition, rank_position};
use field_core::{
    halo_exchange_forward_all, halo_exchange_forward_all_with_corners, halo_exchange_reverse_all,
    halo_exchange_reverse_all_with_corners,
};
use grass_mpi::CommBackend;

#[derive(FieldData, Default)]
struct Probe {
    #[forward]
    u: Vec<f64>,
    #[reverse]
    r: Vec<f64>,
}

const GLOBAL: [usize; 3] = [7, 5, 4];
const NG: usize = 2;
const SENTINEL: f64 = -1.0;

fn config() -> UniformMeshConfig {
    UniformMeshConfig {
        nx: GLOBAL[0],
        ny: GLOBAL[1],
        nz: GLOBAL[2],
        ng: NG,
        bounds_lo: [0.0; 3],
        // Unit spacing, so a centroid's global index is `x - 0.5`.
        bounds_hi: [GLOBAL[0] as f64, GLOBAL[1] as f64, GLOBAL[2] as f64],
        y_edges: None,
        z_edges: None,
    }
}

fn encode(g: [i64; 3]) -> f64 {
    (g[0] * 10_000 + g[1] * 100 + g[2] + 1) as f64
}

fn all_periodic_combos() -> impl Iterator<Item = [bool; 3]> {
    (0..8u8).map(|m| [m & 1 != 0, m & 2 != 0, m & 4 != 0])
}

/// Build this rank's mesh, exchange forward and reverse, and check every cell.
/// Returns the number of checked face-ghost cells (so callers can assert the
/// test actually looked at something).
fn check(comm: &dyn CommBackend, periodic: [bool; 3]) -> usize {
    let cfg = config();
    let decomp = factor_decomposition(comm.size(), GLOBAL).expect("decomposition");
    let pos = rank_position(comm.rank(), decomp);
    let mesh = UniformMesh::from_config_periodic(&cfg, decomp, pos, periodic);

    let mut reg = FieldRegistry::new();
    reg.register(Probe::default());
    reg.resize_all(mesh.n_cells_total());

    let n = mesh.dims();
    let tot = mesh.total_dims();
    let ng = mesh.n_ghost();
    let global_of = |c: usize| -> [i64; 3] {
        let x = mesh.cell_centroid(c);
        [0, 1, 2].map(|a| (x[a] - 0.5).round() as i64)
    };
    // Per axis: -1 below the local interior, +1 above, 0 inside.
    let side = |r: usize, a: usize| -> i32 {
        if r < ng {
            -1
        } else if r >= ng + n[a] {
            1
        } else {
            0
        }
    };

    {
        let mut p = reg.get_mut::<Probe>().unwrap();
        for ir in 0..tot[0] {
            for jr in 0..tot[1] {
                for kr in 0..tot[2] {
                    let c = mesh.idx_raw(ir, jr, kr);
                    let ghost = [side(ir, 0), side(jr, 1), side(kr, 2)] != [0, 0, 0];
                    p.u[c] = if ghost {
                        SENTINEL
                    } else {
                        encode(global_of(c))
                    };
                    p.r[c] = if ghost { 1.0 } else { 0.0 };
                }
            }
        }
    }

    halo_exchange_forward_all(mesh.halo_plan(), &reg, comm);
    halo_exchange_reverse_all(mesh.halo_plan(), &reg, comm);

    let p = reg.get::<Probe>().unwrap();
    let mut checked = 0;
    for ir in 0..tot[0] {
        for jr in 0..tot[1] {
            for kr in 0..tot[2] {
                let raw = [ir, jr, kr];
                let c = mesh.idx_raw(ir, jr, kr);
                let sides = [0, 1, 2].map(|a| side(raw[a], a));
                let out: Vec<usize> = (0..3).filter(|&a| sides[a] != 0).collect();
                let g = global_of(c);
                match out.as_slice() {
                    [] => {
                        // Owned cell: one reverse contribution per link that
                        // sends this cell.
                        let mut want = 0.0;
                        for a in 0..3 {
                            let i = raw[a] - ng;
                            if i < ng && mesh.neighbor_rank(a, false).is_some() {
                                want += 1.0;
                            }
                            if i + ng >= n[a] && mesh.neighbor_rank(a, true).is_some() {
                                want += 1.0;
                            }
                        }
                        assert_eq!(
                            p.r[c],
                            want,
                            "rank {} periodic {periodic:?}: reverse sum at global {g:?}",
                            comm.rank()
                        );
                    }
                    [a] => {
                        let a = *a;
                        let inside = (0..GLOBAL[a] as i64).contains(&g[a]);
                        let want = if inside {
                            encode(g)
                        } else if periodic[a] {
                            let mut w = g;
                            w[a] = g[a].rem_euclid(GLOBAL[a] as i64);
                            encode(w)
                        } else {
                            SENTINEL
                        };
                        assert_eq!(
                            p.u[c],
                            want,
                            "rank {} periodic {periodic:?}: face ghost at global {g:?}",
                            comm.rank()
                        );
                        checked += 1;
                    }
                    // Edge/corner ghosts are not part of a face-only halo.
                    _ => {}
                }
            }
        }
    }
    checked
}

/// The corner-complete plans: after a staged forward exchange every ghost
/// (face, edge or corner) whose out-of-range axes all have links holds the
/// value of the global cell it images, and a staged reverse exchange folds
/// every such ghost back onto exactly that owner. Returns the number of
/// edge/corner ghosts checked.
fn check_corners(comm: &dyn CommBackend, periodic: [bool; 3]) -> usize {
    let cfg = config();
    let decomp = factor_decomposition(comm.size(), GLOBAL).expect("decomposition");
    let pos = rank_position(comm.rank(), decomp);
    let mesh = UniformMesh::from_config_periodic(&cfg, decomp, pos, periodic);
    let plans = mesh.corner_halo_plans();

    let mut reg = FieldRegistry::new();
    reg.register(Probe::default());
    reg.resize_all(mesh.n_cells_total());

    let n = mesh.dims();
    let tot = mesh.total_dims();
    let ng = mesh.n_ghost();
    let global_of = |c: usize| -> [i64; 3] {
        let x = mesh.cell_centroid(c);
        [0, 1, 2].map(|a| (x[a] - 0.5).round() as i64)
    };
    let is_ghost = |raw: [usize; 3]| (0..3).any(|a| raw[a] < ng || raw[a] >= ng + n[a]);

    {
        let mut p = reg.get_mut::<Probe>().unwrap();
        for ir in 0..tot[0] {
            for jr in 0..tot[1] {
                for kr in 0..tot[2] {
                    let c = mesh.idx_raw(ir, jr, kr);
                    let ghost = is_ghost([ir, jr, kr]);
                    p.u[c] = if ghost {
                        SENTINEL
                    } else {
                        encode(global_of(c))
                    };
                    p.r[c] = if ghost { 1.0 } else { 0.0 };
                }
            }
        }
    }

    halo_exchange_forward_all_with_corners(plans, &reg, comm);
    halo_exchange_reverse_all_with_corners(plans, &reg, comm);

    let p = reg.get::<Probe>().unwrap();
    let mut checked = 0;
    let mut owned_sum = 0.0;
    let mut linked_ghosts = 0.0;
    for ir in 0..tot[0] {
        for jr in 0..tot[1] {
            for kr in 0..tot[2] {
                let raw = [ir, jr, kr];
                let c = mesh.idx_raw(ir, jr, kr);
                let g = global_of(c);
                if !is_ghost(raw) {
                    // Images of this cell: per axis, itself plus one per
                    // linked side it is within `ng` of.
                    let mut images = 1.0;
                    for a in 0..3 {
                        let i = raw[a] - ng;
                        let mut m = 1.0;
                        if i < ng && mesh.neighbor_rank(a, false).is_some() {
                            m += 1.0;
                        }
                        if i + ng >= n[a] && mesh.neighbor_rank(a, true).is_some() {
                            m += 1.0;
                        }
                        images *= m;
                    }
                    assert_eq!(
                        p.r[c],
                        images - 1.0,
                        "rank {} periodic {periodic:?}: corner reverse sum at global {g:?}",
                        comm.rank()
                    );
                    owned_sum += p.r[c];
                    continue;
                }
                let resolvable =
                    (0..3).all(|a| periodic[a] || (0..GLOBAL[a] as i64).contains(&g[a]));
                let want = if resolvable {
                    linked_ghosts += 1.0;
                    encode([0, 1, 2].map(|a| g[a].rem_euclid(GLOBAL[a] as i64)))
                } else {
                    SENTINEL
                };
                assert_eq!(
                    p.u[c],
                    want,
                    "rank {} periodic {periodic:?}: ghost at global {g:?}",
                    comm.rank()
                );
                let out = (0..3)
                    .filter(|&a| raw[a] < ng || raw[a] >= ng + n[a])
                    .count();
                if out > 1 {
                    checked += 1;
                }
            }
        }
    }
    // Conservation: everything deposited in a resolvable ghost lands on an
    // owner somewhere.
    let owned = comm.all_reduce_sum_f64(owned_sum);
    let linked = comm.all_reduce_sum_f64(linked_ghosts);
    assert_eq!(
        owned, linked,
        "periodic {periodic:?}: reverse corner exchange lost charge"
    );
    checked
}

#[test]
fn serial_corner_plans_fill_and_fold_edge_and_corner_ghosts() {
    let comm = grass_mpi::SingleProcessComm::new();
    for periodic in all_periodic_combos() {
        let checked = check_corners(&comm, periodic);
        if periodic.iter().filter(|&&p| p).count() >= 2 {
            assert!(checked > 0, "periodic {periodic:?}: no edge ghost checked");
        }
    }
}

#[test]
fn serial_periodic_self_links_fill_ghosts() {
    let comm = grass_mpi::SingleProcessComm::new();
    for periodic in all_periodic_combos() {
        assert!(check(&comm, periodic) > 0);
    }
    // A serial, non-periodic mesh still has nothing to exchange.
    let mesh = UniformMesh::from_config(&config());
    assert!(mesh.halo_plan().is_serial());
}

#[test]
fn periodic_axis_that_is_not_split_gets_one_self_link() {
    let mesh =
        UniformMesh::from_config_periodic(&config(), [1, 1, 1], [0, 0, 0], [true, false, true]);
    assert_eq!(mesh.halo_plan().links.len(), 2);
    assert!(mesh.halo_plan().links.iter().all(|l| l.rank == 0));
    assert_eq!(mesh.neighbor_rank(0, false), Some(0));
    assert_eq!(mesh.neighbor_rank(1, true), None);
}

#[test]
fn periodic_sides_emit_no_boundary_patch() {
    let mesh =
        UniformMesh::from_config_periodic(&config(), [1, 1, 1], [0, 0, 0], [true, false, false]);
    let mut patches = std::collections::BTreeSet::new();
    mesh.for_each_face(&mut |f| {
        if let Some(p) = f.patch {
            patches.insert(p);
        }
    });
    let want: std::collections::BTreeSet<u32> = [
        BoundarySide::YLo,
        BoundarySide::YHi,
        BoundarySide::ZLo,
        BoundarySide::ZHi,
    ]
    .into_iter()
    .map(|s| s as u32)
    .collect();
    assert_eq!(patches, want);
}

#[cfg(feature = "mpi_backend")]
mod mpi {
    use super::*;
    use std::process::Command;

    const CHILD_ENV: &str = "FIELD_HALO_MPI_CHILD";
    // 2: both sides of a periodic x talk to one rank. 3: non-adjacent wrap.
    // 4 and 8: several split axes at once.
    const RANK_COUNTS: [usize; 4] = [2, 3, 4, 8];

    fn run_child() {
        let comm = grass_mpi::MpiCommBackend::new(grass_mpi::get_mpi_world());
        for periodic in all_periodic_combos() {
            let local = check(&comm, periodic);
            let total = comm.all_reduce_sum_f64(local as f64);
            assert!(total > 0.0);
            check_corners(&comm, periodic);
        }
        drop(comm);
        grass_mpi::finalize_mpi();
    }

    #[test]
    fn mpi_halo_exchange_matches_global_field() {
        if std::env::var_os(CHILD_ENV).is_some() {
            run_child();
            return;
        }
        if Command::new("mpirun").arg("--version").output().is_err() {
            eprintln!("SKIP mpi_halo_exchange_matches_global_field: `mpirun` not on PATH");
            return;
        }
        let exe = std::env::current_exe().expect("locate test binary");
        for np in RANK_COUNTS {
            let status = Command::new("mpirun")
                .args(["--oversubscribe", "-np", &np.to_string()])
                .arg(&exe)
                .args([
                    "--exact",
                    "mpi::mpi_halo_exchange_matches_global_field",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, "1")
                .status()
                .expect("spawn mpirun");
            assert!(
                status.success(),
                "halo exchange check failed on {np} ranks: {status}"
            );
        }
    }
}
