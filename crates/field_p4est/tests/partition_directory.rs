//! Coupling ownership directory tests, valid both serially and under mpirun.

use field_core::Bounds3;
use field_p4est::{ForestGrid, ForestLayout, ForestPartitionDirectory};

#[test]
fn directory_queries_current_adaptive_partition_without_communication() {
    let mut grid = ForestGrid::new(ForestLayout {
        trees_x: 4,
        trees_y: 2,
        trees_z: 1,
        xmin: -1.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 2.0,
        zmax: 2.5,
        min_level: 1,
    })
    .expect("forest creation");
    grid.refine(3, |cx, cy, _cz, h| cx < 0.25 && cy < 0.75 && h > 0.04);

    let size = grid.mpisize() as i32;
    let rank = grid.mpirank() as i32;
    let probe_points = [
        [-1.0, 0.0, 2.0],
        [-0.75, 0.2, 2.1],
        [0.0, 0.5, 2.25],
        [0.8, 0.9, 2.4],
        [1.0, 1.0, 2.5],
    ];
    // Cross-check the global owner directory against p8est's independent
    // rank-local leaf search on every process.
    for point in &probe_points[1..4] {
        let point = *point;
        let owner = grid.owner_rank_at(point).unwrap();
        assert_eq!(
            grid.find_leaf_at(point[0], point[1], point[2]).is_some(),
            owner == rank,
            "global owner must match local leaf search at {point:?}"
        );
    }

    let directory = ForestPartitionDirectory::new(&grid);
    assert_eq!(directory.role_size(), size);
    for point in probe_points {
        let owner = directory
            .owner_rank(point)
            .expect("in-domain point has an owner");
        assert!((0..size).contains(&owner));
        assert_eq!(
            directory.overlapping_ranks(Bounds3::new(point, point)),
            vec![owner]
        );
    }
    assert_eq!(directory.owner_rank([-1.01, 0.5, 2.25]), None);
    assert_eq!(directory.owner_rank([0.0, 1.01, 2.25]), None);

    let all: Vec<i32> = (0..size).collect();
    assert_eq!(
        directory.overlapping_ranks(Bounds3::new([-1.0, 0.0, 2.0], [1.0, 1.0, 2.5])),
        all,
    );

    let support = Bounds3::new([-0.1, 0.1, 2.1], [0.1, 0.9, 2.4]);
    let overlapping = directory.overlapping_ranks(support);
    assert!(!overlapping.is_empty());
    assert!(overlapping.windows(2).all(|pair| pair[0] < pair[1]));
    for point in [[0.0, 0.2, 2.2], [0.0, 0.5, 2.25], [0.0, 0.8, 2.3]] {
        assert!(overlapping.contains(&directory.owner_rank(point).unwrap()));
    }
}
