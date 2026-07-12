//! Spatial ownership directory for coupling packages.
//!
//! FIELD describes which role-rank owns or overlaps a spatial support. It does
//! not choose what data to exchange or how to interpolate it.

use crate::{UniformMeshConfig, Vec3};

/// Axis-aligned bounds in physical coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds3 {
    /// Lower corner.
    pub lo: Vec3,
    /// Upper corner.
    pub hi: Vec3,
}

impl Bounds3 {
    /// Construct validated bounds.
    pub fn new(lo: Vec3, hi: Vec3) -> Self {
        assert!(
            (0..3).all(|axis| lo[axis] <= hi[axis]),
            "partition bounds require lo <= hi, got {lo:?}..{hi:?}"
        );
        Self { lo, hi }
    }

    fn has_positive_overlap(self, other: Self) -> bool {
        (0..3).all(|axis| self.lo[axis] < other.hi[axis] && other.lo[axis] < self.hi[axis])
    }
}

/// One owned spatial extent for one rank in a FIELD role communicator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartitionExtent {
    /// Rank within the FIELD role communicator.
    pub role_rank: i32,
    /// Owned physical bounds.
    pub bounds: Bounds3,
}

/// Searchable directory of FIELD role-rank ownership.
///
/// A rank may own several extents, which accommodates adaptive forests and
/// disconnected partitions without changing the coupling-facing API.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionDirectory {
    role_size: i32,
    global: Bounds3,
    extents: Vec<PartitionExtent>,
}

impl PartitionDirectory {
    /// Build a directory from explicit rank extents.
    pub fn new(role_size: i32, global: Bounds3, extents: Vec<PartitionExtent>) -> Self {
        assert!(role_size > 0, "partition role size must be positive");
        assert!(
            extents
                .iter()
                .all(|extent| extent.role_rank >= 0 && extent.role_rank < role_size),
            "partition extent contains an out-of-range role rank"
        );
        Self {
            role_size,
            global,
            extents,
        }
    }

    /// Construct the exact Cartesian ownership directory used by
    /// [`crate::UniformMesh::from_config_decomposed`].
    pub fn from_uniform_config(cfg: &UniformMeshConfig, decomp: [i32; 3]) -> Self {
        assert!(
            (0..3).all(|axis| decomp[axis] > 0),
            "decomposition entries must be positive"
        );
        let global_n = [cfg.nx, cfg.ny, cfg.nz];
        assert!(
            (0..3).all(|axis| decomp[axis] as usize <= global_n[axis]),
            "decomposition cannot create zero-cell partitions"
        );
        let edges = [
            uniform_edges(cfg.nx, cfg.bounds_lo[0], cfg.bounds_hi[0]),
            cfg.y_edges
                .clone()
                .unwrap_or_else(|| uniform_edges(cfg.ny, cfg.bounds_lo[1], cfg.bounds_hi[1])),
            cfg.z_edges
                .clone()
                .unwrap_or_else(|| uniform_edges(cfg.nz, cfg.bounds_lo[2], cfg.bounds_hi[2])),
        ];
        assert_eq!(edges[1].len(), cfg.ny + 1, "y_edges must have length ny+1");
        assert_eq!(edges[2].len(), cfg.nz + 1, "z_edges must have length nz+1");

        let mut extents = Vec::with_capacity((decomp[0] * decomp[1] * decomp[2]) as usize);
        for px in 0..decomp[0] {
            for py in 0..decomp[1] {
                for pz in 0..decomp[2] {
                    let pos = [px, py, pz];
                    let mut lo = [0.0; 3];
                    let mut hi = [0.0; 3];
                    for axis in 0..3 {
                        let (offset, count) = partition_span(
                            global_n[axis],
                            decomp[axis] as usize,
                            pos[axis] as usize,
                        );
                        lo[axis] = edges[axis][offset];
                        hi[axis] = edges[axis][offset + count];
                    }
                    let role_rank = px * (decomp[1] * decomp[2]) + py * decomp[2] + pz;
                    extents.push(PartitionExtent {
                        role_rank,
                        bounds: Bounds3::new(lo, hi),
                    });
                }
            }
        }
        Self::new(
            decomp[0] * decomp[1] * decomp[2],
            Bounds3::new(cfg.bounds_lo, cfg.bounds_hi),
            extents,
        )
    }

    /// Number of ranks in the FIELD role communicator.
    pub fn role_size(&self) -> i32 {
        self.role_size
    }

    /// All recorded extents in deterministic role-rank construction order.
    pub fn extents(&self) -> &[PartitionExtent] {
        &self.extents
    }

    /// Unique owner of a point under half-open partition semantics. Shared
    /// internal boundaries belong to the high-side partition; the global high
    /// boundary remains inside the domain.
    pub fn owner_rank(&self, point: Vec3) -> Option<i32> {
        if !(0..3)
            .all(|axis| point[axis] >= self.global.lo[axis] && point[axis] <= self.global.hi[axis])
        {
            return None;
        }
        self.extents
            .iter()
            .find(|extent| {
                (0..3).all(|axis| {
                    point[axis] >= extent.bounds.lo[axis]
                        && (point[axis] < extent.bounds.hi[axis]
                            || (extent.bounds.hi[axis] == self.global.hi[axis]
                                && point[axis] <= extent.bounds.hi[axis]))
                })
            })
            .map(|extent| extent.role_rank)
    }

    /// Sorted, deduplicated role-ranks with positive-volume overlap with
    /// `support`. A zero-volume support is treated as a point ownership query.
    pub fn overlapping_ranks(&self, support: Bounds3) -> Vec<i32> {
        if support.lo == support.hi {
            return self.owner_rank(support.lo).into_iter().collect();
        }
        let mut ranks: Vec<i32> = self
            .extents
            .iter()
            .filter(|extent| extent.bounds.has_positive_overlap(support))
            .map(|extent| extent.role_rank)
            .collect();
        ranks.sort_unstable();
        ranks.dedup();
        ranks
    }
}

fn uniform_edges(n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..=n)
        .map(|index| lo + index as f64 * (hi - lo) / n as f64)
        .collect()
}

fn partition_span(global_n: usize, parts: usize, position: usize) -> (usize, usize) {
    let base = global_n / parts;
    let remainder = global_n % parts;
    if position < remainder {
        (position * (base + 1), base + 1)
    } else {
        (remainder * (base + 1) + (position - remainder) * base, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn x_split() -> PartitionDirectory {
        PartitionDirectory::from_uniform_config(
            &UniformMeshConfig {
                nx: 10,
                ny: 2,
                nz: 2,
                ng: 1,
                bounds_lo: [0.0, 0.0, 0.0],
                bounds_hi: [1.0, 1.0, 1.0],
                y_edges: None,
                z_edges: None,
            },
            [2, 1, 1],
        )
    }

    #[test]
    fn point_owner_uses_half_open_internal_boundary() {
        let directory = x_split();
        assert_eq!(directory.owner_rank([0.0, 0.5, 0.5]), Some(0));
        assert_eq!(directory.owner_rank([0.499, 0.5, 0.5]), Some(0));
        assert_eq!(directory.owner_rank([0.5, 0.5, 0.5]), Some(1));
        assert_eq!(directory.owner_rank([1.0, 0.5, 0.5]), Some(1));
        assert_eq!(directory.owner_rank([1.1, 0.5, 0.5]), None);
    }

    #[test]
    fn finite_support_can_overlap_both_owners() {
        let directory = x_split();
        assert_eq!(
            directory.overlapping_ranks(Bounds3::new([0.45, 0.2, 0.2], [0.55, 0.8, 0.8])),
            vec![0, 1]
        );
        assert_eq!(
            directory.overlapping_ranks(Bounds3::new([0.1, 0.2, 0.2], [0.2, 0.8, 0.8])),
            vec![0]
        );
    }

    #[test]
    fn uneven_cell_counts_produce_exact_partition_edges() {
        let directory = PartitionDirectory::from_uniform_config(
            &UniformMeshConfig {
                nx: 5,
                ny: 1,
                nz: 1,
                ng: 1,
                bounds_lo: [0.0; 3],
                bounds_hi: [1.0; 3],
                y_edges: None,
                z_edges: None,
            },
            [2, 1, 1],
        );
        assert_eq!(directory.extents()[0].bounds.hi[0], 0.6);
        assert_eq!(directory.extents()[1].bounds.lo[0], 0.6);
    }
}
