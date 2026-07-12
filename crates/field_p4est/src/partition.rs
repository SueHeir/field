//! Coupling-facing spatial ownership queries backed by p4est's global
//! partition markers.

use crate::ForestGrid;
use field_core::{Bounds3, Vec3};

/// Live view of a p4est forest's role-rank ownership.
///
/// Unlike [`field_core::PartitionDirectory`], this view does not materialize
/// one extent per adaptive leaf. p4est already replicates a compact
/// `global_first_position` directory on every rank; queries traverse those
/// markers locally and therefore remain cheap as the leaf count grows.
#[derive(Clone, Copy)]
pub struct ForestPartitionDirectory<'a> {
    grid: &'a ForestGrid,
}

impl<'a> ForestPartitionDirectory<'a> {
    /// Borrow the current partition directory of `grid`.
    pub fn new(grid: &'a ForestGrid) -> Self {
        Self { grid }
    }

    /// Number of ranks in the forest communicator.
    pub fn role_size(self) -> i32 {
        self.grid.mpisize() as i32
    }

    /// Unique owner of `point`, or `None` outside the physical domain.
    pub fn owner_rank(self, point: Vec3) -> Option<i32> {
        self.grid.owner_rank_at(point)
    }

    /// Sorted, deduplicated ranks with positive-volume overlap with `support`.
    /// A zero-volume support is treated as a point query.
    pub fn overlapping_ranks(self, support: Bounds3) -> Vec<i32> {
        self.grid.overlapping_ranks(support.lo, support.hi)
    }
}
