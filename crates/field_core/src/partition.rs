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

impl PartitionExtent {
    /// Bytes to send to another code: the rank and the six bounds,
    /// little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(52);
        b.extend_from_slice(&self.role_rank.to_le_bytes());
        for x in self.bounds.lo.iter().chain(&self.bounds.hi) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }

    /// Read what [`to_bytes`](Self::to_bytes) wrote.
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() != 52 {
            return Err(format!("a partition extent is 52 bytes, got {}", b.len()));
        }
        let f = |i: usize| f64::from_le_bytes(b[4 + 8 * i..12 + 8 * i].try_into().unwrap());
        let (lo, hi) = ([f(0), f(1), f(2)], [f(3), f(4), f(5)]);
        if !(0..3).all(|a| lo[a] <= hi[a]) {
            return Err(format!(
                "partition extent bounds out of order: {lo:?}..{hi:?}"
            ));
        }
        Ok(Self {
            role_rank: i32::from_le_bytes(b[..4].try_into().unwrap()),
            bounds: Bounds3::new(lo, hi),
        })
    }
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

    /// A directory from extents gathered from the owning code (for example
    /// each rank's [`crate::UniformMesh::owned_extent`], sent across a
    /// coupling). The global bounds are their union.
    pub fn from_extents(role_size: i32, extents: Vec<PartitionExtent>) -> Self {
        assert!(!extents.is_empty(), "a partition needs at least one extent");
        let mut lo = extents[0].bounds.lo;
        let mut hi = extents[0].bounds.hi;
        for e in &extents {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(e.bounds.lo[axis]);
                hi[axis] = hi[axis].max(e.bounds.hi[axis]);
            }
        }
        let mut extents = extents;
        extents.sort_by_key(|e| e.role_rank);
        Self::new(role_size, Bounds3::new(lo, hi), extents)
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

/// The coupling contract: couplers address records through
/// [`grass_mpi::OwnerMap`] and never need to name this type.
impl grass_mpi::OwnerMap for PartitionDirectory {
    fn owner_rank(&self, point: [f64; 3]) -> Option<i32> {
        PartitionDirectory::owner_rank(self, point)
    }

    /// A fingerprint of the extents: equal on every rank for the same
    /// partition, different after FIELD re-partitions.
    fn version(&self) -> u64 {
        // FNV-1a over the bit patterns, so it is identical on every rank.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: [u8; 8]| {
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        eat((self.role_size as u64).to_le_bytes());
        let boxes = std::iter::once((-1, self.global))
            .chain(self.extents.iter().map(|e| (e.role_rank, e.bounds)));
        for (rank, b) in boxes {
            eat((rank as i64).to_le_bytes());
            for v in b.lo.iter().chain(&b.hi) {
                eat(v.to_bits().to_le_bytes());
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each rank's real mesh, sent as bytes and reassembled by the other
    /// code, gives the same directory as rebuilding it from the config.
    #[test]
    fn extents_sent_by_each_rank_rebuild_the_directory() {
        let cfg = UniformMeshConfig {
            nx: 7,
            ny: 5,
            nz: 4,
            ng: 2,
            bounds_lo: [-1.0, 0.0, 2.0],
            bounds_hi: [2.5, 1.0, 3.0],
            y_edges: Some(vec![0.0, 0.1, 0.3, 0.6, 0.8, 1.0]),
            z_edges: None,
        };
        let decomp = [2, 2, 1];
        let mut received = Vec::new();
        for px in 0..2 {
            for py in 0..2 {
                let mesh = crate::UniformMesh::from_config_decomposed(&cfg, decomp, [px, py, 0]);
                let rank = px * 2 + py;
                let bytes = mesh.owned_extent(rank).to_bytes();
                received.push(PartitionExtent::from_bytes(&bytes).unwrap());
            }
        }
        received.reverse(); // arrival order does not matter
        let sent = PartitionDirectory::from_extents(4, received);
        let rebuilt = PartitionDirectory::from_uniform_config(&cfg, decomp);
        assert_eq!(sent.extents().len(), 4);
        for (a, b) in sent.extents().iter().zip(rebuilt.extents()) {
            assert_eq!(a.role_rank, b.role_rank);
            for axis in 0..3 {
                assert!(
                    (a.bounds.lo[axis] - b.bounds.lo[axis]).abs() < 1e-12,
                    "{a:?} vs {b:?}"
                );
                assert!(
                    (a.bounds.hi[axis] - b.bounds.hi[axis]).abs() < 1e-12,
                    "{a:?} vs {b:?}"
                );
            }
        }
        for i in 0..50 {
            let t = i as f64 / 49.0;
            let p = [-1.0 + 3.5 * t, (t * 7.3).fract(), 2.0 + (t * 3.1).fract()];
            assert_eq!(sent.owner_rank(p), rebuilt.owner_rank(p), "{p:?}");
        }
        assert!(PartitionExtent::from_bytes(&[0u8; 10]).is_err());
    }

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

    #[test]
    fn usable_through_the_grass_owner_map_contract() {
        use grass_mpi::OwnerMap;
        let config = UniformMeshConfig {
            nx: 4,
            ny: 2,
            nz: 1,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0; 3],
            ..Default::default()
        };
        let a = PartitionDirectory::from_uniform_config(&config, [2, 1, 1]);
        let b = PartitionDirectory::from_uniform_config(&config, [1, 2, 1]);
        let map: &dyn OwnerMap = &a;
        for p in [
            [0.1, 0.1, 0.5],
            [0.9, 0.1, 0.5],
            [0.6, 0.9, 0.5],
            [2.0, 0.5, 0.5],
        ] {
            assert_eq!(map.owner_rank(p), a.owner_rank(p));
        }
        assert_eq!(map.owner_rank([0.9, 0.1, 0.5]), Some(1));
        assert_eq!(map.version(), a.clone().version());
        assert_ne!(OwnerMap::version(&a), OwnerMap::version(&b));
    }
}
