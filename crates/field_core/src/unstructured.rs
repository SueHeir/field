//! [`UnstructuredMesh`] — a partitioned, face-based polyhedral mesh behind
//! [`FvMesh`].
//!
//! This is the "unstructured meshes would implement `FvMesh` alone" claim in
//! [`crate::mesh`], made concrete. It is deliberately the OpenFOAM data model:
//! cells carry a centroid and a volume, faces carry an owner, an optional
//! neighbour, an area-weighted normal and an optional boundary patch. Nothing
//! here knows about `(i, j, k)`, ghost *layers*, or the six box sides.
//!
//! # Global description → per-rank mesh
//!
//! A [`PolyMesh`] is the whole mesh, identical on every rank. A
//! [`UnstructuredMesh`] is one rank's slice of it, built from an ownership array
//! (`cell → rank`, e.g. from [`PolyMesh::rcb_partition`]). The local storage is
//!
//! ```text
//! [ owned cells (by global id) | halo cells (by owner rank, global id) | boundary ghosts (one per boundary face) ]
//! ```
//!
//! Owned cells *are* a contiguous prefix here, unlike [`crate::UniformMesh`];
//! callers still use [`FvMesh::is_local_cell`], so the difference is invisible.
//!
//! # Halo
//!
//! One face-neighbour layer. For neighbour rank `q`, this rank receives the
//! halo cells `q` owns and sends the owned cells that some `q`-owned cell
//! touches, both sorted by global id, so the two ends of a link enumerate the
//! same cells in the same order (the [`HaloPlan`] producer contract).
//!
//! # Faces across a rank boundary
//!
//! A face between an owned cell and a halo cell is stored on *both* ranks, each
//! oriented so `owner` is that rank's owned cell. The generic flux loop scatters
//! only into owned cells, so each rank updates its own side exactly once.
//!
//! # Limits of this prototype
//!
//! - Every rank holds the whole [`PolyMesh`]; a production version would read
//!   and partition in parallel (ParMETIS/PT-Scotch style).
//! - No periodic faces and no point location (`locate`) yet. Particle–mesh
//!   transfer on an unstructured mesh needs the latter.
//! - Cell geometry is precomputed (centroid + volume), not derived from
//!   vertices.

use crate::field_data::FieldRegistry;
use crate::halo::{halo_exchange_forward_all, HaloLink, HaloPlan};
use crate::mesh::{BoundarySide, Face, FvMesh, PatchId, Vec3};
use crate::uniform::UniformMeshConfig;
use grass_mpi::CommBackend;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One control volume of a [`PolyMesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyCell {
    /// Cell centroid.
    pub centroid: Vec3,
    /// Cell volume (positive).
    pub volume: f64,
}

/// One face of a [`PolyMesh`]. Exactly one of `neighbour` and `patch` is `Some`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyFace {
    /// Global index of the cell the normal points *out of*.
    pub owner: usize,
    /// Global index of the cell on the other side, for an interior face.
    pub neighbour: Option<usize>,
    /// Area-weighted normal, pointing `owner → neighbour` (outward on a boundary).
    pub area_normal: Vec3,
    /// Face centroid.
    pub centroid: Vec3,
    /// Boundary patch, for a boundary face.
    pub patch: Option<PatchId>,
}

/// A whole unstructured mesh: the input every rank partitions from.
#[derive(Debug, Clone, Default)]
pub struct PolyMesh {
    /// All cells, indexed by global id.
    pub cells: Vec<PolyCell>,
    /// All faces, each listed once.
    pub faces: Vec<PolyFace>,
    /// Patch names, indexed by [`PatchId`].
    pub patch_names: Vec<String>,
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `(triangle, from vertex, to vertex)`: one triangle's counter-clockwise use of an edge.
type EdgeUse = (usize, usize, usize);

/// Deterministic pseudo-random number in `[-1, 1)` from an integer key.
fn hash_unit(key: u64) -> f64 {
    let mut x = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

impl PolyMesh {
    /// Check the mesh is well formed: indices in range, positive volumes, one
    /// of neighbour/patch per face, normals pointing away from the owner, and
    /// every cell closed (its outward area normals sum to zero).
    pub fn validate(&self) -> Result<(), String> {
        let n = self.cells.len();
        let mut closure = vec![[0.0f64; 3]; n];
        let mut scale = vec![0.0f64; n];
        for (i, c) in self.cells.iter().enumerate() {
            if c.volume <= 0.0 || !c.volume.is_finite() {
                return Err(format!("cell {i} has non-positive volume {}", c.volume));
            }
        }
        for (k, f) in self.faces.iter().enumerate() {
            if f.owner >= n {
                return Err(format!("face {k}: owner {} out of range", f.owner));
            }
            match (f.neighbour, f.patch) {
                (Some(nb), None) => {
                    if nb >= n || nb == f.owner {
                        return Err(format!("face {k}: bad neighbour {nb}"));
                    }
                }
                (None, Some(p)) => {
                    if p as usize >= self.patch_names.len() {
                        return Err(format!("face {k}: patch {p} has no name"));
                    }
                }
                _ => return Err(format!("face {k}: needs exactly one of neighbour/patch")),
            }
            let out = sub(f.centroid, self.cells[f.owner].centroid);
            if dot(out, f.area_normal) <= 0.0 {
                return Err(format!("face {k}: normal does not point away from owner"));
            }
            let a = dot(f.area_normal, f.area_normal).sqrt();
            for (acc, a) in closure[f.owner].iter_mut().zip(f.area_normal) {
                *acc += a;
            }
            scale[f.owner] += a;
            if let Some(nb) = f.neighbour {
                for (acc, a) in closure[nb].iter_mut().zip(f.area_normal) {
                    *acc -= a;
                }
                scale[nb] += a;
            }
        }
        for i in 0..n {
            let r = dot(closure[i], closure[i]).sqrt();
            if r > 1e-10 * scale[i].max(1.0) {
                return Err(format!("cell {i} is not closed (|Σ A n| = {r:e})"));
            }
        }
        Ok(())
    }

    /// The hexahedral cells of a uniform box (`y_edges`/`z_edges` must be
    /// `None`), as a face list. Patches are the six [`BoundarySide`]s, so the
    /// same physics sees the same patch ids as on a [`crate::UniformMesh`].
    pub fn hex_box(cfg: &UniformMeshConfig) -> Self {
        assert!(
            cfg.y_edges.is_none() && cfg.z_edges.is_none(),
            "hex_box supports uniform spacing only"
        );
        let n = [cfg.nx, cfg.ny, cfg.nz];
        let h: Vec3 = std::array::from_fn(|d| (cfg.bounds_hi[d] - cfg.bounds_lo[d]) / n[d] as f64);
        let id = |i: usize, j: usize, k: usize| (i * n[1] + j) * n[2] + k;
        let mut cells = vec![
            PolyCell {
                centroid: [0.0; 3],
                volume: h[0] * h[1] * h[2]
            };
            n[0] * n[1] * n[2]
        ];
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let g = [i, j, k];
                    cells[id(i, j, k)].centroid =
                        std::array::from_fn(|d| cfg.bounds_lo[d] + (g[d] as f64 + 0.5) * h[d]);
                }
            }
        }
        let mut faces = Vec::new();
        for i in 0..n[0] {
            for j in 0..n[1] {
                for k in 0..n[2] {
                    let g = [i, j, k];
                    let c = id(i, j, k);
                    for axis in 0..3 {
                        let area = h[(axis + 1) % 3] * h[(axis + 2) % 3];
                        let mut an = [0.0; 3];
                        an[axis] = area;
                        let mut fc = cells[c].centroid;
                        fc[axis] += 0.5 * h[axis];
                        // High-side face: interior if there is a neighbour.
                        if g[axis] + 1 < n[axis] {
                            let mut gn = g;
                            gn[axis] += 1;
                            faces.push(PolyFace {
                                owner: c,
                                neighbour: Some(id(gn[0], gn[1], gn[2])),
                                area_normal: an,
                                centroid: fc,
                                patch: None,
                            });
                        } else {
                            faces.push(PolyFace {
                                owner: c,
                                neighbour: None,
                                area_normal: an,
                                centroid: fc,
                                patch: Some(2 * axis as u32 + 1),
                            });
                        }
                        if g[axis] == 0 {
                            let mut fc = cells[c].centroid;
                            fc[axis] -= 0.5 * h[axis];
                            let mut an = [0.0; 3];
                            an[axis] = -area;
                            faces.push(PolyFace {
                                owner: c,
                                neighbour: None,
                                area_normal: an,
                                centroid: fc,
                                patch: Some(2 * axis as u32),
                            });
                        }
                    }
                }
            }
        }
        let patch_names = (0..6)
            .map(|p| format!("{:?}", BoundarySide::from_index(p).expect("valid side")))
            .collect();
        Self {
            cells,
            faces,
            patch_names,
        }
    }

    /// A genuinely unstructured mesh: an `nx × ny` grid of points over
    /// `[lo, hi]` in x–y, interior points moved by up to `jitter` of a spacing,
    /// each quad split into two triangles along a pseudo-randomly chosen
    /// diagonal, then extruded one layer of thickness `dz` in z. Cells are
    /// triangular prisms; patches are the six [`BoundarySide`]s.
    pub fn triangle_prisms(
        nx: usize,
        ny: usize,
        lo: [f64; 2],
        hi: [f64; 2],
        dz: f64,
        jitter: f64,
    ) -> Self {
        assert!(nx >= 1 && ny >= 1 && jitter < 0.5);
        let hx = (hi[0] - lo[0]) / nx as f64;
        let hy = (hi[1] - lo[1]) / ny as f64;
        let pid = |i: usize, j: usize| i * (ny + 1) + j;
        let mut pts = vec![[0.0f64; 2]; (nx + 1) * (ny + 1)];
        for i in 0..=nx {
            for j in 0..=ny {
                let mut p = [lo[0] + i as f64 * hx, lo[1] + j as f64 * hy];
                if i > 0 && i < nx && j > 0 && j < ny {
                    let key = pid(i, j) as u64;
                    p[0] += jitter * hx * hash_unit(2 * key);
                    p[1] += jitter * hy * hash_unit(2 * key + 1);
                }
                pts[pid(i, j)] = p;
            }
        }
        // Triangles as counter-clockwise vertex triples.
        let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * nx * ny);
        for i in 0..nx {
            for j in 0..ny {
                let (a, b, c, d) = (pid(i, j), pid(i + 1, j), pid(i + 1, j + 1), pid(i, j + 1));
                if hash_unit(1_000_003 + (i * ny + j) as u64) >= 0.0 {
                    tris.push([a, b, c]);
                    tris.push([a, c, d]);
                } else {
                    tris.push([a, b, d]);
                    tris.push([b, c, d]);
                }
            }
        }
        let zc = 0.5 * dz;
        let cells: Vec<PolyCell> = tris
            .iter()
            .map(|t| {
                let [p, q, r] = [pts[t[0]], pts[t[1]], pts[t[2]]];
                let area = 0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]));
                assert!(area > 0.0, "jitter folded a triangle");
                PolyCell {
                    centroid: [(p[0] + q[0] + r[0]) / 3.0, (p[1] + q[1] + r[1]) / 3.0, zc],
                    volume: area * dz,
                }
            })
            .collect();
        let mut faces = Vec::new();
        // Side faces: one per triangle edge, shared edges paired up.
        let mut edges: BTreeMap<(usize, usize), Vec<EdgeUse>> = BTreeMap::new();
        for (t, tri) in tris.iter().enumerate() {
            for e in 0..3 {
                let (u, v) = (tri[e], tri[(e + 1) % 3]);
                edges
                    .entry((u.min(v), u.max(v)))
                    .or_default()
                    .push((t, u, v));
            }
        }
        for (&(a, b), users) in &edges {
            // CCW edge u→v of triangle t: outward normal is (dy, -dx).
            let (t, u, v) = users[0];
            let (pu, pv) = (pts[u], pts[v]);
            let an = [(pv[1] - pu[1]) * dz, -(pv[0] - pu[0]) * dz, 0.0];
            let fc = [0.5 * (pu[0] + pv[0]), 0.5 * (pu[1] + pv[1]), zc];
            if users.len() == 2 {
                faces.push(PolyFace {
                    owner: t,
                    neighbour: Some(users[1].0),
                    area_normal: an,
                    centroid: fc,
                    patch: None,
                });
            } else {
                let (pa, pb) = (pts[a], pts[b]);
                let side = if pa[0] == lo[0] && pb[0] == lo[0] {
                    BoundarySide::XLo
                } else if pa[0] == hi[0] && pb[0] == hi[0] {
                    BoundarySide::XHi
                } else if pa[1] == lo[1] && pb[1] == lo[1] {
                    BoundarySide::YLo
                } else {
                    BoundarySide::YHi
                };
                faces.push(PolyFace {
                    owner: t,
                    neighbour: None,
                    area_normal: an,
                    centroid: fc,
                    patch: Some(side as u32),
                });
            }
        }
        // Bottom and top caps.
        for (t, c) in cells.iter().enumerate() {
            let a = c.volume / dz;
            for (z, s, side) in [(0.0, -1.0, BoundarySide::ZLo), (dz, 1.0, BoundarySide::ZHi)] {
                faces.push(PolyFace {
                    owner: t,
                    neighbour: None,
                    area_normal: [0.0, 0.0, s * a],
                    centroid: [c.centroid[0], c.centroid[1], z],
                    patch: Some(side as u32),
                });
            }
        }
        let patch_names = (0..6)
            .map(|p| format!("{:?}", BoundarySide::from_index(p).expect("valid side")))
            .collect();
        Self {
            cells,
            faces,
            patch_names,
        }
    }

    /// Recursive coordinate bisection of the cells into `nranks` parts of
    /// near-equal size (sizes differ by at most one). Deterministic.
    pub fn rcb_partition(&self, nranks: usize) -> Vec<i32> {
        assert!(nranks >= 1);
        let mut owner = vec![0i32; self.cells.len()];
        let ids: Vec<usize> = (0..self.cells.len()).collect();
        self.rcb(ids, 0, nranks, &mut owner);
        owner
    }

    fn rcb(&self, mut ids: Vec<usize>, first: usize, k: usize, owner: &mut [i32]) {
        if k == 1 {
            for i in ids {
                owner[i] = first as i32;
            }
            return;
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for &i in &ids {
            for d in 0..3 {
                lo[d] = lo[d].min(self.cells[i].centroid[d]);
                hi[d] = hi[d].max(self.cells[i].centroid[d]);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .expect("three axes");
        ids.sort_by(|&a, &b| {
            self.cells[a].centroid[axis]
                .total_cmp(&self.cells[b].centroid[axis])
                .then(a.cmp(&b))
        });
        let k1 = k / 2;
        let split = ids.len() * k1 / k;
        let right = ids.split_off(split);
        self.rcb(ids, first, k1, owner);
        self.rcb(right, first + k1, k - k1, owner);
    }
}

/// One rank's slice of a [`PolyMesh`]; implements [`FvMesh`].
pub struct UnstructuredMesh {
    n_owned: usize,
    /// Global ids of owned then halo cells (boundary ghosts have none).
    global_ids: Vec<usize>,
    volumes: Vec<f64>,
    centroids: Vec<Vec3>,
    faces: Vec<Face>,
    halo: HaloPlan,
    patch_names: Vec<String>,
}

impl UnstructuredMesh {
    /// The whole mesh on one rank (no halo).
    pub fn serial(global: &PolyMesh) -> Self {
        Self::partition(global, &vec![0; global.cells.len()], 0)
    }

    /// Rank `rank`'s slice of `global`, given each cell's owning rank.
    pub fn partition(global: &PolyMesh, owner: &[i32], rank: i32) -> Self {
        assert_eq!(owner.len(), global.cells.len(), "one owner per cell");
        let owned: Vec<usize> = (0..owner.len()).filter(|&g| owner[g] == rank).collect();
        let mut halo_set: BTreeSet<(i32, usize)> = BTreeSet::new();
        // For each neighbour rank: owned cells it can see (send list).
        let mut send: BTreeMap<i32, BTreeSet<usize>> = BTreeMap::new();
        for f in &global.faces {
            if let Some(nb) = f.neighbour {
                let (ro, rn) = (owner[f.owner], owner[nb]);
                if ro == rank && rn != rank {
                    halo_set.insert((rn, nb));
                    send.entry(rn).or_default().insert(f.owner);
                } else if rn == rank && ro != rank {
                    halo_set.insert((ro, f.owner));
                    send.entry(ro).or_default().insert(nb);
                }
            }
        }
        let mut global_ids = owned.clone();
        global_ids.extend(halo_set.iter().map(|&(_, g)| g));
        let local: HashMap<usize, usize> = global_ids
            .iter()
            .enumerate()
            .map(|(l, &g)| (g, l))
            .collect();
        let mut volumes: Vec<f64> = global_ids.iter().map(|&g| global.cells[g].volume).collect();
        let mut centroids: Vec<Vec3> = global_ids
            .iter()
            .map(|&g| global.cells[g].centroid)
            .collect();

        let mut faces = Vec::new();
        for f in &global.faces {
            let own_here = owner[f.owner] == rank;
            match f.neighbour {
                Some(nb) => {
                    let nb_here = owner[nb] == rank;
                    if own_here {
                        faces.push(Face {
                            owner: local[&f.owner],
                            other: local[&nb],
                            area_normal: f.area_normal,
                            centroid: f.centroid,
                            patch: None,
                        });
                    } else if nb_here {
                        // Flip so `owner` is the cell this rank owns.
                        let an = f.area_normal;
                        faces.push(Face {
                            owner: local[&nb],
                            other: local[&f.owner],
                            area_normal: [-an[0], -an[1], -an[2]],
                            centroid: f.centroid,
                            patch: None,
                        });
                    }
                }
                None if own_here => {
                    // One ghost per boundary face, mirrored through the face.
                    let o = local[&f.owner];
                    let g = volumes.len();
                    let c = global.cells[f.owner].centroid;
                    let n2 = dot(f.area_normal, f.area_normal);
                    let s = 2.0 * dot(sub(f.centroid, c), f.area_normal) / n2;
                    volumes.push(volumes[o]);
                    centroids.push(std::array::from_fn(|d| c[d] + s * f.area_normal[d]));
                    faces.push(Face {
                        owner: o,
                        other: g,
                        area_normal: f.area_normal,
                        centroid: f.centroid,
                        patch: f.patch,
                    });
                }
                None => {}
            }
        }

        let mut links = Vec::new();
        for (&q, cells) in &send {
            let recv_cells = halo_set
                .range((q, 0)..=(q, usize::MAX))
                .map(|&(_, g)| local[&g])
                .collect();
            links.push(HaloLink {
                rank: q,
                send_cells: cells.iter().map(|g| local[g]).collect(),
                recv_cells,
            });
        }
        Self {
            n_owned: owned.len(),
            global_ids,
            volumes,
            centroids,
            faces,
            halo: HaloPlan { links },
            patch_names: global.patch_names.clone(),
        }
    }

    /// Global id of local cell `c`, or `None` for a boundary ghost.
    pub fn global_id(&self, c: usize) -> Option<usize> {
        self.global_ids.get(c).copied()
    }

    /// Number of halo (neighbour-rank) cells.
    pub fn n_halo_cells(&self) -> usize {
        self.global_ids.len() - self.n_owned
    }

    /// Name of boundary patch `p`.
    pub fn patch_name(&self, p: PatchId) -> &str {
        &self.patch_names[p as usize]
    }
    /// Move this rank's cells to a new partition of `global`, carrying every
    /// registered field's **forward** columns with them, and return the new
    /// local mesh. Collective over `comm`.
    ///
    /// Afterwards the registry is sized to the new mesh, owned and halo cells
    /// hold their migrated values, and `#[zero]`/`#[reverse]` columns are
    /// zeroed: anything that must survive a repartition has to be a forward
    /// column. Boundary ghosts are left for the next `Boundary` phase. The
    /// caller replaces the mesh resource with the result; any system that
    /// cached mesh-shaped data (a `CellVector`, an owned-cell list) must
    /// rebuild it.
    pub fn repartition(
        &self,
        global: &PolyMesh,
        new_owner: &[i32],
        registry: &FieldRegistry,
        comm: &dyn CommBackend,
    ) -> UnstructuredMesh {
        let (rank, size) = (comm.rank(), comm.size());
        let next = UnstructuredMesh::partition(global, new_owner, rank);
        // Pack [count, gids.., forward data..] per destination, in old layout.
        let mut cells_to: Vec<Vec<usize>> = vec![Vec::new(); size as usize];
        for c in 0..self.n_owned {
            let g = self.global_ids[c];
            cells_to[new_owner[g] as usize].push(c);
        }
        let outgoing: Vec<Vec<f64>> = cells_to
            .iter()
            .map(|cells| {
                let mut buf = Vec::with_capacity(1 + cells.len() * (1 + registry.forward_size()));
                buf.push(cells.len() as f64);
                buf.extend(cells.iter().map(|&c| self.global_ids[c] as f64));
                registry.pack_forward_all_cells(cells, &mut buf);
                buf
            })
            .collect();
        // Pairwise exchange: in round k send to rank + k, receive from rank - k.
        let mut incoming = Vec::with_capacity(size as usize);
        incoming.push(outgoing[rank as usize].clone());
        for k in 1..size {
            let dest = (rank + k) % size;
            let src = (rank - k + size) % size;
            incoming.push(comm.sendrecv_f64(dest, &outgoing[dest as usize], src));
        }
        // Unpack into the new layout.
        let local_of: HashMap<usize, usize> =
            (0..next.n_owned).map(|c| (next.global_ids[c], c)).collect();
        registry.resize_all(next.n_cells_total());
        registry.zero_all(next.n_cells_total());
        let mut received = 0;
        for buf in &incoming {
            let n = buf[0] as usize;
            let cells: Vec<usize> = buf[1..1 + n]
                .iter()
                .map(|&g| local_of[&(g as usize)])
                .collect();
            let used = registry.unpack_forward_all_cells(&cells, &buf[1 + n..]);
            debug_assert_eq!(used, buf.len() - 1 - n, "repartition message size");
            received += n;
        }
        assert_eq!(received, next.n_owned, "every new owned cell arrives once");
        halo_exchange_forward_all(&next.halo, registry, comm);
        next
    }
}

impl FvMesh for UnstructuredMesh {
    fn n_local_cells(&self) -> usize {
        self.n_owned
    }
    fn n_cells_total(&self) -> usize {
        self.volumes.len()
    }
    fn is_local_cell(&self, c: usize) -> bool {
        c < self.n_owned
    }
    fn cell_volume(&self, c: usize) -> f64 {
        self.volumes[c]
    }
    fn cell_centroid(&self, c: usize) -> Vec3 {
        self.centroids[c]
    }
    fn for_each_face(&self, f: &mut dyn FnMut(&Face)) {
        for face in &self.faces {
            f(face);
        }
    }
    fn halo_plan(&self) -> &HaloPlan {
        &self.halo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_cfg() -> UniformMeshConfig {
        UniformMeshConfig {
            nx: 4,
            ny: 3,
            nz: 2,
            ng: 1,
            bounds_lo: [0.0; 3],
            bounds_hi: [2.0, 1.5, 1.0],
            y_edges: None,
            z_edges: None,
        }
    }

    #[test]
    fn generated_meshes_are_closed() {
        PolyMesh::hex_box(&box_cfg()).validate().unwrap();
        PolyMesh::triangle_prisms(9, 7, [0.0, 0.0], [1.0, 0.8], 0.1, 0.3)
            .validate()
            .unwrap();
    }

    #[test]
    fn rcb_balances_any_rank_count() {
        let m = PolyMesh::triangle_prisms(9, 7, [0.0, 0.0], [1.0, 1.0], 0.1, 0.3);
        for k in 1..=9 {
            let own = m.rcb_partition(k);
            let mut count = vec![0usize; k];
            for &r in &own {
                count[r as usize] += 1;
            }
            let (mn, mx) = (count.iter().min().unwrap(), count.iter().max().unwrap());
            assert!(mx - mn <= 1, "{k} ranks: {count:?}");
        }
    }

    #[test]
    fn partitions_cover_every_cell_and_halo_lists_match() {
        let m = PolyMesh::triangle_prisms(8, 6, [0.0, 0.0], [1.0, 1.0], 0.1, 0.3);
        let k = 5;
        let own = m.rcb_partition(k);
        let parts: Vec<_> = (0..k as i32)
            .map(|r| UnstructuredMesh::partition(&m, &own, r))
            .collect();
        assert_eq!(
            parts.iter().map(|p| p.n_local_cells()).sum::<usize>(),
            m.cells.len()
        );
        // Each link's send list on r equals the matching recv list on q, by global id.
        for (r, p) in parts.iter().enumerate() {
            for link in &p.halo_plan().links {
                let q = &parts[link.rank as usize];
                let back = q
                    .halo_plan()
                    .links
                    .iter()
                    .find(|l| l.rank == r as i32)
                    .expect("links are symmetric");
                let mine: Vec<_> = link.send_cells.iter().map(|&c| p.global_id(c)).collect();
                let theirs: Vec<_> = back.recv_cells.iter().map(|&c| q.global_id(c)).collect();
                assert_eq!(mine, theirs);
            }
        }
    }
}
