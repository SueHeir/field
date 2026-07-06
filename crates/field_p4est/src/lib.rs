//! # `field_p4est` — p4est-backed AMR for the FIELD substrate
//!
//! Wraps the [p4est](https://www.p4est.org/) forest-of-octrees C library (via a
//! small C shim, FFI in [`forest`]) and exposes [`ForestMesh`], which implements
//! `field_core`'s `FvMesh + AdaptiveMesh`. Unlike the pure-Rust `field_amr`
//! demonstration mesh, this is real, dynamically-refinable, 2:1-balanced
//! arbitrary-depth AMR — the same machinery toy-cfd's `cfd_p4est_solver` uses.
//!
//! ```rust,ignore
//! use field_p4est::{ForestGrid, ForestLayout, ForestMesh};
//!
//! let mut grid = ForestGrid::new(ForestLayout { /* brick + box + min_level */ })?;
//! grid.refine(max_level, |cx, cy, cz, h| near_body(cx, cy, cz)); // build mesh
//! let mesh = ForestMesh::new(grid); // implements FvMesh + AdaptiveMesh
//! ```
//!
//! Build: links the prebuilt `libp4est.a`/`libsc.a` located via `P4EST_PREFIX`
//! (see `build.rs`). Multi-rank forests expose p4est's ghost layer as FIELD
//! ghost cells and a `HaloPlan`.

mod forest;
mod mesh;

pub use forest::{
    finalize, init, Error, Face, FinerSet, ForestGrid, ForestLayout, LeafInfo, NeighborSet,
};
pub use mesh::{locate_in, transfer_recursive, ForestMesh};
