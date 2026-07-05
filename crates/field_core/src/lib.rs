//! # `field_core` — the FIELD substrate
//!
//! FIELD is to mesh/Eulerian methods what SOIL is to particle/Lagrangian methods:
//! a **physics-agnostic substrate** riding the GRASS framework. SOIL owns atoms,
//! domain decomposition, neighbor lists, and halo comm, and knows nothing about
//! contact forces; FIELD owns meshes, per-cell fields, and halo comm, and knows
//! nothing about fluxes, equations of state, or boundary conditions.
//!
//! ```text
//! GRASS    framework: App + Plugin + Scheduler, IO, MPI coupling
//!   └─ FIELD   substrate: Mesh, FieldData, halo exchange      (this crate)
//!        └─ test-cfd   physics: fluxes, EOS, BCs, integrators
//! ```
//!
//! ## What FIELD is *not* a copy of
//!
//! FIELD was designed from the contracts, not transcribed from toy-cfd:
//!
//! - **The mesh is abstracted, not hardcoded.** [`FvMesh`] is a trait (the
//!   cell-centered finite-volume view), with [`StructuredMesh`] / [`AdaptiveMesh`]
//!   extensions, so structured grids, octree-AMR forests, and unstructured meshes
//!   all satisfy the same physics. toy-cfd hardcoded one `UniformGrid`. (Non-FV
//!   discretizations — FEM/DG — are out of scope; they would be a sibling trait on
//!   the same substrate, not a change to `FvMesh`.)
//! - **No physics leaks into the substrate.** toy-cfd's grid `use`d `ConsVar` and
//!   its ghost exchange hardcoded five `f64`s per cell. Here [`halo`] moves
//!   whatever [`FieldData`] the registry holds — the state vector is just another
//!   registered field.
//! - **The field contract is smaller than `AtomData`.** Cells never migrate or
//!   reorder, so FIELD's [`FieldData`] drops `pack`/`unpack`/`truncate`/
//!   `swap_remove`/`apply_permutation` and keeps only resize + forward/reverse/zero.
//!
//! ## Layers in this crate
//!
//! | Module | Role | FV-specific? |
//! |--------|------|--------------|
//! | [`mesh`] | the [`FvMesh`] trait family — the finite-volume topology + geometry | **yes** |
//! | [`field_data`] | [`FieldData`] + [`FieldRegistry`] — per-cell extension state | no (method-agnostic) |
//! | [`component`] | [`Component`] — f64-serializable column element types | no (method-agnostic) |
//! | [`halo`] | static halo plan + generic forward/reverse exchange | no (method-agnostic) |
//! | [`uniform`] | [`UniformMesh`] — the structured Cartesian implementation | yes |
//! | [`schedule`] | [`MeshScheduleSet`] — explicit-step phase ordering | no |
//! | [`plugin`] | GRASS plugins that install the mesh and registry | — |
//!
//! The "method-agnostic" rows are the substrate a future non-FV topology (FEM/DG)
//! would reuse unchanged; only [`mesh`] and [`uniform`] assume finite volume.

pub mod component;
pub mod field_data;
pub mod halo;
pub mod hierarchy;
pub mod mesh;
pub mod operator;
pub mod plugin;
pub mod schedule;
pub mod uniform;

pub use component::Component;
pub use field_data::{FieldData, FieldRegistry, FieldRegistryError};
/// `#[derive(FieldData)]` — generates the trait impl from `#[forward]` /
/// `#[reverse]` / `#[zero]`-tagged `Vec<C: Component>` columns. (Same name as the
/// [`FieldData`] trait, different namespace — like `serde::Serialize`.)
pub use field_derive::FieldData;
pub use halo::{
    halo_exchange_forward, halo_exchange_forward_all, halo_exchange_reverse_all, HaloLink, HaloPlan,
};
pub use hierarchy::MeshHierarchy;
pub use mesh::{
    AdaptiveMesh, BoundarySide, CartesianMesh, CoarseFineFace, Face, FvMesh, PatchId,
    StructuredMesh, Vec3,
};
pub use operator::{CellVector, LinearOperator, Vector};
pub use plugin::{
    resize_fields, CommPlugin, FieldDefaultPlugins, FieldRegistryPlugin, UniformMeshPlugin,
};
pub use schedule::MeshScheduleSet;
pub use uniform::{UniformMesh, UniformMeshConfig};

/// Commonly used FIELD imports.
pub mod prelude {
    pub use crate::{
        register_field_data, try_register_field_data, BoundarySide, CommPlugin, Component, Face,
        FieldData, FieldDefaultPlugins, FieldRegistry, FieldRegistryError, FieldRegistryPlugin,
        FvMesh, MeshScheduleSet, StructuredMesh, UniformMesh, UniformMeshConfig, UniformMeshPlugin,
        Vec3,
    };
}
