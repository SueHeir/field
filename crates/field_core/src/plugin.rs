//! GRASS plugins that wire the FIELD substrate into an [`App`].
//!
//! These install the comm backend, the mesh, and the field registry, and keep
//! field storage sized to the mesh. They contain no physics; a CFD crate adds its
//! own plugins (EOS, flux, integrator, …) on top, exactly as DIRT layers granular
//! plugins onto SOIL's `CorePlugins`.
//!
//! The easiest entry point is [`FieldDefaultPlugins`], which bundles the three in
//! the right order:
//!
//! ```rust,ignore
//! App::new()
//!     .add_plugins(FieldDefaultPlugins { mesh: my_config })
//!     .add_plugins(MyCfdPlugins)   // registers ConsVar, fluxes, BCs, …
//!     .start();
//! ```

use crate::uniform::{factor_decomposition, rank_position};
use crate::{FieldRegistry, FvMesh, UniformMesh, UniformMeshConfig};
use grass_app::prelude::*;
use grass_mpi::CommResource;
use grass_scheduler::{Res, ResMut};

/// Installs the [`grass_mpi::CommResource`] — the serial no-op backend by default,
/// or the real MPI backend under the `mpi_backend` feature. Provides the `"comm"`
/// capability the mesh plugin requires.
pub struct CommPlugin;

impl Plugin for CommPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(not(feature = "mpi_backend"))]
        app.add_resource(CommResource(Box::new(grass_mpi::SingleProcessComm::new())));
        #[cfg(feature = "mpi_backend")]
        app.add_resource(CommResource(Box::new(grass_mpi::MpiCommBackend::new(
            grass_mpi::get_mpi_world(),
        ))));
    }

    fn provides(&self) -> Vec<&str> {
        vec!["comm"]
    }
}

/// Installs the [`FieldRegistry`] resource so physics can `register_field_data!`.
pub struct FieldRegistryPlugin;

impl Plugin for FieldRegistryPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(FieldRegistry::new());
    }

    fn provides(&self) -> Vec<&str> {
        vec!["field_registry"]
    }
}

/// Installs a [`UniformMesh`] resource. The mesh is built single-process at
/// `build` time and then **decomposed in place** by a `PreSetup` system once the
/// comm backend's rank count is known — so the same plugin works serial or under
/// MPI with no change at the call site.
pub struct UniformMeshPlugin {
    /// Grid configuration used to build the mesh.
    pub config: UniformMeshConfig,
}

impl UniformMeshPlugin {
    /// Creates the plugin from a mesh configuration.
    pub fn new(config: UniformMeshConfig) -> Self {
        Self { config }
    }
}

impl Plugin for UniformMeshPlugin {
    fn build(&self, app: &mut App) {
        // Config is stashed as a resource so the setup systems can read it; the
        // mesh starts single-process and is replaced once the rank grid is known.
        app.add_resource(self.config.clone());
        app.add_resource(UniformMesh::from_config(&self.config));
        app.add_setup_system(decompose_mesh, ScheduleSetupSet::PreSetup);
        // `resize_fields` is mesh-type-generic (it only needs `n_cells_total`), so
        // a future `ForestMeshPlugin` reuses it as `resize_fields::<ForestMesh>`.
        app.add_setup_system(resize_fields::<UniformMesh>, ScheduleSetupSet::Setup);
    }

    fn provides(&self) -> Vec<&str> {
        vec!["mesh", "structured_mesh"]
    }

    fn requires(&self) -> Vec<&str> {
        vec!["field_registry", "comm"]
    }
}

/// `PreSetup`: factor the rank count into a process grid, record it on the comm
/// backend, and rebuild the local mesh partition. A serial run (`size == 1`)
/// leaves the single-process mesh untouched.
fn decompose_mesh(
    cfg: Res<UniformMeshConfig>,
    mut comm: ResMut<CommResource>,
    mut mesh: ResMut<UniformMesh>,
) {
    let size = comm.size();
    if size <= 1 {
        return;
    }
    let rank = comm.rank();
    let decomp = factor_decomposition(size, [cfg.nx, cfg.ny, cfg.nz]);
    let pos = rank_position(rank, decomp);
    comm.set_processor_grid(decomp, pos);
    *mesh = UniformMesh::from_config_decomposed(&cfg, decomp, pos);
}

/// `Setup`: size every registered field to the (now-final) mesh cell count, once.
/// Generic over the mesh type so any `FvMesh` plugin can reuse it (e.g. a forest
/// solver bundle registers `resize_fields::<ForestMesh>`).
pub fn resize_fields<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>) {
    reg.resize_all(mesh.n_cells_total());
}

/// The standard FIELD substrate bundle: comm backend, field registry, and a
/// structured mesh built from `mesh`. Mirror of DIRT's `CorePlugins`.
pub struct FieldDefaultPlugins {
    /// Grid configuration for the structured mesh in the bundle.
    pub mesh: UniformMeshConfig,
}

impl PluginGroup for FieldDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(CommPlugin)
            .add(FieldRegistryPlugin)
            .add(UniformMeshPlugin::new(self.mesh))
    }
}
