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
use crate::{FieldRegistry, FvMesh, MeshPeriodicity, UniformMesh, UniformMeshConfig};
use grass_app::prelude::*;
use grass_mpi::CommResource;
use grass_scheduler::{Res, ResMut};

/// Makes sure the App has GRASS's shared [`grass_mpi::CommResource`] and
/// provides the `"comm"` capability the mesh plugin requires. It adds
/// [`grass_mpi::MpiPlugin`], which reuses a communicator another library (SOIL,
/// a coupler) or the application already installed, and otherwise creates one:
/// the real MPI backend under the `mpi_backend` feature, else the serial one.
pub struct CommPlugin;

impl Plugin for CommPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(grass_mpi::MpiPlugin);
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
        // The mesh decomposes over the App's shared communicator, whichever
        // library installed it; add GRASS's if nobody has yet.
        app.add_plugins(grass_mpi::MpiPlugin);
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
/// backend, and rebuild the local mesh partition, wrapping any axes named by an
/// optional [`MeshPeriodicity`] resource. A serial, non-periodic run leaves the
/// single-process mesh untouched.
fn decompose_mesh(
    cfg: Res<UniformMeshConfig>,
    periodic: Option<Res<MeshPeriodicity>>,
    mut comm: ResMut<CommResource>,
    mut mesh: ResMut<UniformMesh>,
) {
    let periodic = periodic.map(|p| p.0).unwrap_or([false; 3]);
    let size = comm.size();
    if size <= 1 {
        if periodic != [false; 3] {
            *mesh = UniformMesh::from_config_periodic(&cfg, [1, 1, 1], [0, 0, 0], periodic);
        }
        return;
    }
    let rank = comm.rank();
    let decomp = factor_decomposition(size, [cfg.nx, cfg.ny, cfg.nz])
        .unwrap_or_else(|e| panic!("UniformMesh decomposition failed: {e}"));
    let pos = rank_position(rank, decomp);
    comm.set_processor_grid(decomp, pos);
    *mesh = UniformMesh::from_config_periodic(&cfg, decomp, pos, periodic);
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
