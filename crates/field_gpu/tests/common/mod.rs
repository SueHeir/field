//! One explicit heat-diffusion step on a mesh, done on the CPU and on the GPU, side by
//! side in one schedule, so the two can be compared cell by cell.

use field_core::{
    halo_exchange_forward_all, FieldDefaultPlugins, FieldRegistry, FvMesh, MeshPeriodicity,
    MeshScheduleSet, UniformMesh, UniformMeshConfig,
};
use field_gpu::{axis_neighbor_table, GpuColumn, GpuColumnPlugin};
use grass_app::prelude::*;
use grass_gpu::cubecl::{self, prelude::*};
use grass_gpu::{DualBuffer, GpuDevice, GpuReal};
use grass_mpi::CommResource;
use grass_scheduler::prelude::*;

pub struct OnGpu;
pub struct OnCpu;

const R: GpuReal = 0.1;

#[cube(launch_unchecked)]
fn diffuse<F: Float>(t: &[F], neighbors: &[u32], r: &[F], next: &mut [F]) {
    let c = ABSOLUTE_POS;
    if c < t.len() {
        if neighbors[6 * c] == 0xFFFFFFFFu32 {
            next[c] = t[c];
        } else {
            let mut sum = F::new(0.0f32);
            let mut side = 0;
            while side < 6 {
                sum += t[neighbors[6 * c + side] as usize] - t[c];
                side += 1;
            }
            next[c] = t[c] + r[0] * sum;
        }
    }
}

#[cube(launch_unchecked)]
fn commit<F: Float>(next: &[F], t: &mut [F]) {
    if ABSOLUTE_POS < t.len() {
        t[ABSOLUTE_POS] = next[ABSOLUTE_POS];
    }
}

/// Kernel state built during setup, once the mesh is final.
#[derive(Default)]
pub struct Heat {
    gpu: Option<HeatGpu>,
    table: Vec<u32>,
    next_cpu: Vec<GpuReal>,
}

struct HeatGpu {
    next: DualBuffer<GpuReal>,
    neighbors: DualBuffer<u32>,
    r: DualBuffer<GpuReal>,
}

fn initial(mesh: &UniformMesh, c: usize) -> GpuReal {
    let [x, y, z] = mesh.cell_centroid(c);
    let tau = std::f64::consts::TAU;
    ((tau * x).sin() * (tau * y).cos() * (tau * z).sin() + 2.0) as GpuReal
}

fn init_system(
    mesh: Res<UniformMesh>,
    registry: Res<FieldRegistry>,
    gpu: Res<GpuDevice>,
    mut heat: ResMut<Heat>,
) {
    let mut on_gpu = registry.expect_mut::<GpuColumn<OnGpu>>("init");
    let mut on_cpu = registry.expect_mut::<GpuColumn<OnCpu>>("init");
    for c in 0..mesh.n_cells_total() {
        let value = if mesh.is_local_cell(c) { initial(&mesh, c) } else { 0.0 };
        on_gpu.data.host_mut()[c] = value;
        on_cpu.data.host_mut()[c] = value;
    }
    heat.table = axis_neighbor_table(&*mesh);
    heat.next_cpu = vec![0.0; mesh.n_cells_total()];
    heat.gpu = Some(HeatGpu {
        next: DualBuffer::filled(&gpu, mesh.n_cells_total(), 0.0),
        neighbors: DualBuffer::new(&gpu, heat.table.clone()),
        r: DualBuffer::new(&gpu, vec![R]),
    });
}

fn halo(mesh: Res<UniformMesh>, registry: Res<FieldRegistry>, comm: Res<CommResource>) {
    halo_exchange_forward_all(mesh.halo_plan(), &registry, comm.0.as_ref());
}

fn gpu_step(registry: Res<FieldRegistry>, gpu: Res<GpuDevice>, mut heat: ResMut<Heat>) {
    let mut column = registry.expect_mut::<GpuColumn<OnGpu>>("gpu_step");
    let groups = CubeCount::Static((column.data.len() as u32).div_ceil(64), 1, 1);
    let dim = CubeDim::new_1d(64);
    let state = heat.gpu.as_mut().expect("init ran");
    // SAFETY: indices are cells of the mesh these columns are sized to.
    unsafe {
        diffuse::launch_unchecked::<GpuReal>(
            &gpu.client,
            groups.clone(),
            dim,
            column.data.gpu_write(),
            state.neighbors.gpu_read(),
            state.r.gpu_read(),
            state.next.gpu_overwrite(),
        );
        commit::launch_unchecked::<GpuReal>(&gpu.client, groups, dim, state.next.arg(), column.data.arg());
    }
}

fn cpu_step(registry: Res<FieldRegistry>, mut heat: ResMut<Heat>) {
    let mut column = registry.expect_mut::<GpuColumn<OnCpu>>("cpu_step");
    let Heat { table, next_cpu, .. } = &mut *heat;
    let t = column.data.host_mut();
    for c in 0..t.len() {
        next_cpu[c] = if table[6 * c] == u32::MAX {
            t[c]
        } else {
            let sum: GpuReal = (0..6).map(|side| t[table[6 * c + side] as usize] - t[c]).sum();
            t[c] + R * sum
        };
    }
    t.copy_from_slice(next_cpu);
}

/// An App with both columns, set up and ready. `app.run()` advances one step.
pub fn heat_app(n: usize, periodic: bool) -> App {
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins {
        mesh: UniformMeshConfig { nx: n, ny: n, nz: n, ng: 1, ..Default::default() },
    });
    app.add_resource(MeshPeriodicity([periodic; 3]));
    app.add_plugins(GpuColumnPlugin::<OnGpu, UniformMesh>::default());
    app.add_plugins(GpuColumnPlugin::<OnCpu, UniformMesh>::default());
    app.add_resource(Heat::default());
    app.add_setup_system(init_system, ScheduleSetupSet::PostSetup);
    app.add_update_system(halo, MeshScheduleSet::Halo);
    app.add_update_system(gpu_step, MeshScheduleSet::Flux);
    app.add_update_system(cpu_step, MeshScheduleSet::Flux);
    app.organize_systems();
    app.setup();
    app
}

/// Largest difference between the two columns over local cells, and the total heat.
pub fn compare(app: &App) -> (GpuReal, f64) {
    let registry = app.get_resource_ref::<FieldRegistry>().unwrap();
    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let mut on_gpu = registry.expect_mut::<GpuColumn<OnGpu>>("compare");
    let mut on_cpu = registry.expect_mut::<GpuColumn<OnCpu>>("compare");
    let on_gpu = on_gpu.data.host();
    let on_cpu = on_cpu.data.host();
    let local = || (0..mesh.n_cells_total()).filter(|&c| mesh.is_local_cell(c));
    let max_diff = local().map(|c| (on_gpu[c] - on_cpu[c]).abs()).fold(0.0, GpuReal::max);
    let total = local().map(|c| on_gpu[c] as f64).sum();
    (max_diff, total)
}
