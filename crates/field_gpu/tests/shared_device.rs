//! A kernel that reads SOIL particle columns and a FIELD mesh column at once.
//!
//! This is the point of one shared device: a particle-mesh transfer binds both libraries'
//! buffers in one kernel. Each particle gathers the value of the cell it sits in.

use field_core::{FieldDefaultPlugins, FieldRegistry, FvMesh, UniformMesh, UniformMeshConfig};
use field_gpu::{GpuColumn, GpuColumnPlugin};
use grass_app::prelude::*;
use grass_gpu::cubecl::{self, prelude::*};
use grass_gpu::{DualBuffer, GpuDevice, GpuReal};
use soil_gpu::{GpuState, Grid};

struct Temperature;

/// A 4 x 4 x 4 unit cube with one ghost layer, in the mesh's own raw ordering.
#[cube(launch_unchecked)]
fn gather<F: Float>(pos: &[F], temperature: &[F], gathered: &mut [F]) {
    let i = ABSOLUTE_POS;
    if i < gathered.len() {
        let four = F::new(4.0f32);
        let cx = u32::cast_from(F::floor(pos[3 * i] * four)) + 1u32;
        let cy = u32::cast_from(F::floor(pos[3 * i + 1] * four)) + 1u32;
        let cz = u32::cast_from(F::floor(pos[3 * i + 2] * four)) + 1u32;
        gathered[i] = temperature[(((cx * 6u32 + cy) * 6u32) + cz) as usize];
    }
}

#[test]
fn one_kernel_reads_particle_and_mesh_columns() {
    let Ok(gpu) = GpuDevice::default_device() else {
        eprintln!("skipping: no adapter");
        return;
    };
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins {
        mesh: UniformMeshConfig { nx: 4, ny: 4, nz: 4, ng: 1, ..Default::default() },
    });
    app.add_plugins(GpuColumnPlugin::<Temperature, UniformMesh>::default());
    app.organize_systems();
    app.setup();

    let registry = app.get_resource_ref::<FieldRegistry>().unwrap();
    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let field_device = app.get_resource_ref::<GpuDevice>().unwrap().clone();

    // Temperature is a function of position, so the right cell is checkable.
    let temperature_at = |p: [f64; 3]| (p[0] + 10.0 * p[1] + 100.0 * p[2]) as GpuReal;
    let mut column = registry.expect_mut::<GpuColumn<Temperature>>("test");
    for c in 0..mesh.n_cells_total() {
        column.data.host_mut()[c] = temperature_at(mesh.cell_centroid(c));
    }

    let n = 1000;
    let pos: Vec<GpuReal> = (0..n)
        .flat_map(|i| {
            let f = i as GpuReal;
            [(f * 0.13).fract(), (f * 0.37).fract(), (f * 0.71).fract()]
        })
        .collect();
    let grid = Grid::from_positions(&pos, 0.25);
    // SOIL's columns, created on the device FIELD installed.
    let mut particles = GpuState::new(field_device.clone(), n, grid);
    particles.set_params(1e-3, [0.0; 3]);
    particles.set_state(&pos, &vec![0.0; 3 * n], &vec![1.0; n]);
    particles.run_steps(0);

    let mut gathered = DualBuffer::filled(&field_device, n, 0.0 as GpuReal);
    let pos_arg = particles.pos_arg();
    // SAFETY: particle cells are inside the mesh, whose column is `temperature`.
    unsafe {
        gather::launch_unchecked::<GpuReal>(
            &gpu.client,
            CubeCount::Static((n as u32).div_ceil(64), 1, 1),
            CubeDim::new_1d(64),
            pos_arg,
            column.data.gpu_read(),
            gathered.gpu_overwrite(),
        );
    }

    let got = gathered.host().to_vec();
    for (i, p) in pos.as_chunks::<3>().0.iter().enumerate() {
        let center = |x: GpuReal| (((x * 4.0).floor() + 0.5) / 4.0) as f64;
        let expected = temperature_at([center(p[0]), center(p[1]), center(p[2])]);
        assert!((got[i] - expected).abs() < 1e-3, "particle {i} at {p:?}: {} vs {expected}", got[i]);
    }
}
