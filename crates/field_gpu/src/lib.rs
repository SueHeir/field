//! GPU storage for FIELD, on the shared `grass_gpu` device.
//!
//! A [`GpuColumn`] is an ordinary [`FieldData`] store whose data can live on the GPU, so
//! it is sized, registered and halo-exchanged like any other store and physics kernels
//! bind it directly. The halo goes through CPU memory for now: [`GpuColumnPlugin`]
//! downloads the column before the exchange when the mesh has any links, and the next
//! kernel uploads it again. A mesh with no links, such as one serial non-periodic rank,
//! makes no transfers at all.
//!
//! Kernels use `GpuReal`: `f32` on the Mac, `f64` with the `cuda` feature. The halo wire
//! format stays `f64`, so values cross it unchanged.
//!
//! [`axis_neighbor_table`] gives kernels a mesh's stencil, so a flux kernel can read a
//! cell's six neighbors the same way the CPU path does.

#![warn(missing_docs)]

use std::any::Any;
use std::marker::PhantomData;

use field_core::{CartesianMesh, FieldData, FieldRegistry, FvMesh, MeshScheduleSet};
use grass_app::prelude::*;
use grass_gpu::{DualBuffer, GpuDevice, GpuPlugin, GpuReal};
use grass_scheduler::prelude::*;

/// One per-cell [`GpuReal`] column with a CPU and a GPU copy. `K` is any marker type naming
/// the column, since the [`FieldRegistry`] keeps one store per type.
pub struct GpuColumn<K: 'static> {
    /// The data. Say what each kernel does with [`DualBuffer::gpu_read`] or
    /// [`DualBuffer::gpu_write`], which return its launch argument.
    pub data: DualBuffer<GpuReal>,
    _kind: PhantomData<fn() -> K>,
}

impl<K: 'static> GpuColumn<K> {
    /// An empty column on `device`. The mesh plugin sizes it during setup.
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            data: DualBuffer::new(device, Vec::new()),
            _kind: PhantomData,
        }
    }

    fn host_for_halo(&self) -> &[GpuReal] {
        self.data.host_if_current().expect(
            "field_gpu: halo exchange read a column whose GPU copy is newer. Add \
             GpuColumnPlugin, which downloads it first.",
        )
    }
}

impl<K: 'static> FieldData for GpuColumn<K> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn resize(&mut self, n: usize) {
        self.data.resize(n, 0.0);
    }
    fn forward_size(&self) -> usize {
        1
    }
    fn pack_forward(&self, c: usize, buf: &mut Vec<f64>) {
        buf.push(self.host_for_halo()[c] as f64);
    }
    fn unpack_forward(&mut self, c: usize, buf: &[f64]) -> usize {
        self.data.host_mut()[c] = buf[0] as GpuReal;
        1
    }
    fn pack_forward_cells(&self, cells: &[usize], buf: &mut Vec<f64>) {
        let host = self.host_for_halo();
        buf.extend(cells.iter().map(|&c| host[c] as f64));
    }
    fn unpack_forward_cells(&mut self, cells: &[usize], buf: &[f64]) -> usize {
        let host = self.data.host_mut();
        for (&c, &value) in cells.iter().zip(buf) {
            host[c] = value as GpuReal;
        }
        cells.len()
    }
}

/// Downloads column `K` before the halo exchange, if the mesh has any links.
pub fn sync_column_for_halo<K: 'static, M: FvMesh>(mesh: Res<M>, registry: Res<FieldRegistry>) {
    if !mesh.halo_plan().links.is_empty() {
        registry
            .expect_mut::<GpuColumn<K>>("field_gpu::sync_column_for_halo")
            .data
            .sync_to_host();
    }
}

/// Registers a [`GpuColumn<K>`] on the shared device, and downloads it before each halo
/// exchange on mesh type `M`. Add it after the FIELD plugins.
pub struct GpuColumnPlugin<K, M> {
    _marker: PhantomData<fn() -> (K, M)>,
}

impl<K, M> Default for GpuColumnPlugin<K, M> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<K: 'static, M: FvMesh + 'static> Plugin for GpuColumnPlugin<K, M> {
    fn build(&self, app: &mut App) {
        app.add_plugins(GpuPlugin::default());
        let device = app
            .get_resource_ref::<GpuDevice>()
            .expect("GpuPlugin installs the device")
            .clone();
        field_core::register_field_data!(app, GpuColumn::<K>::new(&device));
        app.add_update_system(sync_column_for_halo::<K, M>, MeshScheduleSet::PreHalo);
    }

    fn requires(&self) -> Vec<&str> {
        vec!["field_registry"]
    }

    fn is_unique(&self) -> bool {
        false
    }
}

/// A mesh's stencil as a flat table for kernels: `table[6 * c + 2 * axis + hi]` is the
/// neighbor of cell `c` on that side, or `u32::MAX` for a cell that has none (ghosts).
pub fn axis_neighbor_table<M: CartesianMesh>(mesh: &M) -> Vec<u32> {
    let n = mesh.n_cells_total();
    let mut table = vec![u32::MAX; 6 * n];
    for c in (0..n).filter(|&c| mesh.is_local_cell(c)) {
        for axis in 0..3 {
            for hi in [false, true] {
                if let Some(&neighbor) = mesh.axis_neighbors(c, axis, hi).first() {
                    table[6 * c + 2 * axis + hi as usize] = neighbor as u32;
                }
            }
        }
    }
    table
}
