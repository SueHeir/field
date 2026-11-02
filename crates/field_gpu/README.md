# field_gpu

GPU storage for FIELD, on the device
[`grass_gpu`](https://github.com/SueHeir/grass/tree/main/crates/grass_gpu) shares.

| item | what it does |
|---|---|
| `GpuColumn<K>` | one per-cell `GpuReal` column (`f32` on the Mac, `f64` with `cuda`) with a CPU and GPU copy, and an ordinary `FieldData` store |
| `GpuColumnPlugin<K, M>` | registers the column, and downloads it before a halo exchange when the mesh has links |
| `axis_neighbor_table` | a mesh's stencil as a flat table a kernel can read |

A `GpuColumn` is sized, registered and halo-exchanged like any other store, and kernels bind
its buffer directly. The halo goes through CPU memory for now: one download and one upload
per exchange, and none at all on a mesh with no links.

## Checked

`tests/heat.rs` and `examples/gpu_heat.rs` run the same diffusion stencil on the CPU and the
GPU, side by side through real halo exchange. After 200 steps on a periodic 32³ mesh, the
largest difference is 1.7e-6 on values near 2, on 1, 2 and 4 ranks. Total heat changes by 3
parts in 10 million.

`tests/shared_device.rs` runs one kernel that reads SOIL's particle positions and a FIELD
column together, on the device both libraries share. That is the shape of a particle-mesh
transfer in MPM or PIC.

## License

MIT OR Apache-2.0
