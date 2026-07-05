# field_amr

Pure-Rust AMR demonstration crate for the FIELD substrate.

## Role

`field_amr` provides a two-level block-adaptive Cartesian `ForestMesh`. It
implements `field_core::FvMesh + field_core::AdaptiveMesh` so finite-volume
physics can run on refined cells through the same mesh contract used by
`field_core::UniformMesh`.

## Contract

This crate proves the AMR substrate shape without bringing physics into FIELD.
It describes refinement geometry, leaf volumes, faces, and coarse/fine
interfaces. It does not own flux formulas, refinement criteria tied to a
particular equation set, or solver state.

## Example

```rust,ignore
use field_amr::{ForestConfig, ForestMesh};
use field_core::FvMesh;

let mesh = ForestMesh::from_config(&ForestConfig {
    ncx: 4,
    ncy: 4,
    ncz: 4,
    bounds_lo: [0.0, 0.0, 0.0],
    bounds_hi: [1.0, 1.0, 1.0],
    refine: vec![[1, 1, 1]],
});

assert!(mesh.n_cells_total() > mesh.n_local_cells());
```

## License

MIT OR Apache-2.0
