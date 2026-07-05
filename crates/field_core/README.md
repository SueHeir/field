# field_core

Core FIELD substrate crate: mesh traits, per-cell field storage, halo exchange,
and GRASS plugin wiring for mesh/Eulerian simulations.

## Role

`field_core` is the substrate layer for FIELD. It owns the finite-volume mesh
trait family (`FvMesh`, `StructuredMesh`, `AdaptiveMesh`), `FieldData` and
`FieldRegistry`, component columns, static halo plans, `UniformMesh`, schedule
sets, and the default plugins that install the mesh and registry into a GRASS
app.

## Contract

This crate is physics-agnostic. It knows about cells, faces, volumes, registered
fields, and halo movement, but not fluxes, equations of state, boundary
conditions, or solver-specific state vectors. Physics crates define their own
`Component` and `FieldData` columns and register them with the substrate.

## Example

```rust,ignore
use field_core::prelude::*;

let mesh = UniformMesh::from_config(&UniformMeshConfig {
    nx: 8,
    ny: 8,
    nz: 8,
    bounds_lo: [0.0, 0.0, 0.0],
    bounds_hi: [1.0, 1.0, 1.0],
});

assert_eq!(mesh.n_local_cells(), 8 * 8 * 8);
```

## License

MIT OR Apache-2.0
