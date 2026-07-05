# field_p4est

p4est-backed AMR crate for the FIELD substrate.

## Role

`field_p4est` wraps the p4est forest-of-octrees library through a small C shim
and exposes a dynamically refinable `ForestMesh` implementing
`field_core::FvMesh + field_core::AdaptiveMesh`. It is the production-oriented
AMR backend, while `field_amr` remains the pure-Rust demonstration mesh.

## Contract

This crate owns p4est integration, forest layout, leaf/neighbor queries, and the
FIELD mesh adapter. It remains substrate-level code: refinement criteria,
equation systems, fluxes, and boundary-condition policies belong in physics
crates layered above FIELD.

## Example

```rust,ignore
use field_p4est::{ForestGrid, ForestLayout, ForestMesh};

let mut grid = ForestGrid::new(ForestLayout {
    trees_x: 1,
    trees_y: 1,
    trees_z: 1,
    xmin: 0.0,
    xmax: 1.0,
    ymin: 0.0,
    ymax: 1.0,
    zmin: 0.0,
    zmax: 1.0,
    min_level: 0,
})?;
grid.refine(2, |cx, cy, cz, _h| cx * cx + cy * cy + cz * cz < 0.25);
let mesh = ForestMesh::new(grid);
```

Builds require p4est and sc libraries discoverable through `P4EST_PREFIX`.

## License

MIT OR Apache-2.0
