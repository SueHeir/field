# field_transfer

Particle ⇄ mesh transfer operators — the solver-agnostic hybrid primitive:
conservative **scatter** (particle → cell) and consistent **gather**
(cell → particle), shared by PIC, MPM, CFD-DEM, and IBM.

## Role

`field_transfer` provides the two operations every particle-in-cell family needs
at the Lagrangian/Eulerian boundary, implemented once so no hybrid re-rolls them
(and re-rolls the conservation bookkeeping subtly wrong):

- [`scatter`] / [`scatter_density`] — deposit a per-particle quantity onto the
  mesh (momentum onto a PIC grid, void fraction / drag reaction onto CFD-DEM
  cells, mass onto an MPM grid);
- [`gather`] — sample a cell field back at the particle positions (grid velocity
  in PIC/MPM, fluid velocity for a drag law in CFD-DEM, image-point value in IBM).

Both use one shape function — cloud-in-cell (trilinear), the eight cells
bracketing a point — exposed as a fixed-capacity `Stencil` produced by the
`TransferMesh` trait.

## Contract

This crate is physics-agnostic and particle-method-agnostic. The **particle side
is plain slices** (`positions: &[[f64; 3]]`, `values: &[f64]`), so it has **no
dependency on SOIL or any particle method**. The **mesh side is the minimal
`TransferMesh` geometry** (cell count, cell volume, stencil for a point); every
FIELD `StructuredMesh` satisfies it via a blanket impl, and a non-FIELD grid can
implement `TransferMesh` directly.

The cloud-in-cell weights are a **partition of unity** (`Σ w == 1`), which is the
single fact behind the two guarantees hybrids rely on: `scatter` is
**conservative** (`Σ_c cells[c]` gains exactly `Σ_p values[p]`) and `gather` is
**constant-preserving** (a uniform field is sampled back exactly). They are a
matched adjoint pair under the same kernel, which is what keeps hybrid coupling
stable.

## Example

```rust,ignore
use field_core::{UniformMesh, UniformMeshConfig};
use field_transfer::{scatter, TransferMesh};

let mesh = UniformMesh::from_config(&UniformMeshConfig {
    nx: 8, ny: 8, nz: 8,
    bounds_lo: [0.0, 0.0, 0.0],
    bounds_hi: [1.0, 1.0, 1.0],
});

let positions = [[0.5, 0.5, 0.5], [0.25, 0.75, 0.1]];
let values = [1.0, 2.0];
let mut cells = vec![0.0; mesh.n_cells()];

let stats = scatter(&mesh, &positions, &values, &mut cells);

// Conservative: what was deposited equals the sum of particle values.
assert_eq!(stats.applied, 2);
let deposited: f64 = cells.iter().sum();
assert!((deposited - 3.0).abs() < 1e-12);
```

## License

MIT OR Apache-2.0
