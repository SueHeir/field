# MPM Elastic Bar Transfer Demonstrator

This example is the particle-mesh hybrid proof for FIELD's `field_transfer`
operator. It builds a one-dimensional elastic bar as material points embedded in
a thin 3-D FIELD `UniformMesh`, then performs the MPM transfer work that every
PIC/MPM-style solver needs:

- scatter particle mass to grid mass;
- scatter particle momentum to grid momentum;
- scatter particle mass as a density and check the volume integral;
- gather an affine grid velocity back to particles with the same kernel.

The local elastic-bar state is intentionally kept in this example rather than in
`crates/*/src`: material points, strain, stress, and elastic energy are
MPM-specific physics, while the reusable piece is the existing
solver-agnostic transfer operator.

Run:

```bash
cargo run --release -p mpm_elastic_bar -- examples/mpm_elastic_bar/config.toml
```

The binary prints one `RESULT` line. `sweep.py` runs a small resolution/particle
ladder and fails unless mass, momentum, density-integral conservation, and
grid-to-particle affine gather all pass at tight tolerances.
