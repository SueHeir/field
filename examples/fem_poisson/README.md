# fem_poisson — the implicit global-solve proof for GRASS + FIELD

Solves the steady-state Poisson problem

```
-∇²u = f    on Ω = (0,1)²
   u = 0    on ∂Ω
```

with continuous **P1 (linear-triangle) finite elements** on the FIELD mesh, and a
**single global sparse linear solve** `K u = b` — no timestepping.

## Why this exists

`field_core`'s `FvMesh` trait is deliberately scoped to *finite volume*, and both
the FIELD and GRASS READMEs carried a caveat that implicit/global solvers (FEM,
spectral) were unproven on the stack. This example refutes that caveat by
construction: an FEM solver that assembles a sparse matrix and factors it lives
happily **as an example on top of the existing substrate**, using only the
method-agnostic pieces the docs already promised such a solver would ride —
the mesh geometry, the GRASS scheduler, and GRASS resources. No `field_core`
`src` was changed.

## How it maps onto GRASS + FIELD

| Concern | Mechanism |
|---|---|
| Mesh geometry & node/element layout | FIELD `UniformMesh` (`dims`, `spacing`, `cell_centroid`) |
| Schedule | a custom `PoissonSolveSet` = `Assemble → Solve → Validate` — the "assemble → linear-solve → converge" anatomy `field_core::schedule` predicted an implicit solver would define for itself |
| Assemble | a GRASS system loops the mesh cells (each split into two P1 triangles) and scatters the element stiffness + consistent load into a sparse `K`, `b` |
| Solve | one GRASS system: COO → CSC → sparse Cholesky (`nalgebra-sparse`) → back-substitution. A **single** global solve |
| Drive | `app.prepare(); app.run();` — the update schedule runs **exactly once**, no pseudo-time loop |

## Validation

Method of manufactured solutions with `u = sin(πx) sin(πy)` (so
`f = 2π² sin(πx) sin(πy)`, and `u = 0` on the unit-square boundary is exact). P1
elements are second-order in L², so the error falls like `O(h²)`.

```bash
# single run
cargo run --release -p fem_poisson -- examples/fem_poisson/config.toml

# convergence gate (PASS/FAIL, exit-coded)
python3 examples/fem_poisson/sweep.py
```

Observed L² convergence (see `sweep.py`):

```
  n   16 ->   32:  observed L2 order p = 1.989
  n   32 ->   64:  observed L2 order p = 1.997
  n   64 ->  128:  observed L2 order p = 1.999
  mean observed order = 1.995   (theory 2.000)
```
