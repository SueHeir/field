# FIELD

<!-- disclaimer-banner -->
> This code was fully written via Claude 4.6,4.8 and Fable 5, and stands as a proof of concept for a bevy-like ecosystem for physics simulation research, with the goal of testing if one scheduler/framework (GRASS) works for most scientific codes. SOIL and FIELD are particle- and mesh-based substrates for physics such as DIRT (DEM) or dev_field_efvm. Note that all other physics based repos I have start with dev_, as I do NOT know these methods. Please read, evaluate, use with a grain of salt, I have not personally read or reviewed everything here.
<!-- /disclaimer-banner -->


**F**ramework for **I**ndexed **E**ulerian **L**attice **D**iscretization — the
mesh/Eulerian substrate, sibling to [SOIL](https://github.com/SueHeir/soil) (the
particle/Lagrangian substrate), both riding the
[GRASS](https://github.com/SueHeir/grass) framework.

```
GRASS    framework: App + Plugin + Scheduler, IO, MPI coupling   (no particles/mesh/physics)
  ├─ SOIL    particle substrate: Atom, domain decomp, neighbors ── DIRT      (DEM physics)
  └─ FIELD   mesh substrate:     Mesh, FieldData, halo exchange ── test-cfd  (CFD physics)
```

FIELD owns meshes, per-cell field storage, and halo communication. It knows
nothing about fluxes, equations of state, or boundary conditions — those live in
the physics crate on top, exactly as DIRT layers granular physics onto SOIL.

## Design decisions (and why they differ from toy-cfd)

FIELD is designed from the contracts, not transcribed from the `toy-cfd`
prototype it replaces. Three deliberate departures:

1. **The mesh is a trait family, not a struct.** Coming from particles, this is
   the new problem: a particle is always a point, but a "mesh" can be structured
   Cartesian, stretched, octree-AMR, or unstructured polyhedral. The unifying
   view (OpenFOAM's) is that a finite-volume update only needs *cells with
   volumes* and *faces connecting owner→other with an area-normal*. That is the
   object-safe `FvMesh` core trait. Capabilities that resist the universal face
   view live in extensions required only by solvers that need them:
   - `StructuredMesh` — `(i,j,k)` indexing for wide/directional stencils.
   - `AdaptiveMesh` — refinement levels + coarse/fine face pairs for AMR reflux.

   `UniformMesh` implements `FvMesh + StructuredMesh`; a future `ForestMesh`
   (p4est) implements `FvMesh + AdaptiveMesh`; an unstructured mesh implements
   `FvMesh` alone. **New mesh = implement the traits; physics is untouched.**

   **Scope: this is finite volume.** `FvMesh`'s primitives (cell volumes, faces)
   are the FV data model. FDM and lattice-Boltzmann ride the same grids via
   `StructuredMesh` and just don't use the face methods. FEM/DG/spectral are
   element/DOF-centric, not control-volume/face-centric, and are deliberately out
   of scope — if ever wanted they would be a *sibling* topology trait on the same
   method-agnostic substrate (`FieldData` + `HaloPlan` + partition, none of which
   depend on `FvMesh`), not a change to `FvMesh`.

2. **No physics leaks into the substrate.** toy-cfd's `cfd_grid` depended on
   `cfd_state::ConsVar` and its ghost exchange hardcoded `FIELDS_PER_CELL = 5` —
   the substrate depended on the physics. In FIELD the halo plan is pure index
   lists and the exchange moves whatever `FieldData` the registry holds; the
   state vector is just another registered field.

3. **`FieldData` is smaller than SOIL's `AtomData`.** Particles migrate between
   ranks and get reordered, so `AtomData` carries
   `pack`/`unpack`/`truncate`/`swap_remove`/`apply_permutation`. Cells never
   move. FIELD's `FieldData` keeps only `resize` + forward/reverse/zero. (It does
   keep SOIL's `#[forward]`/`#[reverse]` split that toy-cfd had collapsed into one
   `#[exchange]` — reverse accumulate is needed for AMR reflux and FEM assembly.)

## Status

- **`field_core`** — `FvMesh`/`StructuredMesh`/`AdaptiveMesh` traits, `FieldData` +
  `FieldRegistry`, the `Component` column trait, static `HaloPlan` + generic
  forward/reverse exchange, `UniformMesh` (physics-free port of toy-cfd's
  `UniformGrid`), `MeshScheduleSet`, and GRASS plugins.
- **`field_derive`** — `#[derive(FieldData)]` generating resize + forward/reverse/
  zero from `#[forward]`/`#[reverse]`/`#[zero]`-tagged `Vec<C: Component>` columns.
  Uniform over `f64`, `[f64; N]`, `bool`, and physics structs (anything `Component`).
- **MPI decomposition** — `CommPlugin` (serial no-op or real MPI backend),
  `FieldDefaultPlugins` bundle, and a `PreSetup` system that factors the rank count
  into a process grid, records it on the comm, and rebuilds the local partition
  (so serial and MPI use the same call site). `mpi_backend` feature compiles.

Builds and tests green (20 tests); `--features mpi_backend` compiles.

The substrate is feature-complete for milestone **M1** (explicit, structured,
serial + MPI). Remaining FIELD work belongs to later milestones:

### Next

- `field_amr` — `ForestMesh` on p4est: `AdaptiveMesh`, refluxing, transfer (M2).
- `field_print` — VTK / dump / restart for the generic field registry.
- `test-cfd` — port the compressible physics onto `FvMesh` (`ConsVar: Component`).
