//! [`MeshScheduleSet`] — the per-step phase ordering for an explicit mesh solver.
//!
//! The mesh counterpart to SOIL's `ParticleSimScheduleSet`. SOIL's phases are
//! shaped by velocity-Verlet (initial/final half-kicks, exchange, neighbor
//! rebuild). A mesh solver's anatomy is different and simpler — there is no
//! neighbor rebuild and no atom migration — so the phases follow the explicit
//! update itself:
//!
//! ```text
//! halo (fill ghosts) → boundary (apply BCs) → flux (build RHS) → update (advance)
//! ```
//!
//! These are generic to any explicit method (FVM, FDM, explicit FEM); the choice
//! of flux scheme and time integrator is physics, layered on top. Multi-stage
//! integrators (RK) drive this set repeatedly per step via the scheduler's loop
//! constructs — the set itself describes one sub-stage.
//!
//! **Scope: explicit time stepping.** The phase list bakes in an explicit update
//! (`Flux → Update`, "advance the state by one sub-stage"). An *implicit* solver
//! (Newton–Krylov for low-Mach / fully-coupled viscous, the M3 linear-solver
//! milestone) has a different anatomy — assemble → linear-solve → converge — and
//! should define its own `ScheduleSet` rather than reuse this one. This is not the
//! universal mesh schedule, only the explicit one.

/// Schedule phases for an explicit mesh time step, in execution order.
#[derive(Debug, Clone, Copy, grass_derive::ScheduleSet)]
pub enum MeshScheduleSet {
    /// Per-step bookkeeping: step counter, CFL-limited `dt`.
    Setup,
    /// Before halo exchange.
    PreHalo,
    /// Forward-exchange ghost cells across rank boundaries.
    Halo,
    /// After halo exchange.
    PostHalo,
    /// Before boundary conditions.
    PreBoundary,
    /// Fill physical-boundary ghost cells from boundary conditions.
    Boundary,
    /// Before flux assembly.
    PreFlux,
    /// Assemble the RHS (flux divergence) into the per-cell accumulator.
    Flux,
    /// After flux assembly — e.g. AMR reflux corrections (reverse exchange).
    PostFlux,
    /// Before the state update.
    PreUpdate,
    /// Advance the conserved state by one (sub-)stage.
    Update,
    /// After the state update.
    PostUpdate,
    /// Diagnostics, dumps, restart writes.
    Output,
}
