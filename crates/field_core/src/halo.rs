//! Halo (ghost-cell) exchange — the parallel boundary, expressed without any
//! physics types.
//!
//! # The particle → mesh inversion
//!
//! In SOIL the halo is *dynamic*: particles move, so every rebuild recomputes
//! which atoms are ghosts and reconstructs the swap lists. In FIELD the mesh is
//! *static*, so the halo is computed **once** and reused every step. A
//! [`HaloPlan`] is just two index lists per neighbor rank: "pack these local
//! cells, send them; unpack what comes back into these ghost cells".
//!
//! Crucially, the plan holds no notion of *what* lives in a cell — it moves
//! whatever the [`FieldRegistry`] says is forward/reverse exchangeable. This is
//! what fixes toy-cfd's layering inversion, where `cfd_grid::ghost_exchange`
//! hardcoded `ConsVar` and `FIELDS_PER_CELL = 5`.
//!
//! **Limitation — face-only halos.** A [`crate::UniformMesh`] plan exchanges only
//! the six axis-face slabs, not corner/edge (diagonal) ghosts. This is sufficient
//! for face-flux FVM and for *separable* stencils (a Laplacian `∂xx+∂yy+∂zz` only
//! reads axis neighbors), so the M1 inviscid + viscous physics are covered. A
//! scheme that reads a diagonal ghost (mixed derivatives) or trilinear sampling
//! straddling a subdomain corner needs a 26-neighbor halo — a substrate addition
//! deferred until general IBM-under-MPI requires it.

use crate::field_data::{FieldData, FieldRegistry};
use grass_mpi::CommBackend;

/// One neighbor-rank link in a [`HaloPlan`].
///
/// `send_cells[k]` on this rank corresponds to `recv_cells[k]` on the neighbor's
/// matching link (and vice-versa), so both ends agree on cell ordering and the
/// flat `f64` buffer lines up. Both lists have equal length.
pub struct HaloLink {
    /// Rank to exchange with.
    pub rank: i32,
    /// Local interior cells to pack and send (in order).
    pub send_cells: Vec<usize>,
    /// Local ghost cells to unpack received data into (in order).
    pub recv_cells: Vec<usize>,
}

/// The full halo communication pattern for one rank's subdomain. A serial
/// single-rank run produces an empty plan (no links), so exchange is a no-op and
/// the serial `CommBackend`'s `unreachable!` point-to-point methods are never hit.
///
/// **Producer contract** (every backend that builds a `HaloPlan` must uphold):
/// for a link to rank `r`, this rank's `recv_cells` order MUST match the order in
/// which rank `r`'s matching link enumerates the *same* cells in its `send_cells`
/// (and vice-versa). The exchange routines move a flat `f64` buffer positionally —
/// a permuted-but-same-length ordering passes the size `debug_assert` yet delivers
/// wrong-neighbour values silently. `reverse` exchange swaps the send/recv roles,
/// so the ordering must hold in both directions. (The p4est backend gets this for
/// free from p4est's matched mirror/ghost ordering; a structured-pencil backend
/// gets it from matching face traversal on both sides.)
#[derive(Default)]
pub struct HaloPlan {
    pub links: Vec<HaloLink>,
}

impl HaloPlan {
    pub fn empty() -> Self {
        Self { links: Vec::new() }
    }

    /// `true` when there is nothing to exchange (serial run).
    pub fn is_serial(&self) -> bool {
        self.links.is_empty()
    }
}

/// Forward-exchange a single [`FieldData`] store: owner cells → ghost cells,
/// overwrite semantics. No-op when the store has no forward fields or the plan
/// is serial.
pub fn halo_exchange_forward(
    plan: &HaloPlan,
    field: &mut dyn FieldData,
    comm: &dyn CommBackend,
) {
    let per = field.forward_size();
    if per == 0 {
        return;
    }
    for link in &plan.links {
        let mut send = Vec::with_capacity(link.send_cells.len() * per);
        for &c in &link.send_cells {
            field.pack_forward(c, &mut send);
        }
        let recv = comm.sendrecv_f64(link.rank, &send, link.rank);
        let mut off = 0;
        for &c in &link.recv_cells {
            off += field.unpack_forward(c, &recv[off..]);
        }
        debug_assert_eq!(off, recv.len(), "forward halo size mismatch");
    }
}

/// Forward-exchange **every** forward-participating store in the registry in one
/// pass per neighbor (cell data interleaved in declaration order). This is the
/// per-step entry point physics calls; it is the mesh counterpart to SOIL's
/// `forward_comm`.
pub fn halo_exchange_forward_all(
    plan: &HaloPlan,
    registry: &FieldRegistry,
    comm: &dyn CommBackend,
) {
    let per = registry.forward_size();
    if per == 0 {
        return;
    }
    for link in &plan.links {
        let mut send = Vec::with_capacity(link.send_cells.len() * per);
        for &c in &link.send_cells {
            registry.pack_forward_all(c, &mut send);
        }
        let recv = comm.sendrecv_f64(link.rank, &send, link.rank);
        let mut off = 0;
        for &c in &link.recv_cells {
            off += registry.unpack_forward_all(c, &recv[off..]);
        }
        debug_assert_eq!(off, recv.len(), "forward halo size mismatch");
    }
}

/// Reverse-exchange every reverse-participating store: ghost contributions →
/// owner cells, **accumulate** (`+=`) semantics. The send/recv roles swap
/// relative to forward — we pack the *ghost* cells and add into the *interior*
/// cells that own them. Used for AMR reflux corrections and FEM-style assembly.
pub fn halo_exchange_reverse_all(
    plan: &HaloPlan,
    registry: &FieldRegistry,
    comm: &dyn CommBackend,
) {
    let per = registry.reverse_size();
    if per == 0 {
        return;
    }
    for link in &plan.links {
        // Reverse: pack what we received as ghosts (recv_cells), add into the
        // interior cells we originally sent from (send_cells).
        let mut send = Vec::with_capacity(link.recv_cells.len() * per);
        for &c in &link.recv_cells {
            registry.pack_reverse_all(c, &mut send);
        }
        let recv = comm.sendrecv_f64(link.rank, &send, link.rank);
        let mut off = 0;
        for &c in &link.send_cells {
            off += registry.unpack_reverse_all(c, &recv[off..]);
        }
        debug_assert_eq!(off, recv.len(), "reverse halo size mismatch");
    }
}
