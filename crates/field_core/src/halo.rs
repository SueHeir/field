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
use grass_mpi::{CommBackend, SendRecvOp};

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
///
/// **Multiple links to one rank.** All links are exchanged concurrently in one
/// untagged batch, so MPI pairs messages between two ranks in posting order: the
/// `k`-th link to rank `r` in this plan pairs with the `k`-th link back to us in
/// `r`'s plan. Producers either keep one link per neighbour rank or emit
/// repeated links in an order both ends agree on ([`crate::UniformMesh`] emits at
/// most one per axis, axes in x, y, z order).
///
/// **Self links.** A link whose `rank` is this rank (a periodic axis that is not
/// split) is exchanged as a local copy, never through MPI, so it works on the
/// serial backend too.
#[derive(Default)]
pub struct HaloPlan {
    /// Per-neighbour exchange links that make up this plan.
    pub links: Vec<HaloLink>,
}

impl HaloPlan {
    /// An empty plan with no links (serial run — nothing to exchange).
    pub fn empty() -> Self {
        Self { links: Vec::new() }
    }

    /// `true` when there is nothing to exchange: a serial, non-periodic run. (A
    /// serial periodic mesh has self links, so it is not "serial" here.)
    pub fn is_serial(&self) -> bool {
        self.links.is_empty()
    }
}

/// Which way a [`HaloLink`] is walked.
#[derive(Clone, Copy)]
enum Direction {
    /// owner (`send_cells`) → ghost (`recv_cells`)
    Forward,
    /// ghost (`recv_cells`) → owner (`send_cells`)
    Reverse,
}

/// Exchange every link of `plan` at once: pack all send buffers, post every
/// remote swap in one non-blocking batch (so neighbours along an axis no longer
/// wait on each other in a chain, and receive sizes are known up front — no
/// `MPI_Probe`), copy self links locally, then unpack in link order.
fn exchange(
    plan: &HaloPlan,
    comm: &dyn CommBackend,
    per: usize,
    dir: Direction,
    mut pack: impl FnMut(&[usize], &mut Vec<f64>),
    mut unpack: impl FnMut(&[usize], &[f64]) -> usize,
) {
    if per == 0 || plan.links.is_empty() {
        return;
    }
    fn cells(link: &HaloLink, dir: Direction) -> (&[usize], &[usize]) {
        match dir {
            Direction::Forward => (&link.send_cells, &link.recv_cells),
            Direction::Reverse => (&link.recv_cells, &link.send_cells),
        }
    }

    let mut sends: Vec<Vec<f64>> = Vec::with_capacity(plan.links.len());
    let mut recvs: Vec<Vec<f64>> = Vec::with_capacity(plan.links.len());
    for link in &plan.links {
        let (out, inn) = cells(link, dir);
        let mut buf = Vec::with_capacity(out.len() * per);
        pack(out, &mut buf);
        debug_assert_eq!(buf.len(), out.len() * per, "halo pack size mismatch");
        recvs.push(vec![0.0; inn.len() * per]);
        sends.push(buf);
    }

    let me = comm.rank();
    let mut ops: Vec<SendRecvOp<'_>> = Vec::with_capacity(plan.links.len());
    for ((link, send), recv) in plan.links.iter().zip(&sends).zip(recvs.iter_mut()) {
        if link.rank == me {
            debug_assert_eq!(send.len(), recv.len(), "self halo link size mismatch");
            recv.copy_from_slice(send);
        } else {
            ops.push(SendRecvOp {
                dest: link.rank,
                send_buf: send,
                source: link.rank,
                recv_buf: recv,
            });
        }
    }
    if !ops.is_empty() {
        comm.sendrecv_batch_f64_into(&mut ops);
    }
    drop(ops);

    for (link, recv) in plan.links.iter().zip(&recvs) {
        let (_, inn) = cells(link, dir);
        let used = unpack(inn, recv);
        debug_assert_eq!(used, recv.len(), "halo size mismatch");
    }
}

/// Forward-exchange a single [`FieldData`] store: owner cells → ghost cells,
/// overwrite semantics. No-op when the store has no forward fields or the plan
/// is empty.
pub fn halo_exchange_forward(
    plan: &HaloPlan,
    field: &mut dyn FieldData,
    comm: &dyn CommBackend,
) {
    let per = field.forward_size();
    // `pack` only reads and `unpack` only runs after every pack, but both
    // closures need the store; route them through a RefCell.
    let field = std::cell::RefCell::new(field);
    exchange(
        plan,
        comm,
        per,
        Direction::Forward,
        |cells, buf| field.borrow().pack_forward_cells(cells, buf),
        |cells, buf| field.borrow_mut().unpack_forward_cells(cells, buf),
    );
}

/// Forward-exchange **every** forward-participating store in the registry in one
/// pass per neighbor (store-major: each store's cells in one block, stores in
/// declaration order). This is the
/// per-step entry point physics calls; it is the mesh counterpart to SOIL's
/// `forward_comm`.
pub fn halo_exchange_forward_all(
    plan: &HaloPlan,
    registry: &FieldRegistry,
    comm: &dyn CommBackend,
) {
    exchange(
        plan,
        comm,
        registry.forward_size(),
        Direction::Forward,
        |cells, buf| registry.pack_forward_all_cells(cells, buf),
        |cells, buf| registry.unpack_forward_all_cells(cells, buf),
    );
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
    exchange(
        plan,
        comm,
        registry.reverse_size(),
        Direction::Reverse,
        |cells, buf| registry.pack_reverse_all_cells(cells, buf),
        |cells, buf| registry.unpack_reverse_all_cells(cells, buf),
    );
}
