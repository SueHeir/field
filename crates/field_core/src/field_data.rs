//! [`FieldData`] — the single seam between the FIELD substrate and physics.
//!
//! This is the mesh analog of SOIL's `AtomData`, but **deliberately smaller**.
//! `AtomData` carries `pack`/`unpack`/`truncate`/`swap_remove`/`apply_permutation`
//! because particles *migrate between ranks* and get *reordered* every rebuild.
//! Mesh cells do neither: the partition is static and cells keep their index for
//! the whole run. So FIELD's contract drops all of that machinery and keeps only:
//!
//! - [`resize`](FieldData::resize) — size the store to the mesh's cell count once,
//! - **forward** (`pack_forward`/`unpack_forward`) — owner → ghost, *overwrite*,
//! - **reverse** (`pack_reverse`/`unpack_reverse_add`) — ghost → owner, *accumulate*,
//! - **zero** (`zero_cell`) — reset per-step accumulators.
//!
//! `#[forward]` covers the everyday FVM halo (a rank needs its neighbor's state
//! to compute boundary fluxes). `#[reverse]` covers contributions computed on a
//! ghost that must sum back to the owner — AMR reflux corrections and node/FEM
//! assembly. M1 physics uses only forward; reverse is kept because the mesh world
//! genuinely needs it (toy-cfd collapsed both into one `#[exchange]`; FIELD does
//! not, to keep the door open for AMR and FEM).
//!
//! Implement this by hand for one-off stores, or derive it (the `field_derive`
//! crate, landing next) from a struct of `Vec<f64>` / `Vec<[f64; N]>` columns
//! tagged `#[forward]` / `#[reverse]` / `#[zero]`.

use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell, RefMut};
use std::fmt;

/// A per-cell field (one component, or several packed into a struct), stored as
/// columns keyed by flat cell index — the same indexing the mesh uses.
pub trait FieldData: Any + Send + Sync {
    /// Upcasts to `&dyn Any` so a caller can downcast to the concrete store type.
    fn as_any(&self) -> &dyn Any;
    /// Mutable counterpart of [`as_any`](Self::as_any).
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Number of cells currently stored.
    fn len(&self) -> usize;

    /// Resize every column to `n` cells (new cells zero-filled). Called once after
    /// the mesh is known so all stores align with [`crate::FvMesh::n_cells_total`].
    fn resize(&mut self, n: usize);

    /// `true` when no cells are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── forward: owner → ghost, overwrite ──────────────────────────────────────

    /// Number of `f64`s per cell moved by forward (halo) exchange.
    fn forward_size(&self) -> usize {
        0
    }
    /// Append cell `c`'s forward data to `buf`.
    fn pack_forward(&self, c: usize, buf: &mut Vec<f64>) {
        let _ = (c, buf);
    }
    /// Overwrite cell `c`'s forward fields from the front of `buf`; return the
    /// number of `f64`s consumed (must equal [`forward_size`](Self::forward_size)).
    fn unpack_forward(&mut self, c: usize, buf: &[f64]) -> usize {
        let _ = (c, buf);
        0
    }

    // ── reverse: ghost → owner, accumulate ─────────────────────────────────────

    /// Number of `f64`s per cell moved by reverse exchange.
    fn reverse_size(&self) -> usize {
        0
    }
    /// Append ghost cell `c`'s reverse contribution to `buf`.
    fn pack_reverse(&self, c: usize, buf: &mut Vec<f64>) {
        let _ = (c, buf);
    }
    /// **Add** the front of `buf` into owner cell `c`'s reverse fields; return the
    /// number of `f64`s consumed (must equal [`reverse_size`](Self::reverse_size)).
    fn unpack_reverse_add(&mut self, c: usize, buf: &[f64]) -> usize {
        let _ = (c, buf);
        0
    }

    // ── zero: reset per-step accumulators ──────────────────────────────────────

    /// Reset cell `c`'s `#[zero]` accumulator columns to zero.
    fn zero_cell(&mut self, c: usize) {
        let _ = c;
    }
    /// Apply [`zero_cell`](Self::zero_cell) over `0..n`.
    fn zero(&mut self, n: usize) {
        for c in 0..n {
            self.zero_cell(c);
        }
    }
}

/// Error returned by the fallible [`FieldRegistry`] / `try_register_field_data!`
/// entry points. Registration is the one place mesh setup can be *misused* by a
/// caller (registering a type twice, or before the registry resource exists), so
/// it gets a typed error instead of aborting the process. The variants carry no
/// physics — they are substrate-generic, so the tier-agnostic contract holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldRegistryError {
    /// A field of this [`TypeId`] is already registered (each field type is unique).
    AlreadyRegistered(TypeId),
    /// The [`FieldRegistry`] resource is absent — its plugin was not added to the
    /// app before registration was attempted.
    RegistryMissing,
}

impl fmt::Display for FieldRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldRegistryError::AlreadyRegistered(id) => {
                write!(f, "FieldRegistry: field type {id:?} already registered")
            }
            FieldRegistryError::RegistryMissing => {
                write!(f, "FieldRegistry not found — FieldRegistryPlugin must be added first")
            }
        }
    }
}

impl std::error::Error for FieldRegistryError {}

/// Type-keyed registry of [`FieldData`] stores — one entry per concrete type.
///
/// Lives as a single `grass_app` resource. Mirrors SOIL's `AtomDataRegistry`,
/// including the cached lists of which stores actually participate in forward /
/// reverse exchange (so the per-step hot path skips no-op stores).
pub struct FieldRegistry {
    stores: Vec<(TypeId, RefCell<Box<dyn FieldData>>)>,
    forward_stores: Vec<usize>,
    reverse_stores: Vec<usize>,
}

impl Default for FieldRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FieldRegistry {
    /// Creates an empty registry with no field stores.
    pub fn new() -> Self {
        Self {
            stores: Vec::new(),
            forward_stores: Vec::new(),
            reverse_stores: Vec::new(),
        }
    }

    /// Register a new typed field, returning
    /// [`FieldRegistryError::AlreadyRegistered`] if its type is already present
    /// (each field type is unique, like SOIL's registry). This is the fallible
    /// path; [`register`](Self::register) is the panicking convenience over it.
    pub fn try_register<F: FieldData + 'static>(
        &mut self,
        field: F,
    ) -> Result<(), FieldRegistryError> {
        let id = TypeId::of::<F>();
        if self.stores.iter().any(|(t, _)| *t == id) {
            return Err(FieldRegistryError::AlreadyRegistered(id));
        }
        self.stores.push((id, RefCell::new(Box::new(field))));
        self.rebuild_comm_caches();
        Ok(())
    }

    /// Register a new typed field. Convenience wrapper over
    /// [`try_register`](Self::try_register).
    ///
    /// # Panics
    /// Panics if the type is already registered. Prefer
    /// [`try_register`](Self::try_register) when duplicate registration is a
    /// recoverable condition rather than a programming error.
    pub fn register<F: FieldData + 'static>(&mut self, field: F) {
        self.try_register(field).unwrap_or_else(|e| panic!("{e}"));
    }

    fn rebuild_comm_caches(&mut self) {
        self.forward_stores = (0..self.stores.len())
            .filter(|&i| self.stores[i].1.borrow().forward_size() > 0)
            .collect();
        self.reverse_stores = (0..self.stores.len())
            .filter(|&i| self.stores[i].1.borrow().reverse_size() > 0)
            .collect();
    }

    fn cell_of<F: FieldData + 'static>(&self) -> Option<&RefCell<Box<dyn FieldData>>> {
        let id = TypeId::of::<F>();
        self.stores.iter().find(|(t, _)| *t == id).map(|(_, c)| c)
    }

    /// Borrow a registered field by type.
    pub fn get<F: FieldData + 'static>(&self) -> Option<Ref<'_, F>> {
        let cell = self.cell_of::<F>()?;
        Some(Ref::map(cell.borrow(), |b| {
            // Provably internal: `cell_of::<F>` selected this entry by
            // `TypeId::of::<F>()`, so the boxed value IS an `F` and the downcast
            // cannot fail. A failure would be a FIELD bug, not user input — so it
            // stays an `expect`, not a typed error.
            b.as_any().downcast_ref::<F>().expect("FieldRegistry downcast — this is a bug in FIELD")
        }))
    }

    /// Mutably borrow a registered field by type.
    pub fn get_mut<F: FieldData + 'static>(&self) -> Option<RefMut<'_, F>> {
        let cell = self.cell_of::<F>()?;
        Some(RefMut::map(cell.borrow_mut(), |b| {
            // Provably internal: same invariant as `get` — `cell_of::<F>` matched
            // this entry by `TypeId::of::<F>()`, so the downcast cannot fail.
            b.as_any_mut().downcast_mut::<F>().expect("FieldRegistry downcast — this is a bug in FIELD")
        }))
    }

    /// Borrow a field, panicking with `context` if it is not registered.
    ///
    /// This is a deliberate assertion accessor (mirrors [`Option::expect`]): the
    /// caller states the field *must* be present and a missing one is a
    /// programming error. For the fallible lookup that returns `None` instead of
    /// panicking, use [`get`](Self::get).
    pub fn expect<F: FieldData + 'static>(&self, context: &str) -> Ref<'_, F> {
        self.get::<F>().unwrap_or_else(|| panic!("{context}"))
    }

    /// Mutably borrow a field, panicking with `context` if it is not registered.
    ///
    /// Deliberate assertion accessor — see [`expect`](Self::expect). For the
    /// fallible path use [`get_mut`](Self::get_mut).
    pub fn expect_mut<F: FieldData + 'static>(&self, context: &str) -> RefMut<'_, F> {
        self.get_mut::<F>().unwrap_or_else(|| panic!("{context}"))
    }

    /// Resize every registered field to `n` cells.
    pub fn resize_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().resize(n);
        }
    }

    /// Zero every registered field's accumulator columns over `0..n`.
    pub fn zero_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().zero(n);
        }
    }

    /// Total forward `f64`s per cell summed across all participating stores.
    pub fn forward_size(&self) -> usize {
        self.forward_stores
            .iter()
            .map(|&i| self.stores[i].1.borrow().forward_size())
            .sum()
    }

    /// Total reverse `f64`s per cell summed across all participating stores.
    pub fn reverse_size(&self) -> usize {
        self.reverse_stores
            .iter()
            .map(|&i| self.stores[i].1.borrow().reverse_size())
            .sum()
    }

    /// Pack every forward-participating store's data for cell `c` (declaration
    /// order) — the determinism guarantee both ends of an exchange rely on.
    pub fn pack_forward_all(&self, c: usize, buf: &mut Vec<f64>) {
        for &i in &self.forward_stores {
            self.stores[i].1.borrow().pack_forward(c, buf);
        }
    }

    /// Unpack every forward-participating store's data into cell `c`; returns the
    /// total `f64`s consumed.
    pub fn unpack_forward_all(&self, c: usize, buf: &[f64]) -> usize {
        let mut off = 0;
        for &i in &self.forward_stores {
            off += self.stores[i].1.borrow_mut().unpack_forward(c, &buf[off..]);
        }
        off
    }

    /// Pack every reverse-participating store's data for ghost cell `c`.
    pub fn pack_reverse_all(&self, c: usize, buf: &mut Vec<f64>) {
        for &i in &self.reverse_stores {
            self.stores[i].1.borrow().pack_reverse(c, buf);
        }
    }

    /// Accumulate every reverse-participating store's data into owner cell `c`;
    /// returns the total `f64`s consumed.
    pub fn unpack_reverse_all(&self, c: usize, buf: &[f64]) -> usize {
        let mut off = 0;
        for &i in &self.reverse_stores {
            off += self.stores[i].1.borrow_mut().unpack_reverse_add(c, &buf[off..]);
        }
        off
    }

    /// `true` when no field stores are registered.
    pub fn is_empty(&self) -> bool {
        self.stores.is_empty()
    }
}

// SAFETY: same promise as SOIL's AtomDataRegistry and toy-cfd's FieldRegistry —
// the registry lives in a single grass_app resource cell and is only ever
// touched on the (single) simulation thread, so the !Sync `RefCell`s inside are
// never shared across threads. The `Sync` impl is required to store it in the
// resource map (which is `Send + Sync`).
unsafe impl Sync for FieldRegistry {}

/// Fallibly register a [`FieldData`] store into an [`App`](grass_app::App)'s
/// [`FieldRegistry`]. Returns [`FieldRegistryError::RegistryMissing`] if the
/// registry resource is absent (its plugin was not added) or
/// [`FieldRegistryError::AlreadyRegistered`] if the field type is already
/// registered. This is the fallible path; [`register_field_data!`] is the
/// panicking convenience over it.
///
/// ```rust,ignore
/// try_register_field_data!(app, ConsState::default())?;
/// ```
#[macro_export]
macro_rules! try_register_field_data {
    ($app:expr, $value:expr) => {
        match $app.get_mut_resource(::std::any::TypeId::of::<$crate::FieldRegistry>()) {
            ::std::option::Option::Some(cell) => {
                let mut binder = cell.borrow_mut();
                binder
                    .downcast_mut::<$crate::FieldRegistry>()
                    // Provably internal: the resource stored under this TypeId is a
                    // FieldRegistry by construction, so the downcast cannot fail.
                    .expect("Failed to downcast FieldRegistry — this is a bug in FIELD")
                    .try_register($value)
            }
            ::std::option::Option::None => {
                ::std::result::Result::Err($crate::FieldRegistryError::RegistryMissing)
            }
        }
    };
}

/// Register a [`FieldData`] store into an [`App`](grass_app::App)'s
/// [`FieldRegistry`]. Mirror of SOIL's `register_atom_data!`. Panicking
/// convenience over [`try_register_field_data!`].
///
/// ```rust,ignore
/// register_field_data!(app, ConsState::default());
/// ```
///
/// # Panics
/// Panics if the registry resource is absent (its plugin was not added) or the
/// field type is already registered. Use [`try_register_field_data!`] to handle
/// those conditions as a typed [`FieldRegistryError`] instead.
#[macro_export]
macro_rules! register_field_data {
    ($app:expr, $value:expr) => {
        $crate::try_register_field_data!($app, $value).unwrap_or_else(|e| ::std::panic!("{e}"))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A forward-exchanged scalar with a separate `#[zero]`-style accumulator.
    struct Scalar {
        value: Vec<f64>,
        accum: Vec<f64>,
    }

    impl FieldData for Scalar {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn len(&self) -> usize {
            self.value.len()
        }
        fn resize(&mut self, n: usize) {
            self.value.resize(n, 0.0);
            self.accum.resize(n, 0.0);
        }
        fn forward_size(&self) -> usize {
            1
        }
        fn pack_forward(&self, c: usize, buf: &mut Vec<f64>) {
            buf.push(self.value[c]);
        }
        fn unpack_forward(&mut self, c: usize, buf: &[f64]) -> usize {
            self.value[c] = buf[0];
            1
        }
        fn zero_cell(&mut self, c: usize) {
            self.accum[c] = 0.0;
        }
    }

    #[test]
    fn register_and_borrow() {
        let mut reg = FieldRegistry::new();
        reg.register(Scalar { value: vec![1.0, 2.0, 3.0], accum: vec![9.0; 3] });
        assert_eq!(reg.forward_size(), 1);
        reg.get_mut::<Scalar>().unwrap().value[1] = 99.0;
        assert_eq!(reg.get::<Scalar>().unwrap().value[1], 99.0);
        reg.resize_all(5);
        assert_eq!(reg.get::<Scalar>().unwrap().len(), 5);
    }

    #[test]
    fn forward_pack_unpack_round_trip() {
        let mut reg = FieldRegistry::new();
        reg.register(Scalar { value: vec![10.0, 20.0, 30.0], accum: vec![0.0; 3] });
        let mut buf = Vec::new();
        reg.pack_forward_all(1, &mut buf);
        assert_eq!(buf, vec![20.0]);
        let consumed = reg.unpack_forward_all(0, &buf);
        assert_eq!(consumed, 1);
        assert_eq!(reg.get::<Scalar>().unwrap().value[0], 20.0);
    }

    #[test]
    fn try_register_rejects_duplicate_without_panicking() {
        let mut reg = FieldRegistry::new();
        assert!(reg.try_register(Scalar { value: vec![1.0], accum: vec![0.0] }).is_ok());
        // A second registration of the same type errors instead of aborting.
        let err = reg
            .try_register(Scalar { value: vec![2.0], accum: vec![0.0] })
            .unwrap_err();
        assert!(matches!(err, FieldRegistryError::AlreadyRegistered(id) if id == TypeId::of::<Scalar>()));
        // The original store is untouched by the rejected second call.
        assert_eq!(reg.get::<Scalar>().unwrap().value, vec![1.0]);
    }

    #[test]
    fn error_display_is_descriptive() {
        let e = FieldRegistryError::AlreadyRegistered(TypeId::of::<Scalar>());
        assert!(e.to_string().contains("already registered"));
        assert!(FieldRegistryError::RegistryMissing
            .to_string()
            .contains("FieldRegistryPlugin"));
    }

    #[test]
    fn zero_resets_only_accumulators() {
        let mut reg = FieldRegistry::new();
        reg.register(Scalar { value: vec![1.0; 3], accum: vec![5.0; 3] });
        reg.zero_all(3);
        let s = reg.get::<Scalar>().unwrap();
        assert_eq!(s.accum, vec![0.0, 0.0, 0.0]);
        assert_eq!(s.value, vec![1.0, 1.0, 1.0]);
    }
}
