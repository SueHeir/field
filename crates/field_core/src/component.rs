//! [`Component`] — the per-cell value types a [`crate::FieldData`] column holds.
//!
//! Every column of a derived `FieldData` is a `Vec<C>` where `C: Component`: a
//! fixed-width value that knows how to (de)serialize itself to a flat `f64`
//! buffer. This is the one place FIELD improves on SOIL's `AtomData` derive,
//! which hardcodes `f64` and `[f64; N]` and branches on them throughout codegen.
//! Here the macro is uniform — it only ever calls `C::WIDTH` / `write` / `read` /
//! `add_from` / `zero` — and physics can make its own state a single component:
//!
//! ```rust,ignore
//! #[derive(Clone, Copy, Default)]
//! struct ConsVar { rho: f64, rho_u: f64, rho_v: f64, rho_w: f64, rho_e: f64 }
//!
//! impl field_core::Component for ConsVar {
//!     const WIDTH: usize = 5;
//!     fn write(&self, b: &mut Vec<f64>) { b.extend_from_slice(&[self.rho, self.rho_u, self.rho_v, self.rho_w, self.rho_e]); }
//!     fn read(b: &[f64]) -> Self { Self { rho: b[0], rho_u: b[1], rho_v: b[2], rho_w: b[3], rho_e: b[4] } }
//!     fn add_from(&mut self, b: &[f64]) { self.rho += b[0]; /* … */ }
//!     fn zero() -> Self { Self::default() }
//! }
//! // then: #[derive(FieldData)] struct State { #[forward] u: Vec<ConsVar> }
//! ```

/// A fixed-width, `f64`-serializable per-cell value.
///
/// `WIDTH` is the number of `f64`s the value occupies in an exchange buffer.
/// Implemented here for `f64`, `[f64; N]`, and `bool`; physics implements it for
/// its own conserved-variable structs.
pub trait Component: Copy + 'static {
    /// Number of `f64`s this component serializes to.
    const WIDTH: usize;
    /// Append this component's `f64`s to `buf`.
    fn write(&self, buf: &mut Vec<f64>);
    /// Read this component from the front of `buf` (overwrite semantics).
    fn read(buf: &[f64]) -> Self;
    /// Accumulate (`+=`) the front of `buf` into `self` (reverse-exchange semantics).
    fn add_from(&mut self, buf: &[f64]);
    /// The additive identity — used to zero accumulators and to fill on resize.
    fn zero() -> Self;
}

impl Component for f64 {
    const WIDTH: usize = 1;
    #[inline]
    fn write(&self, buf: &mut Vec<f64>) {
        buf.push(*self);
    }
    #[inline]
    fn read(buf: &[f64]) -> Self {
        buf[0]
    }
    #[inline]
    fn add_from(&mut self, buf: &[f64]) {
        *self += buf[0];
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl<const N: usize> Component for [f64; N] {
    const WIDTH: usize = N;
    #[inline]
    fn write(&self, buf: &mut Vec<f64>) {
        buf.extend_from_slice(self);
    }
    #[inline]
    fn read(buf: &[f64]) -> Self {
        std::array::from_fn(|i| buf[i])
    }
    #[inline]
    fn add_from(&mut self, buf: &[f64]) {
        for i in 0..N {
            self[i] += buf[i];
        }
    }
    #[inline]
    fn zero() -> Self {
        [0.0; N]
    }
}

/// Boolean masks (e.g. the IBM solid mask) exchange as `1.0`/`0.0`. `add_from` is
/// an *opinionated* logical OR — the sensible reverse-accumulate for a mask, but a
/// baked-in choice. A field needing different `bool` reverse semantics must wrap
/// it in a newtype with its own `Component` impl rather than use this blanket one.
impl Component for bool {
    const WIDTH: usize = 1;
    #[inline]
    fn write(&self, buf: &mut Vec<f64>) {
        buf.push(if *self { 1.0 } else { 0.0 });
    }
    #[inline]
    fn read(buf: &[f64]) -> Self {
        buf[0] > 0.5
    }
    #[inline]
    fn add_from(&mut self, buf: &[f64]) {
        *self = *self || buf[0] > 0.5;
    }
    #[inline]
    fn zero() -> Self {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_round_trip() {
        let mut buf = Vec::new();
        3.5f64.write(&mut buf);
        assert_eq!(buf, vec![3.5]);
        assert_eq!(f64::read(&buf), 3.5);
        let mut acc = 1.0f64;
        acc.add_from(&buf);
        assert_eq!(acc, 4.5);
    }

    #[test]
    fn array_round_trip() {
        let v = [1.0, 2.0, 3.0];
        let mut buf = Vec::new();
        v.write(&mut buf);
        assert_eq!(buf, vec![1.0, 2.0, 3.0]);
        assert_eq!(<[f64; 3]>::read(&buf), v);
        let mut acc = [10.0, 20.0, 30.0];
        acc.add_from(&buf);
        assert_eq!(acc, [11.0, 22.0, 33.0]);
        assert_eq!(<[f64; 3]>::WIDTH, 3);
    }

    #[test]
    fn bool_round_trip() {
        let mut buf = Vec::new();
        true.write(&mut buf);
        assert_eq!(buf, vec![1.0]);
        assert!(bool::read(&buf));
        assert!(!bool::zero());
    }
}
