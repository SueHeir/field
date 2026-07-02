//! End-to-end test of `#[derive(FieldData)]`: a physics-style multi-component
//! state plus scalar accumulators, exercised through the registry the same way
//! the per-step halo exchange will drive it.

use field_core::prelude::*;

/// A 2-component "conserved variable" — stands in for the CFD `ConsVar`, proving
/// the derive works for a physics type, not just `f64`/`[f64; N]`.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Duo {
    a: f64,
    b: f64,
}

impl Component for Duo {
    const WIDTH: usize = 2;
    fn write(&self, buf: &mut Vec<f64>) {
        buf.push(self.a);
        buf.push(self.b);
    }
    fn read(buf: &[f64]) -> Self {
        Duo { a: buf[0], b: buf[1] }
    }
    fn add_from(&mut self, buf: &[f64]) {
        self.a += buf[0];
        self.b += buf[1];
    }
    fn zero() -> Self {
        Duo::default()
    }
}

#[derive(FieldData, Default)]
struct State {
    /// Exchanged conserved state (5-wide in real CFD; 2-wide here).
    #[forward]
    u: Vec<Duo>,
    /// A reverse-accumulated, per-step-zeroed scalar (an AMR-reflux-style column).
    #[reverse]
    #[zero]
    flux: Vec<f64>,
    /// A plain column that resizes with the mesh but never crosses a rank.
    backup: Vec<Duo>,
}

#[test]
fn sizes_count_only_tagged_columns() {
    let s = State::default();
    // forward = Duo (2); reverse = f64 (1); backup contributes to neither.
    assert_eq!(FieldData::forward_size(&s), 2);
    assert_eq!(s.reverse_size(), 1);
}

#[test]
fn resize_and_len_cover_all_columns() {
    let mut s = State::default();
    s.resize(4);
    assert_eq!(s.len(), 4);
    assert_eq!(s.u.len(), 4);
    assert_eq!(s.flux.len(), 4);
    assert_eq!(s.backup.len(), 4);
}

#[test]
fn forward_is_overwrite() {
    let mut s = State::default();
    s.resize(2);
    s.u[0] = Duo { a: 1.0, b: 2.0 };
    let mut buf = Vec::new();
    s.pack_forward(0, &mut buf);
    assert_eq!(buf, vec![1.0, 2.0]);
    // Pre-seed the destination to prove forward overwrites rather than adds.
    s.u[1] = Duo { a: 99.0, b: 99.0 };
    let consumed = s.unpack_forward(1, &buf);
    assert_eq!(consumed, 2);
    assert_eq!(s.u[1], Duo { a: 1.0, b: 2.0 });
}

#[test]
fn reverse_accumulates() {
    let mut s = State::default();
    s.resize(2);
    s.flux[0] = 5.0;
    let mut buf = Vec::new();
    s.pack_reverse(0, &mut buf);
    assert_eq!(buf, vec![5.0]);
    s.flux[1] = 10.0;
    let consumed = s.unpack_reverse_add(1, &buf);
    assert_eq!(consumed, 1);
    assert_eq!(s.flux[1], 15.0); // 10 + 5, accumulate not overwrite
}

#[test]
fn zero_resets_only_zero_tagged_columns() {
    let mut s = State::default();
    s.resize(1);
    s.u[0] = Duo { a: 7.0, b: 8.0 };
    s.flux[0] = 42.0;
    s.zero(1);
    assert_eq!(s.flux[0], 0.0); // #[zero]
    assert_eq!(s.u[0], Duo { a: 7.0, b: 8.0 }); // untouched
}

#[test]
fn round_trips_through_the_registry() {
    let mut reg = FieldRegistry::new();
    reg.register(State::default());
    reg.resize_all(3);
    {
        let mut st = reg.get_mut::<State>().unwrap();
        st.u[0] = Duo { a: 1.5, b: -2.5 };
    }
    assert_eq!(reg.forward_size(), 2);
    let mut buf = Vec::new();
    reg.pack_forward_all(0, &mut buf);
    assert_eq!(buf, vec![1.5, -2.5]);
    let consumed = reg.unpack_forward_all(2, &buf);
    assert_eq!(consumed, 2);
    assert_eq!(reg.get::<State>().unwrap().u[2], Duo { a: 1.5, b: -2.5 });
}
