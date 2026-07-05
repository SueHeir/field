# field_derive

Procedural macro crate for deriving `field_core::FieldData` implementations.

## Role

`field_derive` provides `#[derive(FieldData)]` for structs whose fields are
per-cell `Vec<C>` columns where `C: field_core::Component`. The macro generates
the resize, forward halo, reverse halo, and zeroing methods required by
`field_core::FieldRegistry`.

## Contract

The macro only generates substrate plumbing. It does not define physics fields
or decide what a solver exchanges. Field structs live in the crate that owns the
method or physics, while this crate turns their declarative column attributes
into the `FieldData` implementation.

## Example

```rust,ignore
use field_core::FieldData;

#[derive(FieldData)]
struct Conserved {
    #[forward]
    q: Vec<[f64; 5]>,
    #[reverse]
    #[zero]
    flux_residual: Vec<[f64; 5]>,
}
```

## License

MIT OR Apache-2.0
