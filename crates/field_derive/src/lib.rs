//! Proc-macro crate providing `#[derive(FieldData)]` for the FIELD substrate.
//!
//! The `FieldData` trait and the [`Component`] bound live in `field_core`; this
//! crate only generates the `impl`. Every field of the deriving struct must be a
//! `Vec<C>` where `C: field_core::Component` (`f64`, `[f64; N]`, `bool`, or a
//! physics type that implements `Component`).
//!
//! # Attributes (which way a column travels)
//!
//! - **`#[forward]`** — owner → ghost, *overwrite*. The everyday FVM halo: a rank
//!   needs its neighbor's state to compute boundary fluxes.
//! - **`#[reverse]`** — ghost → owner, *accumulate* (`+=`). Contributions computed
//!   on a ghost that sum back to the owner (AMR reflux, FEM-style assembly).
//!   Usually paired with `#[zero]`.
//! - **`#[zero]`** — a per-step accumulator, reset to `Component::zero()` each step.
//! - **no attribute** — a per-cell column that is resized with the mesh but never
//!   crosses a rank boundary (e.g. a backup/scratch column).
//!
//! Unlike SOIL's `AtomData`, there is **no** `pack`/`unpack`/`truncate`/
//! `swap_remove`/`apply_permutation`: mesh cells never migrate or reorder, so the
//! generated impl is just `resize` + forward/reverse/zero.
//!
//! ```rust,ignore
//! use field_core::prelude::*;
//!
//! #[derive(FieldData)]
//! struct Thermal {
//!     #[forward]
//!     temperature: Vec<f64>,
//!     #[reverse]
//!     #[zero]
//!     heat_flux: Vec<f64>,
//! }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Fields, Type};

/// A parsed struct field: its name, its `Vec<C>` element type `C`, and which
/// exchange attributes it carries.
struct FieldInfo {
    ident: syn::Ident,
    inner: Type,
    is_forward: bool,
    is_reverse: bool,
    is_zero: bool,
}

/// Extract `C` from a `Vec<C>` field type, or `None` if the field is not a `Vec`.
fn vec_inner(ty: &Type) -> Option<Type> {
    let Type::Path(tp) = ty else { return None };
    let seg = tp.path.segments.last()?;
    if seg.ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else { return None };
    match args.args.first()? {
        syn::GenericArgument::Type(inner) => Some(inner.clone()),
        _ => None,
    }
}

fn has_attr(field: &syn::Field, name: &str) -> bool {
    field.attrs.iter().any(|a| a.path().is_ident(name))
}

/// Derive `field_core::FieldData` for a struct of `Vec<C: Component>` columns.
#[proc_macro_derive(FieldData, attributes(forward, reverse, zero))]
pub fn derive_field_data(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return syn::Error::new_spanned(
                    &input,
                    "FieldData can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &input,
                "FieldData can only be derived for structs, not enums or unions",
            )
            .to_compile_error()
            .into();
        }
    };

    let mut infos = Vec::new();
    for field in fields.iter() {
        let ident = field.ident.as_ref().expect("named field").clone();
        let Some(inner) = vec_inner(&field.ty) else {
            return syn::Error::new_spanned(
                field,
                format!(
                    "FieldData: field `{ident}` must be a `Vec<C>` whose element \
                     `C` implements `field_core::Component` (e.g. `Vec<f64>`, \
                     `Vec<[f64; 3]>`, or a physics `Vec<ConsVar>`)."
                ),
            )
            .to_compile_error()
            .into();
        };
        infos.push(FieldInfo {
            ident,
            inner,
            is_forward: has_attr(field, "forward"),
            is_reverse: has_attr(field, "reverse"),
            is_zero: has_attr(field, "zero"),
        });
    }

    // resize (all columns) + len (first column, or 0 for an empty struct).
    let resize_stmts: Vec<_> = infos
        .iter()
        .map(|f| {
            let id = &f.ident;
            let inner = &f.inner;
            quote! { self.#id.resize(n, <#inner as ::field_core::Component>::zero()); }
        })
        .collect();
    let len_body = match infos.first() {
        Some(f) => {
            let id = &f.ident;
            quote! { self.#id.len() }
        }
        None => quote! { 0 },
    };

    let forward_methods = build_comm(&infos, Dir::Forward);
    let reverse_methods = build_comm(&infos, Dir::Reverse);
    let zero_method = build_zero(&infos);

    quote! {
        impl ::field_core::FieldData for #name {
            fn as_any(&self) -> &dyn ::std::any::Any { self }
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any { self }

            fn len(&self) -> usize { #len_body }

            fn resize(&mut self, n: usize) {
                #(#resize_stmts)*
            }

            #forward_methods
            #reverse_methods
            #zero_method
        }
    }
    .into()
}

enum Dir {
    Forward,
    Reverse,
}

/// Build the size / pack / unpack methods for one exchange direction. Returns an
/// empty token stream when no field is tagged for that direction, so the trait's
/// default (size 0, no-op) is used and the registry skips the store.
fn build_comm(infos: &[FieldInfo], dir: Dir) -> proc_macro2::TokenStream {
    let is_forward = matches!(dir, Dir::Forward);
    let selected: Vec<&FieldInfo> = infos
        .iter()
        .filter(|f| if is_forward { f.is_forward } else { f.is_reverse })
        .collect();
    if selected.is_empty() {
        return quote! {};
    }

    let size_terms: Vec<_> = selected
        .iter()
        .map(|f| {
            let inner = &f.inner;
            quote! { <#inner as ::field_core::Component>::WIDTH }
        })
        .collect();

    let pack_stmts: Vec<_> = selected
        .iter()
        .map(|f| {
            let id = &f.ident;
            let inner = &f.inner;
            quote! { <#inner as ::field_core::Component>::write(&self.#id[c], buf); }
        })
        .collect();

    // Each column advances a running offset by its component width.
    let unpack_stmts: Vec<_> = selected
        .iter()
        .map(|f| {
            let id = &f.ident;
            let inner = &f.inner;
            if is_forward {
                quote! {
                    self.#id[c] = <#inner as ::field_core::Component>::read(&buf[off..]);
                    off += <#inner as ::field_core::Component>::WIDTH;
                }
            } else {
                quote! {
                    <#inner as ::field_core::Component>::add_from(&mut self.#id[c], &buf[off..]);
                    off += <#inner as ::field_core::Component>::WIDTH;
                }
            }
        })
        .collect();

    if is_forward {
        quote! {
            fn forward_size(&self) -> usize { 0 #( + #size_terms )* }
            fn pack_forward(&self, c: usize, buf: &mut Vec<f64>) {
                #(#pack_stmts)*
            }
            fn unpack_forward(&mut self, c: usize, buf: &[f64]) -> usize {
                let mut off = 0usize;
                #(#unpack_stmts)*
                off
            }
        }
    } else {
        quote! {
            fn reverse_size(&self) -> usize { 0 #( + #size_terms )* }
            fn pack_reverse(&self, c: usize, buf: &mut Vec<f64>) {
                #(#pack_stmts)*
            }
            fn unpack_reverse_add(&mut self, c: usize, buf: &[f64]) -> usize {
                let mut off = 0usize;
                #(#unpack_stmts)*
                off
            }
        }
    }
}

/// Build `zero_cell` for `#[zero]` columns (empty when none are tagged).
fn build_zero(infos: &[FieldInfo]) -> proc_macro2::TokenStream {
    let zero_fields: Vec<&FieldInfo> = infos.iter().filter(|f| f.is_zero).collect();
    if zero_fields.is_empty() {
        return quote! {};
    }
    let stmts: Vec<_> = zero_fields
        .iter()
        .map(|f| {
            let id = &f.ident;
            let inner = &f.inner;
            quote! { self.#id[c] = <#inner as ::field_core::Component>::zero(); }
        })
        .collect();
    quote! {
        fn zero_cell(&mut self, c: usize) {
            #(#stmts)*
        }
    }
}
