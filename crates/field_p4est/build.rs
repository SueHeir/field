//! Compiles the p4est C shim and links the prebuilt p4est + sc static libraries.
//!
//! The p4est install is located via the `P4EST_PREFIX` env var (a directory
//! containing `include/` and `lib/`). If unset, it falls back to the prebuilt
//! install shipped alongside toy-cfd (`../toy-cfd/third_party/p4est-install`),
//! which is how this dev machine is set up. A reproducible build should set
//! `P4EST_PREFIX` (or vendor a `third_party/p4est-install` and point at it).

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // .../field/crates/field_p4est -> .../field -> .../<GitHub>
    let github = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("manifest dir has three ancestors");
    let default_prefix = github
        .join("toy-cfd")
        .join("third_party")
        .join("p4est-install");

    let p4est_prefix = std::env::var("P4EST_PREFIX")
        .map(PathBuf::from)
        .unwrap_or(default_prefix);
    let p4est_include = p4est_prefix.join("include");
    let p4est_lib = p4est_prefix.join("lib");

    if !p4est_include.exists() {
        panic!(
            "p4est headers not found at {}.\n\
             Set P4EST_PREFIX to a p4est install directory (containing include/ and lib/).",
            p4est_include.display()
        );
    }

    // Ask pkg-config for the active Open MPI installation. Keep the historical
    // Homebrew paths as a fallback for machines where `ompi.pc` is absent.
    let mpi = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("ompi")
        .ok();
    let mpi_includes = mpi
        .as_ref()
        .map(|lib| lib.include_paths.clone())
        .unwrap_or_else(|| vec![PathBuf::from("/opt/homebrew/Cellar/open-mpi/5.0.9/include")]);
    let mpi_libs = mpi
        .as_ref()
        .map(|lib| lib.link_paths.clone())
        .unwrap_or_else(|| vec![PathBuf::from("/opt/homebrew/Cellar/open-mpi/5.0.9/lib")]);

    let mut build = cc::Build::new();
    build.file("csrc/shim.c").include(&p4est_include);
    for include in &mpi_includes {
        build.include(include);
    }
    build
        .flag_if_supported("-Wno-unused-parameter")
        .compile("field_p4est_shim");

    println!("cargo:rustc-link-search=native={}", p4est_lib.display());
    for lib in mpi_libs {
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
    println!("cargo:rustc-link-lib=static=p4est");
    println!("cargo:rustc-link-lib=static=sc");
    println!("cargo:rustc-link-lib=mpi");
    println!("cargo:rustc-link-lib=z");
    println!("cargo:rustc-link-lib=m");

    println!("cargo:rerun-if-changed=csrc/shim.c");
    println!("cargo:rerun-if-changed=csrc/shim.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=P4EST_PREFIX");
}
