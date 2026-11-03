//! Diffusion on the GPU and the CPU side by side, under MPI, with real halo exchange.
//!
//! ```sh
//! cargo build -p field_gpu --features mpi_backend --example gpu_heat --release
//! mpirun -n 4 target/release/examples/gpu_heat
//! ```

#[path = "../tests/common/mod.rs"]
mod common;

use common::{compare, heat_app};
use grass_app::prelude::*;
use grass_mpi::CommResource;

fn main() {
    let mut app = heat_app(32, true);
    let (_, total_start) = compare(&app);
    let steps = 200;
    for _ in 0..steps {
        app.run();
    }
    let (max_diff, total) = compare(&app);
    let comm = app.get_resource_ref::<CommResource>().unwrap();
    let total_start = comm.0.all_reduce_sum_f64(total_start);
    let total = comm.0.all_reduce_sum_f64(total);
    println!(
        "rank {} of {}: {steps} steps, max |GPU - CPU| = {max_diff:.2e}, total heat {total_start:.3} -> {total:.3}",
        comm.0.rank(),
        comm.0.size(),
    );
}
