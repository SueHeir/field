//! A diffusion kernel against the same stencil on the CPU, through real halo exchange.

mod common;

use std::sync::Mutex;

use common::{compare, heat_app};
use grass_gpu::{GpuDevice, GpuReal};

/// The device's transfer counts are shared by the process, so tests take turns.
static TURN: Mutex<()> = Mutex::new(());

const STEPS: usize = 100;
/// Single-precision rounding over this many steps, on values near 2.
const F32_BAND: GpuReal = 2e-4;

#[test]
fn periodic_mesh_matches_the_cpu_through_the_halo() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = GpuDevice::default_device() {
        eprintln!("skipping: {e}");
        return;
    }
    let mut app = heat_app(16, true);
    let (_, total_start) = compare(&app);
    for _ in 0..STEPS {
        app.run();
    }
    let (max_diff, total) = compare(&app);
    assert!(max_diff < F32_BAND, "GPU and CPU differ by {max_diff}");
    // A periodic box conserves heat, which a wrong halo would break.
    assert!(
        ((total - total_start) / total_start).abs() < 1e-5,
        "{total_start} became {total}"
    );
}

#[test]
fn a_mesh_with_no_links_stays_on_the_gpu() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(device) = GpuDevice::default_device() else {
        return;
    };
    let mut app = heat_app(16, false);
    app.run();
    let before = device.counts().snapshot();
    for _ in 0..10 {
        app.run();
    }
    let moved = device.counts().snapshot().since(&before);
    assert_eq!((moved.uploads, moved.downloads), (0, 0));
    let (max_diff, _) = compare(&app);
    assert!(max_diff < F32_BAND, "GPU and CPU differ by {max_diff}");
}

#[test]
fn a_periodic_mesh_transfers_once_each_way_per_step() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(device) = GpuDevice::default_device() else {
        return;
    };
    let mut app = heat_app(8, true);
    app.run();
    let before = device.counts().snapshot();
    for _ in 0..10 {
        app.run();
    }
    let moved = device.counts().snapshot().since(&before);
    assert_eq!((moved.uploads, moved.downloads), (10, 10));
}
