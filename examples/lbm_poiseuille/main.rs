//! lbm_poiseuille — D2Q9 lattice-Boltzmann flow on the FIELD substrate.
//!
//! # What this proves
//!
//! FIELD is not limited to Godunov finite-volume physics. This example keeps the
//! whole D2Q9 BGK lattice-Boltzmann method in `examples/` while using FIELD for
//! exactly the substrate-level pieces a local-reach mesh method needs:
//!
//! * `UniformMesh` provides the structured, ghost-inclusive indexing;
//! * `FieldData` stores the per-cell distribution functions without teaching the
//!   substrate what an LBM population is;
//! * the GRASS schedule drives boundary filling, collide, and stream phases.
//!
//! The validation case is force-driven plane Poiseuille flow with periodic x and
//! halfway bounce-back walls. At steady state the profile is the analytic
//! parabola `u(y) = g y (H-y) / (2 nu)`, where `nu = (tau - 1/2)/3`.

use std::any::TypeId;

use field_core::prelude::*;
use grass_app::prelude::*;
use grass_io::{Config, InputPlugin};
use grass_scheduler::{Res, ResMut};
use serde::Deserialize;

const Q: usize = 9;
const CX: [isize; Q] = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const CY: [isize; Q] = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const W: [f64; Q] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
const OPP: [usize; Q] = [0, 3, 4, 1, 2, 7, 8, 5, 6];

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct LbmConfig {
    steps: usize,
    tau: f64,
    force_x: f64,
}

impl Default for LbmConfig {
    fn default() -> Self {
        Self {
            steps: 16_000,
            tau: 0.8,
            force_x: 1.0e-6,
        }
    }
}

impl LbmConfig {
    fn viscosity(&self) -> f64 {
        (self.tau - 0.5) / 3.0
    }
}

#[derive(FieldData, Default)]
struct LbmFields {
    #[forward]
    f: Vec<[f64; Q]>,
    f_post: Vec<[f64; Q]>,
}

#[derive(Default)]
struct LbmState {
    initialized: bool,
    step: usize,
}

#[derive(Default)]
struct LbmMetrics {
    l2_relative: f64,
    linf_abs: f64,
    ux_max: f64,
    ux_exact_max: f64,
    mass_initial: f64,
    mass_final: f64,
}

fn equilibrium(rho: f64, ux: f64, uy: f64) -> [f64; Q] {
    let usq = ux * ux + uy * uy;
    let mut out = [0.0; Q];
    for q in 0..Q {
        let eu = CX[q] as f64 * ux + CY[q] as f64 * uy;
        out[q] = W[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
    }
    out
}

fn macroscopic(pop: &[f64; Q], force_x: f64) -> (f64, f64, f64) {
    let rho: f64 = pop.iter().sum();
    let mut mx = 0.0;
    let mut my = 0.0;
    for q in 0..Q {
        mx += pop[q] * CX[q] as f64;
        my += pop[q] * CY[q] as f64;
    }
    (rho, mx / rho + 0.5 * force_x / rho, my / rho)
}

fn initialize(
    mesh: Res<UniformMesh>,
    registry: Res<FieldRegistry>,
    cfg: Res<LbmConfig>,
    mut state: ResMut<LbmState>,
    mut metrics: ResMut<LbmMetrics>,
) {
    if state.initialized {
        return;
    }
    assert!(
        cfg.tau > 0.5,
        "D2Q9 BGK requires tau > 0.5 for positive viscosity"
    );
    let dims = mesh.dims();
    assert_eq!(dims[2], 1, "lbm_poiseuille is a 2-D D2Q9 example");
    assert!(mesh.n_ghost() >= 1, "D2Q9 streaming needs one ghost layer");

    let mut fields = registry.expect_mut::<LbmFields>("LbmFields must be registered");
    let feq = equilibrium(1.0, 0.0, 0.0);
    for c in 0..mesh.n_cells_total() {
        fields.f[c] = feq;
        fields.f_post[c] = feq;
    }
    state.initialized = true;
    metrics.mass_initial = total_mass(&mesh, &fields);
}

fn fill_periodic_x_ghosts(mesh: Res<UniformMesh>, registry: Res<FieldRegistry>) {
    let dims = mesh.dims();
    let ng = mesh.n_ghost();
    let mut fields = registry.expect_mut::<LbmFields>("LbmFields must be registered");

    for j in 0..dims[1] {
        let jr = j + ng;
        let lo_ghost = mesh.idx_raw(ng - 1, jr, ng);
        let hi_ghost = mesh.idx_raw(ng + dims[0], jr, ng);
        let lo_src = mesh.idx(dims[0] - 1, j, 0);
        let hi_src = mesh.idx(0, j, 0);
        fields.f[lo_ghost] = fields.f[lo_src];
        fields.f[hi_ghost] = fields.f[hi_src];
    }
}

fn collide(mesh: Res<UniformMesh>, registry: Res<FieldRegistry>, cfg: Res<LbmConfig>) {
    let dims = mesh.dims();
    let omega = 1.0 / cfg.tau;
    let force_x = cfg.force_x;
    let mut fields = registry.expect_mut::<LbmFields>("LbmFields must be registered");

    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let c = mesh.idx(i, j, 0);
            let pop = fields.f[c];
            let (rho, ux, uy) = macroscopic(&pop, force_x);
            let feq = equilibrium(rho, ux, uy);
            let mut post = [0.0; Q];
            for q in 0..Q {
                let ex = CX[q] as f64;
                let ey = CY[q] as f64;
                let eu = ex * ux + ey * uy;
                let guo = W[q]
                    * (1.0 - 0.5 * omega)
                    * (3.0 * (ex - ux) * force_x + 9.0 * eu * ex * force_x);
                post[q] = pop[q] - omega * (pop[q] - feq[q]) + guo;
            }
            fields.f_post[c] = post;
        }
    }
}

fn stream(mesh: Res<UniformMesh>, registry: Res<FieldRegistry>, mut state: ResMut<LbmState>) {
    let dims = mesh.dims();
    let ng = mesh.n_ghost();
    let mut fields = registry.expect_mut::<LbmFields>("LbmFields must be registered");

    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let ir = i + ng;
            let jr = j + ng;
            let c = mesh.idx_raw(ir, jr, ng);
            let mut next = [0.0; Q];
            for q in 0..Q {
                let src_jr = jr as isize - CY[q];
                if src_jr < ng as isize || src_jr >= (ng + dims[1]) as isize {
                    next[q] = fields.f_post[c][OPP[q]];
                } else {
                    let mut src_ir = ir as isize - CX[q];
                    if src_ir < ng as isize {
                        src_ir += dims[0] as isize;
                    } else if src_ir >= (ng + dims[0]) as isize {
                        src_ir -= dims[0] as isize;
                    }
                    let src = mesh.idx_raw(src_ir as usize, src_jr as usize, ng);
                    next[q] = fields.f_post[src][q];
                }
            }
            fields.f[c] = next;
        }
    }
    state.step += 1;
}

fn total_mass(mesh: &UniformMesh, fields: &LbmFields) -> f64 {
    let dims = mesh.dims();
    let mut mass = 0.0;
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            mass += fields.f[mesh.idx(i, j, 0)].iter().sum::<f64>();
        }
    }
    mass
}

fn validate(mesh: &UniformMesh, fields: &LbmFields, cfg: &LbmConfig, metrics: &mut LbmMetrics) {
    let dims = mesh.dims();
    let ny = dims[1];
    let height = ny as f64;
    let nu = cfg.viscosity();
    let mut l2 = 0.0;
    let mut l2_ref = 0.0;
    let mut linf = 0.0f64;
    let mut ux_max = f64::NEG_INFINITY;
    let mut ux_exact_max = f64::NEG_INFINITY;

    for j in 0..ny {
        let mut ux_sum = 0.0;
        for i in 0..dims[0] {
            let (_, ux, _) = macroscopic(&fields.f[mesh.idx(i, j, 0)], cfg.force_x);
            ux_sum += ux;
        }
        let ux = ux_sum / dims[0] as f64;
        let y = j as f64 + 0.5;
        let exact = cfg.force_x * y * (height - y) / (2.0 * nu);
        let err = ux - exact;
        l2 += err * err;
        l2_ref += exact * exact;
        linf = linf.max(err.abs());
        ux_max = ux_max.max(ux);
        ux_exact_max = ux_exact_max.max(exact);
    }

    metrics.l2_relative = (l2 / l2_ref).sqrt();
    metrics.linf_abs = linf;
    metrics.ux_max = ux_max;
    metrics.ux_exact_max = ux_exact_max;
    metrics.mass_final = total_mass(mesh, fields);
}

fn main() {
    let mut app = App::new();
    app.add_plugins(InputPlugin);

    let (grid, lbm): (UniformMeshConfig, LbmConfig) = {
        let cfg = app
            .get_resource_ref::<Config>()
            .expect("InputPlugin installs a Config resource");
        (
            cfg.section::<UniformMeshConfig>("grid"),
            cfg.section::<LbmConfig>("lbm"),
        )
    };
    app.add_resource(lbm.clone());
    app.add_resource(LbmState::default());
    app.add_resource(LbmMetrics::default());
    app.add_plugins(FieldDefaultPlugins { mesh: grid });
    register_field_data!(app, LbmFields::default());

    app.add_update_system(initialize, MeshScheduleSet::Setup);
    app.add_update_system(fill_periodic_x_ghosts, MeshScheduleSet::Boundary);
    app.add_update_system(collide, MeshScheduleSet::Flux);
    app.add_update_system(stream, MeshScheduleSet::Update);

    app.prepare();
    for _ in 0..lbm.steps {
        app.run();
    }
    app.run_cleanup();

    {
        let mesh = app
            .get_resource_ref::<UniformMesh>()
            .expect("UniformMesh resource");
        let registry = app
            .get_resource_ref::<FieldRegistry>()
            .expect("FieldRegistry resource");
        let fields = registry.expect::<LbmFields>("LbmFields must be registered");
        let metrics_cell = app
            .resource_cell(TypeId::of::<LbmMetrics>())
            .expect("LbmMetrics resource");
        let mut metrics_borrow = metrics_cell.borrow_mut();
        let metrics = metrics_borrow
            .downcast_mut::<LbmMetrics>()
            .expect("LbmMetrics downcast");
        validate(&mesh, &fields, &lbm, metrics);
    }

    let mesh = app
        .get_resource_ref::<UniformMesh>()
        .expect("UniformMesh resource");
    let state = app
        .get_resource_ref::<LbmState>()
        .expect("LbmState resource");
    let metrics = app
        .get_resource_ref::<LbmMetrics>()
        .expect("LbmMetrics resource");
    let mass_drift = (metrics.mass_final - metrics.mass_initial).abs() / metrics.mass_initial;
    let dims = mesh.dims();
    println!(
        "RESULT nx={} ny={} steps={} tau={:.6} nu={:.8e} force_x={:.8e} \
         ux_max={:.8e} ux_exact_max={:.8e} l2_relative={:.8e} linf_abs={:.8e} \
         mass_drift={:.8e}",
        dims[0],
        dims[1],
        state.step,
        lbm.tau,
        lbm.viscosity(),
        lbm.force_x,
        metrics.ux_max,
        metrics.ux_exact_max,
        metrics.l2_relative,
        metrics.linf_abs,
        mass_drift,
    );

    let registry = app
        .get_resource_ref::<FieldRegistry>()
        .expect("FieldRegistry resource");
    let fields = registry.expect::<LbmFields>("LbmFields must be registered");
    let height = dims[1] as f64;
    for j in 0..dims[1] {
        let mut ux_sum = 0.0;
        for i in 0..dims[0] {
            let (_, ux, _) = macroscopic(&fields.f[mesh.idx(i, j, 0)], lbm.force_x);
            ux_sum += ux;
        }
        let y = j as f64 + 0.5;
        let ux = ux_sum / dims[0] as f64;
        let ux_exact = lbm.force_x * y * (height - y) / (2.0 * lbm.viscosity());
        println!(
            "PROFILE j={} y={:.8e} ux={:.8e} ux_exact={:.8e} abs_error={:.8e}",
            j,
            y,
            ux,
            ux_exact,
            (ux - ux_exact).abs(),
        );
    }
}
