//! mpm_elastic_bar — a particle <-> mesh transfer proof in MPM clothing.
//!
//! The physics state here is deliberately example-local: particles carry mass,
//! velocity, strain, stress, and volume for a 1-D elastic bar. The reusable
//! operation is FIELD's `field_transfer` scatter/gather primitive, which sees
//! only particle positions, scalar columns, and a mesh implementing
//! `TransferMesh`.

use std::env;

use field_core::{FvMesh, UniformMesh, UniformMeshConfig};
use field_transfer::{gather, scatter, scatter_density, TransferMesh};
use grass_io::{load_toml, Config};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct ParticleConfig {
    count: usize,
    x_min: f64,
    x_max: f64,
    area: f64,
    density: f64,
    base_velocity: f64,
    velocity_amplitude: f64,
    strain_amplitude: f64,
}

impl Default for ParticleConfig {
    fn default() -> Self {
        Self {
            count: 512,
            x_min: 0.15,
            x_max: 0.85,
            area: 0.01,
            density: 1.0,
            base_velocity: 0.08,
            velocity_amplitude: 0.02,
            strain_amplitude: 0.01,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct MaterialConfig {
    young_modulus: f64,
}

impl Default for MaterialConfig {
    fn default() -> Self {
        Self {
            young_modulus: 1000.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct ValidationConfig {
    conservation_tol: f64,
    affine_tol: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            conservation_tol: 1.0e-12,
            affine_tol: 1.0e-12,
        }
    }
}

#[derive(Clone, Debug)]
struct Particle {
    pos: [f64; 3],
    mass: f64,
    volume: f64,
    velocity: f64,
    strain: f64,
    stress: f64,
}

#[derive(Debug)]
struct Metrics {
    particles: usize,
    grid_cells: usize,
    total_mass: f64,
    grid_mass: f64,
    mass_rel_err: f64,
    total_momentum: f64,
    grid_momentum: f64,
    momentum_abs_err: f64,
    density_integral: f64,
    density_rel_err: f64,
    affine_max_err: f64,
    elastic_energy: f64,
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: mpm_elastic_bar <config.toml>");
        std::process::exit(2);
    });
    let cfg = Config::from_table(load_toml(&path));
    let grid: UniformMeshConfig = cfg.section("grid");
    let particles: ParticleConfig = cfg.section("particles");
    let material: MaterialConfig = cfg.section("material");
    let validation: ValidationConfig = cfg.section("validation");

    let mesh = UniformMesh::from_config(&grid);
    let particles = make_elastic_bar(&grid, &particles, &material);
    let metrics = run_transfer_checks(&mesh, &particles);

    println!(
        "RESULT particles={} cells={} mass={:.16e} grid_mass={:.16e} mass_rel_err={:.3e} \
         momentum={:.16e} grid_momentum={:.16e} momentum_abs_err={:.3e} \
         density_integral={:.16e} density_rel_err={:.3e} affine_max_err={:.3e} \
         elastic_energy={:.16e}",
        metrics.particles,
        metrics.grid_cells,
        metrics.total_mass,
        metrics.grid_mass,
        metrics.mass_rel_err,
        metrics.total_momentum,
        metrics.grid_momentum,
        metrics.momentum_abs_err,
        metrics.density_integral,
        metrics.density_rel_err,
        metrics.affine_max_err,
        metrics.elastic_energy
    );

    let mut ok = true;
    if metrics.mass_rel_err > validation.conservation_tol {
        eprintln!(
            "CHECKS FAILED: mass conservation rel_err {:.3e} > {:.3e}",
            metrics.mass_rel_err, validation.conservation_tol
        );
        ok = false;
    }
    if metrics.momentum_abs_err
        > validation.conservation_tol * metrics.total_momentum.abs().max(1.0)
    {
        eprintln!(
            "CHECKS FAILED: momentum conservation abs_err {:.3e} exceeds scaled tolerance",
            metrics.momentum_abs_err
        );
        ok = false;
    }
    if metrics.density_rel_err > validation.conservation_tol {
        eprintln!(
            "CHECKS FAILED: density volume integral rel_err {:.3e} > {:.3e}",
            metrics.density_rel_err, validation.conservation_tol
        );
        ok = false;
    }
    if metrics.affine_max_err > validation.affine_tol {
        eprintln!(
            "CHECKS FAILED: affine gather max_err {:.3e} > {:.3e}",
            metrics.affine_max_err, validation.affine_tol
        );
        ok = false;
    }

    if ok {
        println!("ALL CHECKS PASSED");
    } else {
        std::process::exit(1);
    }
}

fn make_elastic_bar(
    grid: &UniformMeshConfig,
    particles: &ParticleConfig,
    material: &MaterialConfig,
) -> Vec<Particle> {
    assert!(particles.count > 0, "particles.count must be positive");
    assert!(
        particles.x_min < particles.x_max,
        "particles.x_min must be < x_max"
    );
    assert!(particles.area > 0.0, "particles.area must be positive");
    assert!(
        particles.density > 0.0,
        "particles.density must be positive"
    );
    assert!(
        material.young_modulus > 0.0,
        "material.young_modulus must be positive"
    );

    let y = 0.5 * (grid.bounds_lo[1] + grid.bounds_hi[1]);
    let z = 0.5 * (grid.bounds_lo[2] + grid.bounds_hi[2]);
    let length = particles.x_max - particles.x_min;
    let dxp = length / particles.count as f64;
    let volume = particles.area * dxp;
    let mass = particles.density * volume;

    (0..particles.count)
        .map(|p| {
            let xi = (p as f64 + 0.5) / particles.count as f64;
            let x = particles.x_min + length * xi;
            let mode = (std::f64::consts::PI * xi).sin();
            let strain_mode = (std::f64::consts::PI * xi).cos();
            let velocity = particles.base_velocity + particles.velocity_amplitude * mode;
            let strain = particles.strain_amplitude * strain_mode;
            let stress = material.young_modulus * strain;
            Particle {
                pos: [x, y, z],
                mass,
                volume,
                velocity,
                strain,
                stress,
            }
        })
        .collect()
}

fn run_transfer_checks<M: FvMesh + TransferMesh + ?Sized>(
    mesh: &M,
    particles: &[Particle],
) -> Metrics {
    let positions: Vec<[f64; 3]> = particles.iter().map(|p| p.pos).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let momenta: Vec<f64> = particles.iter().map(|p| p.mass * p.velocity).collect();

    let mut grid_mass = vec![0.0; mesh.n_cells()];
    let mass_stats = scatter(mesh, &positions, &masses, &mut grid_mass);
    assert_eq!(
        mass_stats.skipped, 0,
        "all bar particles must be inside the transfer mesh"
    );

    let mut grid_momentum = vec![0.0; mesh.n_cells()];
    let momentum_stats = scatter(mesh, &positions, &momenta, &mut grid_momentum);
    assert_eq!(
        momentum_stats.skipped, 0,
        "all bar particles must be inside the transfer mesh"
    );

    let mut density = vec![0.0; mesh.n_cells()];
    let density_stats = scatter_density(mesh, &positions, &masses, &mut density);
    assert_eq!(
        density_stats.skipped, 0,
        "all bar particles must be inside the transfer mesh"
    );

    // Cell-centered affine velocity field. CIC gather is linearly exact, so this
    // is the cell-to-particle half of the transfer validation.
    let affine = |x: [f64; 3]| 0.12 + 0.7 * x[0] - 0.2 * x[1] + 0.05 * x[2];
    let affine_cells: Vec<f64> = (0..mesh.n_cells())
        .map(|c| affine(mesh.cell_centroid(c)))
        .collect();
    let mut gathered = vec![0.0; particles.len()];
    let gather_stats = gather(mesh, &positions, &affine_cells, &mut gathered, f64::NAN);
    assert_eq!(
        gather_stats.skipped, 0,
        "all bar particles must gather from the transfer mesh"
    );

    let total_mass: f64 = masses.iter().sum();
    let grid_mass_total: f64 = grid_mass.iter().sum();
    let total_momentum: f64 = momenta.iter().sum();
    let grid_momentum_total: f64 = grid_momentum.iter().sum();
    let density_integral: f64 = (0..mesh.n_cells())
        .map(|c| density[c] * FvMesh::cell_volume(mesh, c))
        .sum();
    let affine_max_err = positions
        .iter()
        .zip(gathered.iter())
        .map(|(&x, &v)| (v - affine(x)).abs())
        .fold(0.0, f64::max);
    let elastic_energy: f64 = particles
        .iter()
        .map(|p| 0.5 * p.stress * p.strain * p.volume)
        .sum();

    Metrics {
        particles: particles.len(),
        grid_cells: mesh.n_cells(),
        total_mass,
        grid_mass: grid_mass_total,
        mass_rel_err: (grid_mass_total - total_mass).abs() / total_mass,
        total_momentum,
        grid_momentum: grid_momentum_total,
        momentum_abs_err: (grid_momentum_total - total_momentum).abs(),
        density_integral,
        density_rel_err: (density_integral - total_mass).abs() / total_mass,
        affine_max_err,
        elastic_energy,
    }
}
