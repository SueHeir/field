#!/usr/bin/env python3
"""
mpm_elastic_bar — transfer-conservation gate for the MPM hybrid demonstrator.

The binary scatters an elastic bar's material-point mass and momentum to a FIELD
mesh and gathers an affine grid velocity back to the particles. The gate checks
the transfer identities that make PIC/MPM hybrids reusable:

  * particle mass equals summed grid mass;
  * particle momentum equals summed grid momentum;
  * density deposit preserves the volume integral;
  * grid-to-particle gather is exact for affine velocity fields.

These are mathematical consequences of a partition-of-unity CIC kernel, so the
tolerance is tight and independent of the chosen particle/grid resolution.
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_DIR = os.path.join(SCRIPT_DIR, "sweep")

CASES = [
    (32, 256),
    (64, 512),
    (96, 768),
]

TOL = 1.0e-12

RESULT_RE = re.compile(
    r"particles=(\d+)\s+cells=(\d+)\s+mass=([-\d.eE+]+)\s+grid_mass=([-\d.eE+]+)\s+"
    r"mass_rel_err=([-\d.eE+]+)\s+momentum=([-\d.eE+]+)\s+grid_momentum=([-\d.eE+]+)\s+"
    r"momentum_abs_err=([-\d.eE+]+)\s+density_integral=([-\d.eE+]+)\s+"
    r"density_rel_err=([-\d.eE+]+)\s+affine_max_err=([-\d.eE+]+)\s+"
    r"elastic_energy=([-\d.eE+]+)"
)


def sh(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kw)


def build():
    sh(["cargo", "build", "--release", "-q", "-p", "mpm_elastic_bar"])
    exe = os.path.join(REPO_ROOT, "target", "release", "mpm_elastic_bar")
    if not os.path.exists(exe):
        print(f"CHECKS FAILED: binary not found at {exe}")
        sys.exit(2)
    return exe


def config_for(nx, particles):
    return f"""[grid]
nx = {nx}
ny = 2
nz = 2
ng = 2
bounds_lo = [0.0, 0.0, 0.0]
bounds_hi = [1.0, 0.1, 0.1]

[particles]
count = {particles}
x_min = 0.15
x_max = 0.85
area = 0.01
density = 1.0
base_velocity = 0.08
velocity_amplitude = 0.02
strain_amplitude = 0.01

[material]
young_modulus = 1000.0

[validation]
conservation_tol = {TOL}
affine_tol = {TOL}
"""


def run_cases(exe):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    rows = []
    ok = True
    for nx, n_particles in CASES:
        cfg_path = os.path.join(SWEEP_DIR, f"nx{nx}_np{n_particles}.toml")
        with open(cfg_path, "w") as f:
            f.write(config_for(nx, n_particles))
        out = subprocess.run(
            [exe, cfg_path],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        m = RESULT_RE.search(out)
        if not m:
            print(f"CHECKS FAILED: could not parse RESULT for nx={nx}, particles={n_particles}:\n{out}")
            sys.exit(2)
        particles = int(m.group(1))
        cells = int(m.group(2))
        mass_rel = float(m.group(5))
        momentum_abs = float(m.group(8))
        density_rel = float(m.group(10))
        affine = float(m.group(11))
        energy = float(m.group(12))
        rows.append((nx, particles, cells, mass_rel, momentum_abs, density_rel, affine, energy))
        case_ok = mass_rel <= TOL and momentum_abs <= TOL and density_rel <= TOL and affine <= TOL
        ok = ok and case_ok
        status = "ok" if case_ok else "FAIL"
        print(
            f"  nx={nx:3d} particles={particles:4d} cells={cells:5d} "
            f"mass_rel={mass_rel:.3e} momentum_abs={momentum_abs:.3e} "
            f"density_rel={density_rel:.3e} affine={affine:.3e} "
            f"E_elastic={energy:.6e} [{status}]"
        )
    return rows, ok


def main():
    exe = build()
    rows, ok = run_cases(exe)
    worst_mass = max(r[3] for r in rows)
    worst_momentum = max(r[4] for r in rows)
    worst_density = max(r[5] for r in rows)
    worst_affine = max(r[6] for r in rows)
    print(
        "\nWorst errors: "
        f"mass_rel={worst_mass:.3e}, momentum_abs={worst_momentum:.3e}, "
        f"density_rel={worst_density:.3e}, affine={worst_affine:.3e} "
        f"(tolerance {TOL:.1e})"
    )
    if ok:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    print("\nCHECKS FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
