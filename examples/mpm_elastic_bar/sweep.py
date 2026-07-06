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
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_DIR = os.path.join(SCRIPT_DIR, "sweep")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
PLOT_PATH = os.path.join(PLOTS_DIR, "transfer_conservation_errors.svg")

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


def _svg_polyline(points):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in points)


def plot_results(rows):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    metrics = [
        ("mass", 3, "#1f77b4"),
        ("momentum", 4, "#d62728"),
        ("density integral", 5, "#2ca02c"),
        ("affine gather", 6, "#9467bd"),
    ]
    width = 920
    height = 560
    left = 84
    right = 252
    top = 58
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    nx_values = [r[0] for r in rows]
    x_min = min(nx_values)
    x_max = max(nx_values)
    floor = 1.0e-18
    y_min_log = -18.0
    y_max_log = -11.0

    def sx(nx):
        if x_max == x_min:
            return left + 0.5 * plot_w
        return left + (nx - x_min) / (x_max - x_min) * plot_w

    def sy(value):
        v = max(value, floor)
        logv = math.log10(v)
        return top + (y_max_log - logv) / (y_max_log - y_min_log) * plot_h

    tol_y = sy(TOL)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">MPM elastic bar transfer errors vs 1e-12 tolerance</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#333" stroke-width="1"/>',
    ]

    for exponent in range(-18, -10):
        y = sy(10.0 ** exponent)
        stroke = "#dddddd" if exponent != -12 else "#444444"
        dash = " stroke-dasharray=\"7 5\"" if exponent == -12 else ""
        width_attr = "2" if exponent == -12 else "1"
        elements.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{left + plot_w}" y2="{y:.3f}" '
            f'stroke="{stroke}" stroke-width="{width_attr}"{dash}/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">1e{exponent}</text>'
        )

    for nx in nx_values:
        x = sx(nx)
        elements.append(f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + plot_h}" stroke="#eeeeee"/>')
        elements.append(
            f'<text x="{x:.3f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{nx}</text>'
        )

    elements.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14">grid cells in x (particles scale with resolution)</text>'
    )
    elements.append(
        f'<text x="21" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" transform="rotate(-90 21 {top + plot_h / 2:.1f})">absolute or relative error (log10)</text>'
    )
    elements.append(
        f'<text x="{left + plot_w + 12}" y="{tol_y + 4:.3f}" font-family="sans-serif" '
        f'font-size="13" font-weight="700" fill="#333">pass tolerance = 1e-12</text>'
    )

    for label, idx, color in metrics:
        points = [(sx(r[0]), sy(r[idx])) for r in rows]
        elements.append(
            f'<polyline points="{_svg_polyline(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for row, (x, y) in zip(rows, points):
            elements.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.2" fill="{color}" stroke="white" stroke-width="1.5">'
                f'<title>{label}: nx={row[0]}, particles={row[1]}, error={row[idx]:.3e}</title></circle>'
            )

    legend_x = left + plot_w + 32
    legend_y = top + 70
    elements.append(f'<text x="{legend_x}" y="{legend_y - 26}" font-family="sans-serif" font-size="15" font-weight="700">Measured error</text>')
    for i, (label, idx, color) in enumerate(metrics):
        y = legend_y + i * 30
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')
        elements.append(f'<text x="{legend_x + 38}" y="{y + 5}" font-family="sans-serif" font-size="13">{label}</text>')

    worst = max(max(r[3], r[4], r[5], r[6]) for r in rows)
    status = "PASS" if worst <= TOL else "FAIL"
    elements.append(
        f'<text x="{legend_x}" y="{legend_y + 142}" font-family="sans-serif" font-size="13">'
        f'Worst measured error: {worst:.3e}</text>'
    )
    elements.append(
        f'<text x="{legend_x}" y="{legend_y + 164}" font-family="sans-serif" font-size="13" font-weight="700">'
        f'{status}: all four checks <= {TOL:.1e}</text>'
    )
    elements.append("</svg>\n")
    with open(PLOT_PATH, "w") as f:
        f.write("\n".join(elements))
    print(f"\nWrote plot: {os.path.relpath(PLOT_PATH, REPO_ROOT)}")


def main():
    exe = build()
    rows, ok = run_cases(exe)
    plot_results(rows)
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
