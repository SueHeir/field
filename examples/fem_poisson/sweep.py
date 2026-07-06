#!/usr/bin/env python3
"""
fem_poisson — mesh-refinement convergence gate for the FEM Poisson proof.

The `fem_poisson` example solves -∇²u = f with u = 0 on ∂Ω = (0,1)² by P1 finite
elements, assembling a sparse stiffness matrix on the FIELD mesh and performing a
single global sparse-Cholesky solve inside the GRASS schedule (no timestepping).
Against the manufactured solution u = sin(πx) sin(πy) the P1 discretisation is
second-order accurate in L², so the discrete error must fall like O(h²) as the
mesh is refined.

This driver runs the compiled binary across a resolution ladder, reads the
machine-readable RESULT line each run prints, and PASS/FAIL-gates the observed
convergence order (Richardson self-convergence between successive halvings).

    python3 examples/fem_poisson/sweep.py            # build + run + gate

Gate (none of these weakens the physics — P1's theoretical L² order is exactly 2):
  * L2 error strictly decreases with refinement;
  * every successive observed order p = log2(e_coarse / e_fine) lies in
    [ORDER_LO, ORDER_HI] around the theoretical 2.0;
  * the mean observed order is ≥ ORDER_MEAN_MIN.

Exit code 0 and "ALL CHECKS PASSED" on success; non-zero and "CHECKS FAILED"
otherwise (the run-bench harness contract).

Reference: the method of manufactured solutions; C. Johnson, "Numerical Solution
of Partial Differential Equations by the Finite Element Method" (P1 elements are
O(h²) in L²).
"""

import math
import os
import re
import subprocess
import sys
import xml.sax.saxutils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_DIR = os.path.join(SCRIPT_DIR, "sweep")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
CONVERGENCE_PLOT = os.path.join(PLOTS_DIR, "l2_convergence.svg")

# Resolution ladder (each a halving of h). Kept modest so the whole gate runs in
# a few seconds while still exposing four independent order estimates.
RESOLUTIONS = [16, 32, 64, 128]

# Convergence gate bounds around P1's theoretical L² order of 2.0.
ORDER_LO = 1.8
ORDER_HI = 2.2
ORDER_MEAN_MIN = 1.9

RESULT_RE = re.compile(
    r"nx=(\d+)\s+ny=(\d+)\s+h=([-\d.eE+]+)\s+n_dof=(\d+)\s+nnz=(\d+)\s+"
    r"l2_error=([-\d.eE+]+)\s+linf_error=([-\d.eE+]+)"
)


def sh(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kw)


def config_for(n):
    return (
        "[grid]\n"
        f"nx = {n}\n"
        f"ny = {n}\n"
        "nz = 1\n"
        "ng = 1\n"
        "bounds_lo = [0.0, 0.0, 0.0]\n"
        "bounds_hi = [1.0, 1.0, 1.0]\n"
    )


def build():
    sh(["cargo", "build", "--release", "-q", "-p", "fem_poisson"])
    exe = os.path.join(REPO_ROOT, "target", "release", "fem_poisson")
    if not os.path.exists(exe):
        print(f"CHECKS FAILED: binary not found at {exe}")
        sys.exit(2)
    return exe


def run_ladder(exe):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    results = []
    for n in RESOLUTIONS:
        cfg_path = os.path.join(SWEEP_DIR, f"n{n}.toml")
        with open(cfg_path, "w") as f:
            f.write(config_for(n))
        out = subprocess.run(
            [exe, cfg_path], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        m = RESULT_RE.search(out)
        if not m:
            print(f"CHECKS FAILED: could not parse RESULT for n={n}:\n{out}")
            sys.exit(2)
        h = float(m.group(3))
        l2 = float(m.group(6))
        linf = float(m.group(7))
        ndof = int(m.group(4))
        nnz = int(m.group(5))
        results.append((n, h, ndof, nnz, l2, linf))
        print(f"  n={n:4d}  h={h:.4e}  dof={ndof:6d}  nnz={nnz:7d}  "
              f"L2={l2:.6e}  Linf={linf:.6e}")
    return results


def gate(results):
    ok = True
    print("\nConvergence (P1 FEM, manufactured u = sin(pi x) sin(pi y)):")
    l2s = [r[4] for r in results]

    # 1) monotone decrease
    for a, b in zip(l2s, l2s[1:]):
        if not (b < a):
            print(f"  FAIL: L2 error did not decrease: {a:.3e} -> {b:.3e}")
            ok = False

    # 2) per-step observed order
    orders = []
    for (na, ha, *_ , l2a, _), (nb, hb, *_, l2b, _) in zip(results, results[1:]):
        p = math.log(l2a / l2b) / math.log(ha / hb)
        orders.append(p)
        status = "ok" if (ORDER_LO <= p <= ORDER_HI) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  n {na:4d} -> {nb:4d}:  observed L2 order p = {p:.3f}   [{status}]")

    mean_p = sum(orders) / len(orders)
    print(f"\n  mean observed order = {mean_p:.3f} "
          f"(theory 2.000; band [{ORDER_LO}, {ORDER_HI}], mean >= {ORDER_MEAN_MIN})")
    if mean_p < ORDER_MEAN_MIN:
        print("  FAIL: mean observed order below threshold")
        ok = False

    print(f"\n{len(orders)}/{len(orders)} order checks evaluated; "
          f"finest L2 error = {l2s[-1]:.3e}")
    return ok


def svg_text(x, y, value, size=14, anchor="start", weight="normal", fill="#1f2937"):
    escaped = xml.sax.saxutils.escape(str(value))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{escaped}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke="#334155", width=1.5, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def polyline(points, stroke, width=2.5, dash=None, fill="none"):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def write_convergence_plot(results, ok):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    hs = [r[1] for r in results]
    l2s = [r[4] for r in results]
    orders = [
        math.log(a[4] / b[4]) / math.log(a[1] / b[1])
        for a, b in zip(results, results[1:])
    ]
    mean_p = sum(orders) / len(orders)

    theory = [l2s[0] * (h / hs[0]) ** 2.0 for h in hs]
    band_hi = [l2s[0] * (h / hs[0]) ** ORDER_HI for h in hs]
    band_lo = [l2s[0] * (h / hs[0]) ** ORDER_LO for h in hs]

    width, height = 960, 720
    left, right = 90, 40
    top, plot_h = 70, 330
    order_top, order_h = 470, 130
    plot_w = width - left - right

    log_hs = [math.log10(h) for h in hs]
    all_l2 = l2s + theory + band_hi + band_lo
    log_l2 = [math.log10(e) for e in all_l2]
    x_min, x_max = min(log_hs), max(log_hs)
    y_min, y_max = min(log_l2), max(log_l2)
    y_pad = 0.12 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def xmap(h):
        return left + (math.log10(h) - x_min) / (x_max - x_min) * plot_w

    def ymap(e):
        return top + (y_max - math.log10(e)) / (y_max - y_min) * plot_h

    order_y_min = min(1.6, min(orders) - 0.08)
    order_y_max = max(2.25, max(orders) + 0.08)

    def order_xmap(index):
        if len(orders) == 1:
            return left + plot_w / 2
        return left + index / (len(orders) - 1) * plot_w

    def order_ymap(p):
        return order_top + (order_y_max - p) / (order_y_max - order_y_min) * order_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        svg_text(40, 34, "fem_poisson P1 convergence gate", 22, weight="bold"),
        svg_text(40, 58, "Measured L2 error from sweep.py against manufactured solution; pass band is the configured observed-order tolerance.", 13, fill="#475569"),
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    # Pass band between the configured order slopes, anchored at the coarsest actual L2 error.
    band_points = [(xmap(h), ymap(e)) for h, e in zip(hs, band_hi)]
    band_points += [(xmap(h), ymap(e)) for h, e in reversed(list(zip(hs, band_lo)))]
    band = " ".join(f"{x:.1f},{y:.1f}" for x, y in band_points)
    parts.append(f'<polygon points="{band}" fill="#dcfce7" opacity="0.65" stroke="none"/>')

    # Axes and grid.
    for h in hs:
        x = xmap(h)
        parts.append(svg_line(x, top, x, top + plot_h, "#e2e8f0", 1))
        parts.append(svg_text(x, top + plot_h + 24, f"{h:.5f}", 12, anchor="middle", fill="#475569"))
    y_ticks = sorted({10 ** math.floor(y_min), 10 ** math.ceil(y_max), min(l2s), max(l2s)})
    for e in y_ticks:
        if 10 ** y_min <= e <= 10 ** y_max:
            y = ymap(e)
            parts.append(svg_line(left, y, left + plot_w, y, "#e2e8f0", 1))
            parts.append(svg_text(left - 10, y + 4, f"{e:.1e}", 12, anchor="end", fill="#475569"))
    parts.append(svg_line(left, top + plot_h, left + plot_w, top + plot_h, "#334155", 1.5))
    parts.append(svg_line(left, top, left, top + plot_h, "#334155", 1.5))

    parts.append(polyline([(xmap(h), ymap(e)) for h, e in zip(hs, theory)], "#0f766e", 2.5, dash="7 5"))
    parts.append(polyline([(xmap(h), ymap(e)) for h, e in zip(hs, l2s)], "#1d4ed8", 3))
    for n, h, _, _, e, _ in results:
        x, y = xmap(h), ymap(e)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#1d4ed8" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(svg_text(x, y - 10, f"n={n}", 11, anchor="middle", fill="#1e3a8a"))

    parts.extend([
        svg_text(left + plot_w / 2, top + plot_h + 52, "mesh spacing h", 14, anchor="middle", weight="bold"),
        svg_text(24, top + plot_h / 2, "L2 error", 14, anchor="middle", weight="bold"),
        '<g transform="translate(735,92)">',
        '<rect x="0" y="0" width="184" height="86" fill="#ffffff" stroke="#cbd5e1"/>',
        '<rect x="14" y="18" width="28" height="12" fill="#dcfce7" opacity="0.8"/>',
        svg_text(50, 29, f"pass band p=[{ORDER_LO}, {ORDER_HI}]", 12),
        svg_line(14, 48, 42, 48, "#0f766e", 2.5, "7 5"),
        svg_text(50, 52, "P1 theory O(h^2)", 12),
        svg_line(14, 68, 42, 68, "#1d4ed8", 3),
        svg_text(50, 72, "measured L2", 12),
        '</g>',
    ])

    # Observed-order panel with explicit pass lines.
    parts.append(f'<rect x="{left}" y="{order_top}" width="{plot_w}" height="{order_h}" fill="#f8fafc" stroke="#cbd5e1"/>')
    parts.append(f'<rect x="{left}" y="{order_ymap(ORDER_HI):.1f}" width="{plot_w}" height="{order_ymap(ORDER_LO) - order_ymap(ORDER_HI):.1f}" fill="#dcfce7" opacity="0.65"/>')
    for p, label in [(ORDER_LO, f"lower pass {ORDER_LO}"), (2.0, "theory 2.0"), (ORDER_HI, f"upper pass {ORDER_HI}")]:
        color = "#16a34a" if p != 2.0 else "#0f766e"
        dash = "5 5" if p != 2.0 else "2 5"
        y = order_ymap(p)
        parts.append(svg_line(left, y, left + plot_w, y, color, 1.8, dash))
        parts.append(svg_text(left + plot_w - 104, y - 6, label, 12, fill=color))
    order_points = [(order_xmap(i), order_ymap(p)) for i, p in enumerate(orders)]
    parts.append(polyline(order_points, "#7c3aed", 3))
    for i, ((a, *_), (b, *__), p) in enumerate(zip(results, results[1:], orders)):
        x, y = order_xmap(i), order_ymap(p)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#7c3aed" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(svg_text(x, order_top + order_h + 24, f"{a}->{b}", 12, anchor="middle", fill="#475569"))
        parts.append(svg_text(x, y - 10, f"{p:.3f}", 11, anchor="middle", fill="#581c87"))
    parts.append(svg_line(left, order_top + order_h, left + plot_w, order_top + order_h, "#334155", 1.5))
    parts.append(svg_line(left, order_top, left, order_top + order_h, "#334155", 1.5))
    parts.append(svg_text(left + plot_w / 2, order_top + order_h + 52, "refinement step n_coarse -> n_fine", 14, anchor="middle", weight="bold"))
    parts.append(svg_text(32, order_top + order_h / 2, "observed order p", 14, anchor="middle", weight="bold"))

    status = "PASS" if ok else "FAIL"
    status_fill = "#166534" if ok else "#991b1b"
    parts.append(svg_text(40, 690, f"{status}: mean observed order = {mean_p:.3f}; finest L2 error = {l2s[-1]:.3e}", 15, weight="bold", fill=status_fill))
    parts.append("</svg>\n")

    with open(CONVERGENCE_PLOT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"\nWrote convergence plot: {os.path.relpath(CONVERGENCE_PLOT, REPO_ROOT)}")


def main():
    exe = build()
    results = run_ladder(exe)
    ok = gate(results)
    write_convergence_plot(results, ok)
    if ok:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nCHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
