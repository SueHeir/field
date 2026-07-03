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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_DIR = os.path.join(SCRIPT_DIR, "sweep")

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


def main():
    exe = build()
    results = run_ladder(exe)
    ok = gate(results)
    if ok:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nCHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
