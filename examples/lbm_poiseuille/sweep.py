#!/usr/bin/env python3
"""
lbm_poiseuille — validation gate for D2Q9 LBM on FIELD.

The example runs a force-driven D2Q9 BGK channel flow on FIELD's UniformMesh.
At steady state the analytic plane-Poiseuille profile is

    u(y) = g y (H-y) / (2 nu),  nu = (tau - 1/2) / 3.

The gate builds the example, runs the declarative TOML case, parses its RESULT
line, and checks the velocity profile error and mass conservation.
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CONFIG = os.path.join(SCRIPT_DIR, "config.toml")

MAX_L2_RELATIVE = 0.035
MAX_LINF_ABS = 3.5e-6
MAX_MASS_DRIFT = 1.0e-11

RESULT_RE = re.compile(
    r"nx=(\d+)\s+ny=(\d+)\s+steps=(\d+)\s+tau=([-\d.eE+]+)\s+"
    r"nu=([-\d.eE+]+)\s+force_x=([-\d.eE+]+)\s+"
    r"ux_max=([-\d.eE+]+)\s+ux_exact_max=([-\d.eE+]+)\s+"
    r"l2_relative=([-\d.eE+]+)\s+linf_abs=([-\d.eE+]+)\s+"
    r"mass_drift=([-\d.eE+]+)"
)


def sh(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kw)


def main():
    sh(["cargo", "build", "--release", "-q", "-p", "lbm_poiseuille"])
    exe = os.path.join(REPO_ROOT, "target", "release", "lbm_poiseuille")
    out = subprocess.run(
        [exe, CONFIG],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    print(out, end="")

    m = RESULT_RE.search(out)
    if not m:
        print("CHECKS FAILED: could not parse RESULT line")
        sys.exit(2)

    nx = int(m.group(1))
    ny = int(m.group(2))
    steps = int(m.group(3))
    tau = float(m.group(4))
    nu = float(m.group(5))
    force_x = float(m.group(6))
    ux_max = float(m.group(7))
    ux_exact_max = float(m.group(8))
    l2_relative = float(m.group(9))
    linf_abs = float(m.group(10))
    mass_drift = float(m.group(11))

    print(
        f"\nD2Q9 Poiseuille analytic check: nx={nx} ny={ny} steps={steps} "
        f"tau={tau:.3f} nu={nu:.6e} force_x={force_x:.3e}"
    )
    print(f"  umax numeric/exact = {ux_max:.8e} / {ux_exact_max:.8e}")
    print(f"  relative L2 profile error = {l2_relative:.6e} "
          f"(limit {MAX_L2_RELATIVE:.2e})")
    print(f"  Linf profile error = {linf_abs:.6e} "
          f"(limit {MAX_LINF_ABS:.2e})")
    print(f"  relative mass drift = {mass_drift:.6e} "
          f"(limit {MAX_MASS_DRIFT:.2e})")

    ok = True
    if l2_relative > MAX_L2_RELATIVE:
        print("  FAIL: velocity profile relative L2 error too large")
        ok = False
    if linf_abs > MAX_LINF_ABS:
        print("  FAIL: velocity profile Linf error too large")
        ok = False
    if mass_drift > MAX_MASS_DRIFT:
        print("  FAIL: mass drift too large")
        ok = False

    if ok:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    print("\nCHECKS FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
