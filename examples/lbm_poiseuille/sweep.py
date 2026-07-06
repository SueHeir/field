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
import xml.sax.saxutils as xml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
CONFIG = os.path.join(SCRIPT_DIR, "config.toml")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
PROFILE_PLOT = os.path.join(PLOTS_DIR, "poiseuille_profile.svg")

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
PROFILE_RE = re.compile(
    r"PROFILE\s+j=(\d+)\s+y=([-\d.eE+]+)\s+ux=([-\d.eE+]+)\s+"
    r"ux_exact=([-\d.eE+]+)\s+abs_error=([-\d.eE+]+)"
)


def sh(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kw)


def sx(value, xmin, xmax, left, width):
    return left + (value - xmin) / (xmax - xmin) * width


def sy(value, ymin, ymax, top, height):
    return top + (ymax - value) / (ymax - ymin) * height


def polyline(points, xmin, xmax, ymin, ymax, left, top, width, height):
    return " ".join(
        f"{sx(x, xmin, xmax, left, width):.2f},{sy(y, ymin, ymax, top, height):.2f}"
        for x, y in points
    )


def esc(s):
    return xml.escape(str(s))


def write_profile_plot(profile, metrics, ok):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    ys = [row["y"] for row in profile]
    ux = [row["ux"] for row in profile]
    exact = [row["ux_exact"] for row in profile]

    xmin = min(ys)
    xmax = max(ys)
    ymax = max(max(ux), max(exact) + MAX_LINF_ABS) * 1.08
    ymin = min(0.0, min(ux), min(exact) - MAX_LINF_ABS)

    left = 78
    top = 72
    width = 690
    height = 310
    bar_left = 112
    bar_top = 470
    bar_width = 600
    bar_height = 78

    ref_band_top = [
        (row["y"], row["ux_exact"] + MAX_LINF_ABS) for row in profile
    ]
    ref_band_bottom = [
        (row["y"], row["ux_exact"] - MAX_LINF_ABS) for row in reversed(profile)
    ]
    band = polyline(
        ref_band_top + ref_band_bottom,
        xmin, xmax, ymin, ymax, left, top, width, height,
    )
    measured = polyline(
        [(row["y"], row["ux"]) for row in profile],
        xmin, xmax, ymin, ymax, left, top, width, height,
    )
    reference = polyline(
        [(row["y"], row["ux_exact"]) for row in profile],
        xmin, xmax, ymin, ymax, left, top, width, height,
    )

    l2_ratio = metrics["l2_relative"] / MAX_L2_RELATIVE
    linf_ratio = metrics["linf_abs"] / MAX_LINF_ABS
    ratio_max = max(1.15, l2_ratio, linf_ratio) * 1.08
    limit_x = bar_left + bar_width / ratio_max
    bars = [
        ("relative L2", l2_ratio, metrics["l2_relative"], MAX_L2_RELATIVE, "#2f6fbb"),
        ("absolute Linf", linf_ratio, metrics["linf_abs"], MAX_LINF_ABS, "#b05a2a"),
    ]

    bar_svg = []
    for idx, (label, ratio, value, limit, color) in enumerate(bars):
        y = bar_top + idx * 38
        w = min(ratio / ratio_max * bar_width, bar_width)
        bar_svg.append(
            f'<text x="24" y="{y + 18}" class="tick">{esc(label)}</text>'
            f'<rect x="{bar_left}" y="{y}" width="{w:.2f}" height="22" '
            f'fill="{color}" />'
            f'<text x="{bar_left + w + 8:.2f}" y="{y + 16}" class="tick">'
            f'{value:.3e} / {limit:.2e}</text>'
        )

    status = "PASS" if ok else "FAIL"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="620" viewBox="0 0 900 620">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #222; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .sub {{ font-size: 13px; fill: #555; }}
    .tick {{ font-size: 12px; fill: #333; }}
    .axis {{ stroke: #222; stroke-width: 1.2; }}
    .grid {{ stroke: #d7d7d7; stroke-width: 1; }}
  </style>
  <rect width="900" height="620" fill="#ffffff" />
  <text x="32" y="34" class="title">D2Q9 Poiseuille velocity profile validation</text>
  <text x="32" y="55" class="sub">Measured channel profile vs analytic parabola; shaded band is the Linf tolerance. Gate: {status}.</text>

  <line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" class="axis" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" class="axis" />
'''
    for k in range(6):
        x = left + k * width / 5
        yval = xmin + k * (xmax - xmin) / 5
        svg += (
            f'  <line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
            f'y2="{top + height}" class="grid" />\n'
            f'  <text x="{x:.2f}" y="{top + height + 22}" '
            f'text-anchor="middle" class="tick">{yval:.1f}</text>\n'
        )
    for k in range(5):
        y = top + k * height / 4
        uval = ymax - k * (ymax - ymin) / 4
        svg += (
            f'  <line x1="{left}" y1="{y:.2f}" x2="{left + width}" '
            f'y2="{y:.2f}" class="grid" />\n'
            f'  <text x="{left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end" class="tick">{uval:.2e}</text>\n'
        )
    svg += f'''
  <polygon points="{band}" fill="#f2c94c" fill-opacity="0.28" stroke="none" />
  <polyline points="{reference}" fill="none" stroke="#222222" stroke-width="2.2" />
  <polyline points="{measured}" fill="none" stroke="#2f6fbb" stroke-width="2.4" />
'''
    for row in profile:
        svg += (
            f'  <circle cx="{sx(row["y"], xmin, xmax, left, width):.2f}" '
            f'cy="{sy(row["ux"], ymin, ymax, top, height):.2f}" r="2.6" '
            f'fill="#2f6fbb" />\n'
        )
    svg += f'''
  <text x="{left + width / 2:.2f}" y="{top + height + 48}" text-anchor="middle" class="tick">y cell center</text>
  <text x="22" y="{top + height / 2:.2f}" transform="rotate(-90 22 {top + height / 2:.2f})" text-anchor="middle" class="tick">streamwise velocity ux</text>
  <rect x="608" y="86" width="220" height="70" fill="#ffffff" stroke="#cccccc" />
  <line x1="626" y1="108" x2="680" y2="108" stroke="#2f6fbb" stroke-width="2.4" />
  <text x="690" y="112" class="tick">measured</text>
  <line x1="626" y1="132" x2="680" y2="132" stroke="#222222" stroke-width="2.2" />
  <text x="690" y="136" class="tick">analytic reference</text>
  <rect x="626" y="142" width="54" height="10" fill="#f2c94c" fill-opacity="0.28" />
  <text x="690" y="152" class="tick">+/- Linf limit</text>

  <text x="32" y="430" class="title">Error gates</text>
  <text x="32" y="451" class="sub">Bars show measured error divided by the unchanged pass limit; vertical line is the pass threshold.</text>
  <line x1="{bar_left}" y1="{bar_top - 12}" x2="{bar_left + bar_width}" y2="{bar_top - 12}" class="axis" />
  <line x1="{limit_x:.2f}" y1="{bar_top - 18}" x2="{limit_x:.2f}" y2="{bar_top + bar_height}" stroke="#b00020" stroke-width="2" />
  <text x="{limit_x:.2f}" y="{bar_top + bar_height + 20}" text-anchor="middle" class="tick">pass limit</text>
  {''.join(bar_svg)}

  <text x="32" y="592" class="sub">umax numeric/exact = {metrics["ux_max"]:.6e} / {metrics["ux_exact_max"]:.6e}; mass drift = {metrics["mass_drift"]:.3e}</text>
</svg>
'''
    with open(PROFILE_PLOT, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"  wrote {os.path.relpath(PROFILE_PLOT, REPO_ROOT)}")


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
    profile = [
        {
            "j": int(row.group(1)),
            "y": float(row.group(2)),
            "ux": float(row.group(3)),
            "ux_exact": float(row.group(4)),
            "abs_error": float(row.group(5)),
        }
        for row in PROFILE_RE.finditer(out)
    ]
    if len(profile) != ny:
        print(f"CHECKS FAILED: expected {ny} PROFILE rows, parsed {len(profile)}")
        sys.exit(2)

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

    write_profile_plot(
        profile,
        {
            "ux_max": ux_max,
            "ux_exact_max": ux_exact_max,
            "l2_relative": l2_relative,
            "linf_abs": linf_abs,
            "mass_drift": mass_drift,
        },
        ok,
    )

    if ok:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    print("\nCHECKS FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
