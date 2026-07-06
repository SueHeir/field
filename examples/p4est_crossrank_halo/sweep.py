#!/usr/bin/env python3
"""Run and plot the multi-rank p4est ghost-halo validation."""

import os
import re
import struct
import subprocess
import sys
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
PLOT_PATH = os.path.join(PLOTS_DIR, "crossrank_halo.png")
CONFIG = os.path.join(SCRIPT_DIR, "config.toml")

RESULT_RE = re.compile(
    r"RESULT ranks=(\d+) links=(\d+) recv_ghosts=(\d+) send_mirrors=(\d+) "
    r"changed_ghosts=(\d+) cross_rank_refinement_faces=(\d+) "
    r"max_abs_err=([-\d.eE+]+) tol=([-\d.eE+]+) status=(\w+)"
)


def sh(cmd, **kwargs):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=True, **kwargs)


def build():
    sh(["cargo", "build", "--release", "-q", "-p", "p4est_crossrank_halo"])
    exe = os.path.join(REPO_ROOT, "target", "release", "p4est_crossrank_halo")
    if not os.path.exists(exe):
        print(f"CHECKS FAILED: binary not found at {exe}")
        sys.exit(2)
    return exe


def run(exe):
    proc = subprocess.run(
        ["mpirun", "-np", "2", exe, CONFIG],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    match = RESULT_RE.search(proc.stdout)
    if proc.returncode != 0 or not match:
        print("CHECKS FAILED: mpirun validation failed or RESULT line was not found")
        sys.exit(1)
    result = {
        "ranks": int(match.group(1)),
        "links": int(match.group(2)),
        "recv_ghosts": int(match.group(3)),
        "send_mirrors": int(match.group(4)),
        "changed_ghosts": int(match.group(5)),
        "cross_rank_refinement_faces": int(match.group(6)),
        "max_abs_err": float(match.group(7)),
        "tol": float(match.group(8)),
        "status": match.group(9),
    }
    if result["status"] != "PASS":
        print(f"CHECKS FAILED: validation status is {result['status']}")
        sys.exit(1)
    return result


def write_plot(result):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    w, h = 900, 420
    img = bytearray([255, 255, 255] * w * h)

    def rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def set_px(x, y, color):
        if 0 <= x < w and 0 <= y < h:
            off = (y * w + x) * 3
            img[off : off + 3] = bytes(color)

    def rect(x0, y0, x1, y1, color):
        x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
        y0, y1 = sorted((max(0, y0), min(h - 1, y1)))
        for y in range(y0, y1 + 1):
            off = (y * w + x0) * 3
            img[off : off + (x1 - x0 + 1) * 3] = bytes(color) * (x1 - x0 + 1)

    def line_h(x0, x1, y, color, dash=False):
        for x in range(x0, x1 + 1):
            if not dash or ((x - x0) // 8) % 2 == 0:
                set_px(x, y, color)
                set_px(x, y + 1, color)

    def line_v(x, y0, y1, color):
        for y in range(y0, y1 + 1):
            set_px(x, y, color)
            set_px(x + 1, y, color)

    glyphs = {
        " ": ["00000"] * 7,
        ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
        "<": ["00001", "00010", "00100", "01000", "00100", "00010", "00001"],
        "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
        "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01110"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "X": ["10001", "01010", "00100", "00100", "00100", "01010", "10001"],
        "Y": ["10001", "01010", "00100", "00100", "00100", "00100", "00100"],
    }

    def text(x, y, label, color=None, scale=2):
        if color is None:
            color = slate
        cx = x
        for ch in label.upper():
            pat = glyphs.get(ch, glyphs[" "])
            for gy, row in enumerate(pat):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        rect(cx + gx * scale, y + gy * scale, cx + (gx + 1) * scale - 1, y + (gy + 1) * scale - 1, color)
            cx += 6 * scale

    blue, green, purple = rgb("#2563eb"), rgb("#16a34a"), rgb("#7c3aed")
    red, slate, teal, light = rgb("#dc2626"), rgb("#334155"), rgb("#0f766e"), rgb("#e5e7eb")

    rect(0, 0, w - 1, h - 1, rgb("#ffffff"))
    # Left panel: global topology counts, with the red non-vacuous minimum line.
    lx0, ly0, lx1, ly1 = 65, 70, 430, 345
    rect(lx0, ly0, lx1, ly1, rgb("#f8fafc"))
    line_h(lx0, lx1, ly1, slate)
    line_v(lx0, ly0, ly1, slate)
    counts = [result["recv_ghosts"], result["changed_ghosts"], result["cross_rank_refinement_faces"]]
    max_count = max(counts + [1])
    min_y = ly1 - int((1 / max_count) * (ly1 - ly0))
    line_h(lx0, lx1, min_y, red, dash=True)
    for i, (value, color) in enumerate(zip(counts, [blue, green, purple])):
        bx0 = lx0 + 55 + i * 105
        bx1 = bx0 + 55
        by0 = ly1 - int((value / max_count) * (ly1 - ly0 - 15))
        rect(bx0, by0, bx1, ly1 - 1, color)
    text(lx0, 48, "TOPOLOGY COUNTS")
    text(lx0 + 55, ly1 + 14, "RECV", blue, 1)
    text(lx0 + 160, ly1 + 14, "CHANGED", green, 1)
    text(lx0 + 265, ly1 + 14, "AMR FACES", purple, 1)
    text(lx0 + 8, min_y - 18, "MIN 1", red, 1)

    # Right panel: error-vs-tolerance on a compact log scale. Exact zero is drawn
    # at tol*1e-4 so the successful zero-error bar remains visible below the line.
    rx0, ry0, rx1, ry1 = 540, 70, 835, 345
    rect(rx0, ry0, rx1, ry1, rgb("#f8fafc"))
    line_h(rx0, rx1, ry1, slate)
    line_v(rx0, ry0, ry1, slate)
    plotted_err = max(result["max_abs_err"], result["tol"] * 1.0e-4)
    y_min = result["tol"] * 1.0e-5
    y_max = result["tol"] * 10.0

    def log_y(v):
        import math

        return ry1 - int((math.log10(v) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min)) * (ry1 - ry0))

    tol_y = log_y(result["tol"])
    err_y = log_y(plotted_err)
    line_h(rx0, rx1, tol_y, red, dash=True)
    rect(rx0 + 105, err_y, rx0 + 185, ry1 - 1, teal)
    text(rx0, 48, "GHOST VALUE ERROR")
    text(rx0 + 95, ry1 + 14, "ERR", teal, 1)
    text(rx0 + 8, tol_y - 18, "TOL 1E-12", red, 1)

    # Simple color swatches act as a legend without depending on font rendering.
    for x, color in [(80, blue), (150, green), (220, purple), (565, teal), (635, red)]:
        rect(x, 25, x + 42, 42, color)
        rect(x, 45, x + 42, 48, light)
    text(350, 24, "P4EST HALO PASS", slate, 2)

    raw = b"".join(b"\x00" + img[y * w * 3 : (y + 1) * w * 3] for y in range(h))

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(PLOT_PATH, "wb") as f:
        f.write(png)


def main():
    exe = build()
    result = run(exe)
    write_plot(result)
    print(
        "ALL CHECKS PASSED: "
        f"recv={result['recv_ghosts']} changed={result['changed_ghosts']} "
        f"cross_rank_refinement_faces={result['cross_rank_refinement_faces']} "
        f"max_abs_err={result['max_abs_err']:.3e} <= {result['tol']:.3e}"
    )


if __name__ == "__main__":
    main()
