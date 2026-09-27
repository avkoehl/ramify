"""Progress reporting for long `ramify` runs.

`partition_priority` and `width_regions_interpolate` both take a `progress=`
callback fired once per path/region, just before it's processed -- same
signature, so `path_reporter` below works for either. `width_interpolate`
(method="laplace") takes a different one, fired once per solver iteration.
This script is the worked example: two reporters, and a demo run on an
upscaled copy of the bundled shape (one big mainstem region plus many small
ones -- the same lopsidedness a real basin has, which is what makes naive
progress bars useless).

    python scripts/progress_example.py [zoom]
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as ndzoom
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import ramify
from ramify.data import load


def path_reporter(unit="path", every=1.0):
    """Activity line for partition_priority / width_regions_interpolate --
    deliberately no ETA.

    The caller cannot weight this honestly: a path's cost is the area of the
    search window (for partition_priority) or its region's pixel count (for
    width_regions_interpolate), and both are heavily front-loaded (paths run
    biggest-first, so item 1 of 19,000 can outweigh the next thousand).
    Rather than fake a bar off item count, report what it is actually doing
    and let the reader judge.
    """
    state = {"t0": time.monotonic(), "last": -1e9, "px": 0}

    def progress(i, n, label, size):
        now = time.monotonic()
        state["px"] += size
        if now - state["last"] >= every or i == 0 or i == n - 1:
            state["last"] = now
            el = now - state["t0"]
            rate = (i + 1) / el if el > 0 else 0.0
            print(
                f"\r  {unit} {i + 1:>6,}/{n:,}  window {size:>11,} px"
                f"  {el:6.1f}s  {rate:7.1f} {unit}s/s   ",
                end="", flush=True,
            )
        if i == n - 1:
            print(f"\r  {n:,} {unit}s, {state['px']:,} px visited,"
                  f" {time.monotonic() - state['t0']:.1f}s" + " " * 30, flush=True)

    return progress


def solver_reporter(every=1.0):
    """Iteration counter for width_interpolate's Laplace solve."""
    state = {"t0": time.monotonic(), "last": -1e9}

    def progress(iteration):
        now = time.monotonic()
        if now - state["last"] >= every:
            state["last"] = now
            print(f"\r  cg iteration {iteration:>6,}  {now - state['t0']:6.1f}s   ",
                  end="", flush=True)

    return progress


def _mmss(sec):
    if not np.isfinite(sec):
        return "  --  "
    return f"{int(sec) // 60:02d}m{int(sec) % 60:02d}s"


if __name__ == "__main__":
    z = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    mask, root, tips = load()
    m = np.asarray(mask.values if hasattr(mask, "values") else mask)
    big = (ndzoom(m, z, order=0) == 1).astype(np.uint8)
    pts = np.column_stack(np.nonzero(big))
    tree = cKDTree(pts)
    tips = [tuple(pts[tree.query((int(r * z), int(c * z)))[1]]) for r, c in tips]
    root = tuple(pts[tree.query((int(root[0] * z), int(root[1] * z)))[1]])

    print(f"shape {big.shape} = {big.size / 1e6:.1f}M cells, {int((big == 1).sum()):,} mask px")

    print("partition_priority")
    t = time.monotonic()
    labels, net, lines = ramify.partition_priority(
        big, root, tips=tips, progress=path_reporter(unit="path")
    )
    print(f"  {len(net)} segments, {len(lines)} regions in {time.monotonic() - t:.1f}s")

    print("width_regions_interpolate")
    t = time.monotonic()
    w = ramify.width_regions_interpolate(
        labels, lines, progress=path_reporter(unit="region")
    )
    print(f"  widths {np.nanmin(w):.1f} .. {np.nanmax(w):.1f} in {time.monotonic() - t:.1f}s")

    print("centerline + width_interpolate (single shape, per-iteration progress)")
    line = ramify.centerline(big, root=root)
    w = ramify.width_interpolate(big, line, progress=solver_reporter())
    print(f"\n  widths {np.nanmin(w):.1f} .. {np.nanmax(w):.1f}")
