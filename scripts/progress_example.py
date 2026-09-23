"""Progress reporting for long `ramify` runs.

`partition_by_priority` and `interpolate_widths` (with `regions`) both take an optional `progress=` callback, fired
once per path / per region just before that item is worked. This script is the
worked example: two reporters, and a demo run on an upscaled copy of the bundled
shape (one big mainstem region plus many small ones -- the same lopsidedness a
real basin has, which is what makes naive progress bars useless).

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


def region_reporter(regions, mask, exponent=1.5, every=1.0, width=34):
    """Weighted progress bar for interpolate_widths, with an ETA.

    Region sizes span orders of magnitude and the laplace solve costs ~O(n^1.5)
    in a region's pixel count, so a bar driven by *region count* -- or even by
    pixels -- races to ~99% and then sits on the trunk regions for most of the
    wall time. The caller can weight properly because it already holds the
    regions raster: one bincount gives every region's size up front.
    """
    reg = np.asarray(regions.values if hasattr(regions, "values") else regions)
    m = np.asarray(mask.values if hasattr(mask, "values") else mask) == 1
    weight = np.bincount(reg[m].ravel()).astype(float) ** exponent
    total = float(weight.sum())
    state = {"work": 0.0, "t0": time.monotonic(), "last": -1e9}

    def progress(i, n, label, size):
        now = time.monotonic()
        frac = state["work"] / total
        # print on a cadence, but never skip the first or last item
        if now - state["last"] >= every or i == 0 or i == n - 1:
            state["last"] = now
            elapsed = now - state["t0"]
            eta = elapsed * (1 - frac) / frac if frac > 1e-9 else float("nan")
            filled = int(width * frac)
            bar = "#" * filled + "." * (width - filled)
            print(
                f"\r  [{bar}] {frac:5.1%}  region {i + 1}/{n}"
                f"  {size:>9,} px  eta {_mmss(eta)}   ",
                end="", flush=True,
            )
        state["work"] += weight[label] if label < weight.size else 0.0
        if i == n - 1:
            print(f"\r  [{'#' * width}] 100.0%  {n} regions"
                  f"  in {_mmss(time.monotonic() - state['t0'])}" + " " * 24, flush=True)

    return progress


def path_reporter(every=1.0):
    """Activity line for partition_by_priority -- deliberately no ETA.

    Unlike interpolate_widths, the caller cannot weight this honestly: a path's cost
    is the area of the search window, which depends on the reach radius computed
    inside partition_by_priority. Rather than fake a bar off path count (the work is heavily
    front-loaded -- paths run biggest-first, so path 1 of 19,000 can outweigh the
    next thousand), report what it is actually doing and let the reader judge.
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
                f"\r  path {i + 1:>6,}/{n:,}  window {size:>11,} px"
                f"  {el:6.1f}s  {rate:7.1f} paths/s   ",
                end="", flush=True,
            )
        if i == n - 1:
            print(f"\r  {n:,} paths, {state['px']:,} window px visited,"
                  f" {time.monotonic() - state['t0']:.1f}s" + " " * 30, flush=True)

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

    print("extract_centerlines  (no progress hook -- one opaque skeletonize dominates)")
    t = time.monotonic()
    net = ramify.extract_centerlines(big, root, tips=tips)
    print(f"  {len(net.segments)} segments in {time.monotonic() - t:.1f}s")

    print("partition_by_priority")
    regions = ramify.partition_by_priority(big, net.rasterize(by="path"), progress=path_reporter())

    print("interpolate_widths")
    w = ramify.interpolate_widths(
        big, net.rasterize(), regions, progress=region_reporter(regions, big)
    )
    print(f"  widths {np.nanmin(w):.1f} .. {np.nanmax(w):.1f}")
