import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import ramify
from ramify.data import load

OUT = REPO / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def values(arr):
    return arr.values if hasattr(arr, "values") else np.asarray(arr)


def label_cmap(arr):
    labels = np.unique(arr[arr > 0])
    n = len(labels)
    idx = np.arange(n)
    hsv = np.stack(
        [
            (idx * 0.61803398875) % 1.0,
            0.55 + 0.4 * ((idx * 0.37) % 1.0),
            0.7 + 0.25 * ((idx * 0.23) % 1.0),
        ],
        axis=1,
    )
    cmap = mcolors.ListedColormap(mcolors.hsv_to_rgb(hsv))
    lookup = np.zeros(int(labels.max()) + 1, dtype=int)
    lookup[labels] = idx
    return np.ma.masked_where(arr == 0, lookup[arr]), cmap, n


def draw_labels(ax, labeled, net=None, endpoints=False, centerline=False):
    ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
    mapped, cmap, n = label_cmap(labeled)
    ax.imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
    if centerline and net is not None:
        rr, cc = np.nonzero(values(net.rasterize()))
        ax.plot(cc, rr, "k.", markersize=0.4)
    if endpoints and net is not None:
        ax.plot(net.root[1], net.root[0], "k*", markersize=14, mec="w")
        if net.tips:
            tr, tc = zip(*net.tips)
            ax.plot(tc, tr, "wo", markersize=5, mec="k")
    ax.set_axis_off()


def draw_float(ax, arr, norm, cmap="viridis", colorbar=True):
    im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="width (m)")
    ax.set_axis_off()
    return im


def save(name, fig):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def save_labels(name, labeled, titles=None, **kw):
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_labels(ax, values(labeled), **kw)
    save(name, fig)


# -- pipeline ------------------------------------------------------------------

mask, root, tips = load()
mask_arr = values(mask) == 1
print(f"mask: {mask_arr.shape}, root: {root}, tips: {len(tips)}")

net = ramify.extract_centerlines(mask, root=root, tips=tips)
net_auto = ramify.extract_centerlines(mask, root=root)
regions = ramify.partition_by_priority(mask, net.rasterize(by="path"))
regions_vor = ramify.partition_by_nearest(mask, net.rasterize(by="path"))
seg_regions = ramify.subdivide_regions(regions, net)
w_lap = ramify.interpolate_widths(mask, net.rasterize(), method="laplace")
w_near = ramify.interpolate_widths(mask, net.rasterize(), method="nearest")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rw_lap = ramify.interpolate_widths(mask, net.rasterize(), regions, method="laplace")
    rw_near = ramify.interpolate_widths(mask, net.rasterize(), regions, method="nearest")


# Shared discrete log-spaced bins across every width image. Widths span more than
# a decade, so the ladder is logarithmic and made of preferred numbers to keep
# the legend readable. Try the finest ladder first and fall back to coarser ones
# until the count fits -- a 1-2-5 ladder alone gives only ~4 bands over this
# range, which flattens the whole map into two colours.
def log_bins(vmin, vmax, target=12):
    ladders = [
        (1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8),  # ~10 per decade
        (1, 1.5, 2, 3, 5, 7),  # ~6 per decade
        (1, 2, 5),
        (1,),  # decades
    ]
    for ladder in ladders:
        edges, k = [], int(np.floor(np.log10(max(vmin, 1e-12))))
        while not edges or edges[-1] < vmax:
            edges += [mult * 10.0**k for mult in ladder]
            k += 1
        lo = max((i for i, e in enumerate(edges) if e <= vmin), default=0)
        edges = [e for e in edges[lo:] if e / 1.0001 <= vmax]
        if len(edges) <= target:
            break
    return edges


allw = np.concatenate([values(a).ravel() for a in (w_lap, w_near, rw_lap, rw_near)])
finite = allw[np.isfinite(allw) & (allw > 0)]
BINS = log_bins(float(finite.min()), float(finite.max()))
# sequential magnitude -> a perceptually uniform ramp, dark (narrow) to bright
# (wide); viridis stays legible against the white page at both ends
CMAP = plt.get_cmap("viridis", len(BINS))
NORM = mcolors.BoundaryNorm(BINS, ncolors=len(BINS), extend="max")
print(f"width bins ({len(BINS)}): {BINS}")

# -- graphical abstract ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(
    np.ma.masked_where(~mask_arr, mask_arr.astype(int)), cmap="gray", vmin=0, vmax=2
)
axes[0].plot(net.root[1], net.root[0], "k*", markersize=14, mec="w", label="root")
tr, tc = zip(*net.tips)
axes[0].plot(tc, tr, "wo", markersize=5, mec="k", label="tips")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_title("binary branching shape + root, tips")
axes[0].set_axis_off()
mapped, cmap, n = label_cmap(values(regions))
axes[1].imshow(mask_arr, cmap="gray_r", alpha=0.25)
axes[1].imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
axes[1].set_title("branch segmentation")
axes[1].set_axis_off()
draw_float(axes[2], values(rw_lap), NORM, cmap=CMAP)
axes[2].set_title("interpolated widths")
save("abstract.png", fig)

# -- individual function outputs --------------------------------------------------

save_labels("extract_centerlines_tips.png", net.rasterize(by="path"), net=net, endpoints=True)
save_labels(
    "extract_centerlines_auto.png", net_auto.rasterize(by="path"), net=net_auto, endpoints=True
)

save_labels("partition_by_priority.png", regions, net=net, centerline=True)
save_labels("partition_by_nearest.png", regions_vor, net=net, centerline=True)
save_labels("subdivide_regions.png", seg_regions, net=net, centerline=True)

def save_with_colorbar(name, fig, axes):
    # figure-level colorbar; savefig directly because tight_layout() cannot lay
    # one out and leaves the tick labels sitting on top of the last panel
    fig.colorbar(
        plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
        ax=axes, fraction=0.03, pad=0.04, label="width (m)", extend="max",
    )
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


# domain: whole shape vs per-region, both with the default laplace interpolator
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, arr, title in [
    (axes[0], w_lap, "interpolate_widths(...)"),
    (axes[1], rw_lap, "interpolate_widths(..., regions)"),
]:
    draw_float(ax, values(arr), NORM, cmap=CMAP, colorbar=False)
    ax.set_title(title, fontfamily="monospace", fontsize=11)
save_with_colorbar("interpolate_widths_domain.png", fig, axes)

# interpolator: nearest instead of laplace
fig, ax = plt.subplots(figsize=(7, 6.5))
draw_float(ax, values(w_near), NORM, cmap=CMAP, colorbar=False)
ax.set_title('interpolate_widths(..., method="nearest")', fontfamily="monospace", fontsize=11)
save_with_colorbar("interpolate_widths_nearest.png", fig, ax)

# -- open_boundary: a boundary the shape is truncated by, not a real wall --------
# The mask is cut off at the outlet (the edge the root sits on), so the void just
# past it is not a real wall. Mark it open so half-widths there are measured to
# the true flanking walls instead of collapsing at the cut edge. Built in two
# steps: select the mask pixels on the outlet edge, then buffer outward from them.
# The buffer has to have depth, not be a thin rind along the boundary: distances
# are measured *through* the marked void, so a one-pixel skin would just move the
# wall out by one pixel -- hence the padding, since the toy mask ends at the cut.
PAD, BUF, BAND = 30, 12, 2  # pixels: rows added below, buffer depth, rows above root
res = abs(mask.rio.resolution()[1])
x0, y0, x1, y1 = mask.rio.bounds()
mask_p = mask.rio.pad_box(x0, y0 - PAD * res, x1, y1, constant_values=0)  # rows/cols of
mask_p_arr = values(mask_p) == 1                                          # root unchanged

rr = np.arange(mask_p_arr.shape[0])[:, None]
has_mask_below = np.zeros_like(mask_p_arr)
has_mask_below[:-1] = mask_p_arr[1:]
edge = mask_p_arr & ~has_mask_below & (rr >= root[0] - BAND)  # outlet-edge pixels
open_boundary = ~mask_p_arr & (distance_transform_edt(~edge) <= BUF) & (rr >= root[0] - BAND)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    net_p = ramify.extract_centerlines(mask_p, root, tips=tips)
    regions_p = ramify.partition_by_priority(mask_p, net_p.rasterize(by="path"))
    rw_p = ramify.interpolate_widths(mask_p, net_p.rasterize(), regions_p)
    net_open = ramify.extract_centerlines(mask_p, root, tips=tips, open_boundary=open_boundary)
    regions_open = ramify.partition_by_priority(
        mask_p, net_open.rasterize(by="path"), open_boundary=open_boundary
    )
    rw_open = ramify.interpolate_widths(
        mask_p, net_open.rasterize(), regions_open, open_boundary=open_boundary
    )

# Only the widths are worth showing: the partition is visually identical either
# way on this shape (the skeleton and tip->root routing do not depend on the
# boundary convention at all).
moved = int((values(regions_p) != values(regions_open)).sum())
print(f"pixels whose path changes under open_boundary: {moved} of {int(mask_p_arr.sum())}")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, arr, title, mark in [
    (axes[0], rw_p, "interpolate_widths(..., regions)", False),
    (axes[1], rw_open, "interpolate_widths(..., regions, open_boundary=…)", True),
]:
    draw_float(ax, values(arr), NORM, cmap=CMAP, colorbar=False)
    if mark:
        # fill it rather than outline it -- a contour of the patch reads as a
        # stray circle drawn over blank page, when the point is the area itself
        ax.imshow(
            np.ma.masked_where(~open_boundary, open_boundary.astype(float)),
            cmap=mcolors.ListedColormap(["red"]), alpha=0.35, interpolation="nearest",
        )
    ax.set_title(title, fontfamily="monospace", fontsize=11)
    ax.set_axis_off()
save_with_colorbar("open_boundary.png", fig, axes)
