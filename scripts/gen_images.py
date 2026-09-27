"""Regenerate the README figures against the bundled toy dataset.

    python scripts/gen_images.py

See README-instructions.md for the figure table this fulfills.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import ramify
from ramify.data import load
from ramify._io import make_grid

OUT = REPO / "assets"
OUT.mkdir(parents=True, exist_ok=True)

DEMO_PATH_ID = 3  # mid-sized, single-component branch used for single-shape figures


def values(arr):
    return arr.values if hasattr(arr, "values") else np.asarray(arr)


def save(name, fig, tight=True):
    # tight=False for figures with a colorbar already placed across multiple
    # axes (fig.colorbar(..., ax=axes)) -- tight_layout() run afterward
    # doesn't know about it and can squeeze titles into the colorbar's tick
    # labels. bbox_inches="tight" on savefig still crops the whitespace.
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def bbox_of(mask_bool, pad=6):
    rows, cols = np.nonzero(mask_bool)
    h, w = mask_bool.shape
    r0, r1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, h)
    c0, c1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, w)
    return r0, r1, c0, c1


# -- shared label coloring -------------------------------------------------------


def label_cmap(arr):
    """labeled raster -> (mapped array for imshow, cmap, lookup, n colors)."""
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
    lookup = np.zeros((int(labels.max()) + 1) if n else 1, dtype=int)
    lookup[labels] = idx
    mapped = np.ma.masked_where(arr == 0, lookup[arr])
    return mapped, cmap, lookup, n


def color_for(cmap, lookup, n, label):
    return cmap(lookup[int(label)] / max(n - 1, 1))


def draw_labels(ax, mask_arr, labels_arr, title=None):
    ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
    mapped, cmap, lookup, n = label_cmap(labels_arr)
    ax.imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
    if title:
        ax.set_title(title, fontfamily="monospace", fontsize=9)
    ax.set_axis_off()
    return cmap, lookup, n


def _rings(geom):
    if geom.geom_type == "Polygon":
        yield geom.exterior
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            yield part.exterior
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geom.geoms:
            yield from _rings(part)
    else:
        yield geom


def plot_geoms(ax, geoms, grid, color="black", lw=1.2, ls="-", alpha=1.0, r0=0, c0=0):
    for geom in geoms:
        if geom is None or geom.is_empty:
            continue
        for ring in _rings(geom):
            xs, ys = ring.xy
            rr, cc = grid.to_rc(np.asarray(xs), np.asarray(ys))
            ax.plot(cc - c0, rr - r0, color=color, linewidth=lw, linestyle=ls,
                     alpha=alpha, solid_capstyle="round")


def plot_net_colored(ax, net, column, cmap, lookup, n, grid, r0=0, c0=0, **kw):
    for _, row in net.iterrows():
        color = color_for(cmap, lookup, n, row[column])
        plot_geoms(ax, [row.geometry], grid, color="black", lw=2.4, r0=r0, c0=c0)
        plot_geoms(ax, [row.geometry], grid, color=color, lw=1.2, r0=r0, c0=c0, **kw)


def log_bins(vmin, vmax, target=12):
    # Shared discrete log-spaced bins across the width figures. Widths span
    # more than a decade, so the ladder is logarithmic and made of preferred
    # numbers to keep the legend readable.
    ladders = [
        (1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8),
        (1, 1.5, 2, 3, 5, 7),
        (1, 2, 5),
        (1,),
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


def width_norm_cmap(*arrays):
    finite = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays])
    bins = log_bins(float(finite.min()), float(finite.max()))
    cmap = plt.get_cmap("viridis", len(bins))
    norm = mcolors.BoundaryNorm(bins, ncolors=len(bins), extend="max")
    return norm, cmap, bins


# -- pipeline ------------------------------------------------------------------

mask, root, tips = load()
mask_arr = values(mask) == 1
_, grid = make_grid(mask)
print(f"mask: {mask_arr.shape}, root: {root}, tips: {len(tips)}")

labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
labels_arr = values(labels)
w_whole = values(ramify.width_regions_interpolate(labels, lines))

# 1. abstract.png: mask+root+tips -> labels+lines -> widths ----------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

axes[0].imshow(np.ma.masked_where(~mask_arr, mask_arr.astype(int)), cmap="gray", vmin=0, vmax=2)
axes[0].plot(root[1], root[0], "k*", markersize=14, mec="w", label="root")
tr, tc = zip(*tips)
axes[0].plot(tc, tr, "wo", markersize=5, mec="k", label="tips")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_title("mask, root, tips", fontfamily="monospace", fontsize=10)
axes[0].set_axis_off()

cmap, lookup, n = draw_labels(axes[1], mask_arr, labels_arr, "labels + lines")
plot_geoms(axes[1], lines.geometry, grid, color="black", lw=1.0)

im = axes[2].imshow(w_whole, cmap="viridis", interpolation="nearest")
plt.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02, label="width (m)")
axes[2].set_title("width_regions_interpolate", fontfamily="monospace", fontsize=10)
axes[2].set_axis_off()

save("abstract.png", fig)

# 2. quickstart.png: labels + lines, root and tips marked -------------------------

fig, ax = plt.subplots(figsize=(9, 7))
draw_labels(ax, mask_arr, labels_arr)
plot_geoms(ax, lines.geometry, grid, color="black", lw=1.0)
root_rc = grid.to_rc(*grid.to_xy(np.array([root[0]]), np.array([root[1]])))
ax.plot(root[1], root[0], "k*", markersize=14, mec="w", zorder=4)
tr, tc = zip(*tips)
ax.plot(tc, tr, "wo", markersize=5, mec="k", zorder=4)
ax.set_title("partition_priority(mask, root, tips=tips)", fontfamily="monospace", fontsize=10)
save("quickstart.png", fig)

# 3. net_vs_lines.png: zoom on one junction, net dashed vs lines solid -----------

junction_r0, junction_c0 = 181, 176  # net path 3's outlet, where it meets path 1
r0, r1 = junction_r0 - 22, junction_r0 + 22
c0, c1 = junction_c0 - 15, junction_c0 + 30

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(mask_arr[r0:r1, c0:c1], cmap="gray_r", alpha=0.25)
mapped, cmap, lookup, n = label_cmap(labels_arr[r0:r1, c0:c1])
ax.imshow(mapped, cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
nearby_ids = np.unique(labels_arr[r0:r1, c0:c1])
nearby_ids = nearby_ids[nearby_ids > 0]
plot_geoms(ax, net[net["path_id"].isin(nearby_ids)].geometry, grid,
           color="black", lw=1.6, ls="--", r0=r0, c0=c0)
plot_geoms(ax, lines[lines["region_id"].isin(nearby_ids)].geometry, grid,
           color="crimson", lw=1.6, r0=r0, c0=c0)
ax.plot([], [], color="black", ls="--", lw=1.6, label="net (skeleton)")
ax.plot([], [], color="crimson", lw=1.6, label="lines (region centerline)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("net vs. lines at one junction", fontfamily="monospace", fontsize=10)
ax.set_axis_off()
save("net_vs_lines.png", fig)

# 4. tips_options.png: tips= / min_length= / neither, each showing net -----------

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, net_tips, _ = ramify.partition_priority(mask, root, tips=tips)
    _, net_ml, _ = ramify.partition_priority(mask, root, min_length=15.0)
    _, net_all, _ = ramify.partition_priority(mask, root)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, net_i, title in [
    (axes[0], net_tips, "tips=tips (28 tips)"),
    (axes[1], net_ml, "min_length=15 (61 paths)"),
    (axes[2], net_all, "neither (66 paths)"),
]:
    ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
    cmap = plt.get_cmap("hsv")
    n = net_i["path_id"].nunique()
    ids = sorted(net_i["path_id"].unique())
    lookup = {pid: i for i, pid in enumerate(ids)}
    for _, row in net_i.iterrows():
        color = cmap((lookup[row["path_id"]] * 0.61803398875) % 1.0)
        plot_geoms(ax, [row.geometry], grid, color=color, lw=1.0)
    ax.set_title(title, fontfamily="monospace", fontsize=10)
    ax.set_axis_off()
save("tips_options.png", fig)

# 5. partition_methods.png: partition_priority vs partition_nearest --------------

labels_near, net_near, lines_near = ramify.partition_nearest(mask, root, tips=tips)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
cmap, lookup, n = draw_labels(axes[0], mask_arr, labels_arr, "partition_priority")
plot_net_colored(axes[0], net, "path_id", cmap, lookup, n, grid)
cmap2, lookup2, n2 = draw_labels(axes[1], mask_arr, values(labels_near), "partition_nearest")
plot_net_colored(axes[1], net_near, "path_id", cmap2, lookup2, n2, grid)
save("partition_methods.png", fig)

# 6. level.png: level="path" vs level="segment" -----------------------------------

labels_seg, net_seg, lines_seg = ramify.partition_priority(mask, root, tips=tips, level="segment")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
cmap, lookup, n = draw_labels(axes[0], mask_arr, labels_arr, 'level="path"')
plot_net_colored(axes[0], net, "path_id", cmap, lookup, n, grid)
cmap2, lookup2, n2 = draw_labels(axes[1], mask_arr, values(labels_seg), 'level="segment"')
plot_net_colored(axes[1], net_seg, "segment_id", cmap2, lookup2, n2, grid)
save("level.png", fig)

# 7. widths_regions.png: interpolate laplace / nearest / stations ----------------

w_lap = w_whole
w_near = values(ramify.width_regions_interpolate(labels, lines, method="nearest"))
w_st, stations = ramify.width_regions_stations(labels, lines, spacing=50)
w_st = values(w_st)

NORM, CMAP, BINS = width_norm_cmap(w_lap, w_near, w_st)
print(f"width bins ({len(BINS)}): {BINS}")

fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
for ax, arr, title in [
    (axes[0], w_lap, 'width_regions_interpolate(..., method="laplace")'),
    (axes[1], w_near, 'width_regions_interpolate(..., method="nearest")'),
]:
    im = ax.imshow(arr, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(title, fontfamily="monospace", fontsize=9)
    ax.set_axis_off()
im = axes[2].imshow(w_st, cmap=CMAP, norm=NORM, interpolation="nearest")
xs = [p.x for p in stations["station"]]
ys = [p.y for p in stations["station"]]
rr, cc = grid.to_rc(np.asarray(xs), np.asarray(ys))
axes[2].plot(cc, rr, "k.", markersize=2)
plot_geoms(axes[2], stations.geometry, grid, color="black", lw=0.4, alpha=0.6)
axes[2].set_title("width_regions_stations(..., spacing=50)", fontfamily="monospace", fontsize=9)
axes[2].set_axis_off()
fig.colorbar(
    plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
    ax=axes, fraction=0.025, pad=0.02, label="width (m)", extend="max",
)
save("widths_regions.png", fig, tight=False)

# 8. single_shape.png: one region, standalone centerline/width_interpolate/width_stations

region = labels == DEMO_PATH_ID
region_bool = values(region) > 0
bbox = bbox_of(region_bool)
r0, r1, c0, c1 = bbox

line = ramify.centerline(region)
w_single = values(ramify.width_interpolate(region, line))
w_st_single, stations_single = ramify.width_stations(region, line, spacing=50)
w_st_single = values(w_st_single)

norm_s, cmap_s, _ = width_norm_cmap(w_single, w_st_single)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

axes[0].imshow(region_bool[r0:r1, c0:c1], cmap="gray_r", alpha=0.35)
plot_geoms(axes[0], [line], grid, color="crimson", lw=1.8, r0=r0, c0=c0)
axes[0].set_title("centerline(shape)", fontfamily="monospace", fontsize=10)
axes[0].set_axis_off()

im = axes[1].imshow(w_single[r0:r1, c0:c1], cmap=cmap_s, norm=norm_s, interpolation="nearest")
axes[1].set_title("width_interpolate(shape, line)", fontfamily="monospace", fontsize=10)
axes[1].set_axis_off()

im = axes[2].imshow(w_st_single[r0:r1, c0:c1], cmap=cmap_s, norm=norm_s, interpolation="nearest")
plot_geoms(axes[2], stations_single.geometry, grid, color="black", lw=0.6, r0=r0, c0=c0)
axes[2].set_title("width_stations(shape, line, spacing=50)", fontfamily="monospace", fontsize=10)
axes[2].set_axis_off()

fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_s, cmap=cmap_s),
    ax=axes, fraction=0.025, pad=0.02, label="width (m)", extend="max",
)
save("single_shape.png", fig, tight=False)

# 9. open_boundary.png: widths without / with open_boundary ----------------------

rr, cc = np.ogrid[: mask_arr.shape[0], : mask_arr.shape[1]]
open_boundary = (~mask_arr) & (rr >= root[0] - 10) & (cc <= root[1] + 20)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    labels_ob, net_ob, lines_ob = ramify.partition_priority(
        mask, root, tips=tips, open_boundary=open_boundary
    )
    w_ob = values(
        ramify.width_regions_interpolate(labels_ob, lines_ob, open_boundary=open_boundary)
    )

norm_o, cmap_o, _ = width_norm_cmap(w_whole, w_ob)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, arr, title, mark in [
    (axes[0], w_whole, "width_regions_interpolate(labels, lines)", False),
    (axes[1], w_ob, "..., open_boundary=open_boundary", True),
]:
    ax.imshow(arr, cmap=cmap_o, norm=norm_o, interpolation="nearest")
    if mark:
        ax.imshow(
            np.ma.masked_where(~open_boundary, open_boundary.astype(float)),
            cmap=mcolors.ListedColormap(["red"]), alpha=0.35, interpolation="nearest",
        )
    ax.set_title(title, fontfamily="monospace", fontsize=9)
    ax.set_axis_off()
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm_o, cmap=cmap_o),
    ax=axes, fraction=0.03, pad=0.04, label="width (m)", extend="max",
)
save("open_boundary.png", fig, tight=False)
