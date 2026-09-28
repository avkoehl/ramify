"""Regenerate the README figures against the bundled toy dataset.

    python scripts/gen_images.py

See notes.md for the figure plan this fulfills.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

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


def crop(arr, win):
    r0, r1, c0, c1 = win
    return arr[r0:r1, c0:c1]


def draw_labels_win(ax, mask_arr, labels_arr, win, title=None):
    # Colors come from the full raster so a region keeps its color when zoomed.
    mapped, cmap, lookup, n = label_cmap(labels_arr)
    ax.imshow(crop(mask_arr, win), cmap="gray_r", alpha=0.25)
    ax.imshow(crop(mapped, win), cmap=cmap, vmin=0, vmax=max(n - 1, 1), interpolation="nearest")
    if title:
        ax.set_title(title, fontfamily="monospace", fontsize=9)
    ax.set_axis_off()
    return cmap, lookup, n


def draw_stations(ax, w_arr, stations, grid, norm, cmap, win=None, points=True,
                  cell_lw=0.4, marker=2, centerlines=None):
    r0, c0 = (win[0], win[2]) if win else (0, 0)
    ax.imshow(crop(w_arr, win) if win else w_arr, cmap=cmap, norm=norm, interpolation="nearest")
    plot_geoms(ax, stations.geometry, grid, color="black", lw=cell_lw, alpha=0.6, r0=r0, c0=c0)
    if centerlines is not None:
        plot_geoms(ax, centerlines, grid, color="white", lw=1.6, r0=r0, c0=c0)
        plot_geoms(ax, centerlines, grid, color="crimson", lw=0.9, r0=r0, c0=c0)
    if points:
        xs = np.asarray([p.x for p in stations["station"]])
        ys = np.asarray([p.y for p in stations["station"]])
        rr, cc = grid.to_rc(xs, ys)
        ax.plot(cc - c0, rr - r0, "o", color="black", mfc="white", markersize=marker,
                mew=0.6, zorder=5)
    if win:
        ax.set_xlim(-0.5, win[3] - win[2] - 0.5)
        ax.set_ylim(win[1] - win[0] - 0.5, -0.5)
    ax.set_axis_off()


def arrow_axis(ax, text):
    ax.set_axis_off()
    ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=2, color="0.3", mutation_scale=20))
    ax.text(0.5, 0.56, text, ha="center", va="bottom", fontfamily="monospace",
            fontsize=9, rotation=0, transform=ax.transAxes, wrap=True)


def width_colorbar(fig, norm, cmap, ax, **kw):
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="width (m)", extend="max", **kw)


# -- pipeline ------------------------------------------------------------------

mask, root, tips = load()
mask_arr = values(mask) == 1
_, grid = make_grid(mask)
print(f"mask: {mask_arr.shape}, root: {root}, tips: {len(tips)}")

labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
labels_arr = values(labels)
w_st, stations = ramify.width_regions_stations(labels, lines, spacing=50)
w_st = values(w_st)
w_lap = values(ramify.width_regions_interpolate(labels, lines))
w_near = values(ramify.width_regions_interpolate(labels, lines, method="nearest"))

# One width scale for every figure of the partitioned shape.
NORM, CMAP, BINS = width_norm_cmap(w_st, w_lap, w_near)
print(f"width bins ({len(BINS)}): {BINS}")

FULL = (0, mask_arr.shape[0], 0, mask_arr.shape[1])
JUNCTION_WIN = (115, 215, 150, 265)  # mainstem with paths 3 and 6 joining it
BRANCH_WIN = (140, 215, 150, 275)    # path 3 where it joins the mainstem

# 1. abstract.png: mask+root+tips -> labels+lines -> station widths --------------

fig = plt.figure(figsize=(17, 5.6))
gs = GridSpec(1, 6, figure=fig, width_ratios=[1, 0.22, 1, 0.22, 1, 0.04], wspace=0.02)
ax_in, ax_a1, ax_lab, ax_a2, ax_w, ax_cb = (fig.add_subplot(gs[0, i]) for i in range(6))

ax_in.imshow(np.ma.masked_where(~mask_arr, mask_arr.astype(int)), cmap="gray", vmin=0, vmax=2)
ax_in.plot(root[1], root[0], "k*", markersize=14, mec="w", label="root")
tr, tc = zip(*tips)
ax_in.plot(tc, tr, "wo", markersize=5, mec="k", label="tips")
ax_in.legend(loc="lower right", fontsize=8)
ax_in.set_title("mask, root, tips", fontfamily="monospace", fontsize=10)
ax_in.set_axis_off()

arrow_axis(ax_a1, "partition_\npriority")

draw_labels_win(ax_lab, mask_arr, labels_arr, FULL)
plot_geoms(ax_lab, lines.geometry, grid, color="black", lw=1.0)
ax_lab.set_title("labels + lines", fontfamily="monospace", fontsize=10)

arrow_axis(ax_a2, "width_regions_\nstations")

draw_stations(ax_w, w_st, stations, grid, NORM, CMAP, points=False, cell_lw=0.3)
ax_w.set_title("station widths", fontfamily="monospace", fontsize=10)

fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=ax_cb,
             label="width (m)", extend="max")
save("abstract.png", fig, tight=False)

# 2. quickstart.png: labels + lines | station cells colored by width -------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
draw_labels_win(axes[0], mask_arr, labels_arr, FULL, "labels, lines")
plot_geoms(axes[0], lines.geometry, grid, color="black", lw=1.0)
draw_stations(axes[1], w_st, stations, grid, NORM, CMAP, marker=1.5)
axes[1].set_title("width_regions_stations(labels, lines, spacing=50)",
                  fontfamily="monospace", fontsize=9)
width_colorbar(fig, NORM, CMAP, axes, fraction=0.025, pad=0.02)
save("quickstart.png", fig, tight=False)

# 3. tips_options.png: tips= / min_length= / neither, each showing net -----------

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, net_ml, _ = ramify.partition_priority(mask, root, min_length=15.0)
    _, net_all, _ = ramify.partition_priority(mask, root)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, net_i, title in [
    (axes[0], net, f"tips=tips ({len(tips)} tips)"),
    (axes[1], net_ml, f"min_length=15 ({net_ml['path_id'].nunique()} paths)"),
    (axes[2], net_all, f"neither ({net_all['path_id'].nunique()} paths)"),
]:
    ax.imshow(mask_arr, cmap="gray_r", alpha=0.25)
    cmap = plt.get_cmap("hsv")
    ids = sorted(net_i["path_id"].unique())
    lookup = {pid: i for i, pid in enumerate(ids)}
    for _, row in net_i.iterrows():
        color = cmap((lookup[row["path_id"]] * 0.61803398875) % 1.0)
        plot_geoms(ax, [row.geometry], grid, color=color, lw=1.0)
    ax.set_title(title, fontfamily="monospace", fontsize=10)
    ax.set_axis_off()
save("tips_options.png", fig)

# 4. widths_stations.png: zoom on one junction, centerline + stations + cells -----

win = BRANCH_WIN
near_ids = np.unique(crop(labels_arr, win))
near_ids = near_ids[near_ids > 0]
st_win = stations[stations["region_id"].isin(near_ids)]

fig, ax = plt.subplots(figsize=(10, 6))
draw_stations(ax, w_st, st_win, grid, NORM, CMAP, win=win, cell_lw=0.7, marker=4,
              centerlines=lines[lines["region_id"].isin(near_ids)].geometry)
ax.plot([], [], color="crimson", lw=1.2, label="centerline (lines)")
ax.plot([], [], "o", color="black", mfc="white", markersize=4, mew=0.6, label="station")
ax.plot([], [], color="black", lw=0.7, label="station cell")
ax.legend(loc="center right", fontsize=8)
ax.set_title("width_regions_stations(labels, lines, spacing=50)",
             fontfamily="monospace", fontsize=9)
width_colorbar(fig, NORM, CMAP, ax, fraction=0.03, pad=0.02)
save("widths_stations.png", fig, tight=False)

# 5. partition_methods.png: priority vs nearest, zoomed on a junction -----------

labels_near, net_near, _ = ramify.partition_nearest(mask, root, tips=tips)
win = JUNCTION_WIN

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, lab, net_i, title in [
    (axes[0], labels_arr, net, "partition_priority"),
    (axes[1], values(labels_near), net_near, "partition_nearest"),
]:
    cmap, lookup, n = draw_labels_win(ax, mask_arr, lab, win, title)
    plot_net_colored(ax, net_i, "path_id", cmap, lookup, n, grid, r0=win[0], c0=win[2])
    ax.set_xlim(-0.5, win[3] - win[2] - 0.5)
    ax.set_ylim(win[1] - win[0] - 0.5, -0.5)
save("partition_methods.png", fig)

# 6. level.png: level="path" vs level="segment" -----------------------------------

labels_seg, net_seg, _ = ramify.partition_priority(mask, root, tips=tips, level="segment")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
cmap, lookup, n = draw_labels(axes[0], mask_arr, labels_arr, 'level="path"')
plot_net_colored(axes[0], net, "path_id", cmap, lookup, n, grid)
cmap2, lookup2, n2 = draw_labels(axes[1], mask_arr, values(labels_seg), 'level="segment"')
plot_net_colored(axes[1], net_seg, "segment_id", cmap2, lookup2, n2, grid)
save("level.png", fig)

# 7. widths_regions.png: interpolate laplace / nearest ----------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, arr, title in [
    (axes[0], w_lap, 'width_regions_interpolate(..., method="laplace")'),
    (axes[1], w_near, 'width_regions_interpolate(..., method="nearest")'),
]:
    ax.imshow(arr, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(title, fontfamily="monospace", fontsize=9)
    ax.set_axis_off()
width_colorbar(fig, NORM, CMAP, axes, fraction=0.025, pad=0.02)
save("widths_regions.png", fig, tight=False)

# 8. open_boundary.png: widths without / with open_boundary ----------------------

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

norm_o, cmap_o, _ = width_norm_cmap(w_lap, w_ob)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, arr, title, mark in [
    (axes[0], w_lap, "width_regions_interpolate(labels, lines)", False),
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
width_colorbar(fig, norm_o, cmap_o, axes, fraction=0.03, pad=0.04)
save("open_boundary.png", fig, tight=False)

# 9. single_shape.png: one region; centerline + station widths, then interpolate --

region = labels == DEMO_PATH_ID
region_bool = values(region) > 0
win = bbox_of(region_bool)

line = ramify.centerline(region)
w_st_single, stations_single = ramify.width_stations(region, line, spacing=50)
w_st_single = values(w_st_single)
w_single = values(ramify.width_interpolate(region, line))

norm_s, cmap_s, _ = width_norm_cmap(w_st_single, w_single)

fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
draw_stations(axes[0], w_st_single, stations_single, grid, norm_s, cmap_s, win=win,
              cell_lw=0.7, marker=4, centerlines=[line])
axes[0].set_title("centerline(shape) + width_stations(shape, line, spacing=50)",
                  fontfamily="monospace", fontsize=9)
axes[1].imshow(crop(w_single, win), cmap=cmap_s, norm=norm_s, interpolation="nearest")
axes[1].set_title("width_interpolate(shape, line)", fontfamily="monospace", fontsize=9)
axes[1].set_axis_off()
width_colorbar(fig, norm_s, cmap_s, axes, fraction=0.03, pad=0.02)
save("single_shape.png", fig, tight=False)
