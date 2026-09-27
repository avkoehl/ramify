# partition.py
import warnings
from collections import deque

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import label as cc_label
from skimage.segmentation import watershed
from shapely.geometry import LineString

from ._io import (
    make_grid,
    check_point,
    half_width_pixels,
    local_half_width,
    region_groups,
    relocate_line,
)
from ._skeleton import (
    skeleton_nodes,
    snap_paths,
    dijkstra_tree,
    endpoints,
    break_into_segments,
    prune_short_leaves,
    path_length,
    path_weight,
)
from .centerline import centerline

SQRT2 = np.sqrt(2.0)
# forward-only neighbour offsets; directed=False makes each bidirectional
_EDGES = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)]


def partition_priority(mask, root, tips=None, min_length=None, path_by="area",
                        level="path", open_boundary=None, pixel_size=None,
                        progress=None):
    """Assign every mask pixel to a path.

    Builds the tip -> root network (shared with `partition_nearest`; see
    `_build_network`), then claims territory in ascending `path_id` (the
    mainstem, path 1, first): each path claims mask pixels within some of its
    segments' local half-width, measured geodesically, so a wide-but-farther
    path can reach a pixel a narrow-but-nearer one cannot. Unclaimed remainder
    is watershed-filled.

    Returns `(labels, net, lines)`: `labels` is a `uint32` raster (0 outside
    the mask), `path_id` when `level="path"` or `segment_id` when
    `level="segment"`. `net` is a `GeoDataFrame`, one row per segment -- the
    routing record that explains the decomposition and `path_id` order.
    `lines` is a `GeoDataFrame`, one row per positive label in `labels`, each
    a centerline recomputed from that label's own pixels (not `net`'s
    skeleton, which is partitioning scaffolding) -- see `_region_lines`.
    """
    if level not in ("path", "segment"):
        raise ValueError(f"level must be 'path' or 'segment', got {level!r}")

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0
    df = _build_network(mask_bool, root, tips, min_length, path_by,
                        open_boundary, grid.pixel_size)

    path_seed = _rasterize_labels(df, mask_bool.shape, "path_id")
    allocation = _priority_allocate(mask_bool, path_seed, open_boundary, progress)

    if level == "segment":
        allocation = _subdivide_by_segment(allocation, df, mask_bool.shape)

    lines = _region_lines(df, allocation, grid, level)
    return grid.wrap(allocation), _to_geodataframe(df, grid), lines


def partition_nearest(mask, root, tips=None, min_length=None, path_by="area",
                       level="path", open_boundary=None, pixel_size=None):
    """Assign every mask pixel to the geodesically nearest path (or segment).

    Same network construction as `partition_priority`, but every mask pixel
    goes to whichever path (or, with `level="segment"`, segment) it can reach
    by the shortest within-mask route -- no ordering, no radius limits.
    Returns `(labels, net, lines)`; see `partition_priority`.
    """
    if level not in ("path", "segment"):
        raise ValueError(f"level must be 'path' or 'segment', got {level!r}")

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0
    df = _build_network(mask_bool, root, tips, min_length, path_by,
                        open_boundary, grid.pixel_size)

    path_seed = _rasterize_labels(df, mask_bool.shape, "path_id")
    allocation = _nearest_allocate(mask_bool, path_seed)

    if level == "segment":
        allocation = _subdivide_by_segment(allocation, df, mask_bool.shape)

    lines = _region_lines(df, allocation, grid, level)
    return grid.wrap(allocation), _to_geodataframe(df, grid), lines


# -- shared network construction ---------------------------------------------


def _build_network(mask_bool, root, tips, min_length, path_by, open_boundary, pixel_size):
    if tips is not None and min_length is not None:
        raise ValueError("tips and min_length are mutually exclusive")
    if path_by not in ("area", "length", "strahler"):
        raise ValueError(
            f"path_by must be 'area', 'length', or 'strahler', got {path_by!r}"
        )

    root = check_point(mask_bool, root, "root")
    nodes = skeleton_nodes(mask_bool)
    if not nodes:
        raise ValueError("skeletonization produced no pixels")

    tip_points = [check_point(mask_bool, t, "tip") for t in tips] if tips else []
    points = [root] + tip_points
    traces = snap_paths(points, nodes, mask_bool)
    if traces[0] is None:
        raise ValueError(f"root {root} cannot reach the skeleton within the mask")
    for t, tr in zip(tip_points, traces[1:]):
        if tr is None:
            raise ValueError(f"tip {t} cannot reach the skeleton within the mask")
    for tr in traces:
        nodes.update(tr)

    parent, dist = dijkstra_tree(nodes, root)

    if tip_points:
        for t in tip_points:
            if t not in parent:
                raise ValueError(f"tip {t} is not connected to the root")
        tip_nodes = tip_points
    else:
        reachable = set(parent.keys())
        if min_length is not None:
            reachable = prune_short_leaves(reachable, lambda j: min_length / pixel_size)
        tip_nodes = [n for n in endpoints(reachable) if n != root]
        if not tip_nodes:
            raise ValueError(
                "no tips found: skeleton has no endpoints reachable from root"
            )

    kept = set()
    for t in tip_nodes:
        n = t
        while n is not None and n not in kept:
            kept.add(n)
            n = parent[n]

    segments = break_into_segments(kept, parent, tip_nodes, root)
    edt = local_half_width(mask_bool, open_boundary, pixel_size)
    return _annotate(segments, edt, pixel_size, path_by, root)


def _annotate(segments, edt, pixel_size, path_by, root):
    n = len(segments)
    start_of = {seg[0]: i for i, seg in enumerate(segments)}
    downstream = [start_of.get(seg[-1]) for seg in segments]  # None at outlet
    children = [[] for _ in range(n)]
    for i, d in enumerate(downstream):
        if d is not None:
            children[d].append(i)

    length = np.array([path_length(s, pixel_size) for s in segments])
    weight = np.array([path_weight(s, edt, pixel_size) for s in segments])

    # post-order accumulation (iterative)
    strahler = np.zeros(n, dtype=int)
    sub_length = np.zeros(n)
    sub_weight = np.zeros(n)
    outlets = [i for i, d in enumerate(downstream) if d is None]
    order = []
    stack = list(outlets)
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i])
    for i in reversed(order):  # leaves first
        if not children[i]:
            strahler[i] = 1
            sub_length[i] = length[i]
            sub_weight[i] = weight[i]
        else:
            orders = strahler[children[i]]
            m = orders.max()
            strahler[i] = m + 1 if (orders == m).sum() > 1 else m
            sub_length[i] = length[i] + sub_length[children[i]].max()
            sub_weight[i] = weight[i] + sub_weight[children[i]].sum()

    key = {
        "area": lambda i: (sub_weight[i],),
        "length": lambda i: (sub_length[i],),
        "strahler": lambda i: (strahler[i], sub_length[i]),
    }[path_by]

    # Heavy-path decomposition: walk upstream from each outlet, continuing
    # along the heaviest child; other children start new (raw) paths. `depth`
    # counts position upstream from each path's outlet segment (0-based),
    # used for the final, deterministic segment ordering below.
    raw_path = np.zeros(n, dtype=int)  # 0 = unlabeled
    depth = np.zeros(n, dtype=int)
    outlet_seg = {}
    next_label = 0

    queue = deque(sorted(sorted(outlets), key=key, reverse=True))
    while queue:
        cur = queue.popleft()
        if raw_path[cur]:
            continue
        next_label += 1
        outlet_seg[next_label] = cur
        d = 0
        while True:
            raw_path[cur] = next_label
            depth[cur] = d
            d += 1
            preds = [c for c in children[cur] if not raw_path[c]]
            if not preds:
                break
            preds = sorted(sorted(preds), key=key, reverse=True)
            queue.extend(preds[1:])
            cur = preds[0]

    # Path numbering: sort paths by their statistic at the path's most
    # downstream segment, descending, ties by that segment's (deterministic,
    # pre-renumbering) construction index -- so path_id is a size rank and
    # every parent path is numbered before its tributaries.
    labels = sorted(
        outlet_seg,
        key=lambda p: (tuple(-v for v in key(outlet_seg[p])), outlet_seg[p]),
    )
    new_path_id = {label: rank + 1 for rank, label in enumerate(labels)}
    path_id = np.array([new_path_id[p] for p in raw_path])

    # Segment numbering: ordered by path_id, then downstream -> upstream.
    final_order = sorted(range(n), key=lambda i: (path_id[i], depth[i]))
    final_id = np.empty(n, dtype=int)
    final_id[final_order] = np.arange(1, n + 1)

    df = pd.DataFrame(
        {
            "segment_id": final_id,
            "path_id": path_id,
            "strahler": strahler,
            "length": length,
            "weight": weight,
            "downstream_segment_id": pd.array(
                [int(final_id[d]) if d is not None else pd.NA for d in downstream],
                dtype="Int64",
            ),
            "pixels": segments,
        }
    )
    return df.sort_values("segment_id", ignore_index=True)


# -- rasterizing segments/paths into seed labels ------------------------------


def _rasterize_labels(df, shape, column):
    # One label per pixel; a junction pixel shared between rows takes the
    # smallest `column` value (e.g. the mainstem, path_id == 1, wins a
    # mainstem/tributary junction pixel).
    pixels = [np.asarray(p).reshape(-1, 2) for p in df["pixels"]]
    counts = np.fromiter((len(p) for p in pixels), np.intp, len(pixels))
    rc = np.concatenate(pixels) if pixels else np.empty((0, 2), np.intp)
    flat = rc[:, 0].astype(np.intp) * shape[1] + rc[:, 1].astype(np.intp)
    values = np.repeat(df[column].to_numpy(), counts)

    order = np.lexsort((values, flat))
    flat = flat[order]
    values = values[order]
    winner = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])

    arr = np.zeros(shape, dtype=np.uint32)
    arr.ravel()[flat[winner]] = values[winner]
    return arr


def _to_geodataframe(df, grid):
    # Segment pixels are stored upstream -> downstream internally (how the
    # tree walk naturally produces them); the public geometry is oriented
    # downstream -> upstream (root-first), so each pixel list is reversed here.
    import geopandas as gpd

    keep, geoms = [], []
    for i, pixels in enumerate(df["pixels"]):
        if len(pixels) < 2:
            continue
        rows, cols = zip(*reversed(pixels))
        xs, ys = grid.to_xy(np.array(rows), np.array(cols))
        geoms.append(LineString(np.column_stack([xs, ys])))
        keep.append(i)

    out = df.iloc[keep].drop(columns=["pixels"]).copy()
    out["geometry"] = geoms
    return gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs)


# -- allocation ----------------------------------------------------------------


def _priority_allocate(mask_bool, seed_arr, open_boundary, progress):
    H, W = mask_bool.shape
    radius = half_width_pixels(mask_bool, open_boundary)
    allocation = np.zeros(mask_bool.shape, dtype=np.uint32)

    groups = list(region_groups(seed_arr))  # ascending label == biggest path first
    for i, (label, idx) in enumerate(groups):
        rr, cc = idx // W, idx % W
        keep = mask_bool[rr, cc]
        if not keep.any():
            continue
        rr, cc = rr[keep], cc[keep]
        rad = radius[rr, cc]
        R = float(rad.max())  # farthest this path can reach = search bound

        pad = int(np.ceil(R)) + 1
        r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + pad + 1, H)
        c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + pad + 1, W)

        if progress is not None:
            progress(i, len(groups), int(label), (r1 - r0) * (c1 - c0))

        seed_local = np.stack([rr - r0, cc - c0], axis=1)
        tube = _reach(mask_bool[r0:r1, c0:c1], seed_local, rad, R)
        sub = allocation[r0:r1, c0:c1]
        sub[tube & (sub == 0)] = label  # keep only where no bigger path won

    claimed = allocation > 0
    unclaimed = mask_bool & ~claimed
    if unclaimed.any() and claimed.any():
        # connectivity=2 (8-connected) matches the skeleton/tree/_reach graph
        # above; the 2D default (4-connected) can't flood into a mask pixel
        # that touches the rest of the shape only at a corner.
        allocation = watershed(
            image=distance_transform_edt(~claimed),
            markers=allocation,
            mask=mask_bool,
            connectivity=2,
        ).astype(np.uint32)

    return allocation


def _reach(mask_win, seed_rc, radii, R):
    # Pixels within some seed's local half-width, geodesically, inside the
    # window. A virtual super-source is wired to each seed s with edge weight
    # R - radius(s) (>= 0), so one bounded search from it gives, per pixel q,
    #   dist_V(q) = R + min_s ( geodist(q, s) - radius(s) ),
    # and dist_V(q) <= R is exactly "some seed's radius reaches q".
    h, w = mask_win.shape
    ys, xs = np.nonzero(mask_win)
    M = ys.size
    if M == 0:
        return np.zeros((h, w), dtype=bool)
    ids = np.full((h, w), -1, dtype=np.int64)
    ids[ys, xs] = np.arange(M)
    V = M  # super-source node id

    rows, cols, wts = [], [], []
    for dr, dc, step in _EDGES:
        ny, nx = ys + dr, xs + dc
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nb = np.where(ok, ids[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
        keep = nb >= 0
        rows.append(ids[ys[keep], xs[keep]])
        cols.append(nb[keep])
        wts.append(np.full(int(keep.sum()), step))

    sids = ids[seed_rc[:, 0], seed_rc[:, 1]]
    ok = sids >= 0
    rows.append(np.full(int(ok.sum()), V))
    cols.append(sids[ok])
    wts.append(R - radii[ok])

    n = M + 1
    graph = csr_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    dist = dijkstra(graph, directed=False, indices=V, limit=R)
    out = np.zeros((h, w), dtype=bool)
    out[ys, xs] = dist[:M] <= R
    return out


def _nearest_allocate(mask_bool, seed_arr):
    markers = np.where(mask_bool, seed_arr, 0).astype(np.int64)
    if not (markers > 0).any():
        raise ValueError("no seed pixels found inside the mask")
    return watershed(
        image=distance_transform_edt(markers == 0),
        markers=markers,
        mask=mask_bool,
        connectivity=2,
    ).astype(np.uint32)


def _reference_line_pixels(df, label, column):
    # The ordered network pixels for one region, downstream -> upstream:
    # its segments (one, for level="segment"; all of a path's, concatenated
    # in already-downstream-to-upstream segment_id order, for level="path"),
    # each reversed from its internal upstream->downstream storage, sharing
    # each junction pixel once.
    rows = df[df[column] == label]
    if column == "path_id":
        rows = rows.sort_values("segment_id")
    pixels = []
    for p in rows["pixels"]:
        seg = list(reversed(p))
        if pixels and pixels[-1] == seg[0]:
            seg = seg[1:]
        pixels.extend(seg)
    return pixels


def _compute_region_line(component_bool, start, end, grid):
    # Crop to the component's own bounding box before calling centerline() --
    # skeletonize/label/snap all cost O(array shape), not O(true pixels), so
    # this keeps the total cost close to one pass over the mask rather than
    # (regions x mask area). centerline() has no way to take a custom
    # transform, so it's called on a pixel_size=1, origin-(0, 0) crop and the
    # result is relocated back into `grid`'s true coordinates afterward.
    rows, cols = np.nonzero(component_bool)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    cropped = component_bool[r0:r1, c0:c1].astype(np.uint8)

    if start is None:
        line_local = centerline(cropped)
    else:
        line_local = centerline(
            cropped, root=(start[0] - r0, start[1] - c0), tip=(end[0] - r0, end[1] - c0)
        )
    return relocate_line(line_local, r0, c0, grid)


def _region_lines(df, allocation, grid, level):
    # One recomputed centerline per positive label in `allocation`. See
    # DESIGN.md addendum "Computing region centerlines" for the algorithm and
    # "Fallbacks" for the source="diameter"/"empty" cases below.
    column = "path_id" if level == "path" else "segment_id"
    seg_to_path = dict(zip(df["segment_id"], df["path_id"]))

    labels_present = np.unique(allocation[allocation > 0]).tolist()
    region_id, path_id, segment_id = [], [], []
    length, area, source, island_pixels, geometry = [], [], [], [], []

    n_diameter = n_empty = 0
    total_islands = 0

    for label in labels_present:
        region_bool = allocation == label
        area_px = int(region_bool.sum())
        ref_pixels = _reference_line_pixels(df, label, column)

        comp = cc_label(region_bool, connectivity=2)
        comp_groups = dict(region_groups(comp))
        ref_counts = {}
        for r, c in ref_pixels:
            cid = comp[r, c]
            if cid > 0:
                ref_counts[cid] = ref_counts.get(cid, 0) + 1

        # most reference pixels wins; ties -> larger component, then lower
        # smallest flat index (region_groups' per-component indices are
        # already sorted ascending)
        winner = min(
            comp_groups,
            key=lambda cid: (-ref_counts.get(cid, 0), -comp_groups[cid].size, comp_groups[cid][0]),
        )
        winning_mask = comp == winner
        winning_size = int(comp_groups[winner].size)
        islands = area_px - winning_size
        total_islands += islands

        in_region = [i for i, (r, c) in enumerate(ref_pixels) if comp[r, c] == winner]

        if not in_region:
            line = _compute_region_line(winning_mask, None, None, grid)
            src = "diameter"
            n_diameter += 1
        else:
            start, end = ref_pixels[in_region[0]], ref_pixels[in_region[-1]]
            adjacent = abs(start[0] - end[0]) <= 1 and abs(start[1] - end[1]) <= 1
            if start == end or adjacent or winning_size < 2:
                line = LineString()
                src = "empty"
                n_empty += 1
            else:
                line = _compute_region_line(winning_mask, start, end, grid)
                src = "network"

        region_id.append(int(label))
        path_id.append(int(label) if level == "path" else int(seg_to_path[label]))
        segment_id.append(pd.NA if level == "path" else int(label))
        length.append(float(line.length))
        area.append(area_px * grid.pixel_size**2)
        source.append(src)
        island_pixels.append(int(islands))
        geometry.append(line)

    if n_diameter:
        warnings.warn(
            f"{n_diameter} region(s): the network line never entered the region "
            "(e.g. a short tributary swallowed by its parent's claim); used the "
            "region's own geodesic diameter instead"
        )
    if n_empty:
        warnings.warn(f"{n_empty} region(s) too small for a centerline; empty geometry")
    if total_islands:
        warnings.warn(
            f"{total_islands} region pixel(s), across disconnected components, "
            "were excluded from their region's centerline"
        )

    import geopandas as gpd

    out = pd.DataFrame(
        {
            "region_id": region_id,
            "path_id": path_id,
            "segment_id": pd.array(segment_id, dtype="Int64"),
            "length": length,
            "area": area,
            "source": source,
            "island_pixels": island_pixels,
            "geometry": geometry,
        }
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs)


def _subdivide_by_segment(path_allocation, df, shape):
    # Subdivide each path's territory into segment-level territories. Each
    # territory is seeded only by its own path's segments, so neighbouring
    # paths' labels (e.g. shared junction pixels) never bleed across
    # boundaries.
    territories = dict(region_groups(path_allocation))
    out = np.zeros(shape, dtype=np.uint32)
    _, w = shape
    for path_id, group in df.groupby("path_id"):
        idx = territories.get(int(path_id))
        if idx is None:
            continue
        rows, cols = np.divmod(idx, w)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1

        territory = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
        territory[rows - r0, cols - c0] = 1

        seeds = np.zeros(territory.shape, dtype=np.uint32)
        for _, row in group.iterrows():
            rc = np.asarray(row["pixels"])
            inside = (
                (rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1)
            )
            rc = rc[inside]
            seeds[rc[:, 0] - r0, rc[:, 1] - c0] = row["segment_id"]
        seeds = np.where(territory == 1, seeds, 0)
        if not (seeds > 0).any():
            continue

        sub = _nearest_allocate(territory.astype(bool), seeds)
        hit = sub > 0
        out[r0:r1, c0:c1][hit] = sub[hit]

    return out
