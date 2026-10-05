# partition.py
import warnings
from collections import deque

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import label as cc_label
from shapely.geometry import LineString, Point
from shapely.ops import substring

from ._io import (
    make_grid,
    crop_grid,
    check_point,
    half_width_pixels,
    local_half_width,
    region_groups,
    relocate_line,
    rasterize_line,
    polygonize_mask,
    smoother,
    smooth_line,
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
    disk_cut,
)
from .centerline import centerline

SQRT2 = np.sqrt(2.0)
# forward-only neighbour offsets; directed=False makes each bidirectional
_EDGES = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)]


def partition_priority(mask, root, tips=None, min_length=None, path_by="area",
                        level="path", open_boundary=None, smooth=None,
                        pixel_size=None, progress=None):
    """Assign every mask pixel to a path.

    Builds the tip -> root network (shared with `partition_nearest`; see
    `_build_network`), then claims territory in ascending `path_id` (the
    mainstem, path 1, first): each path claims mask pixels within some of its
    segments' local half-width, measured geodesically, so a wide-but-farther
    path can reach a pixel a narrow-but-nearer one cannot. Each unclaimed
    pixel then joins the geodesically nearest claimed one.

    Returns `(labels, net, lines)`: `labels` is a `uint32` raster (0 outside
    the mask), `path_id` when `level="path"` or `segment_id` when
    `level="segment"`. `net` is a `GeoDataFrame`, one row per segment -- the
    routing record that explains the decomposition and `path_id` order.
    `lines` is a `GeoDataFrame`, one row per positive label in `labels`. At
    path level each is a centerline recomputed from that path's own pixels
    (not `net`'s skeleton, which is partitioning scaffolding) -- see
    `_region_lines`. At segment level, each path's line is cut where its
    junctions project onto it, and each segment's region is the part of its
    path's region geodesically nearest its piece -- see `_split_by_segment`.

    `smooth` smooths each path line after it is computed, as in `centerline`
    (`None`, `"chaikin"`, `"taubin"`, or a callable), before any segment
    cuts; `length` is measured on the smoothed line. `net` is never smoothed.
    Raises `ValueError` if a smoothed line leaves the mask.
    """
    if level not in ("path", "segment"):
        raise ValueError(f"level must be 'path' or 'segment', got {level!r}")
    smooth_fn = smoother(smooth)

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0
    df = _build_network(mask_bool, root, tips, min_length, path_by,
                        open_boundary, grid.pixel_size)

    path_seed = _rasterize_labels(df, mask_bool.shape, "path_id")
    allocation = _priority_allocate(mask_bool, path_seed, open_boundary, progress)

    lines = _region_lines(df, allocation, grid, smooth_fn)
    if level == "segment":
        allocation, lines = _split_by_segment(allocation, df, lines, grid)
    return grid.wrap(allocation), _to_geodataframe(df, grid), lines


def partition_nearest(mask, root, tips=None, min_length=None, path_by="area",
                       level="path", open_boundary=None, smooth=None,
                       pixel_size=None):
    """Assign every mask pixel to the geodesically nearest path (or segment).

    Same network construction as `partition_priority`, but every mask pixel
    goes to whichever path it can reach by the shortest within-mask route --
    no ordering, no radius limits. `level="segment"` then splits each path's
    region as in `partition_priority`.
    Returns `(labels, net, lines)`; `smooth` as in `partition_priority`.
    """
    if level not in ("path", "segment"):
        raise ValueError(f"level must be 'path' or 'segment', got {level!r}")
    smooth_fn = smoother(smooth)

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0
    df = _build_network(mask_bool, root, tips, min_length, path_by,
                        open_boundary, grid.pixel_size)

    path_seed = _rasterize_labels(df, mask_bool.shape, "path_id")
    allocation = _nearest_allocate(mask_bool, path_seed)

    lines = _region_lines(df, allocation, grid, smooth_fn)
    if level == "segment":
        allocation, lines = _split_by_segment(allocation, df, lines, grid)
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
    segments = _cut_end_segments(
        segments, root, set(tip_points), half_width_pixels(mask_bool, None), mask_bool
    )
    edt = local_half_width(mask_bool, open_boundary, pixel_size)
    return _annotate(segments, edt, pixel_size, path_by, root)


def _cut_end_segments(segments, root, anchors, half_width, mask_bool):
    # Straighten the network where it meets the root and each given tip (see
    # `disk_cut`). Each cut is confined to the one segment touching that
    # point: a segment's other end is a junction, so the tree's topology and
    # shared junction pixels are untouched. Auto-detected tips are skeleton
    # endpoints already and have nothing to cut toward. Segments run
    # upstream -> downstream, so the root is at seg[-1] and a tip at seg[0].
    out = []
    for seg in segments:
        if seg[-1] == root:
            seg = disk_cut(seg[::-1], half_width, mask_bool, fill=True)[::-1]
        if seg[0] in anchors:
            seg = disk_cut(seg, half_width, mask_bool, fill=True)
        out.append(seg)
    return out


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
        # each unclaimed pixel joins the geodesically nearest claimed one;
        # claimed pixels are seeds of themselves and keep their label
        allocation = _nearest_allocate(mask_bool, allocation)

    return allocation


def _reach(mask_win, seed_rc, radii, R):
    # Pixels within some seed's local half-width, geodesically, inside the
    # window. A virtual super-source is wired to each seed s with edge weight
    # R - radius(s) (>= 0), so one bounded search from it gives, per pixel q,
    #   dist_V(q) = R + min_s ( geodist(q, s) - radius(s) ),
    # and dist_V(q) <= R is exactly "some seed's radius reaches q".
    h, w = mask_win.shape
    ys, xs, ids, rows, cols, wts = _pixel_graph(mask_win)
    M = ys.size
    if M == 0:
        return np.zeros((h, w), dtype=bool)
    V = M  # super-source node id

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


def _pixel_graph(mask_bool):
    # The mask's pixels as graph nodes, 8-connected, steps of 1 / sqrt(2):
    # the same metric as the skeleton tree. Returns node coordinates, the
    # (h, w) node-id raster (-1 off the mask), and forward-only edge lists
    # (row ids, col ids, weights) to be used with directed=False.
    h, w = mask_bool.shape
    ys, xs = np.nonzero(mask_bool)
    ids = np.full((h, w), -1, dtype=np.int64)
    ids[ys, xs] = np.arange(ys.size)

    rows, cols, wts = [], [], []
    for dr, dc, step in _EDGES:
        ny, nx = ys + dr, xs + dc
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nb = np.where(ok, ids[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
        keep = nb >= 0
        rows.append(ids[ys[keep], xs[keep]])
        cols.append(nb[keep])
        wts.append(np.full(int(keep.sum()), step))
    return ys, xs, ids, rows, cols, wts


def _nearest_allocate(mask_bool, seed_arr):
    # Geodesic Voronoi: every mask pixel takes the label of the seed pixel it
    # reaches by the shortest within-mask route (one multi-source Dijkstra;
    # `sources` records which seed each pixel's shortest path started from).
    # Seed pixels keep their own label; pixels no seed can reach stay 0.
    ys, xs, _, rows, cols, wts = _pixel_graph(mask_bool)
    seed_vals = seed_arr[ys, xs]
    seed_ids = np.flatnonzero(seed_vals > 0)
    if seed_ids.size == 0:
        raise ValueError("no seed pixels found inside the mask")

    M = ys.size
    graph = csr_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(M, M),
    )
    _, _, sources = dijkstra(
        graph, directed=False, indices=seed_ids, min_only=True, return_predecessors=True
    )
    out = np.zeros(mask_bool.shape, dtype=np.uint32)
    reached = sources >= 0
    out[ys[reached], xs[reached]] = seed_vals[sources[reached]]
    return out


def _reference_line_pixels(df, path_id):
    # The ordered network pixels for one path, downstream -> upstream: its
    # segments, concatenated in (already downstream-to-upstream) segment_id
    # order, each reversed from its internal upstream->downstream storage,
    # sharing each junction pixel once.
    rows = df[df["path_id"] == path_id].sort_values("segment_id")
    pixels = []
    for p in rows["pixels"]:
        seg = list(reversed(p))
        if pixels and pixels[-1] == seg[0]:
            seg = seg[1:]
        pixels.extend(seg)
    return pixels


def _winning_component(region_bool, ref_pixels):
    # The connected component of a region its line belongs to: most
    # reference pixels wins; ties -> larger component, then lower smallest
    # flat index (region_groups' per-component indices are already sorted
    # ascending). Returns (component labels, winner id, winner size).
    comp = cc_label(region_bool, connectivity=2)
    comp_groups = dict(region_groups(comp))
    ref_counts = {}
    for r, c in ref_pixels:
        cid = comp[r, c]
        if cid > 0:
            ref_counts[cid] = ref_counts.get(cid, 0) + 1
    winner = min(
        comp_groups,
        key=lambda cid: (-ref_counts.get(cid, 0), -comp_groups[cid].size, comp_groups[cid][0]),
    )
    return comp, winner, int(comp_groups[winner].size)


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


def _region_lines(df, allocation, grid, smooth_fn=None):
    # One recomputed centerline per path label in `allocation`. See
    # DESIGN.md addendum "Computing region centerlines" for the algorithm and
    # "Fallbacks" for the source="diameter"/"empty" cases below.
    records = []
    n_diameter = n_empty = 0
    total_islands = 0
    shape_poly = polygonize_mask(allocation > 0, grid) if smooth_fn is not None else None

    for label in np.unique(allocation[allocation > 0]).tolist():
        region_bool = allocation == label
        area_px = int(region_bool.sum())
        ref_pixels = _reference_line_pixels(df, label)

        comp, winner, winning_size = _winning_component(region_bool, ref_pixels)
        winning_mask = comp == winner
        islands = area_px - winning_size
        total_islands += islands

        in_region = [i for i, (r, c) in enumerate(ref_pixels) if comp[r, c] == winner]

        if not in_region:
            try:
                line = _compute_region_line(winning_mask, None, None, grid)
                src = "diameter"
                n_diameter += 1
            except ValueError:
                # a blob of a few pixels skeletonizes to one pixel: no ends
                line = LineString()
                src = "empty"
                n_empty += 1
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

        line = smooth_line(
            line, smooth_fn, shape_poly, grid.pixel_size, f"line for region {label}"
        )
        records.append((int(label), int(label), pd.NA, line, area_px, src, islands))

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
    return _lines_frame(records, grid)


def _lines_frame(records, grid):
    # records: (region_id, path_id, segment_id, line, area_px, source, islands)
    import geopandas as gpd

    region_id, path_id, segment_id, geometry, area_px, source, islands = (
        zip(*records) if records else ([],) * 7
    )
    out = pd.DataFrame(
        {
            "region_id": pd.array(region_id, dtype="int64"),
            "path_id": pd.array(path_id, dtype="int64"),
            "segment_id": pd.array(segment_id, dtype="Int64"),
            "length": [float(g.length) for g in geometry],
            "area": [a * grid.pixel_size**2 for a in area_px],
            "source": list(source),
            "island_pixels": pd.array(islands, dtype="int64"),
            "geometry": list(geometry),
        }
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs)


def _cut_path_line(line, source, group, grid):
    # Split one path's line into one piece per segment (`group`: the path's
    # rows of `df`, in segment_id = downstream -> upstream order), cutting
    # where each junction projects onto the line. The line runs downstream ->
    # upstream when it came from the network; a "diameter" line has no set
    # orientation, so it is flipped if its end is nearer the path's outlet.
    # A segment whose junctions project onto the same spot (e.g. one lying
    # wholly inside its parent's region, downstream of where the line
    # starts) gets an empty piece.
    n = len(group)
    if line.is_empty:
        return [LineString()] * n

    def point(rc):
        x, y = grid.to_xy(np.array([rc[0]]), np.array([rc[1]]))
        return Point(float(x[0]), float(y[0]))

    pixels = list(group["pixels"])
    if source != "network":
        outlet = point(pixels[0][-1])
        if Point(line.coords[-1]).distance(outlet) < Point(line.coords[0]).distance(outlet):
            line = LineString(line.coords[::-1])

    # a segment's junction with the next one upstream is its own first pixel
    cuts = [line.project(point(seg[0])) for seg in pixels[:-1]]
    cuts = np.maximum.accumulate([0.0] + cuts + [line.length])
    cuts[-1] = line.length
    return [
        substring(line, a, b) if b - a > 1e-9 * grid.pixel_size else LineString()
        for a, b in zip(cuts[:-1], cuts[1:])
    ]


def _split_by_segment(path_allocation, df, path_lines, grid):
    # Segment level from path level: each path's line (already smoothed) is
    # cut into per-segment pieces (`_cut_path_line`), and each pixel of the
    # path's region goes to the piece it can reach by the shortest route
    # inside that region. Segment lines are those pieces, so they join end to
    # end into the path line. A path with no usable line falls back to its
    # network segments as seeds; region pixels no seed reaches (islands) go
    # to the nearest labeled pixel.
    shape = path_allocation.shape
    territories = dict(region_groups(path_allocation))
    lines_by_path = {int(r.region_id): (r.geometry, r.source) for r in path_lines.itertuples()}
    out = np.zeros(shape, dtype=np.uint32)
    records = []
    n_no_region = 0
    total_islands = 0
    _, w = shape

    for path_id, group in df.groupby("path_id", sort=True):
        group = group.sort_values("segment_id")
        idx = territories.get(int(path_id))
        if idx is None:
            continue
        rows, cols = np.divmod(idx, w)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        territory = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        territory[rows - r0, cols - c0] = True
        sub_grid = crop_grid(grid, r0, c0, territory.shape)

        line, src = lines_by_path.get(int(path_id), (LineString(), "empty"))
        pieces = _cut_path_line(line, src, group, grid)
        seg_ids = group["segment_id"].tolist()

        # downstream piece wins a pixel two pieces share at a cut
        seeds = np.zeros(territory.shape, dtype=np.uint32)
        for sid, piece in zip(seg_ids, pieces):
            px = rasterize_line(piece, sub_grid, all_touched=True) & territory
            seeds[px & (seeds == 0)] = sid
        if not seeds.any():
            for sid, seg in zip(seg_ids, group["pixels"]):
                rc = np.asarray(seg).reshape(-1, 2) - (r0, c0)
                ok = (
                    (rc[:, 0] >= 0) & (rc[:, 0] < territory.shape[0])
                    & (rc[:, 1] >= 0) & (rc[:, 1] < territory.shape[1])
                )
                rc = rc[ok]
                hit = territory[rc[:, 0], rc[:, 1]] & (seeds[rc[:, 0], rc[:, 1]] == 0)
                seeds[rc[hit, 0], rc[hit, 1]] = sid
        if not seeds.any():
            seeds[np.unravel_index(np.argmax(territory), territory.shape)] = seg_ids[0]

        sub = _nearest_allocate(territory, seeds)
        left = territory & (sub == 0)
        if left.any():
            _, (ii, jj) = distance_transform_edt(sub == 0, return_indices=True)
            sub[left] = sub[ii[left], jj[left]]
        out[r0:r1, c0:c1][territory] = sub[territory]

        for sid, piece in zip(seg_ids, pieces):
            region = sub == sid
            area_px = int(region.sum())
            if area_px == 0:
                n_no_region += 1
                continue
            ref = np.argwhere(rasterize_line(piece, sub_grid, all_touched=True) & region)
            _, _, winning_size = _winning_component(region, map(tuple, ref))
            islands = area_px - winning_size
            total_islands += islands
            piece_src = src if not piece.is_empty else "empty"
            records.append((int(sid), int(path_id), int(sid), piece, area_px, piece_src, islands))

    if n_no_region:
        warnings.warn(
            f"{n_no_region} segment(s) got no part of their path's line or "
            "region (e.g. a segment lying inside its parent's region); they "
            "have no label"
        )
    if total_islands:
        warnings.warn(
            f"{total_islands} segment-region pixel(s) lie in components apart "
            "from their segment's line"
        )
    return out, _lines_frame(records, grid)
