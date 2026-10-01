# centerline.py
import warnings

import numpy as np
from skimage.measure import label as cc_label
from shapely.geometry import LineString

from ._io import make_grid, check_point, half_width_pixels, polygonize_mask, smoother, smooth_line
from ._skeleton import (
    skeleton_nodes,
    snap_paths,
    dijkstra_tree,
    endpoints,
    tree_path,
    prune_short_leaves,
    disk_cut,
)


def centerline(mask, root=None, tip=None, smooth=None, pixel_size=None) -> LineString:
    """One `LineString` through a single-thread shape.

    - `root` and `tip` given: the skeleton path between them.
    - `root` only: the longest geodesic path from root to any skeleton endpoint.
    - neither given: the skeleton's geodesic diameter (longest path overall).
    - `tip` without `root`: raises `ValueError`.

    `root` / `tip` are `(row, col)` pixel indices. Orientation: starts at
    `root` when given; otherwise deterministic but unspecified. Near a given
    `root` / `tip`, the line leaves the skeleton and runs straight to the
    point from the farthest skeleton pixel whose inscribed disk reaches it,
    so it doesn't swerve into the forks a skeleton makes at a shape's end.

    This is the *longest* path through the shape, deliberately not the
    heaviest (widest) one that `partition_priority`/`partition_nearest` would
    pick as the mainstem -- `centerline(mask, root)` may pick a different
    branch than partition path 1, and that is expected.

    `smooth` takes the pixel staircase out of the line: `None` (default, the
    raw pixel path), `"chaikin"`, `"taubin"`, or a `LineString -> LineString`
    callable. Endpoints are kept. Raises `ValueError` if the smoothed line
    leaves the mask.
    """
    if tip is not None and root is None:
        raise ValueError("tip requires root to also be given")
    smooth_fn = smoother(smooth)

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0

    n_components = cc_label(mask_bool, connectivity=2).max()
    if n_components > 1:
        raise ValueError(
            f"mask has {n_components} connected components; centerline requires "
            "a single connected shape"
        )

    nodes = skeleton_nodes(mask_bool)
    if not nodes:
        raise ValueError("skeletonization produced no pixels")

    # trimming compares against tree-geodesic lengths, which are in pixel-index
    # units (steps of 1 / sqrt(2)); the threshold must be in the same units, so
    # this is the pixel-space half-width, not the real-world (pixel_size scaled)
    # `local_half_width` used elsewhere for weights and widths.
    half_width = half_width_pixels(mask_bool, None)

    def threshold_fn(junction):
        return half_width[junction]

    if root is not None:
        root = check_point(mask_bool, root, "root")
        points = [root] + ([check_point(mask_bool, tip, "tip")] if tip is not None else [])
        traces = snap_paths(points, nodes, mask_bool)
        if traces[0] is None:
            raise ValueError(f"root {root} cannot reach the skeleton within the mask")
        if tip is not None and traces[1] is None:
            raise ValueError(f"tip {tip} cannot reach the skeleton within the mask")
        for tr in traces:
            nodes.update(tr)

        parent, dist = dijkstra_tree(nodes, root)

        if tip is not None:
            if tip not in parent:
                raise ValueError(f"tip {tip} is not connected to the root")
            path_nodes = list(reversed(tree_path(parent, tip, stop=root)))
        else:
            pruned = prune_short_leaves(nodes, threshold_fn)
            candidates = [n for n in endpoints(pruned) if n != root and n in dist]
            if not candidates:
                candidates = [n for n in endpoints(nodes) if n != root and n in dist]
            if not candidates:
                raise ValueError(
                    "no tips found: skeleton has no endpoints reachable from root"
                )
            _warn_if_branching(pruned)
            best = max(dist[n] for n in candidates)
            best_tip = min(n for n in candidates if dist[n] == best)
            path_nodes = list(reversed(tree_path(parent, best_tip, stop=root)))
    else:
        start = min(nodes)
        parent0, dist0 = dijkstra_tree(nodes, start)
        pruned = prune_short_leaves(nodes, threshold_fn)
        candidates = [n for n in endpoints(pruned) if n in dist0]
        if not candidates:
            candidates = [n for n in endpoints(nodes) if n in dist0]
        if not candidates:
            raise ValueError("skeleton has no endpoints")
        _warn_if_branching(pruned)

        best = max(dist0[n] for n in candidates)
        a = min(n for n in candidates if dist0[n] == best)

        parent1, dist1 = dijkstra_tree(nodes, a)
        candidates1 = [n for n in candidates if n in dist1]
        best = max(dist1[n] for n in candidates1)
        b = min(n for n in candidates1 if dist1[n] == best)

        path_nodes = list(reversed(tree_path(parent1, b, stop=a)))

    if root is not None:
        path_nodes = disk_cut(path_nodes, half_width, mask_bool)
        if tip is not None:
            path_nodes = disk_cut(path_nodes[::-1], half_width, mask_bool)[::-1]

    rows = np.array([n[0] for n in path_nodes])
    cols = np.array([n[1] for n in path_nodes])
    xs, ys = grid.to_xy(rows, cols)
    line = LineString(np.column_stack([xs, ys]))
    if smooth_fn is not None:
        line = smooth_line(
            line, smooth_fn, polygonize_mask(mask_bool, grid), grid.pixel_size, "centerline"
        )
    return line


def _warn_if_branching(pruned_nodes):
    n = len(endpoints(pruned_nodes))
    if n > 2:
        warnings.warn(
            f"trimmed skeleton has {n} endpoints; the shape looks branching -- "
            "consider ramify.partition_priority / partition_nearest instead"
        )
