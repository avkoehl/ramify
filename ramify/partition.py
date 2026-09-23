import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.segmentation import watershed

from ._io import unwrap, wrap, edt_field, region_groups
from .centerline import Network

SQRT2 = np.sqrt(2.0)
# forward-only neighbour offsets; directed=False makes each bidirectional
_EDGES = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, SQRT2), (1, -1, SQRT2)]


def partition_by_priority(mask, seeds, open_boundary=None, progress=None):
    # Ordered, radius-limited claiming. Each path (seed label) claims the mask
    # pixels within *some* of its seeds' local half-width, measured as a
    # boundary-respecting (geodesic) distance -- so a wide-but-farther seed can
    # reach a pixel a narrow-but-nearer seed cannot. Paths are processed
    # biggest-first (label 1 = mainstem = highest priority) and a pixel is kept
    # by the first (biggest) path to reach it, so wide branches claim
    # proportionally more space at junctions. Unclaimed remainder is
    # watershed-filled so the labels cover the mask completely.
    #
    # Each path's claim is a windowed, radius-bounded Dijkstra (see _reach), so
    # cost is O(tube area) per path rather than O(domain); the mask never
    # changes during the loop, so there is no per-step graph rebuild.
    #
    # `progress`, if given, is called once per path just before its Dijkstra:
    # progress(i, n_paths, label, window_pixel_count). The window area is the
    # honest cost proxy -- a path's seed count is its *length*, which says little
    # about the tube it will claim. Paths run biggest-first, so the early ones are
    # by far the slowest; a bar weighted by window area tracks that, one counting
    # paths does not.
    mask_arr, _, meta = unwrap(mask)
    seed_arr, _, _ = unwrap(seeds)
    mask_bool = mask_arr == 1
    _check(mask_bool, seed_arr)

    H, W = mask_bool.shape
    radius = distance_transform_edt(edt_field(mask_bool, open_boundary))  # local half-width
    allocation = np.zeros(mask_bool.shape, dtype=np.uint32)

    # group seed pixels by label once, so each path works from its own coords
    # (and bbox) instead of scanning the full seed array every iteration
    flat = np.flatnonzero(seed_arr)
    labels_flat = seed_arr.ravel()[flat]
    order = np.argsort(labels_flat, kind="stable")
    flat = flat[order]
    labels_sorted = labels_flat[order]
    uniq, starts = np.unique(labels_sorted, return_index=True)  # ascending
    bounds = np.append(starts, labels_sorted.size)

    for i, label in enumerate(uniq):  # ascending label == biggest path first
        idx = flat[bounds[i]:bounds[i + 1]]
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

        # reported here, not at the top of the loop: the window is the first
        # point where this path's cost is actually known
        if progress is not None:
            progress(i, len(uniq), int(label), (r1 - r0) * (c1 - c0))

        seed_local = np.stack([rr - r0, cc - c0], axis=1)
        tube = _reach(mask_bool[r0:r1, c0:c1], seed_local, rad, R)
        sub = allocation[r0:r1, c0:c1]
        sub[tube & (sub == 0)] = label  # keep only where no bigger path won

    claimed = allocation > 0
    unclaimed = mask_bool & ~claimed
    if unclaimed.any() and claimed.any():
        allocation = watershed(
            image=distance_transform_edt(~claimed),
            markers=allocation,
            mask=mask_bool,
        ).astype(np.uint32)

    return wrap(allocation, meta)


def _reach(mask_win, seed_rc, radii, R):
    # Pixels within some seed's local half-width, geodesically, inside the
    # window. A virtual super-source is wired to each seed s with edge weight
    # R - radius(s) (>= 0), so one bounded search from it gives, per pixel q,
    #   dist_V(q) = R + min_s ( geodist(q, s) - radius(s) ),
    # and dist_V(q) <= R is exactly "some seed's radius reaches q". The bound
    # (limit=R) keeps the search inside the tube. Metric is octile ({1, sqrt2}),
    # matching centerline._dijkstra_tree.
    h, w = mask_win.shape
    ys, xs = np.nonzero(mask_win)
    M = ys.size
    if M == 0:
        return np.zeros((h, w), dtype=bool)
    ids = np.full((h, w), -1, dtype=np.int64)  # pixel -> node id, -1 off-mask
    ids[ys, xs] = np.arange(M)
    V = M  # super-source node id

    rows, cols, wts = [], [], []
    for dr, dc, step in _EDGES:  # grid edges between adjacent in-mask pixels
        ny, nx = ys + dr, xs + dc
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        nb = np.where(ok, ids[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)], -1)
        keep = nb >= 0
        rows.append(ids[ys[keep], xs[keep]])
        cols.append(nb[keep])
        wts.append(np.full(int(keep.sum()), step))

    sids = ids[seed_rc[:, 0], seed_rc[:, 1]]  # super-source -> each seed
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


def partition_by_nearest(mask, seeds):
    # Nearest-seed partition of the mask: every pixel goes to the seed label
    # it can reach by the shortest within-mask route. No ordering, no radius
    # limits. Use for simple subdivision, e.g. splitting a path's territory
    # by segment: partition_by_nearest(regions == path_id, segment_seeds).
    mask_arr, _, meta = unwrap(mask)
    seed_arr, _, _ = unwrap(seeds)
    mask_bool = mask_arr == 1
    _check(mask_bool, seed_arr)

    markers = np.where(mask_bool, seed_arr, 0).astype(np.int64)
    if not (markers > 0).any():
        return wrap(np.zeros(mask_bool.shape, dtype=np.uint32), meta)

    out = watershed(
        image=distance_transform_edt(markers == 0),
        markers=markers,
        mask=mask_bool,
    ).astype(np.uint32)
    return wrap(out, meta)


def subdivide_regions(regions, network: Network):
    # Subdivide each path's territory (from partition_by_priority) into segment-level
    # territories. Each territory is seeded only by its own path's segments,
    # so neighboring paths' labels (e.g. shared junction pixels) never bleed
    # across boundaries. Pixels in territories whose path has no segments in
    # the table remain 0.
    reg_arr, _, meta = unwrap(regions)
    if reg_arr.shape != network.shape:
        raise ValueError(
            f"regions shape {reg_arr.shape} does not match network grid {network.shape}"
        )

    # Group the territories once, then work each path inside its own bounding
    # box. Everything below is local: a full-grid pass per path would cost the
    # whole raster ~once per path, and partition_by_nearest() runs a distance transform and
    # a watershed, so that is the expensive kind of pass. Cropping is exact
    # here -- the distance transform measures to the nearest seed and every one
    # of this path's seeds is inside its own bbox, and the watershed only ever
    # floods within the territory.
    territories = dict(region_groups(reg_arr))

    out = np.zeros(network.shape, dtype=np.uint32)
    _, w = network.shape
    for path_id, group in network.segments.groupby("path_id"):
        idx = territories.get(int(path_id))
        if idx is None:
            continue  # path swallowed during allocation
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

        sub = np.asarray(partition_by_nearest(territory, seeds))
        hit = sub > 0
        out[r0:r1, c0:c1][hit] = sub[hit]

    return wrap(out, meta)


def _check(mask_bool, seed_arr):
    if seed_arr.shape != mask_bool.shape:
        raise ValueError(
            f"seeds shape {seed_arr.shape} does not match mask shape {mask_bool.shape}"
        )
    if (seed_arr[mask_bool] > 0).sum() == 0:
        raise ValueError("no seed pixels found inside the mask")
