import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric

from ._io import unwrap, wrap, edt_field, GridMeta

SQRT2 = np.sqrt(2.0)
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

Pixel = tuple[int, int]


@dataclass
class Network:
    segments: pd.DataFrame  # segment_id, path_id, strahler, length, weight,
    #                         downstream_segment_id, pixels
    shape: tuple[int, int]
    pixel_size: float
    root: Pixel
    tips: list[Pixel]
    _meta: GridMeta | None = field(default=None, repr=False)

    def rasterize(self, by=None):
        if by is not None and by not in ("path", "segment"):
            raise ValueError(f"by must be None, 'path', or 'segment', got {by!r}")

        flat, counts = self._pixel_index()
        if by is None:
            arr = np.zeros(self.shape, dtype=np.uint8)
            arr.ravel()[flat] = 1
            return wrap(arr, self._meta)

        column = "path_id" if by == "path" else "segment_id"
        values = np.repeat(self.segments[column].to_numpy(), counts)

        # A junction pixel belongs to several segments, so the writes collide.
        # The old loop wrote paths in descending path_id and let later writes
        # win, which means: lowest path_id takes the pixel (so the mainstem,
        # path_id == 1, wins), and within one path the later row in `segments`
        # takes it. Reproduce that by sorting the collisions together and
        # keeping one winner each, rather than replaying the writes in order --
        # same answer, and no O(paths^2) regrouping to get there.
        path_ids = np.repeat(self.segments["path_id"].to_numpy(), counts)
        seq = np.repeat(np.arange(len(self.segments)), counts)
        order = np.lexsort((-seq, path_ids, flat))  # last key sorts first
        winner = order[np.flatnonzero(
            np.r_[True, flat[order][1:] != flat[order][:-1]]
        )]

        arr = np.zeros(self.shape, dtype=np.uint32)
        arr.ravel()[flat[winner]] = values[winner]
        return wrap(arr, self._meta)

    def _pixel_index(self):
        # Flat indices of every segment's pixels, concatenated in `segments`
        # row order, plus each segment's pixel count so the per-segment columns
        # can be np.repeat'd out to match.
        pixels = [np.asarray(p).reshape(-1, 2) for p in self.segments["pixels"]]
        counts = np.fromiter((len(p) for p in pixels), np.intp, len(pixels))
        rc = np.concatenate(pixels) if pixels else np.empty((0, 2), np.intp)
        return rc[:, 0].astype(np.intp) * self.shape[1] + rc[:, 1], counts

    def to_gdf(self):
        if self._meta is None or self._meta.transform is None:
            raise ValueError(
                "to_gdf requires a georeferenced xr.DataArray input to extract_centerlines()"
            )
        import geopandas as gpd
        import rasterio.transform
        from shapely.geometry import LineString

        keep, geoms = [], []
        for i, pixels in enumerate(self.segments["pixels"]):
            if len(pixels) < 2:
                continue
            rows, cols = zip(*pixels)
            xs, ys = rasterio.transform.xy(self._meta.transform, rows, cols)
            geoms.append(LineString(zip(xs, ys)))
            keep.append(i)

        gdf = self.segments.iloc[keep].drop(columns=["pixels"]).copy()
        gdf["geometry"] = geoms
        return gpd.GeoDataFrame(gdf, geometry="geometry", crs=self._meta.crs)


def extract_centerlines(mask, root, tips=None, path_by="area", pixel_size=None,
            open_boundary=None) -> Network:
    if path_by not in ("area", "length", "strahler"):
        raise ValueError(
            f"path_by must be 'area', 'length', or 'strahler', got {path_by!r}"
        )

    mask_arr, px, meta = unwrap(mask, pixel_size)
    mask_bool = mask_arr == 1
    root = (int(root[0]), int(root[1]))
    if not mask_bool[root]:
        raise ValueError(f"root {root} is not inside the mask")

    # 1. skeletonize
    nodes = _skeleton_nodes(mask_bool)
    if not nodes:
        raise ValueError("skeletonization produced no pixels")

    # 2-3. trace root and provided tips onto the skeleton (mask-constrained)
    points = [root] + ([tuple(map(int, t)) for t in tips] if tips else [])
    traces = _snap_paths(points, nodes, mask_bool)
    if traces[0] is None:
        raise ValueError(f"root {root} cannot reach the skeleton within the mask")
    snapped_tips = []
    if tips:
        for t, tr in zip(points[1:], traces[1:]):
            if tr is None:
                raise ValueError(f"tip {t} cannot reach the skeleton within the mask")
        snapped_tips = points[1:]
    for tr in traces:
        nodes.update(tr)

    # 4-5a. shortest-path tree from root (guarantees the result is a tree,
    # even when mask holes create skeleton loops)
    parent, dist = _dijkstra_tree(nodes, root)

    # 5b. resolve tips
    if tips:
        for t in snapped_tips:
            if t not in parent:
                raise ValueError(f"tip {t} is not connected to the root")
        tip_nodes = snapped_tips
    else:
        tip_nodes = [n for n in _endpoints(nodes) if n != root and n in parent]
        if not tip_nodes:
            raise ValueError(
                "no tips found: skeleton has no endpoints reachable from root"
            )

    # 5c. keep only pixels on some tip -> root path
    kept = set()
    for t in tip_nodes:
        n = t
        while n is not None and n not in kept:
            kept.add(n)
            n = parent[n]

    # 6. orient and break into segments at tips and junctions
    segments = _to_segments(kept, parent, tip_nodes, root)

    # 7. annotate
    edt = distance_transform_edt(edt_field(mask_bool, open_boundary)) * px
    df = _annotate(segments, edt, px, path_by, root)

    return Network(
        segments=df,
        shape=mask_bool.shape,
        pixel_size=px,
        root=root,
        tips=[t for t in tip_nodes],
        _meta=meta,
    )


# -- skeleton and snapping ---------------------------------------------------


def _skeleton_nodes(mask_bool) -> set:
    skel = skeletonize(mask_bool)
    rows, cols = np.nonzero(skel)
    return set(zip(rows.tolist(), cols.tolist()))


def _snap_paths(points, nodes, mask_bool):
    # least-cost path from the skeleton to each point, constrained to the mask;
    # returns list of pixel-paths (or None if unreachable), aligned with points
    penalty = np.where(mask_bool, 1.0, np.inf)
    mcp = MCP_Geometric(penalty)
    # only points off the skeleton need tracing; pass them as ends so the flood
    # stops once they are reached (they sit on/near the skeleton) instead of
    # filling the whole array. Costs and tracebacks for the reached ends are
    # identical to the full-flood result.
    ends = [list(p) for p in points if p not in nodes]
    if ends:
        mcp.find_costs(starts=[list(n) for n in nodes], ends=ends)

    out = []
    for p in points:
        if p in nodes:
            out.append([p])
            continue
        try:
            path = mcp.traceback(list(p))
        except ValueError:
            out.append(None)
            continue
        out.append([(int(r), int(c)) for r, c in path] if path else None)
    return out


# -- tree construction -------------------------------------------------------


def _dijkstra_tree(nodes, root):
    # parent[n] is the neighbor of n one step closer to root (downstream)
    dist = {root: 0.0}
    parent = {root: None}
    heap = [(0.0, root)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist[n]:
            continue
        r, c = n
        for dr, dc in _OFFSETS:
            m = (r + dr, c + dc)
            if m not in nodes:
                continue
            nd = d + (SQRT2 if dr and dc else 1.0)
            if nd < dist.get(m, np.inf):
                dist[m] = nd
                parent[m] = n
                heapq.heappush(heap, (nd, m))
    return parent, dist


def _endpoints(nodes):
    out = []
    for r, c in nodes:
        degree = sum((r + dr, c + dc) in nodes for dr, dc in _OFFSETS)
        if degree == 1:
            out.append((r, c))
    return out


def _to_segments(kept, parent, tip_nodes, root):
    # segments are ordered upstream -> downstream; junction pixels are shared:
    # last pixel of each upstream segment, first pixel of the downstream one
    n_children = {}
    for n in kept:
        p = parent[n]
        if p is not None:
            n_children[p] = n_children.get(p, 0) + 1

    breakpoints = set(tip_nodes) | {n for n, k in n_children.items() if k > 1}
    stops = breakpoints | {root}

    segments = []
    for s in breakpoints:
        if s == root:
            continue
        seg = [s]
        cur = parent[s]
        while cur not in stops:
            seg.append(cur)
            cur = parent[cur]
        seg.append(cur)
        segments.append(seg)
    return segments


# -- annotation ---------------------------------------------------------------


def _annotate(segments, edt, pixel_size, path_by, root):
    n = len(segments)
    start_of = {seg[0]: i for i, seg in enumerate(segments)}
    downstream = [start_of.get(seg[-1]) for seg in segments]  # None at root
    children = [[] for _ in range(n)]
    for i, d in enumerate(downstream):
        if d is not None:
            children[d].append(i)

    length = np.array([_seg_length(s, pixel_size) for s in segments])
    weight = np.array([_seg_weight(s, edt, pixel_size) for s in segments])

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
            sub_weight[i] = weight[i] + sub_weight[children[i]].max()

    key = {
        "area": lambda i: (sub_weight[i],),
        "length": lambda i: (sub_length[i],),
        "strahler": lambda i: (strahler[i], sub_length[i]),
    }[path_by]

    # heavy-path decomposition: walk upstream from each outlet, continuing
    # along the heaviest child; other children start new paths
    path_id = np.zeros(n, dtype=int)
    next_id = 1
    from collections import deque

    queue = deque(sorted(outlets, key=key, reverse=True))
    while queue:
        cur = queue.popleft()
        if path_id[cur]:
            continue
        while True:
            path_id[cur] = next_id
            preds = [c for c in children[cur] if not path_id[c]]
            if not preds:
                break
            preds.sort(key=key, reverse=True)
            queue.extend(preds[1:])
            cur = preds[0]
        next_id += 1

    df = pd.DataFrame(
        {
            "segment_id": np.arange(1, n + 1),
            "path_id": path_id,
            "strahler": strahler,
            "length": length,
            "weight": weight,
            "downstream_segment_id": pd.array(
                [d + 1 if d is not None else pd.NA for d in downstream],
                dtype="Int64",
            ),
            "pixels": segments,
        }
    )
    return df.sort_values(["path_id", "segment_id"], ignore_index=True)


def _seg_length(pixels, pixel_size):
    total = 0.0
    for (r1, c1), (r2, c2) in zip(pixels[:-1], pixels[1:]):
        total += SQRT2 if (r1 != r2 and c1 != c2) else 1.0
    return total * pixel_size


def _seg_weight(pixels, edt, pixel_size):
    # trapezoid rule for integral of distance-to-edge along the segment (~ area/2)
    total = 0.0
    for (r1, c1), (r2, c2) in zip(pixels[:-1], pixels[1:]):
        step = SQRT2 if (r1 != r2 and c1 != c2) else 1.0
        total += 0.5 * (edt[r1, c1] + edt[r2, c2]) * step
    return total * pixel_size
