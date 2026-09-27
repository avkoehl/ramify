# _skeleton.py
"""Skeleton, snapping, and tree primitives shared by `centerline` and the
partition network construction.
"""
import heapq

import numpy as np
from skimage.morphology import skeletonize
from skimage.graph import MCP_Geometric

SQRT2 = np.sqrt(2.0)
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_nodes(mask_bool) -> set:
    skel = skeletonize(mask_bool)
    rows, cols = np.nonzero(skel)
    return set(zip(rows.tolist(), cols.tolist()))


def neighbors(node):
    r, c = node
    return [(r + dr, c + dc) for dr, dc in OFFSETS]


def degree_map(nodes):
    return {n: sum(nb in nodes for nb in neighbors(n)) for n in nodes}


def endpoints(nodes):
    deg = degree_map(nodes)
    return [n for n, d in deg.items() if d == 1]


def snap_paths(points, nodes, mask_bool):
    """Least-cost path from the skeleton to each point, constrained to the mask.

    Returns a list of pixel-paths (or None if unreachable), aligned with
    `points`.
    """
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


def dijkstra_tree(nodes, root):
    """parent[n] is the neighbor of n one step closer to root."""
    dist = {root: 0.0}
    parent = {root: None}
    heap = [(0.0, root)]
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist[n]:
            continue
        for m in neighbors(n):
            if m not in nodes:
                continue
            step = SQRT2 if (m[0] != n[0] and m[1] != n[1]) else 1.0
            nd = d + step
            if nd < dist.get(m, np.inf):
                dist[m] = nd
                parent[m] = n
                heapq.heappush(heap, (nd, m))
    return parent, dist


def tree_path(parent, node, stop=None):
    """Nodes from `node` up to `stop` (inclusive), following `parent`."""
    path = [node]
    cur = node
    while cur != stop:
        cur = parent[cur]
        if cur is None:
            break
        path.append(cur)
    return path


def break_into_segments(kept, parent, tip_nodes, root):
    """Break a rooted tree over `kept` into segments at `tip_nodes` and
    junctions (nodes with >1 tree-child). Segments run upstream -> downstream;
    junction pixels are shared: last pixel of the upstream segment, first
    pixel of the downstream one.
    """
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


def path_length(pixels, pixel_size):
    total = 0.0
    for (r1, c1), (r2, c2) in zip(pixels[:-1], pixels[1:]):
        total += SQRT2 if (r1 != r2 and c1 != c2) else 1.0
    return total * pixel_size


def path_weight(pixels, edt, pixel_size):
    """Trapezoid rule for the integral of distance-to-edge along a path (~ area/2)."""
    total = 0.0
    for (r1, c1), (r2, c2) in zip(pixels[:-1], pixels[1:]):
        step = SQRT2 if (r1 != r2 and c1 != c2) else 1.0
        total += 0.5 * (edt[r1, c1] + edt[r2, c2]) * step
    return total * pixel_size


def prune_short_leaves(nodes, threshold_fn):
    """Iteratively remove terminal branches shorter than a threshold.

    A terminal branch is the run of degree-2 nodes from a degree-1 endpoint up
    to the first node whose degree is not 2 (a junction, or the far endpoint of
    an isolated simple path). `threshold_fn(junction_node)` gives the minimum
    length a branch reattaching at that node must have to survive. Repeats
    until stable (removing one branch can turn its junction into a new,
    shorter-degree node that starts a new terminal branch elsewhere).
    Never empties the node set entirely.
    """
    kept = set(nodes)
    while len(kept) > 1:
        deg = degree_map(kept)
        leaves = [n for n, d in deg.items() if d == 1]
        removed_any = False
        for leaf in leaves:
            if leaf not in kept or len(kept) <= 1:
                continue
            branch = [leaf]
            prev, cur = None, leaf
            length = 0.0
            while True:
                nbrs = [nb for nb in neighbors(cur) if nb in kept and nb != prev]
                if len(nbrs) != 1:
                    junction = cur
                    break
                nxt = nbrs[0]
                length += SQRT2 if (nxt[0] != cur[0] and nxt[1] != cur[1]) else 1.0
                if deg.get(nxt, 0) != 2:
                    junction = nxt
                    break
                branch.append(nxt)
                prev, cur = cur, nxt
            if junction == leaf or length >= threshold_fn(junction):
                continue
            if len(kept) - len(branch) <= 0:
                continue
            kept.difference_update(branch)
            removed_any = True
        if not removed_any:
            break
    return kept
