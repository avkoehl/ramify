import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree

from ._io import unwrap, wrap, edt_field, region_groups

SQRT2 = np.sqrt(2.0)
# 8-connected stencil; diagonals weighted 1/sqrt(2) for isotropy (and so a pixel
# attached only diagonally is never isolated)
_OFFSETS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 1 / SQRT2),
    (-1, 1, 1 / SQRT2),
    (1, -1, 1 / SQRT2),
    (1, 1, 1 / SQRT2),
]


def interpolate_widths(mask, centerline, regions=None, method="laplace",
                       pixel_size=None, open_boundary=None, progress=None):
    # Per-pixel width of the shape. Exact widths (2 * distance-to-boundary)
    # are taken at centerline pixels and interpolated across the mask.
    # With `regions` (e.g. the output of partition_by_priority) the
    # interpolation runs independently within each labeled region, so widths do
    # not diffuse across path boundaries at junctions; `progress` applies only
    # in that case (see _widths_by_region).
    if regions is None:
        return _widths_global(mask, centerline, method, pixel_size, open_boundary)
    return _widths_by_region(mask, centerline, regions, method, pixel_size,
                             open_boundary, progress)


def _widths_global(mask, centerline, method, pixel_size, open_boundary):
    # method="laplace": smooth diffusion (Laplace equation, Dirichlet BCs at
    #   the centerline) — continuous fields, best for downstream analysis.
    # method="nearest": each pixel takes the width of its nearest centerline
    #   pixel (a Voronoi-style assignment, cf. ramify.partition_by_nearest) — piecewise
    #   constant, fast, exact at the centerline.
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got {method!r}")

    mask_arr, px, meta = unwrap(mask, pixel_size)
    cl_arr, _, _ = unwrap(centerline)
    mask_bool = mask_arr == 1
    cl_bool = (cl_arr > 0) & mask_bool
    if cl_arr.shape != mask_bool.shape:
        raise ValueError(
            f"centerline shape {cl_arr.shape} does not match mask shape {mask_bool.shape}"
        )
    if not cl_bool.any():
        raise ValueError("no centerline pixels found inside the mask")

    seed_widths = np.where(
        cl_bool, distance_transform_edt(edt_field(mask_bool, open_boundary)) * px * 2.0, 0.0
    )

    out = np.full(mask_bool.shape, np.nan)
    if method == "nearest":
        out[mask_bool] = _nearest(cl_bool, seed_widths)[mask_bool]
    else:
        idx = np.flatnonzero(mask_bool)
        out.ravel()[idx] = _laplace(
            idx, cl_bool.ravel()[idx], seed_widths.ravel()[idx], mask_bool.shape
        )
        # parts of the mask the centerline cannot reach (a detached blob, an
        # island) have no Dirichlet data at all -- fall back rather than
        # reporting the zero the homogeneous solve would give
        leftover = mask_bool & np.isnan(out)
        if leftover.any():
            warnings.warn(
                f"{int(leftover.sum())} mask pixels are not connected to the "
                "centerline; filled by nearest centerline width"
            )
            out[leftover] = _nearest(cl_bool, seed_widths)[leftover]

    out = wrap(out, meta)
    if meta is not None:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out


def _widths_by_region(mask, centerline, regions, method, pixel_size,
                      open_boundary, progress):
    # Like _widths_global, but interpolated independently within each labeled
    # region, so widths do not diffuse across path boundaries at junctions. Each region is seeded only by the
    # centerline pixels inside it. Regions containing no centerline pixels
    # are filled by nearest-centerline fallback (with a warning) so the
    # output always covers the mask.
    #
    # Regions are solved from their own flat pixel indices (grouped once, up
    # front) rather than by re-scanning the full grid per label, so the cost is
    # O(mask area) in total instead of O(mask area x number of regions).
    #
    # `progress`, if given, is called once per region just before that region is
    # solved: progress(i, n_regions, label, region_pixel_count). Regions are very
    # uneven and the laplace solve costs ~O(n^1.5) in a region's pixel count, so
    # drive a bar off that count -- counting regions makes it race to ~99% and
    # then sit on the few big ones for most of the wall time.
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got {method!r}")

    mask_arr, px, meta = unwrap(mask, pixel_size)
    cl_arr, _, _ = unwrap(centerline)
    reg_arr, _, _ = unwrap(regions)
    mask_bool = mask_arr == 1
    cl_bool = (cl_arr > 0) & mask_bool
    if cl_arr.shape != mask_bool.shape or reg_arr.shape != mask_bool.shape:
        raise ValueError("mask, centerline, and regions must share one shape")
    if not cl_bool.any():
        raise ValueError("no centerline pixels found inside the mask")

    seed_widths = np.where(
        cl_bool, distance_transform_edt(edt_field(mask_bool, open_boundary)) * px * 2.0, 0.0
    )
    cl_flat = cl_bool.ravel()
    seed_flat = seed_widths.ravel()

    out = np.full(mask_bool.shape, np.nan)
    out_flat = out.ravel()
    groups = list(region_groups(reg_arr, mask_bool))  # views, so the total is free
    for i, (label, idx) in enumerate(groups):
        if progress is not None:
            progress(i, len(groups), label, idx.size)
        is_seed = cl_flat[idx]
        if not is_seed.any():
            continue  # filled by fallback below
        seed_vals = seed_flat[idx]
        if method == "laplace":
            out_flat[idx] = _laplace(idx, is_seed, seed_vals, mask_bool.shape)
        else:
            out_flat[idx] = _nearest_flat(idx, is_seed, seed_vals, mask_bool.shape)

    leftover = mask_bool & np.isnan(out)
    if leftover.any():
        warnings.warn(
            f"{int(leftover.sum())} mask pixels fall in regions with no "
            "centerline pixels (or outside any region); filled by nearest "
            "centerline width"
        )
        fallback = _nearest(cl_bool, seed_widths)
        out[leftover] = fallback[leftover]

    out = np.where(mask_bool, out, np.nan)
    out = wrap(out, meta)
    if meta is not None:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out


def _neighbours(qidx, idx, dy, dx, shape):
    # Membership test for one stencil offset: for each query pixel qidx[i], does
    # its neighbour at (dy, dx) belong to the sorted set `idx`? Works purely on
    # flat indices, so nothing the size of the grid (or even of the region's
    # bounding box) is allocated. Returns (has, pos) with idx[pos[i]] the
    # neighbour wherever has[i]. The explicit column check is what stops a
    # dx = -1 step at column 0 from wrapping onto the previous row.
    h, w = shape
    rows, cols = np.divmod(qidx, w)
    nr, nc = rows + dy, cols + dx
    ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    nflat = qidx + (dy * w + dx)
    pos = np.searchsorted(idx, nflat)
    np.clip(pos, 0, idx.size - 1, out=pos)
    return ok & (idx[pos] == nflat), pos


def _nearest(cl_bool, seed_widths):
    # Whole-grid nearest-centerline assignment (used by _widths_global and as the
    # _widths_by_region fallback); the EDT is linear in grid size.
    _, idx = distance_transform_edt(~cl_bool, return_indices=True)
    return seed_widths[idx[0], idx[1]]


def _nearest_flat(idx, is_seed, seed_vals, shape):
    # Same rule restricted to one region's seeds. A KD-tree over the region's
    # centerline pixels costs O(region size); an EDT here would cost O(grid).
    w = shape[1]
    seed_rc = np.column_stack(np.divmod(idx[is_seed], w))
    all_rc = np.column_stack(np.divmod(idx, w))
    _, nn = cKDTree(seed_rc).query(all_rc, k=1)
    return seed_vals[is_seed][nn]


def _laplace(idx, is_seed, seed_vals, shape):
    # Laplace interpolation over the pixel set `idx` (sorted flat indices) with
    # Dirichlet BCs at the seeds. The seed rows are eliminated rather than
    # carried as identity rows, so the system solved is
    #
    #     (D - W_ff) x_f = W_fs s
    #
    # over the free pixels only: symmetric, diagonally dominant, positive
    # definite — which is what cg actually requires, and smaller besides.
    free = ~is_seed
    n_free = int(free.sum())
    out = seed_vals.copy()
    if n_free == 0:
        return out

    fidx = idx[free]
    fpos = np.cumsum(free) - 1  # position in `idx` -> row in the free system
    local = np.arange(n_free)

    rows, cols, data = [], [], []
    diag = np.zeros(n_free)
    b = np.zeros(n_free)
    touches_seed = np.zeros(n_free, dtype=bool)
    for dy, dx, wt in _OFFSETS:
        has, pos = _neighbours(fidx, idx, dy, dx, shape)
        p = pos[has]
        diag[has] += wt
        nbr_seed = is_seed[p]
        rows.append(local[has][~nbr_seed])
        cols.append(fpos[p[~nbr_seed]])
        data.append(np.full(int((~nbr_seed).sum()), -wt))
        # each free pixel meets a given offset at most once, but summing over
        # the eight offsets still accumulates, so go through bincount
        b += np.bincount(
            local[has][nbr_seed],
            weights=wt * seed_vals[p[nbr_seed]],
            minlength=n_free,
        )
        touches_seed[local[has][nbr_seed]] = True

    rows.append(local)
    cols.append(local)
    data.append(diag)

    A = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_free, n_free),
    )

    x = _solve(A, b)

    # A pixel set can fall apart into chunks that no seed touches (a region
    # split by a junction, an island). Such a chunk is a singular Neumann block
    # with a zero rhs, and because A is block diagonal cg leaves it at exactly
    # 0.0 -- so the solver silently reports width 0 there. Exact zero is also
    # the only way a *seeded* block can land on 0 (the maximum principle bounds
    # it below by its smallest seed width), which makes it a cheap filter;
    # confirm against the connectivity, then hand the chunk back as NaN for the
    # caller to fill by fallback.
    if (x == 0.0).any():
        n_comp, comp = connected_components(A, directed=False)
        seeded = np.zeros(n_comp, dtype=bool)
        seeded[comp[touches_seed]] = True
        x[~seeded[comp]] = np.nan

    out[free] = x
    return out


def _solve(A, b):
    # rtol bounds the residual, not the error, and the two diverge as a region
    # grows (A's smallest eigenvalue shrinks): at rtol=1e-4 a 340k-pixel region
    # lands ~1-2 width units off the exact solution. 1e-6 costs ~40% more
    # iterations and pulls that back to ~0.03.
    x, info = cg(A, b, rtol=1e-6)
    if info != 0:
        warnings.warn("conjugate gradient solver did not converge")
    return x
