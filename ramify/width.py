# width.py
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
from shapely import voronoi_polygons
from shapely.geometry import MultiPoint
from shapely.strtree import STRtree

from ._io import make_grid, local_half_width, rasterize_line

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


def width_interpolate(mask, line, method="laplace", open_boundary=None,
                       pixel_size=None, progress=None):
    """Per-pixel width of the shape.

    Exact widths (2 * local half-width) are taken at pixels of `line` and
    interpolated across the mask:

    - `method="laplace"`: smooth diffusion (Laplace equation, Dirichlet BCs at
      the line) -- continuous field, best for downstream analysis. `progress`,
      if given, is called with the solver's iteration count as it runs.
    - `method="nearest"`: each pixel takes the width of its nearest line pixel
      (a Voronoi-style assignment) -- piecewise constant, fast, exact at the
      line.

    `line` is a single `shapely.LineString` in the mask's grid coordinates
    (see `centerline`). Mask pixels the line cannot reach (a detached blob, an
    island) are filled by nearest line width, with a warning.

    Returns a float raster, NaN outside the mask.
    """
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got {method!r}")

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0
    _check_line_bounds(line, grid)

    cl_bool = rasterize_line(line, grid, all_touched=True) & mask_bool
    if not cl_bool.any():
        raise ValueError("line does not intersect the mask")

    seed_widths = np.where(
        cl_bool, local_half_width(mask_bool, open_boundary, grid.pixel_size) * 2.0, 0.0
    )

    out = np.full(mask_bool.shape, np.nan)
    if method == "nearest":
        out[mask_bool] = _nearest(cl_bool, seed_widths)[mask_bool]
    else:
        idx = np.flatnonzero(mask_bool)
        out.ravel()[idx] = _laplace(
            idx, cl_bool.ravel()[idx], seed_widths.ravel()[idx], mask_bool.shape, progress
        )
        leftover = mask_bool & np.isnan(out)
        if leftover.any():
            warnings.warn(
                f"{int(leftover.sum())} mask pixels are not connected to the "
                "line; filled by nearest line width"
            )
            out[leftover] = _nearest(cl_bool, seed_widths)[leftover]

    out = grid.wrap(np.where(mask_bool, out, np.nan))
    if grid.is_xr:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out


def width_stations(mask, line, spacing=None, n_stations=None, pixel_size=None):
    """Voronoi (area / length) width along `line`.

    Exactly one of `spacing` / `n_stations` is required. Stations are placed
    at the midpoints of `n` equal-length intervals along the line (`n =
    max(1, round(length / spacing))` when `spacing` is given). Each station's
    width is the shape area in its Voronoi cell divided by the length of
    `line` in that cell.

    Known method properties: Euclidean cells can cross bends or necks, and end
    cells absorb end-cap area and read wide.

    Returns `(w, stations)`: `w` is a float raster, NaN outside the mask.
    `stations` is a `GeoDataFrame`, one row per station: `station_id`,
    `distance` (along line), `area`, `length`, `width`, active `geometry` =
    cell polygon, plus `station` (Point) and `centerline` (LineString piece in
    the cell) geometry columns.
    """
    if (spacing is None) == (n_stations is None):
        raise ValueError("exactly one of spacing or n_stations is required")

    import geopandas as gpd

    mask_arr, grid = make_grid(mask, pixel_size)
    mask_bool = mask_arr > 0
    _check_line_bounds(line, grid)

    shape_poly = _polygonize_mask(mask_bool, grid)
    if not line.intersects(shape_poly):
        raise ValueError("line does not intersect the mask")

    points, distances, cells, pieces, areas, lengths, widths = _compute_stations(
        line, shape_poly, spacing, n_stations
    )
    if np.isnan(widths).any():
        warnings.warn(
            f"{int(np.isnan(widths).sum())} station(s) have no line inside their "
            "cell; width is NaN there"
        )

    stations = gpd.GeoDataFrame(
        {
            "station_id": np.arange(1, len(points) + 1),
            "distance": distances,
            "area": areas,
            "length": lengths,
            "width": widths,
            "geometry": cells,
        },
        geometry="geometry",
        crs=grid.crs,
    )
    stations["station"] = gpd.GeoSeries(points, crs=grid.crs)
    stations["centerline"] = gpd.GeoSeries(pieces, crs=grid.crs)

    w = _rasterize_widths(cells, widths, mask_bool, grid, fallback=True)
    return grid.wrap(w), stations


def _compute_stations(line, shape_poly, spacing, n_stations):
    # Station placement + Voronoi cells + per-station area/length/width, with
    # no rasterization or GeoDataFrame assembly -- the core shared by
    # width_stations (one shape) and width_regions_stations (one shape's
    # regions, each cropped to its own bounding box; see regions.py).
    length_total = line.length
    if length_total <= 0:
        raise ValueError("line has zero length")

    if spacing is not None:
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        n = max(1, round(length_total / spacing))
    else:
        n = int(n_stations)
        if n < 1:
            raise ValueError("n_stations must be at least 1")

    distances = (np.arange(n) + 0.5) * length_total / n
    points = [line.interpolate(d) for d in distances]

    vor = voronoi_polygons(MultiPoint(points), extend_to=shape_poly)
    cells = list(vor.geoms)
    tree = STRtree(cells)
    matched = tree.query(points, predicate="covered_by")  # (point_idx, cell_idx) pairs
    cell_for = np.full(n, -1, dtype=np.intp)
    cell_for[matched[0]] = matched[1]
    if (cell_for < 0).any():
        raise ValueError("could not match every station to a Voronoi cell")

    areas = np.zeros(n)
    lengths = np.zeros(n)
    clipped, pieces = [], []
    for i in range(n):
        cell = cells[cell_for[i]].intersection(shape_poly)
        piece = line.intersection(cell)
        clipped.append(cell)
        pieces.append(piece)
        areas[i] = cell.area
        lengths[i] = piece.length

    with np.errstate(divide="ignore", invalid="ignore"):
        widths = np.where(lengths > 0, areas / lengths, np.nan)

    return points, distances, clipped, pieces, areas, lengths, widths


# -- shared validation / helpers ----------------------------------------------


def _check_line_bounds(line, grid):
    xmin, ymin, xmax, ymax = grid.bounds()
    lx0, ly0, lx1, ly1 = line.bounds
    tol = 1e-6 * max(grid.pixel_size, 1.0)
    if lx1 < xmin - tol or lx0 > xmax + tol or ly1 < ymin - tol or ly0 > ymax + tol:
        raise ValueError("line bounds fall outside the grid")


def _polygonize_mask(mask_bool, grid):
    from rasterio.features import shapes as raster_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    polys = [
        shapely_shape(geom)
        for geom, _ in raster_shapes(
            mask_bool.astype(np.uint8), mask=mask_bool, transform=grid.transform
        )
    ]
    return unary_union(polys)


def _rasterize_widths(cells, widths, mask_bool, grid, fallback=True):
    # `fallback=False` is used by width_regions_stations, where a pixel a
    # station's cell doesn't cover (an island in a disconnected region) should
    # stay NaN rather than borrow the nearest cell's width regardless of
    # region membership.
    from rasterio.features import rasterize

    shapes = [(cell, i + 1) for i, cell in enumerate(cells) if not cell.is_empty]
    label = rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform, fill=0, dtype=np.int32
    )
    out = np.full(mask_bool.shape, np.nan)
    hit = mask_bool & (label > 0)
    out[hit] = widths[label[hit] - 1]

    if fallback:
        leftover = mask_bool & np.isnan(out)
        if leftover.any():
            _, idx = distance_transform_edt(label == 0, return_indices=True)
            nearest_label = label[idx[0], idx[1]]
            out[leftover] = widths[nearest_label[leftover] - 1]
    return out


# -- interpolation --------------------------------------------------------------


def _neighbours(qidx, idx, dy, dx, shape):
    h, w = shape
    rows, cols = np.divmod(qidx, w)
    nr, nc = rows + dy, cols + dx
    ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    nflat = qidx + (dy * w + dx)
    pos = np.searchsorted(idx, nflat)
    np.clip(pos, 0, idx.size - 1, out=pos)
    return ok & (idx[pos] == nflat), pos


def _nearest(cl_bool, seed_widths):
    _, idx = distance_transform_edt(~cl_bool, return_indices=True)
    return seed_widths[idx[0], idx[1]]


def _nearest_flat(idx, is_seed, seed_vals, shape):
    # Same rule restricted to one region's own seeds (used by
    # width_regions_interpolate). A KD-tree over the region's seed pixels
    # costs O(region size); an EDT here would cost O(grid).
    w = shape[1]
    seed_rc = np.column_stack(np.divmod(idx[is_seed], w))
    all_rc = np.column_stack(np.divmod(idx, w))
    _, nn = cKDTree(seed_rc).query(all_rc, k=1)
    return seed_vals[is_seed][nn]


def _laplace(idx, is_seed, seed_vals, shape, progress):
    # Laplace interpolation over the pixel set `idx` (sorted flat indices) with
    # Dirichlet BCs at the seeds. The seed rows are eliminated rather than
    # carried as identity rows, so the system solved is
    #
    #     (D - W_ff) x_f = W_fs s
    #
    # over the free pixels only: symmetric, diagonally dominant, positive
    # definite -- which is what cg actually requires, and smaller besides.
    free = ~is_seed
    n_free = int(free.sum())
    out = seed_vals.copy()
    if n_free == 0:
        return out

    fidx = idx[free]
    fpos = np.cumsum(free) - 1
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

    x = _solve(A, b, progress)

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


def _solve(A, b, progress):
    # rtol bounds the residual, not the error, and the two diverge as a region
    # grows (A's smallest eigenvalue shrinks): at rtol=1e-4 a 340k-pixel region
    # lands ~1-2 width units off the exact solution. 1e-6 costs ~40% more
    # iterations and pulls that back to ~0.03.
    callback = None
    if progress is not None:
        state = {"i": 0}

        def callback(_xk):
            state["i"] += 1
            progress(state["i"])

    x, info = cg(A, b, rtol=1e-6, callback=callback)
    if info != 0:
        warnings.warn("conjugate gradient solver did not converge")
    return x
