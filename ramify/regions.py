# regions.py
"""Width over a partitioned shape, from `(labels, lines)` -- see the
DESIGN.md addendum "region centerlines and widths by region". Independent of
`net`/`level`: any `(labels, lines)` pair works, including user-edited lines,
as long as `lines` has `region_id` and `geometry` columns.
"""
import warnings

import numpy as np
import pandas as pd

from ._io import make_grid, local_half_width, crop_grid, rasterize_line, region_groups, polygonize_mask
from .width import _laplace, _nearest, _nearest_flat, _compute_stations, _rasterize_widths


def width_regions_interpolate(labels, lines, method="laplace", open_boundary=None,
                               pixel_size=None, progress=None):
    """Per-pixel width over every region of a partitioned shape.

    Equivalent to looping `width_interpolate` per region with
    `open_boundary = (labels > 0) & ~region` (so partition cut lines are
    always open, never walls, and widths don't collapse at junctions), but
    computed in one shared pass so cost is O(mask area) overall rather than
    O(mask area x regions).

    `labels` is a labeled raster (e.g. from `partition_priority`); mask
    pixels are `labels > 0`. `lines` is a `GeoDataFrame` with `region_id` and
    `geometry` columns (e.g. the `lines` `partition_priority` returns) --
    each row's line is rasterized and matched to the pixels of that
    `region_id` in `labels`.

    A region with no row in `lines`, or an empty geometry there, gets no
    seeds of its own; its pixels (and any other pixels no region's line can
    reach) are filled by nearest line width over the whole raster afterward,
    with a warning.

    `progress`, if given, is called once per region, just before it is
    processed, as `progress(i, n_regions, region_id, region_pixel_count)`.

    Returns a float raster, NaN outside the mask, same type as `labels`.
    """
    if method not in ("laplace", "nearest"):
        raise ValueError(f"method must be 'laplace' or 'nearest', got {method!r}")

    labels_arr, grid = make_grid(labels, pixel_size)
    _validate_labels(labels_arr)
    mask_bool = labels_arr > 0
    groups = dict(region_groups(labels_arr))
    lines_by_id = _validate_lines(lines, groups)

    half_width = local_half_width(mask_bool, open_boundary, grid.pixel_size)

    # Seed pixels/widths, built region by region (each cropped to its own
    # bounding box -- rasterizing a line costs O(array shape), not O(line
    # length)) but accumulated into one shared pair of arrays.
    cl_bool = np.zeros(mask_bool.shape, dtype=bool)
    seed_widths = np.zeros(mask_bool.shape, dtype=float)
    w = labels_arr.shape[1]

    for region_id, idx in groups.items():
        line = lines_by_id.get(region_id)
        if line is None or line.is_empty:
            continue
        rows, cols = idx // w, idx % w
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        sub_grid = crop_grid(grid, r0, c0, (r1 - r0, c1 - c0))

        local_cl = rasterize_line(line, sub_grid, all_touched=True)
        local_region = np.zeros(local_cl.shape, dtype=bool)
        local_region[rows - r0, cols - c0] = True
        local_cl &= local_region  # drop line pixels outside their own region

        cl_bool[r0:r1, c0:c1] |= local_cl
        seed_widths[r0:r1, c0:c1][local_cl] = 2.0 * half_width[r0:r1, c0:c1][local_cl]

    cl_flat = cl_bool.ravel()
    seed_flat = seed_widths.ravel()

    out = np.full(mask_bool.shape, np.nan)
    out_flat = out.ravel()
    for i, (region_id, idx) in enumerate(groups.items()):
        if progress is not None:
            progress(i, len(groups), region_id, idx.size)
        is_seed = cl_flat[idx]
        if not is_seed.any():
            continue
        seed_vals = seed_flat[idx]
        if method == "laplace":
            out_flat[idx] = _laplace(idx, is_seed, seed_vals, mask_bool.shape, None)
        else:
            out_flat[idx] = _nearest_flat(idx, is_seed, seed_vals, mask_bool.shape)

    leftover = mask_bool & np.isnan(out)
    if leftover.any():
        warnings.warn(
            f"{int(leftover.sum())} mask pixels had no reachable region line "
            "(an island, or a region with no line); filled by nearest line width"
        )
        out[leftover] = _nearest(cl_bool, seed_widths)[leftover]

    out = grid.wrap(np.where(mask_bool, out, np.nan))
    if grid.is_xr:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out


def width_regions_stations(labels, lines, spacing=None, n_stations=None, pixel_size=None):
    """Voronoi (area / length) width over every region of a partitioned shape.

    Equivalent to `width_stations(region, line, spacing=..., n_stations=...)`
    per region (see DESIGN.md), with cells clipped to the region; `spacing` /
    `n_stations` apply the same way to every region (a region whose line is
    shorter than `spacing` gets one station).

    A region with no row in `lines`, or an empty geometry there, has no
    stations and its pixels stay NaN (there's no seed to borrow a width from,
    unlike `width_regions_interpolate`), with one aggregated warning. Island
    pixels a region's own station cells don't cover also stay NaN.

    Returns `(w, stations)`: `w` is a float raster, NaN outside the mask,
    same type as `labels`. `stations` is the per-region station tables
    concatenated, with a `region_id` column added; `station_id` is unique
    across the whole result. Columns otherwise as in `width_stations`.
    """
    if (spacing is None) == (n_stations is None):
        raise ValueError("exactly one of spacing or n_stations is required")

    import geopandas as gpd

    labels_arr, grid = make_grid(labels, pixel_size)
    _validate_labels(labels_arr)
    mask_bool = labels_arr > 0
    groups = dict(region_groups(labels_arr))
    lines_by_id = _validate_lines(lines, groups)

    out = np.full(mask_bool.shape, np.nan)
    frames = []
    next_id = 1
    n_no_line = 0
    w = labels_arr.shape[1]

    for region_id, idx in groups.items():
        line = lines_by_id.get(region_id)
        if line is None or line.is_empty:
            n_no_line += 1
            continue

        rows, cols = idx // w, idx % w
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        sub_grid = crop_grid(grid, r0, c0, (r1 - r0, c1 - c0))
        region_local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        region_local[rows - r0, cols - c0] = True

        shape_poly = polygonize_mask(region_local, sub_grid)
        if not line.intersects(shape_poly):
            n_no_line += 1
            continue
        points, distances, cells, pieces, areas, lengths, widths = _compute_stations(
            line, shape_poly, spacing, n_stations
        )

        w_local = _rasterize_widths(cells, widths, region_local, sub_grid, fallback=False)
        out[r0:r1, c0:c1][region_local] = w_local[region_local]

        n = len(points)
        frame = gpd.GeoDataFrame(
            {
                "region_id": region_id,
                "station_id": np.arange(next_id, next_id + n),
                "distance": distances,
                "area": areas,
                "length": lengths,
                "width": widths,
                "geometry": cells,
            },
            geometry="geometry",
            crs=grid.crs,
        )
        frame["station"] = gpd.GeoSeries(points, crs=grid.crs)
        frame["centerline"] = gpd.GeoSeries(pieces, crs=grid.crs)
        frames.append(frame)
        next_id += n

    if n_no_line:
        warnings.warn(f"{n_no_line} region(s) have no line; their pixels stay NaN")

    if frames:
        stations = pd.concat(frames, ignore_index=True)
        stations = gpd.GeoDataFrame(stations, geometry="geometry", crs=grid.crs)
    else:
        stations = gpd.GeoDataFrame(
            {
                "region_id": pd.array([], dtype="int64"),
                "station_id": pd.array([], dtype="int64"),
                "distance": pd.array([], dtype="float64"),
                "area": pd.array([], dtype="float64"),
                "length": pd.array([], dtype="float64"),
                "width": pd.array([], dtype="float64"),
                "geometry": [],
                "station": [],
                "centerline": [],
            },
            geometry="geometry",
            crs=grid.crs,
        )

    out = grid.wrap(np.where(mask_bool, out, np.nan))
    if grid.is_xr:
        try:
            out.rio.write_nodata(np.nan, inplace=True)
        except AttributeError:
            pass
    return out, stations


# -- validation ----------------------------------------------------------------


def _validate_labels(labels_arr):
    if not np.issubdtype(labels_arr.dtype, np.integer):
        if not np.all(labels_arr == np.round(labels_arr)):
            raise ValueError("labels must contain integer values")
    if (labels_arr < 0).any():
        raise ValueError("labels must be non-negative")


def _validate_lines(lines, groups):
    ids = lines["region_id"]
    dup = sorted(set(ids[ids.duplicated()].tolist()))
    if dup:
        raise ValueError(f"duplicate region_id in lines: {dup}")
    unknown = sorted(set(int(i) for i in ids) - set(groups))
    if unknown:
        raise ValueError(f"region_id(s) not present in labels: {unknown}")
    return {int(r.region_id): r.geometry for r in lines.itertuples()}
