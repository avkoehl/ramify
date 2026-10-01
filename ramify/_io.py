# _io.py
"""Grid abstraction: the one affine transform every function works through.

Accepts a 2-D ``np.ndarray`` or a georeferenced ``xr.DataArray`` as mask input,
and produces a :class:`Grid` that knows how to convert between pixel indices
and the shape's coordinate space, and how to wrap a numpy result back into the
input's type.
"""
from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString


@dataclass
class Grid:
    """Everything needed to place pixels in coordinate space and back."""

    shape: tuple
    pixel_size: float
    transform: Affine
    crs: object | None
    coords: dict | None
    dims: tuple | None
    is_xr: bool

    def wrap(self, values):
        """np.ndarray -> input type (passthrough for a plain np.ndarray)."""
        if not self.is_xr:
            return values
        import xarray as xr

        out = xr.DataArray(values, coords=self.coords, dims=self.dims)
        if self.crs is not None:
            out.rio.write_crs(self.crs, inplace=True)
        out.rio.write_transform(self.transform, inplace=True)
        return out

    def to_xy(self, rows, cols):
        """Pixel (row, col) indices -> pixel-center coordinates (x, y)."""
        rows = np.asarray(rows, dtype=float) + 0.5
        cols = np.asarray(cols, dtype=float) + 0.5
        t = self.transform
        xs = t.a * cols + t.b * rows + t.c
        ys = t.d * cols + t.e * rows + t.f
        return xs, ys

    def to_rc(self, xs, ys):
        """Coordinates (x, y) -> fractional (row, col) pixel indices."""
        inv = ~self.transform
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        cols = inv.a * xs + inv.b * ys + inv.c - 0.5
        rows = inv.d * xs + inv.e * ys + inv.f - 0.5
        return rows, cols

    def bounds(self):
        """Grid extent (xmin, ymin, xmax, ymax) in coordinate space."""
        h, w = self.shape
        corners_x, corners_y = [], []
        for r, c in [(0, 0), (0, w), (h, 0), (h, w)]:
            t = self.transform
            corners_x.append(t.a * c + t.b * r + t.c)
            corners_y.append(t.d * c + t.e * r + t.f)
        return min(corners_x), min(corners_y), max(corners_x), max(corners_y)


def make_grid(mask, pixel_size=None):
    """Accept np.ndarray or xr.DataArray. Returns (mask_values, Grid)."""
    try:
        import xarray as xr

        is_xr = isinstance(mask, xr.DataArray)
    except ImportError:
        is_xr = False

    if not is_xr:
        arr = np.asarray(mask)
        ps = float(pixel_size) if pixel_size is not None else 1.0
        return arr, Grid(
            shape=arr.shape,
            pixel_size=ps,
            transform=Affine.scale(ps),
            crs=None,
            coords=None,
            dims=None,
            is_xr=False,
        )

    if pixel_size is not None:
        raise ValueError(
            "pixel_size must not be given for a georeferenced xr.DataArray input"
        )
    transform = mask.rio.transform()
    a, b, d, e = transform.a, transform.b, transform.d, transform.e
    if not np.isclose(b, 0.0) or not np.isclose(d, 0.0):
        raise ValueError("mask transform is rotated or sheared; not supported")
    if not np.isclose(abs(a), abs(e)):
        raise ValueError(f"mask pixels are not square: x-scale {a}, y-scale {e}")
    return mask.values, Grid(
        shape=mask.values.shape,
        pixel_size=float(abs(a)),
        transform=transform,
        crs=mask.rio.crs,
        coords=mask.coords,
        dims=mask.dims,
        is_xr=True,
    )


def check_point(mask_bool, point, name="point"):
    """Validate a (row, col) point: in-bounds (no negative wraparound) and in the mask."""
    r, c = int(point[0]), int(point[1])
    h, w = mask_bool.shape
    if not (0 <= r < h and 0 <= c < w):
        raise ValueError(f"{name} {(r, c)} is out of bounds for shape {mask_bool.shape}")
    if not mask_bool[r, c]:
        raise ValueError(f"{name} {(r, c)} is not inside the mask")
    return r, c


def edt_field(mask_bool, open_boundary):
    """Foreground array for the local-half-width distance transform.

    ``distance_transform_edt`` of the returned array gives, at each shape pixel,
    the distance to the nearest *wall*. By default (``open_boundary is None``)
    every non-shape pixel is a wall. When ``open_boundary`` is given, its truthy
    pixels are treated as void (open, not a wall): they join the shape as
    foreground, so half-widths are measured only to the remaining real walls.
    """
    if open_boundary is None:
        return mask_bool
    open_arr, _ = make_grid(open_boundary)
    open_bool = np.asarray(open_arr) > 0
    if open_bool.shape != mask_bool.shape:
        raise ValueError(
            f"open_boundary shape {open_bool.shape} does not match "
            f"mask shape {mask_bool.shape}"
        )
    field = mask_bool | open_bool
    if field.all():
        import warnings

        warnings.warn(
            "open_boundary leaves no wall pixels; local half-widths will be zero"
        )
    return field


def half_width_pixels(mask_bool, open_boundary):
    """Local half-width in pixel units (scale-invariant, for geodesic claiming).

    ``distance_transform_edt`` measures center-to-center distance to the
    nearest background pixel, which overstates the true half-width by about
    half a pixel (the real edge sits halfway between the two pixel centers).
    Corrected by subtracting 0.5 pixel.
    """
    dist = distance_transform_edt(edt_field(mask_bool, open_boundary))
    return np.clip(dist - 0.5, 0.0, None)


def local_half_width(mask_bool, open_boundary, pixel_size):
    """Distance from each mask pixel to the nearest true wall, in map units."""
    return half_width_pixels(mask_bool, open_boundary) * pixel_size


def region_groups(label_arr, mask_bool=None):
    """Group a labeled raster's pixels by label, in one pass.

    Yields ``(label, flat_idx)`` per positive label, ascending, where
    ``flat_idx`` indexes the raveled grid; ``mask_bool`` optionally restricts
    which pixels count. The sort is stable, so each group's indices come out
    ascending -- callers rely on that for ``searchsorted`` neighbour lookups
    and for reading off a bounding box.
    """
    positive = label_arr > 0
    flat = np.flatnonzero(positive if mask_bool is None else positive & mask_bool)
    if flat.size == 0:
        return
    labels = label_arr.ravel()[flat]
    order = np.argsort(labels, kind="stable")
    flat = flat[order]
    labels = labels[order]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    for lo, hi in zip(starts, np.append(starts[1:], flat.size)):
        yield int(labels[lo]), flat[lo:hi]


def crop_grid(grid, r0, c0, shape):
    """A Grid for the (r0:r0+h, c0:c0+w) sub-window of `grid`.

    Same coordinate space and pixel size as `grid` -- only the transform's
    origin shifts -- so any geometry already in `grid`'s coordinates (e.g. a
    line) can be rasterized directly against the cropped array, and anything
    computed on the crop (e.g. a Voronoi cell) is already in the right frame
    with no translation needed.

    Composing the translated transform loses a handful of bits of precision
    in its offset (adding a ~pixel-sized number to a large absolute origin,
    e.g. a UTM easting). For a line whose vertices sit exactly on pixel
    centers -- every line this package returns -- a diagonal segment often
    passes exactly through a shared pixel corner, and `all_touched`
    rasterization of that exact case is sensitive to sub-ULP perturbation:
    rasterizing the same line against the crop vs. the uncropped grid can
    disagree on a handful of corner-touched pixels. Harmless in practice (a
    Laplace solve tolerates a few boundary pixels moving by one cell), but
    it means results are not bit-identical to an uncropped reference.
    """
    return Grid(
        shape=shape,
        pixel_size=grid.pixel_size,
        transform=grid.transform * Affine.translation(c0, r0),
        crs=grid.crs,
        coords=None,
        dims=None,
        is_xr=False,
    )


def relocate_line(line, r0, c0, grid):
    """Map a LineString from a `pixel_size=1`, origin-(0, 0) cropped array
    (e.g. returned by `centerline()`, which has no way to take a custom
    transform) back into `grid`'s true coordinate space, given the crop's
    (r0, c0) pixel offset.
    """
    if line.is_empty:
        return line
    xs, ys = np.asarray(line.coords).T
    rows = ys - 0.5 + r0  # invert Affine.scale(1.0): (col+.5, row+.5) = (x, y)
    cols = xs - 0.5 + c0
    xs2, ys2 = grid.to_xy(rows, cols)
    return LineString(np.column_stack([xs2, ys2]))


def rasterize_line(line, grid, all_touched=True):
    """Rasterize a shapely LineString onto the grid. Returns a bool array."""
    from rasterio.features import rasterize

    if line.is_empty:
        return np.zeros(grid.shape, dtype=bool)
    arr = rasterize(
        [(line, 1)],
        out_shape=grid.shape,
        transform=grid.transform,
        all_touched=all_touched,
        dtype=np.uint8,
    )
    return arr.astype(bool)


def polygonize_mask(mask_bool, grid):
    """The mask's pixels as one shapely (Multi)Polygon in grid coordinates."""
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


def smoother(smooth):
    """Resolve the `smooth` argument to a `LineString -> LineString` callable.

    `None` -> no smoothing (returns None). `"chaikin"` / `"taubin"` ->
    shapelysmooth's corner-cutting / Taubin smoother (both keep the line's
    endpoints). Chaikin runs 3 iterations, not shapelysmooth's 5: the curve
    has converged by then (<0.03% length change on the bundled data) and each
    extra iteration doubles the vertex count. Taubin uses shapelysmooth's
    defaults. Any callable is used as given, e.g.
    `functools.partial(shapelysmooth.chaikin_smooth, iters=2)`.
    """
    if smooth is None or callable(smooth):
        return smooth
    from functools import partial

    from shapelysmooth import chaikin_smooth, taubin_smooth

    named = {"chaikin": partial(chaikin_smooth, iters=3), "taubin": taubin_smooth}
    if smooth not in named:
        raise ValueError(
            f"smooth must be None, 'chaikin', 'taubin', or a callable, got {smooth!r}"
        )
    return named[smooth]


def smooth_line(line, fn, shape_poly, pixel_size, what="line"):
    """Apply smoother `fn` to `line`; raise if the result leaves `shape_poly`.

    The line is first densified to `pixel_size` vertex spacing: the straight
    end cuts (`disk_cut`) are single long segments, and vertex-averaging
    smoothers (Taubin) otherwise drag the vertex beside one far off the axis.
    The containment test allows `1e-6 * pixel_size` of line outside the shape,
    absorbing float specks where the line crosses a diagonal pixel pinch.
    """
    if fn is None or line.is_empty or len(line.coords) < 3:
        return line
    import shapely

    out = fn(shapely.segmentize(line, pixel_size))
    if out.difference(shape_poly).length > 1e-6 * pixel_size:
        raise ValueError(
            f"smoothed {what} leaves the shape; use a lighter smoothing (e.g. "
            "fewer iterations or steps) or smooth=None"
        )
    return out
