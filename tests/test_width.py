import numpy as np
import pytest
from shapely.geometry import LineString

import ramify


def test_invalid_method_raises(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask, root=root, tip=tip)
    with pytest.raises(ValueError, match="method"):
        ramify.width_interpolate(mask, line, method="bogus")


def test_line_outside_grid_raises(straight_bar):
    mask, _, _ = straight_bar
    line = LineString([(1000, 1000), (2000, 2000)])
    with pytest.raises(ValueError, match="bounds"):
        ramify.width_interpolate(mask, line)


def test_line_not_intersecting_mask_raises():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2, 2:8] = 1
    line = LineString([(0.5, 8.5), (8.5, 8.5)])  # inside the grid, off the mask
    with pytest.raises(ValueError, match="does not intersect"):
        ramify.width_interpolate(mask, line)


def test_laplace_and_nearest_cover_the_mask(straight_bar):
    mask, root, tip = straight_bar
    mask_bool = mask > 0
    line = ramify.centerline(mask, root=root, tip=tip)
    for method in ("laplace", "nearest"):
        w = np.asarray(ramify.width_interpolate(mask, line, method=method))
        assert np.isnan(w[~mask_bool]).all()
        assert not np.isnan(w[mask_bool]).any()
        assert (w[mask_bool] > 0).all()


def test_progress_callback_fires_for_laplace(toy_dataset):
    mask, root, _ = toy_dataset
    line = ramify.centerline(mask, root=root)
    calls = []
    ramify.width_interpolate(mask, line, method="laplace", progress=calls.append)
    assert len(calls) > 0
    assert calls == sorted(calls)


def test_width_stations_requires_exactly_one_arg(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask, root=root, tip=tip)
    with pytest.raises(ValueError, match="exactly one"):
        ramify.width_stations(mask, line)
    with pytest.raises(ValueError, match="exactly one"):
        ramify.width_stations(mask, line, spacing=1.0, n_stations=5)


def test_width_stations_n_stations_count(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask, root=root, tip=tip)
    w, stations = ramify.width_stations(mask, line, n_stations=4)
    assert len(stations) == 4
    assert list(stations["station_id"]) == [1, 2, 3, 4]
    assert (stations["area"] > 0).all()
    assert (stations["length"] > 0).all()
    assert (stations["width"] > 0).all()


def test_width_stations_covers_mask(straight_bar):
    mask, root, tip = straight_bar
    mask_bool = mask > 0
    line = ramify.centerline(mask, root=root, tip=tip)
    w, _ = ramify.width_stations(mask, line, n_stations=3)
    arr = np.asarray(w)
    assert np.isnan(arr[~mask_bool]).all()
    assert not np.isnan(arr[mask_bool]).any()


def test_width_stations_geometry_columns(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask, root=root, tip=tip)
    _, stations = ramify.width_stations(mask, line, n_stations=3)
    assert "geometry" in stations.columns
    assert "station" in stations.columns
    assert "centerline" in stations.columns
    assert stations.geometry.name == "geometry"


def test_line_pixel_roundtrip(toy_dataset):
    # "line vertices sit exactly on pixel centers ... so rasterizing a
    # returned line reproduces its pixels exactly"
    from ramify._io import make_grid, rasterize_line

    mask, root, tips = toy_dataset
    line = ramify.centerline(mask, root=root, tip=tips[0])
    _, grid = make_grid(mask)

    coords = np.array(line.coords)
    rows, cols = grid.to_rc(coords[:, 0], coords[:, 1])
    rows_i, cols_i = np.round(rows).astype(int), np.round(cols).astype(int)
    assert np.allclose(rows, rows_i, atol=1e-6)
    assert np.allclose(cols, cols_i, atol=1e-6)

    rasterized = rasterize_line(line, grid, all_touched=False)
    assert rasterized[rows_i, cols_i].all()


def test_numpy_and_xarray_output_types(toy_dataset):
    import xarray as xr

    mask, root, _ = toy_dataset
    mask_np = (np.asarray(mask.values) == 1).astype(np.uint8)
    line = ramify.centerline(mask, root=root)
    line_np = ramify.centerline(mask_np, root=root)

    w_xr = ramify.width_interpolate(mask, line)
    w_np = ramify.width_interpolate(mask_np, line_np)
    assert isinstance(w_xr, xr.DataArray)
    assert isinstance(w_np, np.ndarray)
