import numpy as np
import pytest

from ramify._io import make_grid, check_point


def test_numpy_default_pixel_size():
    mask = np.zeros((5, 5), dtype=np.uint8)
    _, grid = make_grid(mask)
    assert grid.pixel_size == 1.0
    assert grid.crs is None
    assert not grid.is_xr


def test_pixel_center_roundtrip():
    mask = np.zeros((5, 5), dtype=np.uint8)
    _, grid = make_grid(mask, pixel_size=2.0)
    xs, ys = grid.to_xy(np.array([0, 4]), np.array([0, 4]))
    rows, cols = grid.to_rc(xs, ys)
    assert np.allclose(rows, [0, 4])
    assert np.allclose(cols, [0, 4])


def test_wrap_passthrough_for_ndarray():
    mask = np.zeros((3, 3), dtype=np.uint8)
    _, grid = make_grid(mask)
    values = np.ones((3, 3))
    assert grid.wrap(values) is values


def test_check_point_rejects_negative_index():
    mask_bool = np.zeros((5, 5), dtype=bool)
    mask_bool[2, 2] = True
    with pytest.raises(ValueError, match="out of bounds"):
        check_point(mask_bool, (-1, -1), "root")


def test_check_point_rejects_outside_mask():
    mask_bool = np.zeros((5, 5), dtype=bool)
    mask_bool[2, 2] = True
    with pytest.raises(ValueError, match="not inside the mask"):
        check_point(mask_bool, (0, 0), "root")


def test_check_point_accepts_valid_point():
    mask_bool = np.zeros((5, 5), dtype=bool)
    mask_bool[2, 2] = True
    assert check_point(mask_bool, (2, 2), "root") == (2, 2)
