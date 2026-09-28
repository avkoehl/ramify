import numpy as np
import pytest
from shapely.geometry import LineString

import ramify


def test_root_and_tip(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask, root=root, tip=tip)
    assert isinstance(line, LineString)
    coords = list(line.coords)
    assert coords[0] == (1.5, 2.5)  # (col + .5, row + .5) * pixel_size, at root
    assert coords[-1] == (18.5, 2.5)
    assert line.length == pytest.approx(17.0)


def test_root_only_finds_farthest_tip(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask, root=root)
    assert list(line.coords)[0] == (1.5, 2.5)
    assert list(line.coords)[-1] == (18.5, 2.5)


def test_neither_gives_geodesic_diameter(straight_bar):
    mask, root, tip = straight_bar
    line = ramify.centerline(mask)
    assert line.length == pytest.approx(17.0)


def test_tip_without_root_raises(straight_bar):
    mask, _, tip = straight_bar
    with pytest.raises(ValueError, match="root"):
        ramify.centerline(mask, tip=tip)


def test_disconnected_mask_raises():
    mask = np.zeros((5, 20), dtype=np.uint8)
    mask[2, 1:5] = 1
    mask[2, 15:19] = 1
    with pytest.raises(ValueError, match="connected"):
        ramify.centerline(mask)


def test_root_out_of_bounds_raises(straight_bar):
    mask, _, _ = straight_bar
    with pytest.raises(ValueError):
        ramify.centerline(mask, root=(100, 100))


def test_negative_index_does_not_wrap(straight_bar):
    mask, _, _ = straight_bar
    # mask[-1, -1] (i.e. row 4, col 19) is background; a naive numpy wraparound
    # would silently accept (-1, -1) as in-bounds instead of rejecting it.
    with pytest.raises(ValueError, match="out of bounds"):
        ramify.centerline(mask, root=(-1, -1))


def test_root_outside_mask_raises(straight_bar):
    mask, _, _ = straight_bar
    with pytest.raises(ValueError, match="not inside the mask"):
        ramify.centerline(mask, root=(0, 0))


def test_short_stub_is_trimmed(stubbed_bar):
    line = ramify.centerline(stubbed_bar)
    # the diameter should still run the length of the long bar, undeflected by
    # the short stub off the middle
    assert line.length == pytest.approx(17.0)


def test_warns_when_branching(stubbed_bar):
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[5, 1:19] = 1
    mask[0:5, 10] = 1  # a long branch this time, tall enough to survive trimming
    with pytest.warns(UserWarning, match="branching"):
        ramify.centerline(mask)


def test_pixel_size_scales_length(straight_bar):
    mask, root, tip = straight_bar
    line1 = ramify.centerline(mask, root=root, tip=tip)
    line2 = ramify.centerline(mask, root=root, tip=tip, pixel_size=2.0)
    assert line2.length == pytest.approx(line1.length * 2.0)


def test_pixel_size_rejected_for_georeferenced_input(toy_dataset):
    mask, root, _ = toy_dataset
    with pytest.raises(ValueError, match="pixel_size"):
        ramify.centerline(mask, root=root, pixel_size=2.0)


def test_numpy_and_xarray_agree_on_topology(toy_dataset):
    mask, root, _ = toy_dataset
    mask_np = (np.asarray(mask.values) == 1).astype(np.uint8)
    line_geo = ramify.centerline(mask, root=root)
    line_np = ramify.centerline(mask_np, root=root)
    # coordinate scales differ (georeferenced pixel size vs the numpy default
    # of 1), but the underlying pixel path is the same shape
    assert len(line_geo.coords) == len(line_np.coords)


def test_output_is_plain_ndarray_free(toy_dataset):
    # centerline always returns a LineString, regardless of input type
    mask, root, _ = toy_dataset
    mask_np = (np.asarray(mask.values) == 1).astype(np.uint8)
    assert isinstance(ramify.centerline(mask, root=root), LineString)
    assert isinstance(ramify.centerline(mask_np, root=root), LineString)


def test_root_end_skips_skeleton_fork(flared_bar, end_deviation, inside_mask):
    mask, root, _ = flared_bar
    line = ramify.centerline(mask, root=root)
    assert line.coords[0] == (root[1] + 0.5, root[0] + 0.5)
    assert end_deviation(line) < 1.0  # uncut path swerves ~5 px into a fork arm
    assert inside_mask(line, mask)


def test_tip_end_skips_skeleton_fork(flared_bar, end_deviation, inside_mask):
    mask, tip, root = flared_bar  # flared end as the tip this time
    line = ramify.centerline(mask, root=root, tip=tip)
    assert line.coords[-1] == (tip[1] + 0.5, tip[0] + 0.5)
    assert end_deviation(LineString(line.coords[::-1])) < 1.0
    assert inside_mask(line, mask)


def test_root_end_has_no_sideways_snap(toy_dataset, end_deviation, inside_mask):
    # the bundled root sits on the side of a pointed end; the skeleton runs
    # past it toward the point, so the uncut path makes a right-angle turn
    mask, root, _ = toy_dataset
    mask_np = (np.asarray(mask.values) == 1).astype(np.uint8)
    line = ramify.centerline(mask_np, root=root)
    assert end_deviation(line) < 1.0  # ~4.8 px before the cut
    assert inside_mask(line, mask_np)


def test_disk_cut_rejects_chord_leaving_mask():
    from ramify._skeleton import disk_cut

    # a path that detours around a background pixel; every node's disk is
    # large enough to reach the anchor, but the straight chord to the far
    # nodes crosses the hole, so the cut must fall back to a nearer node
    mask = np.ones((5, 7), dtype=bool)
    mask[2, 3] = False
    half_width = np.full(mask.shape, 10.0)
    path = [(2, 1), (2, 2), (1, 3), (2, 4), (2, 5)]
    assert disk_cut(path, half_width, mask) == [(2, 1), (1, 3), (2, 4), (2, 5)]


def test_disk_cut_leaves_path_when_no_disk_reaches_anchor():
    from ramify._skeleton import disk_cut

    mask = np.ones((5, 7), dtype=bool)
    half_width = np.zeros(mask.shape)
    path = [(2, 0), (2, 1), (2, 2), (1, 3), (2, 4)]
    assert disk_cut(path, half_width, mask) == path


def test_disk_cut_fill_keeps_path_pixel_connected():
    from ramify._skeleton import disk_cut

    mask = np.ones((7, 9), dtype=bool)
    half_width = np.full(mask.shape, 10.0)
    path = [(3, 0), (4, 1), (4, 2), (3, 3), (3, 4), (3, 5)]
    assert disk_cut(path, half_width, mask) == [(3, 0), (3, 5)]
    assert disk_cut(path, half_width, mask, fill=True) == [(3, c) for c in range(6)]
