import warnings

import numpy as np
import pandas as pd
import pytest

import ramify


def test_lines_one_row_per_positive_label(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    arr = np.asarray(labels)
    present = set(np.unique(arr[arr > 0]).tolist())
    assert set(lines["region_id"]) == present
    assert len(lines) == len(present)


def test_lines_segment_level_one_row_per_label(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips, level="segment")
    arr = np.asarray(labels)
    present = set(np.unique(arr[arr > 0]).tolist())
    assert set(lines["region_id"]) == present
    assert lines["segment_id"].notna().all()
    assert (lines["segment_id"] == lines["region_id"]).all()


def test_network_line_endpoints_are_first_and_last_in_region_reference(toy_dataset):
    # For every "network"-sourced line, its first vertex should be the
    # region's own claimed territory -- specifically, not off in a
    # neighbouring region's ground (the line must stay inside `region_id`'s
    # own pixels, since it was built by walking the reference line to the
    # first/last in-region pixel).
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    arr = np.asarray(labels)
    from ramify._io import make_grid

    _, grid = make_grid(mask)
    for _, row in lines[lines["source"] == "network"].iterrows():
        coords = np.array(row.geometry.coords)
        rows, cols = grid.to_rc(coords[:, 0], coords[:, 1])
        rows_i = np.round(rows).astype(int)
        cols_i = np.round(cols).astype(int)
        assert arr[rows_i[0], cols_i[0]] == row["region_id"]
        assert arr[rows_i[-1], cols_i[-1]] == row["region_id"]


def test_tributary_line_starts_on_cut_edge(toy_dataset):
    # A tributary's downstream end sits where it was claimed off (a cut, not
    # a skeleton tip): the pixel just downstream of its line's start belongs
    # to a *different* region.
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    arr = np.asarray(labels)
    from ramify._io import make_grid

    _, grid = make_grid(mask)

    tributary = lines[(lines["source"] == "network") & (lines["path_id"] != 1)].iloc[0]
    coords = np.array(tributary.geometry.coords)
    rows, cols = grid.to_rc(coords[:, 0], coords[:, 1])
    r0, c0 = int(round(rows[0])), int(round(cols[0]))
    neighbours = [
        arr[r0 + dr, c0 + dc]
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if 0 <= r0 + dr < arr.shape[0] and 0 <= c0 + dc < arr.shape[1] and not (dr == 0 and dc == 0)
    ]
    assert any(n != tributary["region_id"] and n > 0 for n in neighbours)


def test_fallback_diameter_and_empty_and_islands(toy_dataset):
    # A very small min_length prunes the network into many short tributaries,
    # some of which get swallowed entirely by their parent's priority claim
    # (source="diameter"), some too small for any line (source="empty"), and
    # some regions end up disconnected (island_pixels > 0).
    mask, root, tips = toy_dataset
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels, net, lines = ramify.partition_priority(mask, root, min_length=3.0)
    messages = [str(w.message) for w in caught]
    assert any("geodesic diameter" in m for m in messages)
    assert any("too small for a centerline" in m for m in messages)
    assert any("disconnected components" in m for m in messages)

    assert (lines["source"] == "diameter").any()
    assert (lines["source"] == "empty").any()
    assert (lines.loc[lines["source"] == "empty", "length"] == 0).all()
    assert (lines.loc[lines["source"] == "empty", "geometry"].apply(lambda g: g.is_empty)).all()
    assert (lines["island_pixels"] > 0).any()


def test_net_orientation_starts_at_root(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    from ramify._io import make_grid

    _, grid = make_grid(mask)
    outlet = net[net["downstream_segment_id"].isna()].iloc[0]
    root_xy = grid.to_xy(np.array([root[0]]), np.array([root[1]]))
    first_vertex = np.array(outlet.geometry.coords[0])
    assert np.allclose(first_vertex, [root_xy[0][0], root_xy[1][0]])


def test_net_orientation_junctions_shared(toy_dataset):
    mask, root, tips = toy_dataset
    _, net, _ = ramify.partition_priority(mask, root, tips=tips)
    by_id = net.set_index("segment_id")
    for _, row in net.iterrows():
        d = row["downstream_segment_id"]
        if pd.isna(d):
            continue
        downstream_geom = by_id.loc[int(d), "geometry"]
        first_this = row.geometry.coords[0]
        last_down = downstream_geom.coords[-1]
        assert np.allclose(first_this, last_down)


def test_lines_start_at_downstream_end(toy_dataset):
    # For a "network" line whose region is the mainstem's outlet region
    # (path_id == 1), the first vertex should be at (or very near) the root.
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    from ramify._io import make_grid

    _, grid = make_grid(mask)
    mainstem = lines[lines["path_id"] == 1].iloc[0]
    root_xy = grid.to_xy(np.array([root[0]]), np.array([root[1]]))
    first_vertex = np.array(mainstem.geometry.coords[0])
    assert np.allclose(first_vertex, [root_xy[0][0], root_xy[1][0]])


# -- width_regions_interpolate -------------------------------------------------


def test_width_regions_interpolate_covers_mask(toy_dataset):
    mask, root, tips = toy_dataset
    mask_bool = np.asarray(mask.values) == 1
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    w = np.asarray(ramify.width_regions_interpolate(labels, lines))
    assert np.isnan(w[~mask_bool]).all()
    assert not np.isnan(w[mask_bool]).any()
    assert (w[mask_bool] > 0).all()


def test_width_regions_interpolate_no_near_zero_at_cuts(toy_dataset):
    mask, root, tips = toy_dataset
    mask_bool = np.asarray(mask.values) == 1
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    w = np.asarray(ramify.width_regions_interpolate(labels, lines))
    assert np.nanmin(w[mask_bool]) > 1.0  # no collapse-to-zero at partition cuts


def test_width_regions_interpolate_close_to_manual_loop(toy_dataset):
    # Equivalent in spirit to a manual per-region width_interpolate loop with
    # open_boundary = (labels > 0) & ~region -- not bit-identical (see
    # crop_grid's docstring: all_touched rasterization of a diagonal line
    # through an exact pixel corner is sub-ULP sensitive), but close.
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    labels_arr = np.asarray(labels)
    mask_bool = labels_arr > 0

    w_batch = np.asarray(ramify.width_regions_interpolate(labels, lines))

    w_manual = np.full(labels_arr.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in lines.iterrows():
            if row.geometry.is_empty:
                continue
            region = labels == row["region_id"]
            region_bool = np.asarray(region.values) > 0
            wr = np.asarray(
                ramify.width_interpolate(
                    region, row.geometry, open_boundary=mask_bool & ~region_bool
                )
            )
            w_manual[region_bool] = wr[region_bool]

    both = np.isfinite(w_batch) & np.isfinite(w_manual) & mask_bool
    assert both.sum() / mask_bool.sum() > 0.99
    diff = np.abs(w_batch[both] - w_manual[both])
    assert np.median(diff) < 0.5
    assert diff.mean() < 2.0


def test_width_regions_interpolate_method_nearest(toy_dataset):
    mask, root, tips = toy_dataset
    mask_bool = np.asarray(mask.values) == 1
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    w = np.asarray(ramify.width_regions_interpolate(labels, lines, method="nearest"))
    assert np.isnan(w[~mask_bool]).all()
    assert not np.isnan(w[mask_bool]).any()


def test_width_regions_interpolate_invalid_method(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    with pytest.raises(ValueError, match="method"):
        ramify.width_regions_interpolate(labels, lines, method="bogus")


def test_width_regions_interpolate_progress(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    calls = []
    ramify.width_regions_interpolate(labels, lines, progress=lambda *a: calls.append(a))
    assert len(calls) == len(lines)
    assert calls[0][:2] == (0, len(lines))


def test_width_regions_interpolate_duplicate_region_id_raises(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    dup = pd.concat([lines, lines.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ramify.width_regions_interpolate(labels, dup)


def test_width_regions_interpolate_unknown_region_id_raises(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    bad = lines.copy()
    bad.loc[0, "region_id"] = 9999
    with pytest.raises(ValueError, match="not present"):
        ramify.width_regions_interpolate(labels, bad)


# -- width_regions_stations ----------------------------------------------------


def test_width_regions_stations_station_id_unique_and_region_matches(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    labels_arr = np.asarray(labels)
    w, stations = ramify.width_regions_stations(labels, lines, n_stations=5)

    assert stations["station_id"].is_unique

    from ramify._io import make_grid

    _, grid = make_grid(mask)
    xs = [p.x for p in stations["station"]]
    ys = [p.y for p in stations["station"]]
    rows, cols = grid.to_rc(np.array(xs), np.array(ys))
    rows_i, cols_i = np.round(rows).astype(int), np.round(cols).astype(int)
    under_point = labels_arr[rows_i, cols_i]
    assert np.array_equal(under_point, stations["region_id"].to_numpy())


def test_width_regions_stations_requires_exactly_one_arg(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    with pytest.raises(ValueError, match="exactly one"):
        ramify.width_regions_stations(labels, lines)
    with pytest.raises(ValueError, match="exactly one"):
        ramify.width_regions_stations(labels, lines, spacing=10, n_stations=5)


def test_width_regions_stations_no_line_regions_stay_nan(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, min_length=3.0)
    labels_arr = np.asarray(labels)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w, stations = ramify.width_regions_stations(labels, lines, n_stations=3)
    assert any("have no line" in str(x.message) for x in caught)

    empty_regions = set(lines.loc[lines["source"] == "empty", "region_id"])
    w_arr = np.asarray(w)
    for rid in empty_regions:
        assert np.isnan(w_arr[labels_arr == rid]).all()


def test_width_regions_stations_duplicate_and_unknown_region_id(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    dup = pd.concat([lines, lines.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        ramify.width_regions_stations(labels, dup, n_stations=5)

    bad = lines.copy()
    bad.loc[0, "region_id"] = 9999
    with pytest.raises(ValueError, match="not present"):
        ramify.width_regions_stations(labels, bad, n_stations=5)


def test_labels_validation_negative_raises(toy_dataset):
    mask, root, tips = toy_dataset
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    bad = np.asarray(labels).copy().astype(int)
    bad[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        ramify.width_regions_interpolate(bad, lines)
