import numpy as np
import pandas as pd
import pytest

import ramify
from ramify.partition import _annotate
from ramify._skeleton import path_weight


def test_tips_and_min_length_mutually_exclusive(toy_dataset):
    mask, root, tips = toy_dataset
    with pytest.raises(ValueError, match="mutually exclusive"):
        ramify.partition_priority(mask, root, tips=tips, min_length=5.0)


def test_invalid_level_raises(toy_dataset):
    mask, root, tips = toy_dataset
    with pytest.raises(ValueError, match="level"):
        ramify.partition_priority(mask, root, tips=tips, level="bogus")


def test_invalid_path_by_raises(toy_dataset):
    mask, root, tips = toy_dataset
    with pytest.raises(ValueError, match="path_by"):
        ramify.partition_priority(mask, root, tips=tips, path_by="bogus")


def test_every_mask_pixel_labeled(toy_dataset):
    mask, root, tips = toy_dataset
    mask_bool = np.asarray(mask.values) == 1
    labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
    arr = np.asarray(labels)
    assert (arr[~mask_bool] == 0).all()
    assert (arr[mask_bool] > 0).all()
    assert set(net["path_id"]) == set(np.unique(arr[arr > 0]).tolist())


def test_path_id_starts_at_one_and_parents_precede_tributaries(toy_dataset):
    mask, root, tips = toy_dataset
    _, net, _ = ramify.partition_priority(mask, root, tips=tips)
    assert net["path_id"].min() == 1

    by_id = net.set_index("segment_id")
    for _, row in net.iterrows():
        d = row["downstream_segment_id"]
        if pd.isna(d):
            continue
        downstream_path = by_id.loc[int(d), "path_id"]
        if downstream_path != row["path_id"]:
            assert downstream_path < row["path_id"]


def test_segment_id_deterministic_within_path(toy_dataset):
    mask, root, tips = toy_dataset
    _, net, _ = ramify.partition_priority(mask, root, tips=tips)
    by_id = net.set_index("segment_id")
    for _, row in net.iterrows():
        d = row["downstream_segment_id"]
        if pd.isna(d):
            continue
        if by_id.loc[int(d), "path_id"] == row["path_id"]:
            assert int(d) < row["segment_id"]


def test_repeated_calls_are_identical(toy_dataset):
    mask, root, tips = toy_dataset
    labels1, net1, lines1 = ramify.partition_priority(mask, root, tips=tips)
    labels2, net2, lines2 = ramify.partition_priority(mask, root, tips=tips)
    assert np.array_equal(np.asarray(labels1), np.asarray(labels2))
    pd.testing.assert_frame_equal(
        net1.drop(columns="geometry"), net2.drop(columns="geometry")
    )
    pd.testing.assert_frame_equal(
        lines1.drop(columns="geometry"), lines2.drop(columns="geometry")
    )


def test_partition_nearest_no_ordering_effect(toy_dataset):
    mask, root, tips = toy_dataset
    mask_bool = np.asarray(mask.values) == 1
    labels, net, lines = ramify.partition_nearest(mask, root, tips=tips)
    arr = np.asarray(labels)
    assert (arr[~mask_bool] == 0).all()
    assert (arr[mask_bool] > 0).all()
    assert set(net["path_id"]) == set(np.unique(arr[arr > 0]).tolist())


def test_level_segment_refines_level_path(toy_dataset):
    mask, root, tips = toy_dataset
    path_labels, net, _ = ramify.partition_priority(mask, root, tips=tips)
    seg_labels, _, _ = ramify.partition_priority(mask, root, tips=tips, level="segment")
    assert len(np.unique(np.asarray(seg_labels))) >= len(np.unique(np.asarray(path_labels)))
    seg_to_path = net.set_index("segment_id")["path_id"]
    seg_arr = np.asarray(seg_labels)
    path_arr = np.asarray(path_labels)
    positive = seg_arr > 0
    mapped = seg_to_path.reindex(seg_arr[positive]).to_numpy()
    assert np.array_equal(mapped, path_arr[positive])


def test_min_length_prunes_short_tips(toy_dataset):
    mask, root, tips = toy_dataset
    _, net_all, _ = ramify.partition_priority(mask, root)
    _, net_pruned, _ = ramify.partition_priority(mask, root, min_length=10.0)
    assert len(net_pruned) <= len(net_all)


def test_area_path_by_sums_subtree_not_max():
    # Junction J1 splits into a short-but-heavy leaf (armC) and a connector
    # (armAB) that itself splits into two leaves (branchA, branchB). area's
    # subtree statistic sums the children's weight, so a moderately-heavy leaf
    # (armC) can lose to two lighter siblings combined, where a max-based
    # statistic would have picked armC instead.
    root = (0, 0)
    trunk = [(0, 3), (0, 2), (0, 1), (0, 0)]
    armC = [(2, 3), (1, 3), (0, 3)]
    armAB = [(0, 5), (0, 4), (0, 3)]
    branchA = [(0, 15), (0, 14), (0, 13), (0, 12), (0, 11), (0, 10), (0, 9), (0, 8),
               (0, 7), (0, 6), (0, 5)]
    branchB = [(3, 5), (2, 5), (1, 5), (0, 5)]
    segments = [trunk, armC, armAB, branchA, branchB]

    edt = np.zeros((4, 16))
    edt[1, 3] = edt[2, 3] = 25.0
    edt[0, 6:16] = 1.0
    edt[1, 5] = edt[2, 5] = edt[3, 5] = 12.0

    w_a = path_weight(branchA, edt, 1.0)
    w_b = path_weight(branchB, edt, 1.0)
    w_c = path_weight(armC, edt, 1.0)
    # sanity: the fixture actually distinguishes sum from max
    assert max(w_a, w_b) < w_c < w_a + w_b

    df = _annotate(segments, edt, pixel_size=1.0, path_by="area", root=root)
    by_head = {tuple(row["pixels"][0]): row for _, row in df.iterrows()}
    trunk_path = by_head[(0, 3)]["path_id"]
    assert by_head[(0, 5)]["path_id"] == trunk_path  # armAB continues the mainstem
    assert by_head[(2, 3)]["path_id"] != trunk_path  # armC is its own tributary


def test_length_path_by_uses_max_not_sum():
    # Same shape, but with weights swapped for lengths: branchA is now the
    # long-but-thin arm and armC is a short offshoot. "length" should follow
    # the single longest route (branchA), unaffected by branchB's presence.
    root = (0, 0)
    trunk = [(0, 3), (0, 2), (0, 1), (0, 0)]
    armC = [(2, 3), (1, 3), (0, 3)]
    armAB = [(0, 5), (0, 4), (0, 3)]
    branchA = [(0, 15), (0, 14), (0, 13), (0, 12), (0, 11), (0, 10), (0, 9), (0, 8),
               (0, 7), (0, 6), (0, 5)]
    branchB = [(3, 5), (2, 5), (1, 5), (0, 5)]
    segments = [trunk, armC, armAB, branchA, branchB]
    edt = np.zeros((4, 16))

    df = _annotate(segments, edt, pixel_size=1.0, path_by="length", root=root)
    by_head = {tuple(row["pixels"][0]): row for _, row in df.iterrows()}
    trunk_path = by_head[(0, 3)]["path_id"]
    assert by_head[(0, 5)]["path_id"] == trunk_path
    assert by_head[(0, 15)]["path_id"] == trunk_path  # longest single route wins
    assert by_head[(3, 5)]["path_id"] != trunk_path
