import numpy as np
import pytest


@pytest.fixture
def straight_bar():
    """A 1-pixel-wide horizontal bar, root at one end, tip at the other."""
    mask = np.zeros((5, 20), dtype=np.uint8)
    mask[2, 1:19] = 1
    return mask, (2, 1), (2, 18)


@pytest.fixture
def stubbed_bar():
    """A long horizontal bar with a short stub off the middle -- the stub
    should be trimmed away when choosing endpoints without an explicit tip."""
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[5, 1:19] = 1
    mask[3:5, 10] = 1
    return mask


@pytest.fixture
def toy_dataset():
    from ramify.data import load

    return load()
