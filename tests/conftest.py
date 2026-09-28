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


@pytest.fixture
def flared_bar():
    """A thick horizontal bar whose left end flares into a fishtail, so the
    skeleton forks into a Y there. Root sits on the flared end's edge."""
    from skimage.draw import polygon

    mask = np.zeros((40, 70), dtype=np.uint8)
    rr, cc = polygon([4, 12, 12, 28, 28, 36], [4, 24, 66, 66, 24, 4])
    mask[rr, cc] = 1
    return mask, (20, 4), (20, 65)


@pytest.fixture
def end_deviation():
    """How far the first `length` of a line strays from the straight chord
    over that stretch: ~0 for a clean end, several pixels for a swerve into a
    skeleton fork arm or a sideways snap."""
    from shapely.geometry import LineString, Point

    def fn(line, length=15.0):
        chord = LineString([line.coords[0], line.interpolate(length).coords[0]])
        return max(
            chord.distance(Point(p))
            for p in line.coords
            if line.project(Point(p)) <= length
        )

    return fn


@pytest.fixture
def inside_mask():
    """Whether every pixel a (pixel_size=1, numpy-grid) line crosses is in `mask`."""
    from skimage.draw import line as draw_line

    def fn(line, mask):
        pix = [(int(y - 0.5), int(x - 0.5)) for x, y in line.coords]
        for (r0, c0), (r1, c1) in zip(pix[:-1], pix[1:]):
            rr, cc = draw_line(r0, c0, r1, c1)
            if not mask[rr, cc].all():
                return False
        return True

    return fn
