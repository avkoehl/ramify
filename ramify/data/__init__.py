"""Bundled toy dataset (mask, root, tips) used by the README and scripts."""
from pathlib import Path

_DIR = Path(__file__).parent


def load():
    """Return ``(mask, root, tips)`` for the bundled toy dataset.

    ``mask`` is a georeferenced ``xr.DataArray`` (uint8, 1 = shape), ``root`` a
    ``(row, col)`` tuple, and ``tips`` a list of ``(row, col)`` tuples, ready to
    pass straight to :func:`ramify.extract_centerlines`.
    """
    import numpy as np
    import rioxarray

    with rioxarray.open_rasterio(_DIR / "mask.tif") as da:
        mask = da.squeeze("band", drop=True).load()
    with rioxarray.open_rasterio(_DIR / "root.tif") as da:
        root_rc = np.argwhere(da.squeeze("band", drop=True).values == 1)
    with rioxarray.open_rasterio(_DIR / "tips.tif") as da:
        tips_rc = np.argwhere(da.squeeze("band", drop=True).values == 1)

    (root,) = [tuple(map(int, rc)) for rc in root_rc]
    tips = [tuple(map(int, rc)) for rc in tips_rc]
    return mask, root, tips
