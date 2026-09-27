from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def load():
    # returns (mask, root, tips): mask as xr.DataArray if rioxarray is
    # available (falls back to numpy), root as (row, col), tips as list of
    # (row, col)
    try:
        import rioxarray

        mask = rioxarray.open_rasterio(_HERE / "example.tif").squeeze(drop=True)
        root_arr = rioxarray.open_rasterio(_HERE / "root.tif").squeeze(drop=True).values
        tips_arr = rioxarray.open_rasterio(_HERE / "tips.tif").squeeze(drop=True).values
    except ImportError:
        import tifffile

        mask = tifffile.imread(_HERE / "example.tif")
        root_arr = tifffile.imread(_HERE / "root.tif")
        tips_arr = tifffile.imread(_HERE / "tips.tif")

    rows, cols = np.nonzero(root_arr > 0)
    if len(rows) != 1:
        raise ValueError(
            f"root.tif must contain exactly 1 nonzero pixel, found {len(rows)}"
        )
    root = (int(rows[0]), int(cols[0]))

    rows, cols = np.nonzero(tips_arr > 0)
    tips = list(zip(rows.tolist(), cols.tolist()))
    return mask, root, tips
