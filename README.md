# ramify

Split a branched shape into its branches, and measure each branch.

`ramify` takes a binary mask of a branching shape — a river network, a
floodplain, a glacier, a root system, a leaf's veins — and divides it into one
region per branch. Each region comes with its own centerline, so every branch
can be measured on its own: length, area, and width, along its whole length.

![graphical abstract](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/abstract.png)

## Install

```bash
pip install ramify
```

## Quickstart

```python
import ramify
from ramify.data import load

mask, root, tips = load()   # bundled toy dataset

labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
```

`lines` has one row per branch. Length and area come with it; mean width is
their ratio:

```python
lines["mean_width"] = lines["area"] / lines["length"]
lines[["path_id", "length", "area", "mean_width"]].head()
```

```
 path_id  length     area  mean_width
       1  3356.8 395885.7       117.9
       2  2538.2 282902.2       111.5
       3  1519.3  59523.3        39.2
       4   934.4  29133.2        31.2
       5  1081.2  42812.4        39.6
```

For width everywhere, not just per branch:

```python
widths = ramify.width_regions_interpolate(labels, lines)
```

![quickstart](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/quickstart.png)

## What you get

`partition_priority` returns three things:

- **`labels`**: a raster assigning every pixel of the shape to a branch
  (`path_id`). Path 1 is the mainstem; ids are ordered by branch size, and a
  branch is always numbered before its tributaries.
- **`net`**: the branching network used to build the partition, one row per
  segment (the piece between two junctions, or a junction and a tip). Its
  columns — `segment_id`, `path_id`, `downstream_segment_id`, `strahler`,
  `length`, `weight` — record how the shape was split and why the paths are
  ordered as they are.
- **`lines`**: one centerline per region, recomputed from the region's final
  pixels. Use these to measure branches.

`net` and `lines` differ on purpose. The network is traced through the whole
shape before it is split. Once each branch has its own territory, its true
centerline is not the same as the network line: tributaries start at the edge
of their region rather than in the middle of the mainstem, and regions pick up
pixels during allocation. `lines` is measured from the final regions; `net` is
kept as it was, so its values still explain the partition.

![net vs lines](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/net_vs_lines.png)

The `source` column of `lines` flags regions where the centerline couldn't be
traced from the network (`"diameter"`: a fallback path through the region;
`"empty"`: region too small for a line). `island_pixels` counts region pixels
disconnected from the part the line runs through.

## Inputs

- **`mask`**: 2-D array, shape pixels `> 0`. `np.ndarray` (set `pixel_size=`)
  or a georeferenced `xr.DataArray`; outputs match the input type.
- **`root`**: `(row, col)` of the network's outlet — the downstream end.
- **Tips**, one of:
  - `tips=[(row, col), ...]`: the branch ends you want. Only these branches
    are kept.
  - `min_length=`: use every end of the shape's skeleton, after pruning side
    branches shorter than this (map units).
  - neither: every skeleton end becomes a tip. Noisy boundaries produce many
    short spurious branches; prefer one of the above on real data.

![tips options](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/tips_options.png)

Roots and tips can often be derived automatically — glacier branch tips
([Kienholz et al., 2014](https://tc.copernicus.org/articles/8/503/2014/tc-8-503-2014.pdf)),
channel initiation points, the lowest point on the boundary as the root — or
digitized by hand in GIS software.

## Partitioning options

### Method

```python
labels, net, lines = ramify.partition_priority(mask, root, tips=tips)
labels, net, lines = ramify.partition_nearest(mask, root, tips=tips)
```

- **`partition_priority`**: branches claim space in order, biggest first,
  each out to its local half-width. Wide branches take proportionally more
  space at junctions, and the mainstem is not cut into by its tributaries.
- **`partition_nearest`**: every pixel goes to the nearest branch centerline.
  No ordering, no width limit. Simpler; at junctions, tributaries cut into the
  mainstem.

![partition methods](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/partition_methods.png)

### Which branch is the mainstem (`path_by`)

At each junction, the path continues into one upstream branch; the others
start new paths. `path_by` picks which:

- `"area"` (default): the branch with the most shape upstream of it.
- `"length"`: the branch leading to the farthest tip (Hack's mainstem).
- `"strahler"`: the branch with the higher Strahler order; ties go to the
  longest.

On the bundled toy dataset all three pick the same mainstem, so there's no
figure here worth the difference -- try it on a shape with a short, wide
branch competing against a long, narrow one to see them diverge.

### Paths or segments (`level`)

`level="path"` (default) gives one region per path. `level="segment"` splits
each path's region further, one region per segment; `labels` and `lines` are
then keyed by `segment_id`.

![level](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/level.png)

## Widths

Both width functions take `labels` and `lines` from a partition, and measure
each region from its own centerline.

```python
w = ramify.width_regions_interpolate(labels, lines)                     # laplace
w = ramify.width_regions_interpolate(labels, lines, method="nearest")
w, stations = ramify.width_regions_stations(labels, lines, spacing=50)
```

- **Interpolate**: width is measured exactly on the centerline (twice the
  distance to the nearest wall) and spread across the region, either smoothly
  (`method="laplace"`) or by copying the nearest centerline value
  (`method="nearest"`, faster, piecewise constant). Boundaries between regions
  are not treated as walls, so widths don't shrink at junctions.
- **Stations**: stations are placed along each centerline every `spacing`
  (or `n_stations` per region). Each station gets the part of the region
  closest to it; its width is that cell's area divided by the centerline
  length inside it. `stations` is a `GeoDataFrame`, one row per station, with
  `region_id`, `station_id`, `area`, `length`, `width`, the cell polygon, and
  the station point. Cells are straight-line (Euclidean) Voronoi cells, so on
  tight bends they can reach across the shape; cells at the end of a line
  include the end cap and read slightly wide.

![widths](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/widths_regions.png)

## Single shapes

For a shape with no branches (or one region pulled out of a partition), use
the underlying functions directly:

```python
shape = labels == 3   # one region, pulled out of a partition

line = ramify.centerline(shape)                          # LineString
w = ramify.width_interpolate(shape, line)
w, stations = ramify.width_stations(shape, line, spacing=50)
```

`centerline` returns the longest path through the shape's skeleton. Give
`root=` to fix one end, or `root=` and `tip=` to fix both.

![single shape](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/single_shape.png)

## Open boundaries

Branch sizes, claiming distances, and interpolated widths all depend on the
distance from each pixel to the shape's wall. By default every edge of the
mask is a wall. Where part of the edge is not a real wall — the shape is cut
off by open water or the edge of the data — pass `open_boundary`: a mask on
the same grid marking the open area. Distances are then measured only to the
remaining real walls.

```python
import numpy as np

mask_bool = np.asarray(mask.values) == 1
rr, cc = np.ogrid[:mask_bool.shape[0], :mask_bool.shape[1]]
open_boundary = (~mask_bool) & (rr >= root[0] - 10) & (cc <= root[1] + 20)

labels, net, lines = ramify.partition_priority(mask, root, tips=tips,
                                               open_boundary=open_boundary)
w = ramify.width_regions_interpolate(labels, lines, open_boundary=open_boundary)
```

Pass it to both steps so they measure against the same walls. Mark the open
area with some depth, not a thin strip: distances are measured through it, so a
one-pixel strip only moves the wall by one pixel. `width_regions_stations`
does not use wall distances and has no `open_boundary`.

![open boundary](https://raw.githubusercontent.com/avkoehl/ramify/main/assets/open_boundary.png)

## Conventions

- **Coordinates.** Georeferenced input: geometry is in the input's CRS. Plain
  arrays: `x = col * pixel_size`, `y = row * pixel_size` (y increases
  downward), vertices at pixel centers.
- **Units.** Lengths, widths, `spacing`, `min_length` in map units; areas in
  map units squared.
- **Points.** `root`, `tip`, `tips` are `(row, col)` pixel indices.
- **Orientation.** Every line runs downstream to upstream, starting at the
  root end, wherever a root is known (`net`, `lines`, and `centerline` given
  `root`). `centerline(shape)` called with neither `root` nor `tip` has no
  root to start from; its direction is deterministic but arbitrary.
- **Output type.** Rasters match the input (`np.ndarray` or `xr.DataArray`);
  NaN outside the shape for widths, 0 outside for labels.

## Development

```bash
git clone https://github.com/avkoehl/ramify.git
cd ramify
uv sync --extra dev
```
