"""Grid helpers - all rasters in this project share one 10 m grid.

Because the merged habitat raster is written on exactly the PC-raster grid, the
sampling and projection steps never need a coordinate lookup: a habitat cell at
``(row, col)`` is the same cell in every PC raster. These helpers enforce that
assumption and provide the windowed iteration used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import rasterio
from rasterio.windows import Window, from_bounds


@dataclass(frozen=True)
class Grid:
    height: int
    width: int
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    profile: dict

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.width, self.height)
        return (left, bottom, right, top)


def load_grid(template_path: str | Path) -> Grid:
    with rasterio.open(template_path) as src:
        return Grid(
            height=src.height,
            width=src.width,
            transform=src.transform,
            crs=src.crs,
            profile=src.profile.copy(),
        )


def check_same_grid(paths: list[Path], reference: Path | None = None) -> Grid:
    """Raise if any raster in ``paths`` differs in shape/transform/crs.

    Returns the reference grid on success.
    """
    ref_path = reference or paths[0]
    ref = load_grid(ref_path)
    for p in paths:
        with rasterio.open(p) as src:
            if (src.height, src.width) != ref.shape:
                raise ValueError(f"shape mismatch: {Path(p).name} {(src.height, src.width)} vs {ref.shape}")
            if src.transform != ref.transform:
                raise ValueError(f"transform mismatch: {Path(p).name}")
            if src.crs != ref.crs:
                raise ValueError(f"CRS mismatch: {Path(p).name} ({src.crs} vs {ref.crs})")
    return ref


def _clip_window(win: Window, height: int, width: int) -> Window:
    row_off = max(0, int(win.row_off))
    col_off = max(0, int(win.col_off))
    row_end = min(height, int(win.row_off + win.height))
    col_end = min(width, int(win.col_off + win.width))
    return Window(col_off, row_off, max(0, col_end - col_off), max(0, row_end - row_off))


def iter_block_windows(
    grid: Grid,
    block_size: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> Iterator[Window]:
    """Yield block windows tiling the grid (or just the part covering ``bbox``)."""
    if bbox is not None:
        region = _clip_window(
            from_bounds(*bbox, transform=grid.transform).round_offsets().round_lengths(),
            grid.height,
            grid.width,
        )
        r0, c0 = int(region.row_off), int(region.col_off)
        r1, c1 = r0 + int(region.height), c0 + int(region.width)
    else:
        r0, c0, r1, c1 = 0, 0, grid.height, grid.width

    for row in range(r0, r1, block_size):
        h = min(block_size, r1 - row)
        for col in range(c0, c1, block_size):
            w = min(block_size, c1 - col)
            if h > 0 and w > 0:
                yield Window(col, row, w, h)
