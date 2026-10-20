"""One-off data prep: split + resample the climate PCA onto the 10 m PC grid.

``climate_pca.tif`` is a 3-band, 1000 m raster (one band per PC), on the same
CRS but not pixel-aligned with the 10 m grid the rest of the project uses. This
writes one single-band GeoTIFF per PC, resampled onto the exact reference grid
(same transform/shape/crs as the other PC rasters, so ``check_same_grid``
passes and they slot straight into ``pc_rasters`` in a config file).

Bilinear resampling is used: the climate surface is smooth and continuous, so
interpolating between the 1 km centres is more faithful than nearest-neighbour
(which would carve the output into visible 1 km blocks). No information finer
than 1 km is created - it just avoids fabricating artificial hard edges at
income cell boundaries that aren't in the source data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .config import Config
from .grid import load_grid


def prepare_climate_pcs(
    cfg: Config,
    source: str | Path,
    out_names: list[str] | None = None,
    resampling: str = "bilinear",
    force: bool = False,
) -> list[Path]:
    src_path = Path(source)
    if not src_path.is_absolute():
        src_path = cfg.pc_raster_dir / src_path
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    grid = load_grid(cfg.reference_path)
    resamp = getattr(Resampling, resampling)

    with rasterio.open(src_path) as src:
        n_bands = src.count
        names = out_names or [f"climate_PC{i}" for i in range(1, n_bands + 1)]
        if len(names) != n_bands:
            raise ValueError(f"{src_path.name} has {n_bands} bands but {len(names)} names given")

        out_paths = []
        for b, name in enumerate(names, start=1):
            out_path = cfg.pc_raster_dir / f"{name}.tif"
            out_paths.append(out_path)
            if out_path.exists() and not force:
                print(f"[climate-prep] exists, skipping: {out_path}")
                continue

            src_band = src.read(b)
            dst = np.full((grid.height, grid.width), np.float32(cfg.nodata), dtype=np.float32)
            reproject(
                source=src_band,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                dst_nodata=np.float32(cfg.nodata),
                resampling=resamp,
            )

            profile = dict(grid.profile, count=1, dtype="float32", nodata=cfg.nodata,
                           compress="deflate", predictor=3, tiled=True,
                           blockxsize=512, blockysize=512, BIGTIFF="YES")
            with rasterio.open(out_path, "w", **profile) as ds:
                ds.write(dst, 1)
            valid = dst != np.float32(cfg.nodata)
            print(f"[climate-prep] {name}: {src_path.name} band {b} "
                  f"({resampling}) -> {out_path.name}  "
                  f"[{valid.sum():,} valid cells, range {dst[valid].min():.2f}..{dst[valid].max():.2f}]")
            del dst

    return out_paths
