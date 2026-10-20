"""Step 8 - habitat richness / restoration-potential rasters.

Re-aggregates the per-habitat suitability rasters from ``project`` (no need to
re-read the PC rasters). For every 10 m cell:

  habitat_richness.tif        int16   number of habitat types with suitability >= threshold
  habitat_richness_open.tif   int16   same, non-forest habitats only (open / low-nutrient)
  habitat_richness_sum.tif    float32 sum of suitability over all habitats
                                      (threshold-free "effective number of habitats")

High values = ground whose geophysics could support several valuable habitat
types - a type-agnostic signal for where new nature is worth establishing.
"""
from __future__ import annotations

import glob
import re

import numpy as np
import rasterio

from .config import Config
from .grid import iter_block_windows, load_grid
from .ioretry import read_band
from .lookup import load_lookup

NODATA_F = -1.0


def _per_class_paths(cfg: Config) -> dict[str, str]:
    out = {}
    for p in sorted(glob.glob(str(cfg.dir_fuzzy / "*_suitability.tif"))):
        m = re.match(r"(\d+)_(.+)_suitability\.tif$", p.rsplit("/", 1)[-1])
        if m:
            out[m.group(2)] = p
    return out


def build_richness(cfg: Config, force: bool = False) -> None:
    out_path = cfg.dir_potential / "habitat_richness.tif"
    if out_path.exists() and not force:
        print(f"[richness] exists, skipping: {out_path}")
        return

    paths = _per_class_paths(cfg)
    if not paths:
        raise FileNotFoundError("no per-class suitability rasters - run `signatur project` first")

    lut = load_lookup(cfg)
    code2ht = dict(zip(lut["naturkode"], lut["hovednaturtype"]))
    open_set = set(cfg.richness.open_hovednaturtyper)
    codes = sorted(paths)
    is_open = np.array([code2ht.get(c) in open_set for c in codes])
    thr = cfg.richness.threshold
    print(f"[richness] {len(codes)} habitats, threshold {thr}; "
          f"{int(is_open.sum())} count as 'open' ({', '.join(sorted(open_set))})")

    grid = load_grid(cfg.reference_path)
    srcs = [rasterio.open(paths[c]) for c in codes]

    prof_i = dict(grid.profile, count=1, dtype="int16", nodata=-1,
                  compress="DEFLATE", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    prof_f = dict(prof_i, dtype="float32", nodata=NODATA_F)

    d_all = rasterio.open(out_path, "w", **prof_i)
    d_open = rasterio.open(cfg.dir_potential / "habitat_richness_open.tif", "w", **prof_i)
    d_sum = rasterio.open(cfg.dir_potential / "habitat_richness_sum.tif", "w", **prof_f)

    bbox = cfg.processing.test_bbox
    windows = list(iter_block_windows(grid, cfg.processing.block_size, bbox))
    try:
        for bi, win in enumerate(windows, 1):
            h, w = int(win.height), int(win.width)
            stack = np.empty((len(srcs), h, w), dtype=np.float32)
            for i, s in enumerate(srcs):
                stack[i] = read_band(s, 1, window=win)
            valid = stack[0] != NODATA_F                       # same land mask for every layer

            over = (stack >= thr) & (stack != NODATA_F)
            cnt_all = over.sum(0).astype(np.int16)
            cnt_open = over[is_open].sum(0).astype(np.int16)
            ssum = np.where(stack != NODATA_F, stack, 0.0).sum(0).astype(np.float32)

            cnt_all[~valid] = -1
            cnt_open[~valid] = -1
            ssum[~valid] = NODATA_F

            d_all.write(cnt_all, 1, window=win)
            d_open.write(cnt_open, 1, window=win)
            d_sum.write(ssum, 1, window=win)
            if bi % 25 == 0 or bi == len(windows):
                print(f"[richness]   block {bi}/{len(windows)}", flush=True)
    finally:
        for s in srcs:
            s.close()
        for d in (d_all, d_open, d_sum):
            d.close()

    print(f"[richness] wrote habitat_richness / _open / _sum -> {cfg.dir_potential}")
