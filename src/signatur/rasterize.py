"""Step 1 - priority-merge the habitat layers onto the 10 m PC grid.

Produces ``merged_class_id.tif`` (int16, nodata 0) where every cell carries the
``class_id`` of the highest-priority habitat polygon covering it, plus
``overlap_report.csv`` quantifying how much area each layer lost to a
higher-priority layer.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import bounds as window_bounds, transform as window_transform

from .config import Config
from .grid import load_grid, iter_block_windows
from .ioretry import to_csv
from .lookup import load_lookup


def _load_layer(cfg: Config, layer, code2id: dict[str, int], target_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(cfg.layer_path(layer))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    codes = gdf[cfg.naturkode_field]
    if pd.api.types.is_float_dtype(codes) or pd.api.types.is_integer_dtype(codes):
        codes = codes.astype("Int64").astype(str)
    else:
        codes = codes.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    gdf["class_id"] = codes.map(code2id).astype("Int64")

    unknown = gdf["class_id"].isna().sum()
    if unknown:
        print(f"[rasterize]   {layer.name}: dropping {unknown} features with unmapped naturkode")
        gdf = gdf[gdf["class_id"].notna()]
    gdf["class_id"] = gdf["class_id"].astype("int32")

    if gdf.crs is None:
        raise ValueError(f"{layer.name} has no CRS")
    from pyproj import CRS
    tgt = CRS.from_user_input(target_crs)
    if CRS.from_user_input(gdf.crs) != tgt:
        gdf = gdf.to_crs(tgt)

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"[rasterize]   {layer.name}: repairing {int(invalid.sum())} invalid geometries")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]

    return gdf[["geometry", "class_id"]].reset_index(drop=True)


def build_merged_raster(cfg: Config, force: bool = False) -> None:
    out_path = cfg.dir_merged / "merged_class_id.tif"
    report_path = cfg.dir_merged / "overlap_report.csv"
    if out_path.exists() and report_path.exists() and not force:
        print(f"[rasterize] exists, skipping: {out_path}")
        return

    lut = load_lookup(cfg)
    code2id = dict(zip(lut["naturkode"], lut["class_id"]))
    max_class = int(lut["class_id"].max())

    grid = load_grid(cfg.reference_path)

    # High priority first: index 0 = highest priority layer.
    layers = cfg.layers_by_priority
    gdfs: list[gpd.GeoDataFrame] = []
    sindexes = []
    for layer in layers:
        print(f"[rasterize] loading {layer.name} (priority {layer.priority})")
        gdf = _load_layer(cfg, layer, code2id, grid.crs)
        gdfs.append(gdf)
        sindexes.append(gdf.sindex)

    profile = grid.profile
    profile.update(count=1, dtype="int16", nodata=0, compress="lzw",
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")

    burned = np.zeros(len(layers), dtype=np.int64)      # cells a layer's polygons cover
    survived = np.zeros(len(layers), dtype=np.int64)    # cells that layer actually owns in the merge
    class_counts = np.zeros(max_class + 1, dtype=np.int64)

    bbox = cfg.processing.test_bbox
    with rasterio.open(out_path, "w", **profile) as dst:
        for win in iter_block_windows(grid, cfg.processing.block_size, bbox):
            h, w = int(win.height), int(win.width)
            wt = window_transform(win, grid.transform)
            wb = window_bounds(win, grid.transform)

            merged = np.zeros((h, w), dtype=np.int16)
            owner = np.full((h, w), -1, dtype=np.int8)

            for li, (gdf, sidx) in enumerate(zip(gdfs, sindexes)):
                cand = list(sidx.intersection(wb))
                if not cand:
                    continue
                sub = gdf.iloc[cand]
                block = features.rasterize(
                    ((geom, cid) for geom, cid in zip(sub.geometry, sub["class_id"])),
                    out_shape=(h, w),
                    transform=wt,
                    fill=0,
                    all_touched=cfg.all_touched,
                    dtype="int32",
                )
                covered = block > 0
                burned[li] += int(covered.sum())
                take = covered & (owner < 0)       # only where no higher-priority layer already claimed
                merged[take] = block[take].astype(np.int16)
                owner[take] = li

            for li in range(len(layers)):
                survived[li] += int((owner == li).sum())
            class_counts += np.bincount(merged.ravel(), minlength=max_class + 1)

            dst.write(merged, 1, window=win)

    cell_ha = 0.01  # 10 m x 10 m
    rows = []
    for li, layer in enumerate(layers):
        rows.append({
            "layer": layer.name,
            "priority": layer.priority,
            "cells_covered": int(burned[li]),
            "cells_owned_after_merge": int(survived[li]),
            "cells_lost_to_higher_priority": int(burned[li] - survived[li]),
            "ha_covered": round(burned[li] * cell_ha, 2),
            "ha_owned_after_merge": round(survived[li] * cell_ha, 2),
        })
    to_csv(pd.DataFrame(rows), report_path, index=False)

    per_class = lut[["class_id", "naturkode", "navn", "hovednaturtype"]].copy()
    per_class["n_cells"] = per_class["class_id"].map(lambda c: int(class_counts[c]))
    per_class["area_ha"] = (per_class["n_cells"] * cell_ha).round(2)
    to_csv(per_class, cfg.dir_merged / "class_cells.csv", index=False)

    total = int(class_counts[1:].sum())
    print(f"[rasterize] wrote {out_path}")
    print(f"[rasterize] {total:,} habitat cells ({total * cell_ha:,.0f} ha); report -> {report_path}")
