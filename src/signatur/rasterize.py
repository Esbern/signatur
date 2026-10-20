"""Step 1 - priority-merge the habitat layers onto the 10 m PC grid.

Produces:
  merged_class_id.tif   int16, nodata 0 - class_id of the highest-priority
                        habitat polygon covering each cell
  feature_id.tif        int32, nodata 0 - which SOURCE POLYGON won that cell
                        (paired 1:1 with merged_class_id.tif; same priority
                        resolution, decoded from a single rasterize pass so the
                        two rasters can never disagree about which polygon won)
  feature_split.csv     feature_id -> train/test, the 80/20 HOLDOUT SPLIT
                        assigned per polygon (stratified within each class,
                        seeded). Carried through `extract` as pixel column
                        `split` and used by `holdout` to fit the envelope model
                        on 80% of polygons and evaluate purely on the other 20%
                        - pixels from the same polygon are near-identical, so a
                        pixel-level split would leak information.
  overlap_report.csv    per-layer burnt-cell counts, as before.
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

FEATURE_ID_SCALE = 1_000_000  # packed burn value = class_id * SCALE + feature_id


def _load_layer(cfg: Config, layer, code2id: dict[str, int], target_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(cfg.layer_path(layer))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    codes = gdf[cfg.naturkode_field]
    if pd.api.types.is_float_dtype(codes) or pd.api.types.is_integer_dtype(codes):
        codes = codes.astype("Int64").astype(str)
    else:
        codes = codes.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    gdf["class_id"] = codes.map(code2id).astype("Int64")
    gdf["naturkode"] = codes

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

    gdf = gdf[["geometry", "class_id", "naturkode"]].reset_index(drop=True)
    if len(gdf) >= FEATURE_ID_SCALE:
        raise ValueError(f"{layer.name} has {len(gdf)} features - raise FEATURE_ID_SCALE")
    return gdf


def _assign_holdout_split(gdfs: list[gpd.GeoDataFrame], layer_names: list[str],
                          holdout_frac: float, seed: int) -> pd.DataFrame:
    """One row per source polygon, globally unique feature_id, stratified train/test split."""
    rng = np.random.default_rng(seed)
    rows = []
    next_id = 1
    for gdf, name in zip(gdfs, layer_names):
        fid = np.arange(next_id, next_id + len(gdf))
        gdf["feature_id"] = fid
        next_id += len(gdf)
        rows.append(pd.DataFrame({
            "feature_id": fid, "layer": name,
            "class_id": gdf["class_id"].to_numpy(), "naturkode": gdf["naturkode"].to_numpy(),
        }))
    table = pd.concat(rows, ignore_index=True)
    table["split"] = 0
    for cid, idx in table.groupby("class_id").groups.items():
        idx = np.array(idx)
        is_test = rng.random(len(idx)) < holdout_frac
        table.loc[idx[is_test], "split"] = 1
    return table


def build_merged_raster(cfg: Config, force: bool = False) -> None:
    out_path = cfg.dir_merged / "merged_class_id.tif"
    feat_path = cfg.dir_merged / "feature_id.tif"
    report_path = cfg.dir_merged / "overlap_report.csv"
    if out_path.exists() and feat_path.exists() and report_path.exists() and not force:
        print(f"[rasterize] exists, skipping: {out_path}")
        return

    lut = load_lookup(cfg)
    code2id = dict(zip(lut["naturkode"], lut["class_id"]))
    max_class = int(lut["class_id"].max())

    grid = load_grid(cfg.reference_path)

    # High priority first: index 0 = highest priority layer.
    layers = cfg.layers_by_priority
    gdfs: list[gpd.GeoDataFrame] = []
    for layer in layers:
        print(f"[rasterize] loading {layer.name} (priority {layer.priority})")
        gdfs.append(_load_layer(cfg, layer, code2id, grid.crs))

    split_table = _assign_holdout_split(
        gdfs, [ly.name for ly in layers], cfg.analysis.holdout_frac, cfg.analysis.random_seed)
    to_csv(split_table, cfg.dir_merged / "feature_split.csv", index=False)
    n_train, n_test = int((split_table.split == 0).sum()), int((split_table.split == 1).sum())
    print(f"[rasterize] holdout split: {n_train} train polygons, {n_test} test polygons "
          f"({cfg.analysis.holdout_frac:.0%} target, per-class stratified)")

    for gdf in gdfs:
        gdf["packed"] = gdf["class_id"].astype(np.int64) * FEATURE_ID_SCALE + gdf["feature_id"]
    sindexes = [gdf.sindex for gdf in gdfs]

    cls_profile = dict(grid.profile, count=1, dtype="int16", nodata=0, compress="lzw",
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")
    feat_profile = dict(cls_profile, dtype="int32")

    burned = np.zeros(len(layers), dtype=np.int64)      # cells a layer's polygons cover
    survived = np.zeros(len(layers), dtype=np.int64)    # cells that layer actually owns in the merge
    class_counts = np.zeros(max_class + 1, dtype=np.int64)

    bbox = cfg.processing.test_bbox
    with rasterio.open(out_path, "w", **cls_profile) as dst, \
         rasterio.open(feat_path, "w", **feat_profile) as fdst:
        for win in iter_block_windows(grid, cfg.processing.block_size, bbox):
            h, w = int(win.height), int(win.width)
            wt = window_transform(win, grid.transform)
            wb = window_bounds(win, grid.transform)

            merged = np.zeros((h, w), dtype=np.int64)
            owner = np.full((h, w), -1, dtype=np.int8)

            for li, (gdf, sidx) in enumerate(zip(gdfs, sindexes)):
                cand = list(sidx.intersection(wb))
                if not cand:
                    continue
                sub = gdf.iloc[cand]
                block = features.rasterize(
                    ((geom, pid) for geom, pid in zip(sub.geometry, sub["packed"])),
                    out_shape=(h, w),
                    transform=wt,
                    fill=0,
                    all_touched=cfg.all_touched,
                    dtype="int64",
                )
                covered = block > 0
                burned[li] += int(covered.sum())
                take = covered & (owner < 0)       # only where no higher-priority layer already claimed
                merged[take] = block[take]
                owner[take] = li

            for li in range(len(layers)):
                survived[li] += int((owner == li).sum())

            class_block = (merged // FEATURE_ID_SCALE).astype(np.int16)
            feat_block = (merged % FEATURE_ID_SCALE).astype(np.int32)
            class_counts += np.bincount(class_block.ravel(), minlength=max_class + 1)

            dst.write(class_block, 1, window=win)
            fdst.write(feat_block, 1, window=win)

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
    print(f"[rasterize] wrote {out_path} + {feat_path}")
    print(f"[rasterize] {total:,} habitat cells ({total * cell_ha:,.0f} ha); report -> {report_path}")
