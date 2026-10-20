"""Step 2 - build the compact training store.

Every habitat cell in Denmark (not a capped sample) is written to a single table
with its ``class_id`` and the 8 PC values. Because ``merged_class_id.tif`` shares
the PC grid exactly, this is a straight block-by-block read with no coordinate
lookup - the fast path from the reference project's ``project_cluster_distances``.

Outputs (``output/02_training/``):
  training_pixels.parquet   row, col, x, y, class_id, naturkode, feature_id, split
                            (split: 0 = train polygon, 1 = held-out test polygon)
                            + one column per PC
  training_pixels.npz       X (float32 N x 8), class_id, split, xy, feature_names
  training_meta.json        grid + feature order + provenance
  class_counts.csv          per-class cell counts, area, and train/test split
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as transform_xy

from .config import Config
from .grid import load_grid, check_same_grid, iter_block_windows
from .ioretry import read_band, savez, to_csv, to_parquet, write_text
from .lookup import load_lookup


def build_training_store(cfg: Config, force: bool = False) -> None:
    parquet_path = cfg.dir_training / "training_pixels.parquet"
    if parquet_path.exists() and not force:
        print(f"[extract] exists, skipping: {parquet_path}")
        return

    merged_path = cfg.dir_merged / "merged_class_id.tif"
    feat_path = cfg.dir_merged / "feature_id.tif"
    split_csv = cfg.dir_merged / "feature_split.csv"
    if not (merged_path.exists() and feat_path.exists() and split_csv.exists()):
        raise FileNotFoundError("merged raster / feature_id / feature_split missing - "
                                "run `signatur rasterize` first")

    grid = load_grid(cfg.reference_path)
    check_same_grid([merged_path, feat_path, *cfg.pc_paths], reference=cfg.reference_path)

    split_table = pd.read_csv(split_csv)
    fid_lut = np.full(int(split_table["feature_id"].max()) + 1, -1, dtype=np.int8)
    fid_lut[split_table["feature_id"].to_numpy()] = split_table["split"].to_numpy()

    feats = cfg.feature_names
    nodata = cfg.nodata
    bbox = cfg.processing.test_bbox

    chunks_meta: list[np.ndarray] = []   # (row, col) int32
    chunks_cls: list[np.ndarray] = []    # class_id int16
    chunks_fid: list[np.ndarray] = []    # feature_id int32
    chunks_X: list[np.ndarray] = []      # (n, 8) float32

    merged_src = rasterio.open(merged_path)
    feat_src = rasterio.open(feat_path)
    pc_srcs = [rasterio.open(p) for p in cfg.pc_paths]
    try:
        n_blocks = 0
        for win in iter_block_windows(grid, cfg.processing.block_size, bbox):
            cls = read_band(merged_src, 1, window=win)
            hab = cls > 0
            if not hab.any():
                continue

            fid = read_band(feat_src, 1, window=win)
            lr, lc = np.where(hab)
            stack = np.empty((lr.size, len(feats)), dtype=np.float32)
            valid = np.ones(lr.size, dtype=bool)
            for i, src in enumerate(pc_srcs):
                band = read_band(src, 1, window=win).astype(np.float32)
                vals = band[lr, lc]
                bad = ~np.isfinite(vals)
                if nodata is not None:
                    bad |= vals == np.float32(nodata)
                valid &= ~bad
                stack[:, i] = vals

            if not valid.any():
                continue

            gr = (lr[valid] + int(win.row_off)).astype(np.int32)
            gc = (lc[valid] + int(win.col_off)).astype(np.int32)
            chunks_meta.append(np.column_stack([gr, gc]))
            chunks_cls.append(cls[lr[valid], lc[valid]].astype(np.int16))
            chunks_fid.append(fid[lr[valid], lc[valid]].astype(np.int32))
            chunks_X.append(stack[valid])
            n_blocks += 1
            if n_blocks % 200 == 0:
                got = sum(c.shape[0] for c in chunks_X)
                print(f"[extract]   {n_blocks} blocks, {got:,} cells so far")
    finally:
        merged_src.close()
        feat_src.close()
        for s in pc_srcs:
            s.close()

    if not chunks_X:
        raise RuntimeError("no valid habitat cells found - check inputs / test_bbox")

    rc = np.concatenate(chunks_meta)
    class_id = np.concatenate(chunks_cls)
    feature_id = np.concatenate(chunks_fid)
    X = np.concatenate(chunks_X)
    split = fid_lut[feature_id]
    unresolved = split < 0
    if unresolved.any():
        # a habitat pixel whose winning polygon has no split assignment should
        # not happen (every rasterised feature gets one) - fail loudly rather
        # than silently mixing an unvalidated cell into train.
        raise RuntimeError(f"{int(unresolved.sum())} habitat cells have a feature_id with no "
                           f"train/test split - re-run `signatur rasterize --force`")

    xs, ys = transform_xy(grid.transform, rc[:, 0], rc[:, 1], offset="center")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    lut = load_lookup(cfg)
    id2code = dict(zip(lut["class_id"], lut["naturkode"]))

    df = pd.DataFrame({
        "row": rc[:, 0],
        "col": rc[:, 1],
        "x": xs,
        "y": ys,
        "class_id": class_id,
        "naturkode": pd.Series(class_id).map(id2code).astype(str).values,
        "feature_id": feature_id,
        "split": split.astype(np.int8),
    })
    for i, name in enumerate(feats):
        df[name] = X[:, i]

    to_parquet(df, parquet_path, compression="snappy", index=False)
    savez(
        cfg.dir_training / "training_pixels.npz",
        X=X,
        class_id=class_id,
        split=split.astype(np.int8),
        xy=np.column_stack([xs, ys]).astype(np.float64),
        feature_names=np.array(feats),
    )

    meta = {
        "n_cells": int(X.shape[0]),
        "feature_names": feats,
        "nodata": nodata,
        "crs": str(grid.crs),
        "grid_shape": [grid.height, grid.width],
        "transform": list(grid.transform)[:6],
        "test_bbox": list(bbox) if bbox else None,
        "merged_raster": str(merged_path),
        "pc_rasters": {n: str(p) for n, p in zip(feats, cfg.pc_paths)},
    }
    write_text(cfg.dir_training / "training_meta.json", json.dumps(meta, indent=2))

    counts = (
        pd.Series(class_id).value_counts().rename_axis("class_id").reset_index(name="n_pixels")
    )
    train_counts = (pd.Series(class_id[split == 0]).value_counts()
                    .rename_axis("class_id").reset_index(name="n_pixels_train"))
    test_counts = (pd.Series(class_id[split == 1]).value_counts()
                   .rename_axis("class_id").reset_index(name="n_pixels_test"))
    counts = lut.merge(counts, on="class_id", how="left") \
                .merge(train_counts, on="class_id", how="left") \
                .merge(test_counts, on="class_id", how="left")
    for c in ("n_pixels", "n_pixels_train", "n_pixels_test"):
        counts[c] = counts[c].fillna(0).astype(int)
    counts["area_ha"] = (counts["n_pixels"] * 0.01).round(2)
    counts["below_min"] = counts["n_pixels"] < cfg.analysis.min_pixels_per_class
    to_csv(counts, cfg.dir_training / "class_counts.csv", index=False)

    print(f"[extract] {X.shape[0]:,} habitat cells x {len(feats)} PCs "
          f"({int((split == 0).sum()):,} train / {int((split == 1).sum()):,} test) -> {parquet_path}")
    n_small = int(counts["below_min"].sum())
    if n_small:
        print(f"[extract] {n_small} classes below min_pixels_per_class={cfg.analysis.min_pixels_per_class}")


def load_training(cfg: Config) -> pd.DataFrame:
    p = cfg.dir_training / "training_pixels.parquet"
    if not p.exists():
        raise FileNotFoundError("training store missing - run `signatur extract` first")
    return pd.read_parquet(p)
