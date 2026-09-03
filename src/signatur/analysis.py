"""Shared preparation for the plotting and statistics steps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .extract import load_training
from .lookup import load_lookup


@dataclass
class Prepared:
    df: pd.DataFrame            # per-cell: class_id, naturkode, navn, hovednaturtype, raw + z-scored PCs
    features: list[str]         # PC feature names
    zcols: list[str]            # matching "<feat>_z" column names
    lookup: pd.DataFrame        # full lookup table
    kept_classes: list[int]     # class_ids with >= min_pixels_per_class
    class_order: list[int]      # kept class_ids sorted by hovednaturtype then naturkode
    palette: dict[str, str]     # hovednaturtype -> hex colour


def _winsorize(a: np.ndarray, z: float) -> np.ndarray:
    mean, std = np.nanmean(a), np.nanstd(a)
    if not np.isfinite(std) or std == 0:
        return a
    return np.clip(a, mean - z * std, mean + z * std)


def prepare(cfg: Config) -> Prepared:
    df = load_training(cfg)
    lut = load_lookup(cfg)
    feats = cfg.feature_names

    df = df.merge(lut[["class_id", "naturkode", "navn", "hovednaturtype", "prioriteret"]],
                  on="class_id", how="left", suffixes=("", "_lut"))
    if "naturkode_lut" in df.columns:
        df = df.drop(columns=["naturkode_lut"])

    z = cfg.analysis.winsor_z
    zcols = []
    for f in feats:
        w = _winsorize(df[f].to_numpy(dtype=np.float64), z)
        mean, std = np.nanmean(w), np.nanstd(w)
        std = std if (np.isfinite(std) and std != 0) else 1.0
        col = f"{f}_z"
        df[col] = (w - mean) / std
        zcols.append(col)

    counts = df["class_id"].value_counts()
    kept = sorted(c for c in counts.index if counts[c] >= cfg.analysis.min_pixels_per_class)

    order_df = (lut[lut["class_id"].isin(kept)]
                .sort_values(["hovednaturtype", "naturkode"]))
    class_order = order_df["class_id"].tolist()

    htypes = sorted(lut["hovednaturtype"].dropna().unique())
    try:
        import seaborn as sns
        colors = sns.color_palette("tab10", max(len(htypes), 3)).as_hex()
    except Exception:  # pragma: no cover
        colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
                  "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]
    palette = {h: colors[i % len(colors)] for i, h in enumerate(htypes)}

    return Prepared(df=df, features=feats, zcols=zcols, lookup=lut,
                    kept_classes=kept, class_order=class_order, palette=palette)


def class_label(lut: pd.DataFrame, class_id: int) -> str:
    row = lut.loc[lut["class_id"] == class_id]
    if not len(row):
        return str(class_id)
    navn = row["navn"].iloc[0]
    code = row["naturkode"].iloc[0]
    return f"{code} {navn}" if isinstance(navn, str) and navn else str(code)
