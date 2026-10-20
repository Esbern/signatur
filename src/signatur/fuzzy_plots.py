"""Diagnostics for the fuzzy projection.

Samples the projected ``best_*`` rasters at the habitat training cells and plots
the distribution per habitat type. ``best_suitability`` is the "how close is this
pixel to supporting *some* natural habitat" layer (max suitability over all 44
envelopes) - the single number that matters for delineating areas worth
including in a new protected area, regardless of which habitat type they would
become.

Outputs (``output/06_fuzzy/``):
  best_by_habitat.png / .svg   violin of best_suitability (and best_distance) per type
  best_by_habitat.csv          per-type quantile summary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .analysis import class_label
from .config import Config
from .extract import load_training
from .ioretry import save_fig, to_csv
from .lookup import load_lookup


def _sample_full(path, rows, cols) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1)
        nd = src.nodata
    v = arr[rows, cols].astype(np.float64)
    if nd is not None:
        v[v == nd] = np.nan
    del arr
    return v


def make_fuzzy_plots(cfg: Config, force: bool = False) -> None:
    out_png = cfg.dir_fuzzy / "best_by_habitat.png"
    if out_png.exists() and not force:
        print(f"[fuzzyplots] exists, skipping: {out_png}")
        return

    best_s = cfg.dir_fuzzy / "best_suitability.tif"
    best_d = cfg.dir_fuzzy / "best_distance.tif"
    if not best_s.exists():
        raise FileNotFoundError("run `signatur project` first")

    tr = load_training(cfg)[["row", "col", "class_id", "naturkode"]]
    lut = load_lookup(cfg)
    tr = tr.merge(lut[["class_id", "navn", "hovednaturtype"]], on="class_id", how="left")

    rows = tr["row"].to_numpy()
    cols = tr["col"].to_numpy()
    tr["best_suitability"] = _sample_full(best_s, rows, cols)
    tr["best_distance"] = _sample_full(best_d, rows, cols)
    tr = tr.dropna(subset=["best_suitability"])

    order = (lut[lut["class_id"].isin(tr["class_id"].unique())]
             .sort_values(["hovednaturtype", "naturkode"]))
    cids = order["class_id"].tolist()
    labels = [class_label(lut, c)[:36] for c in cids]

    htypes = sorted(order["hovednaturtype"].dropna().unique())
    import seaborn as sns
    pal = {h: c for h, c in zip(htypes, sns.color_palette("tab10", max(len(htypes), 3)).as_hex())}
    hmap = dict(zip(lut["class_id"], lut["hovednaturtype"]))
    colors = [pal.get(hmap.get(c), "#888") for c in cids]

    rng = np.random.default_rng(cfg.analysis.random_seed)
    cap = 40000

    def series(col):
        out = []
        for c in cids:
            v = tr.loc[tr["class_id"] == c, col].to_numpy()
            v = v[np.isfinite(v)]
            if v.size > cap:
                v = v[rng.choice(v.size, cap, replace=False)]
            out.append(v if v.size else np.zeros(1))
        return out

    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.34 * len(cids)), 12))
    for ax, col, ylab, ylim in [
        (axes[0], "best_suitability", "best_suitability\n(max of 44 envelopes)", (0, 1)),
        (axes[1], "best_distance", "best_distance\n(Mahalanobis)", (0, 6)),
    ]:
        parts = ax.violinplot(series(col), showmedians=True, widths=0.85)
        for pc, cc in zip(parts["bodies"], colors):
            pc.set_facecolor(cc)
            pc.set_alpha(0.75)
        if col == "best_suitability":
            ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(1, len(cids) + 1))
        ax.set_xticklabels(labels if ax is axes[1] else [], rotation=90, fontsize=7)
    axes[0].set_title("Projected suitability sampled inside each habitat's training polygons")
    axes[0].legend([Rectangle((0, 0), 1, 1, color=pal[h]) for h in htypes], htypes,
                   fontsize=7, ncol=min(len(htypes), 7), loc="lower center")
    save_fig(fig, out_png, dpi=150, bbox_inches="tight")
    save_fig(fig, out_png.with_suffix(".svg"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summ = (tr.groupby(["class_id", "naturkode", "navn", "hovednaturtype"])["best_suitability"]
            .agg(n="count", p10=lambda s: s.quantile(.1), median="median",
                 p90=lambda s: s.quantile(.9), frac_gt_0_5=lambda s: (s > 0.5).mean())
            .reset_index().sort_values("median"))
    to_csv(summ, cfg.dir_fuzzy / "best_by_habitat.csv", index=False)
    print(f"[fuzzyplots] {out_png}")
    print(f"[fuzzyplots] median best_suitability over all habitat cells: "
          f"{tr['best_suitability'].median():.3f}")
