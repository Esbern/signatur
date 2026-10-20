"""Step 9 - residual / "outside the model" analysis.

For the ministry write-up:
  - points 7/8 (the "fejl/residual kort"): ``best_suitability.tif`` /
    ``best_distance.tif`` (written by `project`) already ARE the residual map -
    1 = as typical as the average occupied cell of the closest-matching
    habitat, ~0 = resembles none of the 44 surveyed habitat types at all. This
    step adds a categorical, confidence-interval-based reading of that map.
  - point 9: bins the residual into chi-square confidence bands, then clusters
    and profiles (spider charts + per-PC histograms) the cells that fall
    outside EVERY habitat's envelope - candidate "dark diversity" ground the
    model can't explain. Mirrors the clustering + spider-chart method in
    ``~/dev/Groen_trepart/src/plot_cluster_spider.py``.

Categories (from best_suitability = max_c chi2.sf(mahalanobis_d2_c, df)):
  1 core            suit >= 0.5             - as typical as an average occupied cell
  2 marginal        alpha_outside <= suit < 0.5   - resembles some habitat, atypically
  3 outside 95% CI  alpha_strong <= suit < alpha_outside
  4 outside 99% CI  suit < alpha_strong

Outputs (``output/09_residual/``):
  residual_category.tif    categorical raster, all Denmark, nodata 0
  residual_category.qml    QGIS style
  residual_summary.csv     cells/ha per category
  outside_sample.parquet   sampled cells (category >= 3): raw PCs + cluster_id
  cluster_centers.csv      k-means centers in standardised PC space
  cluster_profile.csv      per cluster: size, nearest habitat envelope, mean PCs
  cluster_spider_*.png     one radar chart per cluster + an overlay
  pc_histograms.png        per-PC distribution, split by cluster
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .config import Config
from .grid import check_same_grid, iter_block_windows, load_grid
from .ioretry import read_band, save_fig, to_csv, to_parquet, write_text
from .lookup import load_lookup
from .project import _chi2_sf
from .scaler import Scaler

CATEGORY_LABELS = {
    1: "core (suit >= 0.5)",
    2: "marginal",
    3: "outside 95% CI",
    4: "outside 99% CI",
}
CATEGORY_COLORS = {1: "#1a9850", 2: "#fee08b", 3: "#fc8d59", 4: "#a50026"}


def _categorize(suit: np.ndarray, alpha_outside: float, alpha_strong: float) -> np.ndarray:
    cat = np.zeros(suit.shape, dtype=np.uint8)
    valid = suit >= 0  # best_suitability nodata is -1
    cat[valid & (suit >= 0.5)] = 1
    cat[valid & (suit >= alpha_outside) & (suit < 0.5)] = 2
    cat[valid & (suit >= alpha_strong) & (suit < alpha_outside)] = 3
    cat[valid & (suit < alpha_strong)] = 4
    return cat


def _write_qml(path, entries: list[tuple[int, str, str]]) -> None:
    import xml.sax.saxutils as sx
    rows = "\n".join(
        f'        <paletteEntry label="{sx.escape(lbl)}" value="{v}" alpha="255" color="{c}"/>'
        for v, c, lbl in entries
    )
    qml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <pipe>
    <rasterrenderer alphaBand="-1" nodataColor="" type="paletted" band="1" opacity="1">
      <colorPalette>
{rows}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
'''
    write_text(path, qml)


def build_residual_category(cfg: Config, force: bool = False) -> pd.DataFrame:
    out_path = cfg.dir_residual / "residual_category.tif"
    summary_path = cfg.dir_residual / "residual_summary.csv"
    if out_path.exists() and summary_path.exists() and not force:
        print(f"[residual] exists, skipping: {out_path}")
        return pd.read_csv(summary_path)

    suit_path = cfg.dir_fuzzy / "best_suitability.tif"
    if not suit_path.exists():
        raise FileNotFoundError("best_suitability.tif missing - run `signatur project` first")

    grid = load_grid(cfg.reference_path)
    profile = dict(grid.profile, count=1, dtype="uint8", nodata=0, compress="DEFLATE",
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF="YES")

    counts = np.zeros(5, dtype=np.int64)
    bbox = cfg.processing.test_bbox
    with rasterio.open(suit_path) as src, rasterio.open(out_path, "w", **profile) as dst:
        for win in iter_block_windows(grid, cfg.processing.block_size, bbox):
            suit = read_band(src, 1, window=win)
            cat = _categorize(suit, cfg.residual.alpha_outside, cfg.residual.alpha_strong)
            counts += np.bincount(cat.ravel(), minlength=5)
            dst.write(cat, 1, window=win)

    _write_qml(cfg.dir_residual / "residual_category.qml",
              [(c, CATEGORY_COLORS[c], f"{c}: {CATEGORY_LABELS[c]}") for c in (1, 2, 3, 4)])

    cell_ha = 0.01
    rows = [{"category": c, "label": CATEGORY_LABELS.get(c, "no data / sea"),
             "n_cells": int(counts[c]), "area_ha": round(counts[c] * cell_ha, 1)}
            for c in range(5)]
    summary = pd.DataFrame(rows)
    to_csv(summary, summary_path, index=False)
    print(f"[residual] wrote {out_path}")
    for r in rows[1:]:
        print(f"[residual]   cat {r['category']} {r['label']}: {r['n_cells']:,} cells ({r['area_ha']:,.0f} ha)")
    return summary


def _evaluate_k(Z: np.ndarray, k_min: int, k_max: int, seed: int) -> tuple[int, MiniBatchKMeans]:
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(Z), size=min(25_000, len(Z)), replace=False)
    best_k, best_score, best_model = k_min, -1.0, None
    for k in range(k_min, k_max + 1):
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto", batch_size=4096)
        model.fit(Z)
        labels = model.predict(Z[eval_idx])
        score = silhouette_score(Z[eval_idx], labels) if len(np.unique(labels)) > 1 else -1
        print(f"[residual]   k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, model
    return best_k, best_model


def profile_residual(cfg: Config, force: bool = False) -> None:
    out_csv = cfg.dir_residual / "cluster_profile.csv"
    if out_csv.exists() and not force:
        print(f"[residual] exists, skipping: {out_csv}")
        return

    summary = build_residual_category(cfg, force=False)
    n_outside = int(summary.loc[summary["category"].isin([3, 4]), "n_cells"].sum())
    if n_outside == 0:
        raise RuntimeError("no cells fall outside every habitat envelope - nothing to profile")
    p_keep = min(1.0, cfg.residual.cluster_sample_cap / n_outside)
    print(f"[residual] {n_outside:,} outside-model cells; sampling at p={p_keep:.4f}")

    cat_path = cfg.dir_residual / "residual_category.tif"
    grid = load_grid(cfg.reference_path)
    check_same_grid([cat_path, *cfg.pc_paths], reference=cfg.reference_path)
    feats = cfg.feature_names
    nodata = cfg.nodata
    rng = np.random.default_rng(cfg.analysis.random_seed)
    bbox = cfg.processing.test_bbox

    cat_src = rasterio.open(cat_path)
    pc_srcs = [rasterio.open(p) for p in cfg.pc_paths]
    chunks_xy, chunks_X = [], []
    try:
        n_blocks = 0
        for win in iter_block_windows(grid, cfg.processing.block_size, bbox):
            cat = read_band(cat_src, 1, window=win)
            mask = cat >= 3
            if not mask.any():
                continue
            lr, lc = np.where(mask)
            keep = rng.random(lr.size) < p_keep
            if not keep.any():
                continue
            lr, lc = lr[keep], lc[keep]

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
            chunks_xy.append(np.column_stack([gr, gc]))
            chunks_X.append(stack[valid])
            n_blocks += 1
            if n_blocks % 200 == 0:
                print(f"[residual]   {n_blocks} blocks, {sum(c.shape[0] for c in chunks_X):,} sampled cells")
    finally:
        cat_src.close()
        for s in pc_srcs:
            s.close()

    rc = np.concatenate(chunks_xy)
    X = np.concatenate(chunks_X)
    print(f"[residual] collected {len(X):,} sample cells")

    scaler = Scaler.from_json(cfg.dir_model / "scaler.json")
    Z = scaler.transform(X.astype(np.float64))

    k, model = _evaluate_k(Z, cfg.residual.k_min, cfg.residual.k_max, cfg.analysis.random_seed)
    labels = model.predict(Z)
    centers = model.cluster_centers_
    print(f"[residual] selected k={k}")

    df = pd.DataFrame({"row": rc[:, 0], "col": rc[:, 1], "cluster_id": labels})
    for i, f in enumerate(feats):
        df[f] = X[:, i]
        df[f"{f}_z"] = Z[:, i]
    to_parquet(df, cfg.dir_residual / "outside_sample.parquet", compression="snappy", index=False)

    centers_df = pd.DataFrame(centers, columns=feats)
    centers_df.insert(0, "cluster_id", range(k))
    to_csv(centers_df, cfg.dir_residual / "cluster_centers.csv", index=False)

    # nearest known habitat envelope for each cluster center, by Mahalanobis distance
    npz = np.load(cfg.dir_model / "envelopes.npz")
    class_ids = npz["class_ids"].astype(int)
    means, precisions, df_n = npz["means"], npz["precisions"], int(npz["df"])
    lut = load_lookup(cfg)
    id2label = {int(r.class_id): f"{r.naturkode} {r.navn}" for r in lut.itertuples()}

    diff = centers[:, None, :] - means[None, :, :]
    m = np.einsum("ncj,cjk->nck", diff, precisions)
    d2 = np.einsum("nck,nck->nc", m, diff)
    suit = _chi2_sf(d2, df_n)
    nearest_idx = np.argmin(d2, axis=1)

    rows = []
    for c in range(k):
        n_sampled = int((labels == c).sum())
        est_cells = n_sampled / p_keep
        rows.append({
            "cluster_id": c,
            "n_cells_sampled": n_sampled,
            "est_total_cells": int(round(est_cells)),
            "est_area_ha": round(est_cells * 0.01, 1),
            "nearest_habitat_class_id": int(class_ids[nearest_idx[c]]),
            "nearest_habitat": id2label.get(int(class_ids[nearest_idx[c]]), ""),
            "suitability_to_nearest_habitat": float(suit[c, nearest_idx[c]]),
            **{f: float(v) for f, v in zip(feats, centers[c])},
        })
    profile = pd.DataFrame(rows).sort_values("est_total_cells", ascending=False)
    to_csv(profile, out_csv, index=False)

    _plot_spiders(centers, feats, profile, cfg.dir_residual)
    _plot_histograms(df, feats, k, cfg.dir_residual)

    print(f"[residual] {k} clusters -> {out_csv}")
    print(profile[["cluster_id", "est_total_cells", "est_area_ha", "nearest_habitat",
                   "suitability_to_nearest_habitat"]].to_string(index=False))


def _plot_spiders(centers: np.ndarray, feats: list[str], profile: pd.DataFrame, out_dir) -> None:
    k = centers.shape[0]
    axis_range = float(np.ceil(np.abs(centers).max() * 2) / 2)
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 3)))

    def _draw(ax, vals, color, title):
        v = vals.tolist() + vals[:1].tolist()
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(feats, size=8)
        ax.set_ylim(-axis_range, axis_range)
        ticks = np.linspace(-axis_range, axis_range, 5)
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:.1f}" for t in ticks], size=6, color="grey")
        ax.plot(angles, [0] * len(angles), color="grey", lw=0.8, ls="--")
        ax.plot(angles, v, color=color, lw=2)
        ax.fill(angles, v, color=color, alpha=0.15)
        ax.set_title(title, size=10, pad=12)

    ncol = min(k, 4)
    nrow = (k + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, subplot_kw={"projection": "polar"}, figsize=(4.2 * ncol, 4.2 * nrow))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= k:
            ax.axis("off"); continue
        row = profile[profile.cluster_id == i].iloc[0]
        title = f"cluster {i} (~{row.est_area_ha:,.0f} ha)\nnearest: {row.nearest_habitat}"
        _draw(ax, centers[i], colors[i], title)
    fig.suptitle("Outside-the-model clusters - standardised PC profile (SD from Danish mean)")
    plt.tight_layout()
    save_fig(fig, out_dir / "cluster_spiders.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    for i in range(k):
        v = centers[i].tolist() + centers[i, :1].tolist()
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(feats, size=8)
        ax.set_ylim(-axis_range, axis_range)
        ax.plot(angles, v, color=colors[i], lw=2, label=f"cluster {i}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("All clusters overlaid")
    save_fig(fig, out_dir / "cluster_spiders_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[residual] {out_dir / 'cluster_spiders.png'}")


def _plot_histograms(df: pd.DataFrame, feats: list[str], k: int, out_dir) -> None:
    fig, axes = plt.subplots(len(feats), 1, figsize=(9, 2.6 * len(feats)))
    colors = plt.cm.tab10(np.linspace(0, 1, max(k, 3)))
    for ax, f in zip(np.atleast_1d(axes), feats):
        for c in range(k):
            v = df.loc[df.cluster_id == c, f"{f}_z"]
            ax.hist(v, bins=40, range=(-5, 5), histtype="step", color=colors[c], lw=1.5, label=f"cluster {c}")
        ax.set_title(f, fontsize=9)
        ax.set_xlim(-5, 5)
    axes[0].legend(fontsize=7, ncol=min(k, 5))
    fig.suptitle("Outside-the-model cells: per-PC distribution by cluster (z-scores)")
    plt.tight_layout()
    save_fig(fig, out_dir / "pc_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
