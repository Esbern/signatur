"""Step 3 - exploratory plots.

Headline question: do the habitat types occupy visibly different geophysical
conditions? Figures are written to ``output/03_plots/`` as PNG + SVG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage

from .analysis import Prepared, class_label, prepare
from .config import Config


def _save(fig, cfg: Config, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(cfg.dir_plots / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {name}")


def _subsample(prep: Prepared, cfg: Config) -> pd.DataFrame:
    cap = cfg.analysis.plot_sample_per_class
    rng = np.random.default_rng(cfg.analysis.random_seed)
    parts = []
    for cid, g in prep.df[prep.df["class_id"].isin(prep.kept_classes)].groupby("class_id"):
        parts.append(g.iloc[rng.choice(len(g), min(cap, len(g)), replace=False)] if len(g) > cap else g)
    return pd.concat(parts, ignore_index=True)


def plot_class_support(prep: Prepared, cfg: Config) -> None:
    counts = prep.df["class_id"].value_counts()
    lut = prep.lookup.copy()
    lut["n"] = lut["class_id"].map(counts).fillna(0).astype(int)
    lut = lut.sort_values(["hovednaturtype", "naturkode"])
    colors = [prep.palette.get(h, "#888888") for h in lut["hovednaturtype"]]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(lut))))
    y = np.arange(len(lut))
    ax.barh(y, lut["n"].clip(lower=0.5), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} {n}"[:42] for c, n in zip(lut["naturkode"], lut["navn"])], fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.axvline(cfg.analysis.min_pixels_per_class, color="red", ls=":", lw=1)
    ax.set_xlabel("habitat cells (10 m, log scale)")
    ax.set_title("Sample support per habitat class")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in prep.palette.values()]
    ax.legend(handles + [Line2D([], [], color="red", ls=":")],
              list(prep.palette) + [f"min for stats ({cfg.analysis.min_pixels_per_class})"],
              loc="lower right", fontsize=7)
    _save(fig, cfg, "class_support")


def plot_pc_distributions(prep: Prepared, cfg: Config) -> None:
    sub = _subsample(prep, cfg)
    order = prep.class_order
    labels = [class_label(prep.lookup, c)[:34] for c in order]
    hmap = dict(zip(prep.lookup["class_id"], prep.lookup["hovednaturtype"]))
    colors = [prep.palette.get(hmap.get(c), "#888888") for c in order]

    n = len(prep.features)
    fig, axes = plt.subplots(n, 1, figsize=(max(8, 0.32 * len(order)), 3.1 * n))
    for ax, feat in zip(np.atleast_1d(axes), prep.features):
        data = [sub.loc[sub["class_id"] == c, f"{feat}_z"].to_numpy() for c in order]
        data = [d[np.isfinite(d)] if np.isfinite(d).any() else np.zeros(1) for d in data]
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax.axhline(0, color="grey", ls=":", lw=1)
        ax.set_ylabel(f"{feat}\n(z)", fontsize=9)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_ylim(-4, 4)
        if ax is np.atleast_1d(axes)[-1]:
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
        else:
            ax.set_xticklabels([])
    fig.suptitle("Per-PC distribution by habitat class (winsorised z-scores)", y=0.995)
    _save(fig, cfg, "pc_distributions")


def plot_pc_ridgelines(prep: Prepared, cfg: Config) -> None:
    from scipy.stats import gaussian_kde

    sub = _subsample(prep, cfg)
    htypes = [h for h in prep.palette if (sub["hovednaturtype"] == h).any()]
    n = len(prep.features)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    xs = np.linspace(-4, 4, 200)
    for ax, feat in zip(axes, prep.features):
        for h in htypes:
            v = sub.loc[sub["hovednaturtype"] == h, f"{feat}_z"].to_numpy()
            v = v[np.isfinite(v)]
            if v.size > 5:
                d = gaussian_kde(v)(xs)
                ax.plot(xs, d, color=prep.palette[h], lw=2, label=h)
                ax.fill_between(xs, d, color=prep.palette[h], alpha=0.15)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("z-score")
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Geophysical conditions by main habitat group", y=0.999)
    _save(fig, cfg, "pc_ridgelines")


def _class_means(prep: Prepared) -> tuple[np.ndarray, list[int]]:
    rows, ids = [], []
    for c in prep.class_order:
        g = prep.df.loc[prep.df["class_id"] == c, prep.zcols]
        rows.append(g.mean().to_numpy())
        ids.append(c)
    return np.vstack(rows), ids


def plot_class_mean_heatmap(prep: Prepared, cfg: Config) -> None:
    M, ids = _class_means(prep)
    order = leaves_list(linkage(M, method="ward")) if len(ids) > 2 else np.arange(len(ids))
    M, ids = M[order], [ids[i] for i in order]
    labels = [class_label(prep.lookup, c)[:40] for c in ids]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(ids))))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(prep.features)))
    ax.set_xticklabels(prep.features, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Mean geophysical position per habitat class\n(z-scores, ward-clustered)")
    fig.colorbar(im, ax=ax, shrink=0.5, label="class mean (z)")
    _save(fig, cfg, "class_mean_heatmap")


def plot_sphere_biplots(prep: Prepared, cfg: Config) -> None:
    sub = _subsample(prep, cfg)
    spheres = [("hydro", "hydro_PC1_z", "hydro_PC2_z"),
               ("soil", "soil_PC1_z", "soil_PC2_z"),
               ("toposphere", "toposphere_PC1_z", "toposphere_PC2_z")]
    spheres = [s for s in spheres if s[1] in sub.columns and s[2] in sub.columns]
    fig, axes = plt.subplots(1, len(spheres), figsize=(6 * len(spheres), 5.5))
    for ax, (name, cx, cy) in zip(np.atleast_1d(axes), spheres):
        ax.hexbin(sub[cx], sub[cy], gridsize=45, cmap="Greys", bins="log", mincnt=1)
        for c in prep.class_order:
            g = sub.loc[sub["class_id"] == c, [cx, cy]].to_numpy()
            if len(g) < 10:
                continue
            mu = g.mean(0)
            cov = np.cov(g.T)
            vals, vecs = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
            h = prep.lookup.loc[prep.lookup["class_id"] == c, "hovednaturtype"].iloc[0]
            ax.add_patch(Ellipse(mu, 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]), angle=ang,
                                 fill=False, edgecolor=prep.palette.get(h, "#888"), lw=1.2, alpha=0.8))
        ax.set_xlabel(cx.replace("_z", " (z)"))
        ax.set_ylabel(cy.replace("_z", " (z)"))
        ax.set_title(f"{name} sphere")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in prep.palette.values()]
    fig.legend(handles, list(prep.palette), loc="lower center", ncol=len(prep.palette), fontsize=8)
    fig.suptitle("Habitat niches per sphere (1-SD ellipses over cell density)")
    _save(fig, cfg, "sphere_biplots")


def plot_global_ordination(prep: Prepared, cfg: Config) -> None:
    from sklearn.decomposition import PCA

    sub = _subsample(prep, cfg)
    X = sub[prep.zcols].to_numpy()
    ok = np.isfinite(X).all(1)
    X, sub = X[ok], sub.loc[ok]
    pcs = PCA(n_components=2, random_state=cfg.analysis.random_seed).fit(X)
    Y = pcs.transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    for h in prep.palette:
        m = (sub["hovednaturtype"] == h).to_numpy()
        if m.sum() < 5:
            continue
        ax.scatter(Y[m, 0], Y[m, 1], s=3, alpha=0.25, color=prep.palette[h], label=h)
    var = pcs.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 of 8 PCs ({var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 of 8 PCs ({var[1]:.0f}%)")
    ax.set_title("Global ordination of habitat cells in geophysical space")
    lg = ax.legend(fontsize=8, markerscale=3)
    for lh in lg.legend_handles:
        lh.set_alpha(1)
    _save(fig, cfg, "global_ordination")


def plot_pairwise_distance(prep: Prepared, cfg: Config) -> None:
    M, ids = _class_means(prep)
    if len(ids) < 3:
        print("[plots] pairwise_distance skipped (need >=3 classes)")
        return
    order = leaves_list(linkage(M, method="ward"))
    M, ids = M[order], [ids[i] for i in order]
    D = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(-1))
    labels = [class_label(prep.lookup, c)[:36] for c in ids]

    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(ids)), max(5, 0.3 * len(ids))))
    im = ax.imshow(D, cmap="magma_r")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title("Euclidean distance between class-mean geophysical vectors")
    fig.colorbar(im, ax=ax, shrink=0.6, label="distance (z units)")
    _save(fig, cfg, "pairwise_distance")


def make_all_plots(cfg: Config, force: bool = False) -> None:
    prep = prepare(cfg)
    print(f"[plots] {len(prep.kept_classes)} classes >= min support "
          f"(of {prep.lookup.shape[0]}); {len(prep.df):,} cells")
    plot_class_support(prep, cfg)
    plot_pc_distributions(prep, cfg)
    plot_pc_ridgelines(prep, cfg)
    plot_class_mean_heatmap(prep, cfg)
    plot_sphere_biplots(prep, cfg)
    plot_global_ordination(prep, cfg)
    plot_pairwise_distance(prep, cfg)
