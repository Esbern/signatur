"""Step - 80/20 polygon-level holdout validation.

Habitat polygons (not pixels) were split into train/test in `rasterize`
(stratified per class, seeded) - pixels within one polygon are spatially
autocorrelated (near-identical geophysics), so a pixel-level split would leak
information between train and test and inflate every accuracy number reported
so far in this project.

This step fits the envelope model AND a RandomForest on the 80% train split
only, then evaluates both purely on the 20% test split - polygons the model
never saw during fitting. It is the honest counterpart to `stats.py` (which
uses ordinary k-fold CV, and pixels from one polygon can land in different
folds) and to the informal "suitability at occupied cells" check.

Outputs (``output/08_holdout/``):
  holdout_report.md          headline numbers, written for direct reading
  holdout_per_class.csv      per-class: n_test cells, top-1 accuracy, median
                             suitability of the true class on held-out cells
  holdout_confusion_matrix.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from .analysis import class_label
from .config import Config
from .extract import load_training
from .ioretry import save_fig, to_csv, write_text
from .lookup import load_lookup
from .model import build_envelopes
from .scaler import Scaler

RF_MAX_TRAIN = 120_000


def _chi2_sf(d2: np.ndarray, df: int) -> np.ndarray:
    if df % 2 == 0:
        t = d2 / 2.0
        term = np.ones_like(t)
        acc = np.ones_like(t)
        for i in range(1, df // 2):
            term *= t / i
            acc += term
        return np.exp(-t) * acc
    from scipy.stats import chi2
    return chi2.sf(d2, df)


def run_holdout(cfg: Config, force: bool = False) -> None:
    out_md = cfg.dir_holdout / "holdout_report.md"
    if out_md.exists() and not force:
        print(f"[holdout] exists, skipping: {out_md}")
        return

    df = load_training(cfg)
    if "split" not in df.columns:
        raise RuntimeError("training_pixels.parquet has no 'split' column - re-run "
                           "`signatur rasterize --force` and `signatur extract --force`")
    lut = load_lookup(cfg)
    feats = cfg.feature_names
    ok = np.isfinite(df[feats].to_numpy()).all(1)
    df = df[ok]
    train = df[df["split"] == 0]
    test = df[df["split"] == 1]
    print(f"[holdout] {len(train):,} train cells / {len(test):,} test cells "
          f"({test['class_id'].nunique()} classes in test)")

    # ---- 1. envelope model, fit on train polygons only ----------------------
    build_envelopes(cfg, force=force, subset="train")
    npz = np.load(cfg.dir_model / "envelopes_train.npz")
    class_ids = npz["class_ids"].astype(int)
    means = npz["means"]
    precisions = npz["precisions"]
    df_n = int(npz["df"])
    scaler = Scaler.from_json(cfg.dir_model / "scaler_train.json")
    id2idx = {c: i for i, c in enumerate(class_ids)}

    test_env = test[test["class_id"].isin(id2idx)].copy()
    dropped = len(test) - len(test_env)
    X = scaler.transform(test_env[feats].to_numpy(dtype=np.float64))
    diff = X[:, None, :] - means[None, :, :]
    m = np.einsum("ncj,cjk->nck", diff, precisions)
    d2 = np.einsum("nck,nck->nc", m, diff)
    suit = _chi2_sf(d2, df_n)
    pred_idx = np.argmin(d2, axis=1)
    pred_class = class_ids[pred_idx]
    true_idx = test_env["class_id"].map(id2idx).to_numpy()
    true_class = test_env["class_id"].to_numpy()
    true_suit = suit[np.arange(len(test_env)), true_idx]

    env_top1 = float((pred_class == true_class).mean())
    env_bal_acc = balanced_accuracy_score(true_class, pred_class)

    work = test_env.assign(pred=pred_class, true_suit=true_suit,
                           hit=(pred_class == test_env["class_id"].to_numpy()))
    per_class = work.groupby("class_id").agg(
        n_test_cells=("hit", "size"),
        top1_acc=("hit", "mean"),
        median_suit_true_class=("true_suit", "median"),
    ).reset_index()
    per_class = lut.merge(per_class, on="class_id", how="right")
    to_csv(per_class, cfg.dir_holdout / "holdout_per_class.csv", index=False)

    labels = np.unique(np.concatenate([true_class, pred_class]))
    cm = confusion_matrix(true_class, pred_class, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), max(5, 0.4 * len(labels))))
    lbl_names = [class_label(lut, c)[:34] for c in labels]
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(lbl_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(lbl_names, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Holdout confusion matrix (envelope model, unseen polygons)\n"
                 f"top-1 {env_top1:.2f}  balanced acc {env_bal_acc:.2f}")
    fig.colorbar(im, ax=ax, shrink=0.6)
    save_fig(fig, cfg.dir_holdout / "holdout_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. RandomForest, fit on train polygons only, honest headline number --
    kept = train["class_id"].value_counts()
    kept = kept[kept >= cfg.analysis.min_pixels_per_class].index
    tr = train[train["class_id"].isin(kept)]
    te = test[test["class_id"].isin(kept)]
    Xtr = tr[feats].to_numpy(dtype=np.float32)
    ytr = tr["class_id"].to_numpy()
    if len(ytr) > RF_MAX_TRAIN:
        idx = np.random.default_rng(cfg.analysis.random_seed).choice(len(ytr), RF_MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    Xte = te[feats].to_numpy(dtype=np.float32)
    yte = te["class_id"].to_numpy()

    clf = RandomForestClassifier(n_estimators=150, max_depth=25, min_samples_leaf=5,
                                 max_samples=0.5, n_jobs=4, class_weight="balanced",
                                 random_state=cfg.analysis.random_seed)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    rf_bal_acc = balanced_accuracy_score(yte, yp)
    rf_top1 = float((yp == yte).mean())

    n_classes_test = test["class_id"].nunique()
    chance = 1.0 / n_classes_test
    lines = [
        "# Holdout validation (80/20 polygon split, unseen test polygons)",
        "",
        f"- Train polygons -> {len(train):,} cells; test polygons -> {len(test):,} cells "
        f"({n_classes_test} classes represented)",
        f"- Envelope model classes with too few train cells for a holdout test: {dropped:,} test cells "
        f"from those classes excluded",
        "",
        "## Envelope model (the one that drives the fuzzy maps)",
        "",
        f"- Top-1 accuracy on held-out polygons: **{env_top1:.3f}**",
        f"- Balanced accuracy: **{env_bal_acc:.3f}** (chance {chance:.3f}, "
        f"{env_bal_acc / chance:.1f}x)",
        f"- Median suitability of the TRUE class at held-out cells: "
        f"**{float(np.median(true_suit)):.3f}**",
        "",
        "## RandomForest diagnostic (separability check, trained on 80% only)",
        "",
        f"- Top-1 accuracy: **{rf_top1:.3f}**",
        f"- Balanced accuracy: **{rf_bal_acc:.3f}** (chance {chance:.3f}, {rf_bal_acc / chance:.1f}x)",
        "",
        "## Reading this",
        "",
        "These numbers use only cells whose whole POLYGON was held out of model "
        "fitting - no spatial leakage. Compare directly against the in-sample "
        "numbers in `04_stats/summary.md`: a much lower holdout accuracy would mean "
        "the earlier numbers were inflated by pixel-level autocorrelation; a similar "
        "number means the model generalises to genuinely new sites.",
        "",
    ]
    write_text(out_md, "\n".join(lines))
    print(f"[holdout] envelope top1={env_top1:.3f} bal_acc={env_bal_acc:.3f} | "
          f"RF top1={rf_top1:.3f} bal_acc={rf_bal_acc:.3f} -> {out_md}")
