"""Step 4 - separability statistics.

Quantifies the headline question: are the habitat classes actually distinguishable
by the 8 PCs? Outputs to ``output/04_stats/``:
  per_pc_tests.csv       ANOVA + Kruskal-Wallis + effect sizes per PC
  classifier_report.txt  RandomForest cross-validated class prediction
  confusion_matrix.png   row-normalised confusion matrix
  pc_importance.csv      permutation importance per PC
  summary.md             plain-language verdict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .analysis import Prepared, class_label, prepare
from .config import Config
from .ioretry import save_fig, to_csv, write_text

MAX_TRAIN = 120_000


def _per_pc_tests(prep: Prepared) -> pd.DataFrame:
    rows = []
    kept = prep.df[prep.df["class_id"].isin(prep.kept_classes)]
    for feat, zc in zip(prep.features, prep.zcols):
        groups = [g[zc].to_numpy() for _, g in kept.groupby("class_id")]
        groups = [g[np.isfinite(g)] for g in groups]
        groups = [g for g in groups if g.size > 1]
        if len(groups) < 2:
            continue
        f, p_anova = sps.f_oneway(*groups)
        allv = np.concatenate(groups)
        gm = allv.mean()
        ss_between = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
        ss_total = ((allv - gm) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan

        h, p_kw = sps.kruskal(*groups)
        n = allv.size
        k = len(groups)
        eps2 = (h - k + 1) / (n - k) if n > k else np.nan

        rows.append({
            "PC": feat,
            "anova_F": f, "anova_p": p_anova, "eta2": eta2,
            "kruskal_H": h, "kruskal_p": p_kw, "epsilon2": eps2,
        })
    return pd.DataFrame(rows).sort_values("eta2", ascending=False)


def _classifier(prep: Prepared, cfg: Config):
    d = prep.df[prep.df["class_id"].isin(prep.kept_classes)]
    X = d[prep.zcols].to_numpy()
    y = d["class_id"].to_numpy()
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]

    if len(y) > MAX_TRAIN:
        idx, _ = train_test_split(np.arange(len(y)), train_size=MAX_TRAIN,
                                  stratify=y, random_state=cfg.analysis.random_seed)
        X, y = X[idx], y[idx]

    # Bounded forest: unbounded depth on 100k+ rows produces a multi-GB model
    # that joblib then memmaps to /tmp (small tmpfs) for every worker.
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=25, min_samples_leaf=5, max_samples=0.5,
        n_jobs=4, class_weight="balanced", random_state=cfg.analysis.random_seed,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg.analysis.random_seed)

    tmp = cfg.out("04_stats", ".joblib_tmp")
    with joblib.parallel_config(backend="loky", n_jobs=4, temp_folder=str(tmp)):
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        bal_acc = balanced_accuracy_score(y, y_pred)
        n_classes = len(np.unique(y))
        chance = 1.0 / n_classes

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y,
                                              random_state=cfg.analysis.random_seed)
        clf.fit(Xtr, ytr)
        perm = permutation_importance(clf, Xte, yte, n_repeats=5,
                                      random_state=cfg.analysis.random_seed, n_jobs=4,
                                      scoring="balanced_accuracy")
    imp = (pd.DataFrame({"PC": prep.features,
                         "perm_importance": perm.importances_mean,
                         "perm_std": perm.importances_std})
           .sort_values("perm_importance", ascending=False))

    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
    return dict(bal_acc=bal_acc, chance=chance, n_classes=n_classes, n=len(y),
                cm=cm, cm_labels=labels, importance=imp, y=y, y_pred=y_pred)


def _plot_cm(res, prep: Prepared, cfg: Config) -> None:
    labels = [class_label(prep.lookup, c)[:34] for c in res["cm_labels"]]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), max(5, 0.4 * len(labels))))
    im = ax.imshow(res["cm"], cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Row-normalised confusion matrix\nbalanced accuracy {res['bal_acc']:.2f} "
                 f"(chance {res['chance']:.2f})")
    fig.colorbar(im, ax=ax, shrink=0.6)
    for ext in ("png", "svg"):
        save_fig(fig, cfg.dir_stats / f"confusion_matrix.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summary_md(tests: pd.DataFrame, res, prep: Prepared, cfg: Config) -> str:
    strong = tests[tests["eta2"] >= 0.06]["PC"].tolist()
    weak = tests[tests["eta2"] < 0.06]["PC"].tolist()
    lift = res["bal_acc"] / res["chance"]
    verdict = ("clearly separable" if lift >= 3 and res["bal_acc"] >= 0.4
               else "partially separable" if lift >= 1.8
               else "weakly separable")
    lines = [
        "# Geophysical separability of habitat types",
        "",
        f"- Habitat cells analysed: **{len(prep.df):,}**",
        f"- Classes with >= {cfg.analysis.min_pixels_per_class} cells: **{len(prep.kept_classes)}** "
        f"of {prep.lookup.shape[0]}",
        f"- Cross-validated RandomForest balanced accuracy: **{res['bal_acc']:.2f}** "
        f"vs chance {res['chance']:.2f}  (**{lift:.1f}x** chance)",
        f"- Verdict: habitat types are **{verdict}** in the 8-PC geophysical space.",
        "",
        "## Per-PC discrimination (one-way ANOVA effect size eta^2)",
        "",
        "| PC | eta^2 | ANOVA p | Kruskal epsilon^2 |",
        "|----|------:|--------:|------------------:|",
    ]
    for _, r in tests.iterrows():
        lines.append(f"| {r['PC']} | {r['eta2']:.3f} | {r['anova_p']:.1e} | {r['epsilon2']:.3f} |")
    lines += [
        "",
        f"Strongest discriminators (eta^2 >= 0.06): {', '.join(strong) or 'none'}.",
        f"Weak (eta^2 < 0.06): {', '.join(weak) or 'none'}.",
        "",
        "## Permutation importance (RandomForest, balanced accuracy drop)",
        "",
        "| PC | importance | std |",
        "|----|-----------:|----:|",
    ]
    for _, r in res["importance"].iterrows():
        lines.append(f"| {r['PC']} | {r['perm_importance']:.3f} | {r['perm_std']:.3f} |")
    lines += [
        "",
        "## Reading this",
        "",
        "eta^2 is the fraction of a PC's variance explained by habitat class "
        "(0.01 small, 0.06 medium, 0.14 large). A balanced accuracy well above "
        "chance means the habitat types really do sit in distinct geophysical "
        "envelopes, which is the premise the fuzzy distance-to-envelope map "
        "depends on. Classes that are heavily confused in the confusion matrix "
        "share a geophysical niche and will need habitat-specific priors or "
        "extra spheres (e.g. climate) to separate.",
        "",
    ]
    return "\n".join(lines)


def make_stats(cfg: Config, force: bool = False) -> None:
    out = cfg.dir_stats / "summary.md"
    if out.exists() and not force:
        print(f"[stats] exists, skipping: {out}")
        return

    prep = prepare(cfg)
    if len(prep.kept_classes) < 2:
        raise RuntimeError("need >= 2 classes above min_pixels_per_class for statistics")

    tests = _per_pc_tests(prep)
    to_csv(tests, cfg.dir_stats / "per_pc_tests.csv", index=False)
    print("[stats] per-PC tests done")

    res = _classifier(prep, cfg)
    to_csv(res["importance"], cfg.dir_stats / "pc_importance.csv", index=False)
    _plot_cm(res, prep, cfg)

    report = [
        f"RandomForest 5-fold CV on {res['n']:,} cells, {res['n_classes']} classes",
        f"balanced accuracy : {res['bal_acc']:.4f}",
        f"chance            : {res['chance']:.4f}",
        f"lift over chance  : {res['bal_acc'] / res['chance']:.2f}x",
        "",
        "permutation importance (balanced-accuracy drop):",
        res["importance"].to_string(index=False),
    ]
    write_text(cfg.dir_stats / "classifier_report.txt", "\n".join(report))

    write_text(out, _summary_md(tests, res, prep, cfg))
    print(f"[stats] balanced accuracy {res['bal_acc']:.2f} "
          f"({res['bal_acc'] / res['chance']:.1f}x chance) -> {out}")
