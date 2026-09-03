"""Step 6 (fuzzy, part 1) - fit one Gaussian envelope per habitat type.

Each habitat class is modelled as a single multivariate Gaussian in the
standardised 8-PC space: a mean vector and a (shrunk) covariance. The
Mahalanobis distance of any Denmark cell to that Gaussian, mapped through the
chi-square survival function, becomes the fuzzy suitability for that habitat.

Outputs (``output/05_model/``):
  scaler.json        winsor + z-score parameters (shared with the projection)
  envelopes.npz      class_ids, means (C x 8), precisions (C x 8 x 8), df
  model_report.csv   per-class calibration: n, Mahalanobis d^2 quantiles,
                     fraction of training cells inside the chi-square 95% shell
                     (near 0.95 => the single-Gaussian assumption fits well)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from .config import Config
from .extract import load_training
from .ioretry import to_csv
from .lookup import load_lookup
from .scaler import Scaler


def _fit_covariance(Xz: np.ndarray, method: str):
    est = LedoitWolf() if method == "ledoit_wolf" else EmpiricalCovariance()
    est.fit(Xz)
    return est.location_, est.precision_


def build_envelopes(cfg: Config, force: bool = False) -> None:
    npz_path = cfg.dir_model / "envelopes.npz"
    if npz_path.exists() and not force:
        print(f"[model] exists, skipping: {npz_path}")
        return

    df = load_training(cfg)
    lut = load_lookup(cfg)
    feats = cfg.feature_names
    df_ok = df[np.isfinite(df[feats].to_numpy()).all(axis=1)]

    scaler = Scaler.fit(df_ok, feats, cfg.analysis.winsor_z)
    scaler.to_json(cfg.dir_model / "scaler.json")

    Z_all = scaler.transform(df_ok[feats].to_numpy(dtype=np.float64))
    class_ids = df_ok["class_id"].to_numpy()

    df_n = len(feats)
    chi2_95 = chi2.ppf(0.95, df_n)

    ids, means, precisions, rows = [], [], [], []
    id2meta = lut.set_index("class_id")[["naturkode", "navn", "hovednaturtype"]].to_dict("index")

    for cid in sorted(np.unique(class_ids)):
        Xz = Z_all[class_ids == cid]
        meta = id2meta.get(int(cid), {"naturkode": str(cid), "navn": "", "hovednaturtype": ""})
        if len(Xz) < cfg.fuzzy.min_cells_for_envelope:
            rows.append({"class_id": int(cid), **meta, "n_cells": len(Xz),
                         "envelope": 0, "d2_median": np.nan, "d2_p90": np.nan,
                         "frac_within_chi2_95": np.nan})
            continue

        mu, prec = _fit_covariance(Xz, cfg.fuzzy.covariance)
        diff = Xz - mu
        d2 = np.einsum("ni,ij,nj->n", diff, prec, diff)

        ids.append(int(cid))
        means.append(mu)
        precisions.append(prec)
        rows.append({
            "class_id": int(cid), **meta, "n_cells": len(Xz), "envelope": 1,
            "d2_median": float(np.median(d2)), "d2_p90": float(np.percentile(d2, 90)),
            "frac_within_chi2_95": float(np.mean(d2 <= chi2_95)),
        })

    np.savez_compressed(
        npz_path,
        class_ids=np.array(ids, dtype=np.int32),
        means=np.array(means, dtype=np.float64),
        precisions=np.array(precisions, dtype=np.float64),
        feature_names=np.array(feats),
        df=df_n,
    )
    report = pd.DataFrame(rows).sort_values("class_id")
    to_csv(report, cfg.dir_model / "model_report.csv", index=False)

    n_env = int(report["envelope"].sum())
    poor = report.loc[(report["envelope"] == 1) & (report["frac_within_chi2_95"] < 0.8),
                      "naturkode"].tolist()
    print(f"[model] {n_env} envelopes fitted ({len(report) - n_env} classes skipped, too few cells)")
    if poor:
        print(f"[model] non-Gaussian niches (chi2-95 coverage < 0.8, treat suitability with care): {poor}")
    print(f"[model] -> {npz_path}")
