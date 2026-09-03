"""Feature standardisation shared by the EDA and the fuzzy projection.

The transform is: winsorise each raw PC to mean +/- ``winsor_z`` standard
deviations (limits learned from the habitat training cells), then z-score using
the mean/std of the winsorised values. Persisted so the full-Denmark projection
applies exactly the same transform as model fitting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Scaler:
    features: list[str]
    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str], winsor_z: float) -> "Scaler":
        lo, hi, mean, std = [], [], [], []
        for f in features:
            a = df[f].to_numpy(dtype=np.float64)
            a = a[np.isfinite(a)]
            m, s = a.mean(), a.std()
            s = s if (np.isfinite(s) and s != 0) else 1.0
            l, h = m - winsor_z * s, m + winsor_z * s
            w = np.clip(a, l, h)
            mw, sw = w.mean(), w.std()
            sw = sw if (np.isfinite(sw) and sw != 0) else 1.0
            lo.append(l); hi.append(h); mean.append(mw); std.append(sw)
        return cls(list(features), np.array(lo), np.array(hi), np.array(mean), np.array(std))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n, n_features) raw PC values -> standardised."""
        Xc = np.clip(X, self.lo, self.hi)
        return (Xc - self.mean) / self.std

    def to_json(self, path: Path) -> None:
        payload = {
            "features": self.features,
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Scaler":
        d = json.loads(Path(path).read_text())
        return cls(d["features"], np.array(d["lo"]), np.array(d["hi"]),
                   np.array(d["mean"]), np.array(d["std"]))
