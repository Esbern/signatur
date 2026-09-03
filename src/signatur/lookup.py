"""Build the ``naturkode`` -> ``class_id`` lookup table.

``naturkode`` holds Natura 2000 habitat-type codes. Most are 4-digit but some are
alphanumeric (``91D0``, ``91E0``), so they cannot be burned into a raster
directly. This step assigns each distinct code a small integer ``class_id``
(1..N; 0 means "no habitat") and joins human-readable Danish names from the
editable seed table ``data/naturkode_lookup_seed.csv``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import Config
from .ioretry import to_csv


def _read_codes(path: Path, field: str) -> pd.Series:
    s = pq.read_table(path, columns=[field]).column(field).to_pandas()
    # normalise to a clean string ("9180.0" -> "9180", " 6230 " -> "6230")
    s = s.dropna()
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
        s = s.astype("Int64").astype(str)
    else:
        s = s.astype(str).str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
    return s[s != ""]


def build_lookup(cfg: Config, force: bool = False) -> pd.DataFrame:
    out_path = cfg.dir_lookup / "naturkode_lookup.csv"
    if out_path.exists() and not force:
        print(f"[lookup] exists, skipping: {out_path}")
        return pd.read_csv(out_path, dtype={"naturkode": str})

    seed = pd.read_csv(cfg.lookup_seed, dtype={"naturkode": str})
    seed["naturkode"] = seed["naturkode"].str.strip()

    per_layer_counts: dict[str, pd.Series] = {}
    all_codes: set[str] = set()
    for layer in cfg.habitat_layers:
        codes = _read_codes(cfg.layer_path(layer), cfg.naturkode_field)
        per_layer_counts[layer.name] = codes.value_counts()
        all_codes.update(codes.unique())
        print(f"[lookup] {layer.name}: {len(codes):>7,} features, {codes.nunique()} codes")

    rows = []
    for code in sorted(all_codes):
        seed_row = seed.loc[seed["naturkode"] == code]
        rec: dict[str, object] = {
            "naturkode": code,
            "navn": seed_row["navn"].iloc[0] if len(seed_row) else "",
            "hovednaturtype": seed_row["hovednaturtype"].iloc[0] if len(seed_row) else "Ukendt",
            "prioriteret": int(seed_row["prioriteret"].iloc[0]) if len(seed_row) else 0,
        }
        total = 0
        for layer in cfg.habitat_layers:
            n = int(per_layer_counts[layer.name].get(code, 0))
            rec[f"n_{layer.name}"] = n
            total += n
        rec["n_features_total"] = total
        rows.append(rec)

    lut = pd.DataFrame(rows).sort_values("naturkode").reset_index(drop=True)
    lut.insert(0, "class_id", range(1, len(lut) + 1))

    missing = lut.loc[lut["navn"] == "", "naturkode"].tolist()
    if missing:
        print(f"[lookup] WARNING: no seed entry for codes {missing} - add them to {cfg.lookup_seed.name}")

    to_csv(lut, out_path, index=False)
    print(f"[lookup] wrote {len(lut)} classes -> {out_path}")
    return lut


def load_lookup(cfg: Config) -> pd.DataFrame:
    p = cfg.dir_lookup / "naturkode_lookup.csv"
    if not p.exists():
        raise FileNotFoundError(f"lookup not built yet - run `signatur lookup` ({p})")
    return pd.read_csv(p, dtype={"naturkode": str})


def code_to_class_id(cfg: Config) -> dict[str, int]:
    lut = load_lookup(cfg)
    return dict(zip(lut["naturkode"], lut["class_id"]))
