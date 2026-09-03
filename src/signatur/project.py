"""Step 7 (fuzzy, part 2) - project the Gaussian envelopes across all of Denmark.

Streams the 8 PC rasters block by block (same grid, no coordinate lookup),
standardises with the fitted scaler, and for every land cell computes the
Mahalanobis distance to each habitat envelope. Suitability = chi-square survival
function of that distance (1 = centre of the occupied niche, ~0.5 = a typical
occupied cell, ~0 = far outside).

Outputs (``output/06_fuzzy/``):
  <NN>_<code>_suitability.tif   float32 0..1, one per habitat type (optional)
  best_class.tif                int16   class_id of the closest habitat envelope
  best_suitability.tif          float32 max suitability over all habitats
  best_distance.tif             float32 Mahalanobis distance to that closest envelope
"""
from __future__ import annotations

import re

import numpy as np
import rasterio

from .config import Config
from .grid import check_same_grid, iter_block_windows, load_grid
from .lookup import load_lookup

NODATA_F = -1.0


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_") or "x"


def _chi2_sf(d2: np.ndarray, df: int) -> np.ndarray:
    """Survival function of the chi-square distribution.

    Exact closed form for even ``df`` (always the case here: df = number of PCs),
    which is ~100x faster than ``scipy.stats.chi2.sf`` on large arrays.
    """
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


def project_fuzzy(cfg: Config, force: bool = False) -> None:
    best_class_path = cfg.dir_fuzzy / "best_class.tif"
    if best_class_path.exists() and not force:
        print(f"[project] exists, skipping: {best_class_path}")
        return

    npz = np.load(cfg.dir_model / "envelopes.npz", allow_pickle=True)
    class_ids = npz["class_ids"].astype(int)
    means = npz["means"].astype(np.float32)          # (C, F)
    precisions = npz["precisions"].astype(np.float32)  # (C, F, F)
    df_n = int(npz["df"])
    C = len(class_ids)
    if C == 0:
        raise RuntimeError("no envelopes in model - run `signatur model` first")

    from .scaler import Scaler
    scaler = Scaler.from_json(cfg.dir_model / "scaler.json")

    lut = load_lookup(cfg)
    id2code = dict(zip(lut["class_id"], lut["naturkode"]))

    grid = load_grid(cfg.reference_path)
    check_same_grid([*cfg.pc_paths], reference=cfg.reference_path)
    nodata = cfg.nodata
    bbox = cfg.processing.test_bbox

    suit_profile = grid.profile
    suit_profile.update(count=1, dtype="float32", nodata=NODATA_F,
                        compress=cfg.fuzzy.compression, tiled=True,
                        blockxsize=512, blockysize=512, BIGTIFF="YES")
    cls_profile = dict(suit_profile, dtype="int16", nodata=0)

    pc_srcs = [rasterio.open(p) for p in cfg.pc_paths]
    class_dsts: dict[int, rasterio.io.DatasetWriter] = {}
    if cfg.fuzzy.per_class_rasters:
        for cid in class_ids:
            name = f"{cid:02d}_{_slug(id2code.get(cid, cid))}_suitability.tif"
            class_dsts[cid] = rasterio.open(cfg.dir_fuzzy / name, "w", **suit_profile)

    best_c_dst = rasterio.open(best_class_path, "w", **cls_profile)
    best_s_dst = rasterio.open(cfg.dir_fuzzy / "best_suitability.tif", "w", **suit_profile)
    best_d_dst = rasterio.open(cfg.dir_fuzzy / "best_distance.tif", "w", **suit_profile)

    windows = list(iter_block_windows(grid, cfg.processing.block_size, bbox))
    batch = cfg.fuzzy.batch_px
    nd = None if nodata is None else np.float32(nodata)
    F = len(pc_srcs)
    n_land = 0
    try:
        for bi, win in enumerate(windows, 1):
            h, w = int(win.height), int(win.width)

            # Read one band first; most of the grid bbox is sea/abroad (nodata),
            # so skip the other 7 reads entirely where there is no land.
            b0 = pc_srcs[0].read(1, window=win)
            valid = np.isfinite(b0)
            if nd is not None:
                valid &= b0 != nd
            flat_valid = valid.ravel()
            nv = int(flat_valid.sum())

            cls_blk = np.zeros(h * w, dtype=np.int16)
            bs_blk = np.full(h * w, NODATA_F, dtype=np.float32)
            bd_blk = np.full(h * w, NODATA_F, dtype=np.float32)
            suit_cols = np.full((h * w, C), NODATA_F, dtype=np.float32) if class_dsts else None

            if nv:
                n_land += nv
                Xr = np.empty((nv, F), dtype=np.float32)
                Xr[:, 0] = b0.ravel()[flat_valid]
                for i in range(1, F):
                    band = pc_srcs[i].read(1, window=win).ravel()[flat_valid]
                    ok = np.isfinite(band) & (band != nd if nd is not None else True)
                    band[~ok] = np.nan
                    Xr[:, i] = band
                row_ok = np.isfinite(Xr).all(1)
                idx_valid = np.flatnonzero(flat_valid)

                Xz = scaler.transform(Xr[row_ok]).astype(np.float32)
                nvz = Xz.shape[0]
                pos = idx_valid[row_ok]
                best_s = np.zeros(nvz, dtype=np.float32)
                best_i = np.zeros(nvz, dtype=np.int64)
                best_d2 = np.zeros(nvz, dtype=np.float32)
                for s0 in range(0, nvz, batch):
                    sl = slice(s0, min(s0 + batch, nvz))
                    diff = Xz[sl][:, None, :] - means[None, :, :]        # (nb, C, F)
                    m = np.einsum("ncj,cjk->nck", diff, precisions, optimize=True)
                    d2 = np.einsum("nck,nck->nc", m, diff, optimize=True)  # (nb, C)
                    suit = _chi2_sf(d2, df_n).astype(np.float32)         # (nb, C)
                    bidx = np.argmax(suit, axis=1)
                    rng = np.arange(suit.shape[0])
                    best_s[sl] = suit[rng, bidx]
                    best_i[sl] = bidx
                    best_d2[sl] = d2[rng, bidx]
                    if suit_cols is not None:
                        suit_cols[pos[sl]] = suit

                cls_blk[pos] = class_ids[best_i].astype(np.int16)
                bs_blk[pos] = best_s
                bd_blk[pos] = np.sqrt(best_d2)

            best_c_dst.write(cls_blk.reshape(h, w), 1, window=win)
            best_s_dst.write(bs_blk.reshape(h, w), 1, window=win)
            best_d_dst.write(bd_blk.reshape(h, w), 1, window=win)
            if suit_cols is not None:
                for j, cid in enumerate(class_ids):
                    class_dsts[cid].write(suit_cols[:, j].reshape(h, w), 1, window=win)

            if bi % 10 == 0 or bi == len(windows):
                print(f"[project]   block {bi}/{len(windows)}  ({n_land:,} land cells)", flush=True)
    finally:
        for s in pc_srcs:
            s.close()
        for d in (*class_dsts.values(), best_c_dst, best_s_dst, best_d_dst):
            d.close()

    print(f"[project] wrote best_class / best_suitability / best_distance"
          + (f" + {len(class_dsts)} per-class rasters" if class_dsts else "")
          + f" -> {cfg.dir_fuzzy}")
