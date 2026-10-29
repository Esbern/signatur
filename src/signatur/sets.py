"""Step 10 - plausible-class sets (conformal prediction) instead of one winner.

``best_class.tif`` is winner-takes-all: a cell where the runner-up is almost as
good as the winner looks exactly like a cell with one clear winner. This step
replaces "the best class" with "the set of classes the geophysics cannot rule
out", with a size that varies per cell (1 where one habitat clearly wins, many
where several are near-tied).

How the set is formed
  1. Posterior per class, equal priors, from the same Gaussian envelopes:
       log L_k = -0.5 * d2_k + 0.5 * log|precision_k|,   p(k|x) = softmax_k(log L_k)
     Unlike suitability (absolute: "is this cell typical of k?") the posterior is
     relative ("does this cell look more like k than like the other classes?").
  2. Per-class cutoff q_k, calibrated on the HELD-OUT polygons (split == 1)
     with envelopes fitted on the train polygons only (split conformal,
     Mondrian = one cutoff per true class). q_k is chosen so that, on held-out
     cells of class k, p(k|x) >= q_k for ``coverage`` (e.g. 90%) of them. Each
     polygon gets equal total weight, so large polygons don't dominate.
  3. Set(x) = {k : p(k|x) >= q_k}, always including the highest-posterior
     class. Coverage guarantee (per class): a new site of habitat k has its
     true class in the set with probability >= coverage.
  4. Member envelope (``sets.member_envelope``, default on): the posterior is
     relative, so a class can win in a cell outside its own niche simply
     because the others are further away. A class is therefore only allowed
     where the cell is inside its own envelope; a cell inside no envelope gets
     the empty set. The envelope radius is CALIBRATED on the held-out polygons
     too (the d2 containing ``sets.envelope_coverage`` = 99 % of the class's
     unseen sites), because the Gaussian chi2 radius is far too tight for new
     sites. Same rule, same radii, at the group level in ``groups.py``.

Honest coverage numbers: the held-out polygons are split in two halves; cutoffs
from half A are evaluated on half B and vice versa (cross-conformal). The final
cutoffs used for the map are fitted on all held-out polygons.

For comparison, the calibration-free likelihood-ratio rule is evaluated too:
include k if L_k / L_best >= 1 / lr_ratio (Royall: 8 = "fairly strong evidence").

Steps / outputs:
  setcalib -> ``output/08_holdout/``
    set_calibration.csv    class_id, naturkode, n_test_polygons, q (cutoff), source
    set_coverage.csv       per class: cross-validated coverage + mean set size,
                           conformal vs likelihood-ratio rule
    set_report.md          headline numbers, written for direct reading
  sets -> ``output/10_sets/``
    set_size.tif           int16   number of plausible classes (0 = inside no
                                   habitat's envelope; nodata -1)
    set_mask.tif           uint64  bit j set = class set_bits.csv[j] plausible
                                   (nodata 2**63)
    best_posterior.tif     float32 p(best|x) - how clearly the winner wins
    map_class.tif          int16   allowed class_id with the highest posterior (nodata 0)
    lr_set_size.tif        int16   set size under the likelihood-ratio rule
    set_bits.csv           bit -> class_id, naturkode, navn
    set_size_summary.csv   cells / ha per set size
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from scipy.special import logsumexp
from scipy.stats import chi2

from .config import Config
from .extract import load_training
from .grid import check_same_grid, iter_block_windows, load_grid
from .ioretry import read_band, to_csv, write_text
from .lookup import load_lookup
from .model import build_envelopes
from .scaler import Scaler

NODATA_F = -1.0
NODATA_MASK = np.uint64(1 << 63)
CALIB_BATCH = 200_000


# ---------------------------------------------------------------------------
# shared maths
# ---------------------------------------------------------------------------

def _load_envelopes(path):
    npz = np.load(path, allow_pickle=True)
    prec = npz["precisions"]
    half_logdet = 0.5 * np.linalg.slogdet(prec)[1]                # (C,)
    return (npz["class_ids"].astype(int), npz["means"], prec, half_logdet, int(npz["df"]))


def _d2(Xz: np.ndarray, means: np.ndarray, precisions: np.ndarray) -> np.ndarray:
    diff = Xz[:, None, :] - means[None, :, :]                       # (n, C, F)
    m = np.einsum("ncj,cjk->nck", diff, precisions, optimize=True)
    return np.einsum("nck,nck->nc", m, diff, optimize=True)         # (n, C)


def _log_posterior(d2: np.ndarray, half_logdet: np.ndarray) -> np.ndarray:
    """log p(k|x) with equal priors, (n, C)."""
    logl = -0.5 * d2 + half_logdet[None, :]
    return logl - logsumexp(logl, axis=1, keepdims=True)


def _radius(d2_true: np.ndarray, poly: np.ndarray, level: float) -> float:
    """d2 radius containing ``level`` of polygon-weighted held-out cells (finite-sample corrected)."""
    _, inv, cnt = np.unique(poly, return_inverse=True, return_counts=True)
    return _weighted_quantile(d2_true, 1.0 / cnt[inv], min(1.0, level * (1.0 + 1.0 / len(cnt))))


def _fit_radii(d2_true, true_idx, poly, n_classes, level, min_polys):
    """Per-class CALIBRATED envelope radius: the Mahalanobis d2 that contains ``level``
    of the class's held-out sites. The Gaussian chi2 radius is far too tight for new
    polygons (9.5 % of held-out cells fall outside their own 99 % Gaussian envelope),
    so "inside its own envelope" is checked against real unseen sites instead.
    Classes with too few held-out polygons get the pooled radius."""
    pooled = _radius(d2_true, poly, level)
    r = np.full(n_classes, pooled)
    source = np.array(["pooled"] * n_classes, dtype=object)
    for k in range(n_classes):
        sel = true_idx == k
        if sel.any() and len(np.unique(poly[sel])) >= min_polys:
            r[k] = _radius(d2_true[sel], poly[sel], level)
            source[k] = "class"
    return r, source


def _best_allowed(logp: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Highest-posterior allowed class per row (plain argmax where none is allowed)."""
    any_ok = allowed.any(1)
    best = np.where(allowed, logp, -np.inf).argmax(1)
    return np.where(any_ok, best, logp.argmax(1)), any_ok


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / w.sum()
    return float(v[min(np.searchsorted(cw, level), len(v) - 1)])


def _cutoff(logp_true: np.ndarray, poly: np.ndarray, coverage: float) -> float:
    """Log-posterior cutoff so that >= coverage of polygon-weighted cells are kept.

    Score = -log p(true); the conformal quantile is taken at the finite-sample
    corrected level coverage * (1 + 1/n_polygons), with every polygon weighted
    equally (cells weighted 1 / cells-in-polygon).
    """
    _, inv, cnt = np.unique(poly, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    level = min(1.0, coverage * (1.0 + 1.0 / len(cnt)))
    return -_weighted_quantile(-logp_true, w, level)


def _fit_cutoffs(logp_true, true_idx, poly, n_classes, coverage, min_polys):
    """Per-class (Mondrian) cutoffs; classes with too few polygons get the pooled cutoff."""
    pooled = _cutoff(logp_true, poly, coverage)
    q = np.full(n_classes, pooled)
    source = np.array(["pooled"] * n_classes, dtype=object)
    n_poly = np.zeros(n_classes, dtype=int)
    for k in range(n_classes):
        sel = true_idx == k
        if not sel.any():
            continue
        n_poly[k] = len(np.unique(poly[sel]))
        if n_poly[k] >= min_polys:
            q[k] = _cutoff(logp_true[sel], poly[sel], coverage)
            source[k] = "class"
    return q, source, n_poly, pooled


def _calibrated_radii(calib: pd.DataFrame, class_ids: np.ndarray, sc) -> np.ndarray:
    """Envelope radius per class from a calibration table (inf = rule off / no radius)."""
    if not sc.member_envelope:
        return np.full(len(class_ids), np.inf, dtype=np.float32)
    if "envelope_d2" not in calib.columns:
        raise RuntimeError("calibration table has no envelope_d2 - re-run the calibration step with --force")
    pooled = float(calib["envelope_d2"].max())
    c2r = dict(zip(calib["class_id"].astype(int), calib["envelope_d2"]))
    return np.array([c2r.get(int(c), pooled) for c in class_ids], dtype=np.float32)


# ---------------------------------------------------------------------------
# setcalib
# ---------------------------------------------------------------------------

def calibrate_sets(cfg: Config, force: bool = False) -> None:
    out_csv = cfg.dir_holdout / "set_calibration.csv"
    if out_csv.exists() and not force:
        print(f"[setcalib] exists, skipping: {out_csv}")
        return
    sc = cfg.sets

    build_envelopes(cfg, force=False, subset="train")
    class_ids, means, prec, half_logdet, df_n = _load_envelopes(cfg.dir_model / "envelopes_train.npz")
    scaler = Scaler.from_json(cfg.dir_model / "scaler_train.json")
    id2idx = {c: i for i, c in enumerate(class_ids)}
    C = len(class_ids)

    df = load_training(cfg)
    feats = cfg.feature_names
    df = df[np.isfinite(df[feats].to_numpy()).all(1)]
    test = df[(df["split"] == 1) & df["class_id"].isin(id2idx)]
    true_idx = test["class_id"].map(id2idx).to_numpy()
    poly = test["feature_id"].to_numpy()
    n = len(test)
    print(f"[setcalib] {n:,} held-out cells, {len(np.unique(poly)):,} held-out polygons, {C} classes")

    # posteriors under the TRAIN envelopes - these cells were never seen in fitting
    logp_raw = np.empty((n, C), dtype=np.float32)
    d2_all = np.empty((n, C), dtype=np.float32)
    X = scaler.transform(test[feats].to_numpy(dtype=np.float64))
    for s0 in range(0, n, CALIB_BATCH):
        sl = slice(s0, min(s0 + CALIB_BATCH, n))
        d2_all[sl] = _d2(X[sl], means, prec)
        logp_raw[sl] = _log_posterior(d2_all[sl], half_logdet)
    rows = np.arange(n)
    d2_true = d2_all[rows, true_idx].astype(np.float64)

    # cross-conformal: split held-out polygons into halves (per class, seeded)
    rng = np.random.default_rng(cfg.analysis.random_seed)
    half = np.zeros(n, dtype=np.int8)
    for k in range(C):
        sel = true_idx == k
        polys = np.unique(poly[sel])
        b = set(rng.permutation(polys)[: len(polys) // 2].tolist())
        half[sel] = np.isin(poly[sel], list(b)).astype(np.int8)

    # envelope radii AND cutoffs are both fitted on one half, evaluated on the other
    in_set = np.zeros(n, dtype=bool)
    set_size = np.zeros(n, dtype=np.int16)
    allowed = np.ones((n, C), dtype=bool)
    for h in (0, 1):
        cal, ev = half == h, half != h
        if sc.member_envelope:
            r_h, _ = _fit_radii(d2_true[cal], true_idx[cal], poly[cal], C,
                                sc.envelope_coverage, sc.min_test_polygons)
            allowed[cal] = d2_all[cal] <= r_h[None, :]      # radii from this half, for its cutoffs
        lp_cal = np.where(allowed[cal], logp_raw[cal], -np.inf)
        q_h, *_ = _fit_cutoffs(lp_cal[np.arange(int(cal.sum())), true_idx[cal]].astype(np.float64),
                               true_idx[cal], poly[cal], C, sc.coverage, sc.min_test_polygons)
        if sc.member_envelope:
            allowed[ev] = d2_all[ev] <= r_h[None, :]        # evaluation half: radii it never saw
        lp_ev = np.where(allowed[ev], logp_raw[ev], -np.inf)
        best_ev, ok_ev = _best_allowed(logp_raw[ev], allowed[ev])
        member = lp_ev >= q_h[None, :]
        member[np.flatnonzero(ok_ev), best_ev[ok_ev]] = True
        in_set[ev] = member[np.arange(int(ev.sum())), true_idx[ev]]
        set_size[ev] = member.sum(1)

    # `allowed` now holds, for every cell, the radii fitted on the OTHER half
    best, any_ok = _best_allowed(logp_raw, allowed)
    logp = np.where(allowed, logp_raw, -np.inf).astype(np.float32)
    true_blocked = ~allowed[rows, true_idx]
    outside = ~any_ok
    lr_member = (logp >= (logp.max(1, keepdims=True) - np.log(sc.lr_ratio))) & allowed
    lr_in = lr_member[rows, true_idx]
    lr_size = lr_member.sum(1)
    top1 = (best == true_idx) & any_ok

    # final radii + cutoffs from ALL held-out polygons - these drive the map
    if sc.member_envelope:
        radius, radius_src = _fit_radii(d2_true, true_idx, poly, C,
                                        sc.envelope_coverage, sc.min_test_polygons)
    else:
        radius, radius_src = np.full(C, np.inf), np.array(["off"] * C, dtype=object)
    logp_true = np.where(d2_true <= radius[true_idx], logp_raw[rows, true_idx], -np.inf).astype(np.float64)
    q, source, n_poly, pooled = _fit_cutoffs(logp_true, true_idx, poly, C,
                                             sc.coverage, sc.min_test_polygons)

    lut = load_lookup(cfg)
    meta = lut.set_index("class_id")[["naturkode", "navn", "hovednaturtype"]]
    gauss = chi2.isf(1.0 - sc.envelope_coverage, df_n)
    calib = pd.DataFrame({"class_id": class_ids, "n_test_polygons": n_poly,
                          "log_q": q, "q": np.exp(q), "source": source,
                          "pooled_log_q": pooled, "envelope_d2": radius,
                          "envelope_source": radius_src, "envelope_vs_gaussian": radius / gauss})
    calib = calib.join(meta, on="class_id")
    calib = calib[["class_id", "naturkode", "navn", "hovednaturtype", "n_test_polygons",
                   "q", "log_q", "source", "pooled_log_q",
                   "envelope_d2", "envelope_source", "envelope_vs_gaussian"]]
    to_csv(calib, out_csv, index=False)

    # polygon-weighted per-class summary (each polygon counts once)
    _, inv, cnt = np.unique(poly, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    work = pd.DataFrame({"k": true_idx, "w": w, "cov": in_set * w, "size": set_size * w,
                         "lr_cov": lr_in * w, "lr_size": lr_size * w, "top1": top1 * w})
    g = work.groupby("k").sum()
    per = pd.DataFrame({
        "class_id": class_ids[g.index],
        "n_test_polygons": n_poly[g.index],
        "conformal_coverage": g["cov"] / g["w"],
        "conformal_mean_set_size": g["size"] / g["w"],
        "lr_coverage": g["lr_cov"] / g["w"],
        "lr_mean_set_size": g["lr_size"] / g["w"],
        "top1_posterior_acc": g["top1"] / g["w"],
    }).join(meta, on="class_id")
    to_csv(per, cfg.dir_holdout / "set_coverage.csv", index=False)

    def wmean(v):
        return float((v * w).sum() / w.sum())

    sizes = np.bincount(set_size, weights=w, minlength=C + 1) / w.sum()
    size_lines = [f"| {s} | {sizes[s] * 100:.1f} % |" for s in range(1, min(len(sizes), 16)) if sizes[s] > 0]
    lines = [
        "# Plausible-class sets - calibration on held-out polygons",
        "",
        f"- Target coverage: **{sc.coverage:.0%}** per class (Mondrian split conformal)",
        f"- Held-out cells: {n:,} in {len(cnt):,} polygons; every polygon weighted equally",
        f"- Classes with their own cutoff (>= {sc.min_test_polygons} held-out polygons): "
        f"{int((source == 'class').sum())} of {C}; the rest use the pooled cutoff q = {np.exp(pooled):.3g}",
        "",
        "## Cross-validated results (cutoffs from one half of the held-out polygons, evaluated on the other)",
        "",
        "| rule | coverage (true class in set) | mean set size |",
        "|---|---|---|",
        f"| conformal, {sc.coverage:.0%} | **{wmean(in_set):.3f}** | **{wmean(set_size):.2f}** |",
        f"| likelihood ratio 1/{sc.lr_ratio:g} | {wmean(lr_in):.3f} | {wmean(lr_size):.2f} |",
        f"| best class only (posterior) | {wmean(top1):.3f} | 1 |",
        "",
        f"- Classes whose cross-validated coverage falls below {sc.coverage - 0.05:.0%}: "
        + (", ".join(per.loc[per["conformal_coverage"] < sc.coverage - 0.05, "naturkode"].astype(str)) or "none"),
        f"- Member-envelope condition: **{'on' if sc.member_envelope else 'off'}** - a class needs the "
        f"cell inside its own CALIBRATED envelope (radius containing {sc.envelope_coverage:.1%} of its "
        f"held-out sites; median {np.median(radius / gauss):.1f}x the Gaussian {sc.envelope_coverage:.0%} "
        f"radius in d2). Cross-validated: held-out cells where that rules out the TRUE class "
        f"{wmean(true_blocked):.1%}; inside no envelope at all {wmean(outside):.1%}",
        "",
        "## Set-size distribution on held-out cells (conformal)",
        "",
        "| set size | share of cells |",
        "|---|---|",
        *size_lines,
        "",
        "## Reading this",
        "",
        f"Coverage close to {sc.coverage:.2f} means the guarantee holds on polygons the model never saw. "
        "Mean set size is the price paid for that guarantee: the larger it is, the less the "
        "geophysics distinguishes between habitat types. Per-class numbers are in `set_coverage.csv`, "
        "the cutoffs used for the map in `set_calibration.csv`.",
        "",
    ]
    write_text(cfg.dir_holdout / "set_report.md", "\n".join(lines))
    print(f"[setcalib] coverage {wmean(in_set):.3f} (target {sc.coverage}), mean set size "
          f"{wmean(set_size):.2f} | LR 1/{sc.lr_ratio:g}: coverage {wmean(lr_in):.3f}, "
          f"size {wmean(lr_size):.2f} -> {out_csv}")


# ---------------------------------------------------------------------------
# sets - project across Denmark
# ---------------------------------------------------------------------------

def build_sets(cfg: Config, force: bool = False) -> None:
    out_dir = cfg.dir_sets
    size_path = out_dir / "set_size.tif"
    if size_path.exists() and not force:
        print(f"[sets] exists, skipping: {size_path}")
        return
    calib_path = cfg.dir_holdout / "set_calibration.csv"
    if not calib_path.exists():
        raise FileNotFoundError(f"{calib_path} missing - run `signatur setcalib` first")
    sc = cfg.sets

    # production envelopes (all polygons), cutoffs from the train/held-out calibration
    class_ids, means, prec, half_logdet, df_n = _load_envelopes(cfg.dir_model / "envelopes.npz")
    means, prec = means.astype(np.float32), prec.astype(np.float32)
    C = len(class_ids)
    if C > 63:
        raise ValueError(f"{C} classes do not fit in a uint64 bitmask (max 63)")
    scaler = Scaler.from_json(cfg.dir_model / "scaler.json")

    calib = pd.read_csv(calib_path)
    pooled = float(calib["pooled_log_q"].iloc[0])
    cid2q = dict(zip(calib["class_id"], calib["log_q"]))
    log_q = np.array([cid2q.get(c, pooled) for c in class_ids], dtype=np.float32)
    radius = _calibrated_radii(calib, class_ids, sc)
    missing = [int(c) for c in class_ids if c not in cid2q]
    if missing:
        print(f"[sets] classes without a calibrated cutoff (pooled used): {missing}")
    log_lr = np.float32(np.log(sc.lr_ratio))
    bits = (np.uint64(1) << np.arange(C, dtype=np.uint64))

    lut = load_lookup(cfg)
    bit_tab = pd.DataFrame({"bit": np.arange(C), "class_id": class_ids})
    bit_tab = bit_tab.merge(lut[["class_id", "naturkode", "navn", "hovednaturtype"]],
                            on="class_id", how="left")
    to_csv(bit_tab, out_dir / "set_bits.csv", index=False)

    grid = load_grid(cfg.reference_path)
    check_same_grid([*cfg.pc_paths], reference=cfg.reference_path)
    base = dict(grid.profile, count=1, compress=cfg.fuzzy.compression, tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="YES")
    p_i16 = dict(base, dtype="int16", nodata=-1)
    d_size = rasterio.open(size_path, "w", **p_i16)
    d_lr = rasterio.open(out_dir / "lr_set_size.tif", "w", **p_i16)
    d_mask = rasterio.open(out_dir / "set_mask.tif", "w", **dict(base, dtype="uint64", nodata=int(NODATA_MASK)))
    d_post = rasterio.open(out_dir / "best_posterior.tif", "w", **dict(base, dtype="float32", nodata=NODATA_F))
    d_cls = rasterio.open(out_dir / "map_class.tif", "w", **dict(base, dtype="int16", nodata=0))

    pc_srcs = [rasterio.open(p) for p in cfg.pc_paths]
    nd = None if cfg.nodata is None else np.float32(cfg.nodata)
    F = len(pc_srcs)
    batch = cfg.fuzzy.batch_px
    size_counts = np.zeros(C + 1, dtype=np.int64)
    windows = list(iter_block_windows(grid, cfg.processing.block_size, cfg.processing.test_bbox))
    n_land = 0
    try:
        for bi, win in enumerate(windows, 1):
            h, w = int(win.height), int(win.width)
            size_blk = np.full(h * w, -1, dtype=np.int16)
            lr_blk = np.full(h * w, -1, dtype=np.int16)
            mask_blk = np.full(h * w, NODATA_MASK, dtype=np.uint64)
            post_blk = np.full(h * w, NODATA_F, dtype=np.float32)
            cls_blk = np.zeros(h * w, dtype=np.int16)

            b0 = read_band(pc_srcs[0], 1, window=win)
            valid = np.isfinite(b0)
            if nd is not None:
                valid &= b0 != nd
            flat_valid = valid.ravel()
            if flat_valid.any():
                Xr = np.empty((int(flat_valid.sum()), F), dtype=np.float32)
                Xr[:, 0] = b0.ravel()[flat_valid]
                for i in range(1, F):
                    band = read_band(pc_srcs[i], 1, window=win).ravel()[flat_valid]
                    ok = np.isfinite(band) & (band != nd if nd is not None else True)
                    band[~ok] = np.nan
                    Xr[:, i] = band
                row_ok = np.isfinite(Xr).all(1)
                pos = np.flatnonzero(flat_valid)[row_ok]
                Xz = scaler.transform(Xr[row_ok]).astype(np.float32)
                n_land += len(pos)
                for s0 in range(0, len(pos), batch):
                    sl = slice(s0, min(s0 + batch, len(pos)))
                    d2 = _d2(Xz[sl], means, prec)
                    logp = _log_posterior(d2, half_logdet.astype(np.float32))
                    allowed = d2 <= radius[None, :]
                    best, any_ok = _best_allowed(logp, allowed)
                    rows = np.arange(len(best))
                    lp_ok = np.where(allowed, logp, -np.inf)
                    member = (lp_ok >= log_q[None, :]) & allowed
                    member[rows[any_ok], best[any_ok]] = True
                    lr_member = (lp_ok >= lp_ok.max(1, keepdims=True) - log_lr) & allowed
                    p = pos[sl]
                    size_blk[p] = member.sum(1)
                    lr_blk[p] = lr_member.sum(1)
                    mask_blk[p] = (member.astype(np.uint64) * bits[None, :]).sum(1, dtype=np.uint64)
                    post_blk[p] = np.exp(logp[rows, best])
                    cls_blk[p] = class_ids[best]

            land = size_blk >= 0
            size_counts += np.bincount(size_blk[land], minlength=C + 1)
            d_size.write(size_blk.reshape(h, w), 1, window=win)
            d_lr.write(lr_blk.reshape(h, w), 1, window=win)
            d_mask.write(mask_blk.reshape(h, w), 1, window=win)
            d_post.write(post_blk.reshape(h, w), 1, window=win)
            d_cls.write(cls_blk.reshape(h, w), 1, window=win)
            if bi % 10 == 0 or bi == len(windows):
                print(f"[sets]   block {bi}/{len(windows)}  ({n_land:,} land cells)", flush=True)
    finally:
        for s in pc_srcs:
            s.close()
        for d in (d_size, d_lr, d_mask, d_post, d_cls):
            d.close()

    cell_ha = abs(grid.transform.a * grid.transform.e) / 10_000
    summ = pd.DataFrame({"set_size": np.arange(C + 1), "cells": size_counts})
    summ = summ[summ["cells"] > 0].assign(
        ha=lambda t: t["cells"] * cell_ha, share=lambda t: t["cells"] / t["cells"].sum())
    to_csv(summ, out_dir / "set_size_summary.csv", index=False)
    print(f"[sets] wrote set_size / set_mask / best_posterior / map_class / lr_set_size -> {out_dir}")
