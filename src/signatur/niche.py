"""Step 12 - which raw geophysical / climate parameters each habitat "prefers".

The suitability maps are computed from principal components, which are hard to
read. This step translates each habitat's envelope back into the raw parameters
the PCs were built from (clay, groundwater depth, temperature, ...): for every
habitat it shows how those parameters are distributed over the land where the
habitat's suitability exceeds ``niche.threshold`` (default 0.55, the cut used in
the appendix maps).

Because the parameters have unrelated units and ranges, every value is shown as
its **national percentile** (0-100 over all sampled Danish land cells). On that
axis the national distribution is flat with median 50, a violin shifted right
means "high values", and a narrow violin means "specialised". The labels give
the habitat's median in real units next to the national median.

Note: the cells are selected by the model (suitability from the PCs), so the
plots describe the *modelled* niche in physical terms - not an independent test.

Sampling: a regular grid every ``niche.stride`` cells (20 -> every 200 m, about
1 million land cells). Suitability for all habitats is recomputed at those cells
from the production envelopes - identical to the maps, without reading the 44
suitability rasters. Rasters on a different grid (the 1 km climate layers) are
looked up by coordinate.

Every cell counts: ``nichehist`` makes one pass over ALL 10 m land cells and keeps,
per habitat, a histogram of each parameter's national percentile over the cells
above the threshold - exact also for habitats with only a few hectares. The
200 m sample (``nichesample``) is used only as the national reference
distribution (value -> national percentile); with ~1 million points that is
accurate to well under one percentile point.

Steps / outputs (``output/13_niche/``):
  nichesample  niche_sample.parquet          national reference sample (200 m grid)
  nichehist    niche_hist.npz                per habitat x parameter: percentile histograms (all cells)
  nicheplots   NN_<code>_niche.png/.pdf      one figure per habitat
               niche_overview.png/.pdf       heatmap: median national percentile, habitat x parameter
               niche_medians.csv             habitat x parameter: median (real units) and percentile
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .grid import iter_block_windows, load_grid
from .ioretry import read_band, save_fig, to_csv, to_parquet
from .lookup import load_lookup
from .project import _chi2_sf
from .scaler import Scaler
from .sets import _d2, _load_envelopes

SPHERE_COLORS = {"Jordbund": "#a6761d", "Hydrosfære": "#1f78b4",
                 "Toposfære": "#33a02c", "Klima": "#e31a1c"}


def _sample_rows(path, rows: np.ndarray, cols: np.ndarray, band: int = 1) -> np.ndarray:
    """Values at grid cells (rows x cols) of a raster on the reference grid, row by row."""
    out = np.empty((len(rows), len(cols)), dtype=np.float32)
    with rasterio.open(path) as src:
        nd = src.nodata
        for i, r in enumerate(rows):
            line = read_band(src, band, window=((int(r), int(r) + 1), (0, src.width)))[0]
            out[i] = line[cols]
    if nd is not None:
        out[out == nd] = np.nan
    return out.ravel()


def _sample_xy(path, xs: np.ndarray, ys: np.ndarray, band: int = 1) -> np.ndarray:
    """Values at coordinates from a raster on another grid (e.g. the 1 km climate layers)."""
    with rasterio.open(path) as src:
        a = src.read(band).astype(np.float32)
        nd = src.nodata
        r, c = rasterio.transform.rowcol(src.transform, xs, ys)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
    v = np.full(len(xs), np.nan, dtype=np.float32)
    v[ok] = a[r[ok], c[ok]]
    if nd is not None and not np.isnan(nd):
        v[v == nd] = np.nan
    return v


def build_niche_sample(cfg: Config, force: bool = False) -> None:
    out_path = cfg.dir_niche / "niche_sample.parquet"
    if out_path.exists() and not force:
        print(f"[nichesample] exists, skipping: {out_path}")
        return
    nc = cfg.niche
    grid = load_grid(cfg.reference_path)
    rows = np.arange(nc.stride // 2, grid.height, nc.stride)
    cols = np.arange(nc.stride // 2, grid.width, nc.stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    xs, ys = rasterio.transform.xy(grid.transform, rr.ravel(), cc.ravel())
    xs, ys = np.asarray(xs), np.asarray(ys)
    print(f"[nichesample] grid every {nc.stride} cells: {len(rows)} x {len(cols)} = {xs.size:,} points")

    # PCs -> land mask + suitability, exactly as in `project`
    pcs = {}
    for name, path in zip(cfg.feature_names, cfg.pc_paths):
        with rasterio.open(path) as src:
            same = (src.width, src.height) == (grid.width, grid.height)
        pcs[name] = _sample_rows(path, rows, cols) if same else _sample_xy(path, xs, ys)
        print(f"[nichesample]   {name}", flush=True)
    X = np.column_stack([pcs[n] for n in cfg.feature_names])
    land = np.isfinite(X).all(1)
    print(f"[nichesample] {land.sum():,} land points")

    class_ids, means, prec, _, df_n = _load_envelopes(cfg.dir_model / "envelopes.npz")
    scaler = Scaler.from_json(cfg.dir_model / "scaler.json")
    Xz = scaler.transform(X[land].astype(np.float64))
    suit = np.empty((len(Xz), len(class_ids)), dtype=np.float32)
    for s0 in range(0, len(Xz), 200_000):
        sl = slice(s0, s0 + 200_000)
        suit[sl] = _chi2_sf(_d2(Xz[sl], means, prec), df_n)

    df = pd.DataFrame({"x": xs[land], "y": ys[land]})
    for p in nc.parameters:
        path, band = p["file"], int(p.get("band", 1))
        with rasterio.open(path) as src:
            same = (src.width, src.height) == (grid.width, grid.height)
        v = _sample_rows(path, rows, cols, band) if same else _sample_xy(path, xs, ys, band)
        df[p["name"]] = v[land]
        print(f"[nichesample]   {p['name']}", flush=True)
    for j, cid in enumerate(class_ids):
        df[f"s_{cid}"] = suit[:, j]
    to_parquet(df, out_path, index=False)
    print(f"[nichesample] {len(df):,} land points x {len(nc.parameters)} parameters -> {out_path}")


def _fmt(v: float, unit: str, ref: float | None = None) -> str:
    """Danish number; decimals from the larger of |v| and |ref| so a pair reads alike."""
    if not np.isfinite(v):
        return "–"
    a = max(abs(v), abs(ref) if ref is not None and np.isfinite(ref) else 0)
    d = 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3
    return f"{v:.{d}f}".replace("-", "\u2212").replace(".", ",") + (f" {unit}" if unit else "")

NBINS = 200  # percentile histogram bins (0.5 percentile points each)


def _reference(df: pd.DataFrame, params) -> dict:
    """National reference per parameter from the sample: sorted unique values, their mid-rank
    percentile, national median and - for parameters dominated by one value - that floor value
    and the national share above it."""
    ref = {}
    for p in params:
        v = df[p["name"]].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        u, cnt = np.unique(v, return_counts=True)
        below = np.cumsum(cnt) - cnt
        pu = 100.0 * (below + cnt / 2.0) / len(v)  # mid-rank percentile of each unique value
        floor = (float(u[0]), float(np.mean(v > u[0]))) if cnt[0] / len(v) > 0.25 else None
        ref[p["name"]] = {"u": u, "pu": pu, "median": float(np.median(v)), "floor": floor}
    return ref


def _to_pct(values: np.ndarray, r: dict) -> np.ndarray:
    out = np.interp(values, r["u"], r["pu"])
    out[~np.isfinite(values)] = np.nan
    return out


def build_niche_hist(cfg: Config, force: bool = False) -> None:
    """One pass over EVERY 10 m land cell: per habitat, histograms of the parameters' national
    percentiles over the cells with suitability > threshold. Exact also for rare habitats."""
    out_path = cfg.dir_niche / "niche_hist.npz"
    if out_path.exists() and not force:
        print(f"[nichehist] exists, skipping: {out_path}")
        return
    nc = cfg.niche
    params = list(nc.parameters)
    ref = _reference(pd.read_parquet(cfg.dir_niche / "niche_sample.parquet"), params)

    class_ids, means, prec, _, df_n = _load_envelopes(cfg.dir_model / "envelopes.npz")
    means, prec = means.astype(np.float32), prec.astype(np.float32)
    scaler = Scaler.from_json(cfg.dir_model / "scaler.json")
    C, P = len(class_ids), len(params)
    grid = load_grid(cfg.reference_path)
    T = grid.transform

    pc_srcs = [rasterio.open(path) for path in cfg.pc_paths]
    raw_src, raw_small = {}, {}
    for p in params:  # 10 m layers are read per block; others (1 km climate) are held in memory
        src = rasterio.open(p["file"])
        if (src.width, src.height) == (grid.width, grid.height):
            raw_src[p["name"]] = (src, int(p.get("band", 1)))
        else:
            raw_small[p["name"]] = (src.read(int(p.get("band", 1))).astype(np.float32), src.transform, src.nodata)
            src.close()

    counts = np.zeros((C, P, NBINS), dtype=np.int64)
    above = np.zeros((C, P), dtype=np.int64)
    n_cells = np.zeros(C, dtype=np.int64)
    nd = None if cfg.nodata is None else np.float32(cfg.nodata)
    windows = list(iter_block_windows(grid, cfg.processing.block_size, cfg.processing.test_bbox))
    n_land = 0
    try:
        for bi, win in enumerate(windows, 1):
            h, w = int(win.height), int(win.width)
            b0 = read_band(pc_srcs[0], 1, window=win).ravel()
            valid = np.isfinite(b0) & ((b0 != nd) if nd is not None else True)
            if not valid.any():
                continue
            X = np.empty((int(valid.sum()), len(pc_srcs)), dtype=np.float32)
            X[:, 0] = b0[valid]
            for i in range(1, len(pc_srcs)):
                v = read_band(pc_srcs[i], 1, window=win).ravel()[valid].astype(np.float32)
                v[(~np.isfinite(v)) | ((v == nd) if nd is not None else False)] = np.nan
                X[:, i] = v
            ok = np.isfinite(X).all(1)
            idx = np.flatnonzero(valid)[ok]
            X = X[ok]
            n_land += len(idx)

            bins = np.empty((P, len(idx)), dtype=np.int16)
            raw = {}
            for j, p in enumerate(params):
                name = p["name"]
                if name in raw_src:
                    src, band = raw_src[name]
                    v = read_band(src, band, window=win).ravel()[idx].astype(np.float64)
                    if src.nodata is not None:
                        v[v == src.nodata] = np.nan
                else:
                    a, t, snd = raw_small[name]
                    r = win.row_off + idx // w
                    c = win.col_off + idx % w
                    xs = T.c + (c + 0.5) * T.a
                    ys = T.f + (r + 0.5) * T.e
                    rr = np.floor((ys - t.f) / t.e).astype(np.int64)
                    cc = np.floor((xs - t.c) / t.a).astype(np.int64)
                    inside = (rr >= 0) & (rr < a.shape[0]) & (cc >= 0) & (cc < a.shape[1])
                    v = np.full(len(idx), np.nan)
                    v[inside] = a[rr[inside], cc[inside]]
                    if snd is not None and not np.isnan(snd):
                        v[v == snd] = np.nan
                raw[name] = v
                pc_ = _to_pct(v, ref[name])
                b = np.full(len(idx), -1, dtype=np.int16)
                fin = np.isfinite(pc_)
                b[fin] = np.clip((pc_[fin] / 100.0 * NBINS).astype(np.int16), 0, NBINS - 1)
                bins[j] = b

            for s0 in range(0, len(idx), cfg.fuzzy.batch_px):
                sl = slice(s0, s0 + cfg.fuzzy.batch_px)
                Xz = scaler.transform(X[sl]).astype(np.float32)
                M = _chi2_sf(_d2(Xz, means, prec), df_n) > nc.threshold      # (n, C)
                for k in np.flatnonzero(M.any(0)):
                    sel = M[:, k]
                    n_cells[k] += int(sel.sum())
                    for j, p in enumerate(params):
                        bb = bins[j, sl][sel]
                        counts[k, j] += np.bincount(bb[bb >= 0], minlength=NBINS)
                        fl = ref[p["name"]]["floor"]
                        if fl is not None:
                            above[k, j] += int(np.sum(raw[p["name"]][sl][sel] > fl[0]))
            if bi % 10 == 0 or bi == len(windows):
                print(f"[nichehist]   block {bi}/{len(windows)}  ({n_land:,} land cells)", flush=True)
    finally:
        for s_ in pc_srcs:
            s_.close()
        for src, _ in raw_src.values():
            src.close()

    np.savez_compressed(out_path, counts=counts, above=above, n_cells=n_cells, class_ids=class_ids,
                        params=np.array([p["name"] for p in params]), n_land=n_land,
                        threshold=nc.threshold)
    print(f"[nichehist] {n_land:,} land cells; cells above {nc.threshold} per habitat: "
          f"min {n_cells.min():,}, max {n_cells.max():,} -> {out_path}")


def _hist_quantiles(h: np.ndarray, qs) -> list[float]:
    """Quantiles (in percentile units 0-100) of a histogram over NBINS equal bins."""
    cum = np.concatenate([[0], np.cumsum(h)]) / max(h.sum(), 1)
    edges = np.linspace(0, 100, NBINS + 1)
    return [float(np.interp(q, cum, edges)) for q in qs]


def _fmt(v: float, unit: str, ref: float | None = None) -> str:
    """Danish number; decimals from the larger of |v| and |ref| so a pair reads alike."""
    if not np.isfinite(v):
        return "–"
    a = max(abs(v), abs(ref) if ref is not None and np.isfinite(ref) else 0)
    d = 0 if a >= 100 else 1 if a >= 10 else 2 if a >= 1 else 3
    return f"{v:.{d}f}".replace("-", "\u2212").replace(".", ",") + (f" {unit}" if unit else "")


def make_niche_plots(cfg: Config, force: bool = False, only: list[str] | None = None) -> None:
    out_dir = cfg.dir_niche
    if (out_dir / "niche_overview.png").exists() and not force and not only:
        print(f"[nicheplots] exists, skipping: {out_dir}")
        return
    nc = cfg.niche
    params = list(nc.parameters)
    ref = _reference(pd.read_parquet(out_dir / "niche_sample.parquet"), params)
    H = np.load(out_dir / "niche_hist.npz")
    counts, above, n_cells, class_ids = H["counts"], H["above"], H["n_cells"], H["class_ids"]
    lut = load_lookup(cfg)
    cid2 = lut.set_index("class_id")[["naturkode", "navn"]].to_dict("index")
    try:  # proper Danish names (å/æ/ø) from the habitat-group table, as in the maps
        names = pd.read_csv(cfg.groups.file, dtype=str, encoding="utf-8-sig")
        dk_name = {r["Habitat naturtype nr"].strip(): r["Habitat naturtype nr og navn"].split("–", 1)[-1].strip()
                   for _, r in names.iterrows()}
    except (OSError, KeyError):
        dk_name = {}

    rows = []
    for k, cid in enumerate(class_ids):
        code = cid2[int(cid)]["naturkode"]
        name = dk_name.get(code, cid2[int(cid)]["navn"])
        if only and code not in only:
            continue
        n = int(n_cells[k])
        rec = {"class_id": int(cid), "naturkode": code, "navn": name, "n_cells": n, "area_ha": n / 100}
        for j, p in enumerate(params):
            med_pct = _hist_quantiles(counts[k, j], [0.5])[0] if n else np.nan
            rec[f"{p['name']}_pct"] = med_pct
            rec[f"{p['name']}_median"] = float(np.interp(med_pct, ref[p["name"]]["pu"], ref[p["name"]]["u"])) if n else np.nan
        rows.append(rec)
        if n < nc.min_points:
            print(f"[nicheplots] {code}: only {n} cells above {nc.threshold} - figure skipped")
            continue
        _habitat_figure(out_dir / f"{int(cid):02d}_{code}_niche", code, name, n, params, ref,
                        counts[k], above[k], nc.threshold)
        print(f"[nicheplots] {code} {name}: {n:,} cells", flush=True)

    med = pd.DataFrame(rows)
    if not only:
        to_csv(med, out_dir / "niche_medians.csv", index=False)
        _overview_figure(out_dir / "niche_overview", med, params, nc.threshold)
    print(f"[nicheplots] -> {out_dir}")


def _habitat_figure(stem, code, name, n, params, ref, counts, above, thr) -> None:
    k = len(params)
    fig, ax = plt.subplots(figsize=(8.3, 0.36 * k + 1.8))
    ys = np.arange(k)[::-1]
    x = (np.arange(NBINS) + 0.5) * 100.0 / NBINS
    kernel = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    labels = []
    for y, p, h, ab in zip(ys, params, counts, above):
        color = SPHERE_COLORS.get(p["sphere"], "#888888")
        if h.sum():
            dens = np.convolve(h.astype(float), kernel / kernel.sum(), mode="same")
            ax.fill_between(x, y - 0.4 * dens / dens.max(), y + 0.4 * dens / dens.max(),
                            color=color, alpha=0.75, lw=0)
            q1, q2, q3 = _hist_quantiles(h, [0.25, 0.5, 0.75])
            ax.plot([q1, q3], [y, y], color="black", lw=1.6, solid_capstyle="butt")
            ax.plot([q2], [y], marker="o", ms=3.5, color="white", mec="black", mew=0.8)
        r = ref[p["name"]]
        if r["floor"] is not None:  # e.g. CaCO3: median is 0 almost everywhere - give the share above 0
            lo, dk_share = r["floor"]
            share = ab / h.sum() if h.sum() else np.nan
            labels.append(f"> {lo:g}: {share:.0%}  (DK {dk_share:.0%})".replace("%", " %"))
        else:
            med = float(np.interp(_hist_quantiles(h, [0.5])[0], r["pu"], r["u"])) if h.sum() else np.nan
            labels.append(f"{_fmt(med, p.get('unit', ''), r['median'])}  (DK {_fmt(r['median'], '', med)})")
    ax.axvline(50, color="#555555", lw=0.8, ls="--")
    ax.axvspan(25, 75, color="#eeeeee", zorder=0)
    ax.set_xlim(-2, 102)  # markers at 0 / 100 stay visible
    ax.set_ylim(-0.7, k - 0.3)
    ax.set_yticks(ys)
    ax.set_yticklabels([p["label"] for p in params], fontsize=8)
    for tick, p in zip(ax.get_yticklabels(), params):
        tick.set_color(SPHERE_COLORS.get(p["sphere"], "black"))
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(ys)
    right.set_yticklabels(labels, fontsize=7.5)
    right.tick_params(length=0)
    ax.set_xlabel("National percentil (0 = laveste i Danmark, 100 = højeste)\n"
                  "Til højre: habitatets median (Danmarks median i parentes)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    area = f"{n / 100:,.0f}".replace(",", ".")   # ha, Danish thousands separator
    ax.set_title(f"{code} – {name}\n{area} ha med egnethed > {str(thr).replace('.', ',')}"
                 f"  (alle 10 m-celler)", fontsize=10, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        save_fig(fig, f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _overview_figure(stem, med: pd.DataFrame, params, thr) -> None:
    med = med[med["n_cells"] > 0]
    M = med[[f"{p['name']}_pct" for p in params]].to_numpy()
    fig, ax = plt.subplots(figsize=(0.33 * len(params) + 4.5, 0.24 * len(med) + 2.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels([p["label"] for p in params], rotation=60, ha="right", fontsize=7.5)
    for tick, p in zip(ax.get_xticklabels(), params):
        tick.set_color(SPHERE_COLORS.get(p["sphere"], "black"))
    ax.set_yticks(range(len(med)))
    ax.set_yticklabels([f"{c} {n}"[:44] for c, n in zip(med["naturkode"], med["navn"])], fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label("Median national percentil", fontsize=8)
    ax.set_title(f"Habitattypernes geofysiske og klimatiske niche (celler med egnethed > "
                 f"{str(thr).replace('.', ',')})", fontsize=10, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        save_fig(fig, f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
