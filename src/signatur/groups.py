"""Step 11 - two-level plausible sets: habitat GROUPS first, habitats within them.

The habitat-level sets from ``sets.py`` are honest but large (~14 of 44
habitats at 90 % coverage), which gives tens of thousands of distinct
combinations. Grouping the 44 habitats into the 7 classes of
``groups.file`` (``Klasse nr``, e.g. 7 = Skove) and forming the set over
GROUPS gives at most 2**7 - 1 = 127 combinations - small enough to map and
to show as an UpSet plot or Sankey diagram.

Same method as ``sets.py`` (see docs/plausible_class_sets.md), one level up:
  group posterior     P(g|x) = sum of p(k|x) over the habitats k in group g
  group set           {g : P(g|x) >= q_g}   (+ always the best group), per-group
                      cutoffs q_g calibrated on held-out polygons at
                      ``groups.coverage`` (default 80 %)
  habitats in group   within each plausible group g, {k in g : p(k|x, g) >= q_k}
                      with p(k|x, g) = p(k|x) / P(g|x), calibrated on held-out
                      cells of the true group at ``groups.habitat_coverage``
  member envelope     (``sets.member_envelope``, default on, shared with sets.py) the posterior is
                      relative, and summing it over a group can make a group look
                      certain in the gap BETWEEN its habitats' niches. So a habitat
                      also needs the cell inside its own CALIBRATED envelope
                      (sets.envelope_coverage, same radii as sets.py), and a group needs at least one such habitat.
                      Applied in calibration too, so coverage stays honest.

Steps / outputs:
  groupcalib -> ``output/08_holdout/``
    group_calibration.csv  level (group / habitat), id, n_test_polygons, q, source
    group_coverage.csv     per group: cross-validated coverage + mean set size
    group_report.md        headline numbers
  groups -> ``output/11_groups/``
    group_set_mask.tif     uint8   bit g-1 set = group g plausible; 0 = outside
                                   every envelope; nodata 255
    group_set_size.tif     int16   number of plausible groups (nodata -1)
    best_group.tif         uint8   group with the highest posterior (nodata 0)
    best_group_posterior.tif float32 P(best group|x)
    habitat_set_mask.tif   uint64  plausible habitats inside the plausible groups,
                                   same bit order as 10_sets/set_bits.csv (nodata 2**63)
    habitat_set_size.tif   int16
    group_bits.csv         bit -> group number, name, member habitats
    group_combinations.csv every group combination present: groups, cells, ha, share
    group_set_mask.qml     QGIS style (loaded automatically next to the .tif): the
                           ``groups.style_top_n`` largest combinations in distinct
                           colours, the rest light grey, all labelled
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from scipy.special import logsumexp

from .config import Config
from .extract import load_training
from .grid import check_same_grid, iter_block_windows, load_grid
from .ioretry import read_band, to_csv, write_text
from .lookup import load_lookup
from .model import build_envelopes
from .scaler import Scaler
from .sets import (CALIB_BATCH, NODATA_F, NODATA_MASK, _calibrated_radii, _d2, _fit_cutoffs,
                   _fit_radii, _load_envelopes, _log_posterior)

NODATA_GROUP = 255
OUTSIDE_COLOR = "#4d4d4d"
OTHER_COLOR = "#d9d9d9"


def load_groups(cfg: Config) -> pd.DataFrame:
    """naturkode -> group number / name, from ``groups.file`` (``types and classes.csv``)."""
    raw = pd.read_csv(cfg.groups.file, dtype=str, encoding="utf-8-sig")
    tab = pd.DataFrame({"naturkode": raw["Habitat naturtype nr"].str.strip(),
                        "group": raw["Klasse nr"].astype(int),
                        "group_name": raw["Klasse navn"].str.strip()})
    if tab["naturkode"].duplicated().any():
        raise ValueError(f"{cfg.groups.file}: habitat listed in more than one group: "
                         f"{tab.loc[tab['naturkode'].duplicated(), 'naturkode'].tolist()}")
    return tab


def _group_index(cfg: Config, class_ids: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Per envelope: 0-based group index; plus the group table (group, group_name)."""
    lut = load_lookup(cfg)
    cid2code = dict(zip(lut["class_id"], lut["naturkode"]))
    tab = load_groups(cfg)
    code2g = dict(zip(tab["naturkode"], tab["group"]))
    missing = [cid2code.get(c, c) for c in class_ids if cid2code.get(c) not in code2g]
    if missing:
        raise ValueError(f"habitats missing from {cfg.groups.file}: {missing}")
    groups = (tab[["group", "group_name"]].drop_duplicates("group").sort_values("group")
              .reset_index(drop=True))
    if len(groups) > 7:
        raise ValueError(f"{len(groups)} groups do not fit the uint8 group mask (max 7)")
    g2idx = {g: i for i, g in enumerate(groups["group"])}
    gidx = np.array([g2idx[code2g[cid2code[c]]] for c in class_ids])
    return gidx, groups


def _group_log_posterior(logp: np.ndarray, gidx: np.ndarray, G: int) -> np.ndarray:
    return np.stack([logsumexp(logp[:, gidx == g], axis=1) for g in range(G)], axis=1)


def _group_any(flags: np.ndarray, gidx: np.ndarray, G: int) -> np.ndarray:
    return np.stack([flags[:, gidx == g].any(1) for g in range(G)], axis=1)


def _within_group(logp: np.ndarray, logpg: np.ndarray, gidx: np.ndarray) -> np.ndarray:
    """log p(k | x, group(k)) for every habitat k."""
    return logp - logpg[:, gidx]


def write_group_style(combos: pd.DataFrame, path, top_n: int = 20) -> None:
    """Paletted QGIS style for group_set_mask.tif from group_combinations.csv.

    Largest ``top_n`` combinations get distinct colours (tab20), every other
    combination light grey, 0 (outside every envelope) dark grey. Every entry is
    labelled with its groups and share of the land area, largest first.
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    from .residual import _write_qml

    tab20 = [to_hex(c) for c in colormaps["tab20"].colors]
    palette = tab20[0::2] + tab20[1::2]          # 10 strong colours first, then their light pairs
    entries, i = [], 0
    for r in combos.sort_values("cells", ascending=False).itertuples():
        if r.mask == 0:
            color = OUTSIDE_COLOR
        elif i < top_n:
            color, i = palette[i % len(palette)], i + 1
        else:
            color = OTHER_COLOR
        entries.append((int(r.mask), color, f"{r.groups} ({r.share:.1%})"))
    _write_qml(path, entries)


# ---------------------------------------------------------------------------
# groupcalib
# ---------------------------------------------------------------------------

def calibrate_groups(cfg: Config, force: bool = False) -> None:
    out_csv = cfg.dir_holdout / "group_calibration.csv"
    if out_csv.exists() and not force:
        print(f"[groupcalib] exists, skipping: {out_csv}")
        return
    gc = cfg.groups
    min_polys = cfg.sets.min_test_polygons

    build_envelopes(cfg, force=False, subset="train")
    class_ids, means, prec, half_logdet, df_n = _load_envelopes(cfg.dir_model / "envelopes_train.npz")
    scaler = Scaler.from_json(cfg.dir_model / "scaler_train.json")
    gidx, groups = _group_index(cfg, class_ids)
    C, G = len(class_ids), len(groups)
    id2idx = {c: i for i, c in enumerate(class_ids)}

    df = load_training(cfg)
    feats = cfg.feature_names
    df = df[np.isfinite(df[feats].to_numpy()).all(1)]
    test = df[(df["split"] == 1) & df["class_id"].isin(id2idx)]
    true_k = test["class_id"].map(id2idx).to_numpy()
    true_g = gidx[true_k]
    poly = test["feature_id"].to_numpy()
    n = len(test)
    print(f"[groupcalib] {n:,} held-out cells, {len(np.unique(poly)):,} polygons, "
          f"{C} habitats in {G} groups")

    logp = np.empty((n, C), dtype=np.float64)
    d2_all = np.empty((n, C), dtype=np.float32)
    X = scaler.transform(test[feats].to_numpy(dtype=np.float64))
    for s0 in range(0, n, CALIB_BATCH):
        sl = slice(s0, min(s0 + CALIB_BATCH, n))
        d2_all[sl] = _d2(X[sl], means, prec)
        logp[sl] = _log_posterior(d2_all[sl], half_logdet)
    rows = np.arange(n)
    d2_true = d2_all[rows, true_k].astype(np.float64)
    logpg_raw = _group_log_posterior(logp, gidx, G)
    logpc_raw = _within_group(logp, logpg_raw, gidx)
    in_true_group = gidx[None, :] == true_g[:, None]

    def scores(sel, radius):
        """Masked group / within-true-group scores for rows ``sel`` (-inf = not allowed)."""
        inside = d2_all[sel] <= radius[None, :]
        lg = np.where(_group_any(inside, gidx, G), logpg_raw[sel], -np.inf)
        lc = np.where(in_true_group[sel] & inside, logpc_raw[sel], -np.inf)
        return lg, lc, inside

    # cross-conformal halves of the held-out polygons (per habitat, seeded)
    rng = np.random.default_rng(cfg.analysis.random_seed)
    half = np.zeros(n, dtype=np.int8)
    for k in np.unique(true_k):
        sel = true_k == k
        b = rng.permutation(np.unique(poly[sel]))[: len(np.unique(poly[sel])) // 2]
        half[sel] = np.isin(poly[sel], b)

    def radii(sel):
        if not cfg.sets.member_envelope:
            return np.full(C, np.inf)
        return _fit_radii(d2_true[sel], true_k[sel], poly[sel], C,
                          cfg.sets.envelope_coverage, min_polys)[0]

    def members(lp, q):
        m = (lp >= q[None, :]) & np.isfinite(lp)
        ok = np.isfinite(lp).any(1)
        m[np.flatnonzero(ok), lp[ok].argmax(1)] = True
        return m

    # radii AND cutoffs fitted on one half, evaluated on the other
    g_hit = np.zeros(n, dtype=bool); g_size = np.zeros(n, dtype=np.int16)
    h_hit = np.zeros(n, dtype=bool); h_size = np.zeros(n, dtype=np.int16)
    best_g = np.zeros(n, dtype=bool); best_h = np.zeros(n, dtype=bool)
    true_g_blocked = np.zeros(n, dtype=bool)
    for h in (0, 1):
        cal, ev = half == h, half != h
        r_h = radii(cal)
        lg, lc, _ = scores(cal, r_h)
        ic = np.arange(int(cal.sum()))
        qg_h, *_ = _fit_cutoffs(lg[ic, true_g[cal]], true_g[cal], poly[cal], G, gc.coverage, min_polys)
        qk_h, *_ = _fit_cutoffs(lc[ic, true_k[cal]], true_k[cal], poly[cal], C,
                                gc.habitat_coverage, min_polys)
        lg, lc, _ = scores(ev, r_h)
        ie = np.arange(int(ev.sum()))
        mg, mk = members(lg, qg_h), members(lc, qk_h)
        g_hit[ev], g_size[ev] = mg[ie, true_g[ev]], mg.sum(1)
        h_hit[ev], h_size[ev] = mk[ie, true_k[ev]], mk.sum(1)
        best_g[ev] = (lg.argmax(1) == true_g[ev]) & np.isfinite(lg).any(1)
        best_h[ev] = (lc.argmax(1) == true_k[ev]) & np.isfinite(lc).any(1)
        true_g_blocked[ev] = ~np.isfinite(lg[ie, true_g[ev]])

    # final radii + cutoffs from all held-out polygons (radii identical to setcalib's)
    radius = radii(np.ones(n, dtype=bool))
    logpg, logpc, _ = scores(slice(None), radius)

    # final cutoffs from all held-out polygons
    qg, src_g, npoly_g, pooled_g = _fit_cutoffs(logpg[rows, true_g], true_g, poly, G,
                                                gc.coverage, min_polys)
    qk, src_k, npoly_k, pooled_k = _fit_cutoffs(logpc[rows, true_k], true_k, poly, C,
                                                gc.habitat_coverage, min_polys)
    lut = load_lookup(cfg)
    cid2code = dict(zip(lut["class_id"], lut["naturkode"]))
    calib = pd.concat([
        pd.DataFrame({"level": "group", "id": groups["group"], "name": groups["group_name"],
                      "n_test_polygons": npoly_g, "log_q": qg, "source": src_g,
                      "pooled_log_q": pooled_g}),
        pd.DataFrame({"level": "habitat", "id": class_ids,
                      "name": [cid2code.get(c, str(c)) for c in class_ids],
                      "n_test_polygons": npoly_k, "log_q": qk, "source": src_k,
                      "pooled_log_q": pooled_k, "envelope_d2": radius}),
    ], ignore_index=True).assign(q=lambda t: np.exp(t["log_q"]))
    to_csv(calib, out_csv, index=False)

    _, inv, cnt = np.unique(poly, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]

    def wmean(v, sel=slice(None)):
        return float((v[sel] * w[sel]).sum() / w[sel].sum())

    per = pd.DataFrame([{
        "group": int(groups.loc[g, "group"]), "group_name": groups.loc[g, "group_name"],
        "n_habitats": int((gidx == g).sum()),
        "n_test_polygons": int(npoly_g[g]),
        "group_coverage": wmean(g_hit, true_g == g),
        "group_mean_set_size": wmean(g_size, true_g == g),
        "best_group_acc": wmean(best_g, true_g == g),
        "habitat_coverage_within_group": wmean(h_hit, true_g == g),
        "habitat_mean_set_size_within_group": wmean(h_size, true_g == g),
    } for g in range(G)])
    to_csv(per, cfg.dir_holdout / "group_coverage.csv", index=False)

    sizes = np.bincount(g_size, weights=w, minlength=G + 1) / w.sum()
    lines = [
        "# Two-level plausible sets - habitat groups, then habitats within them",
        "",
        f"- Groups from `{cfg.groups.file.name}`: {G} groups, {C} habitats",
        f"- Target coverage: groups **{gc.coverage:.0%}**, habitats within a group "
        f"**{gc.habitat_coverage:.0%}** (Mondrian split conformal, polygon-weighted)",
        f"- Held-out cells: {n:,} in {len(cnt):,} polygons",
        f"- Member-envelope condition: **{'on' if cfg.sets.member_envelope else 'off'}** - a group needs >= 1 "
        f"habitat whose CALIBRATED envelope (radius holding {cfg.sets.envelope_coverage:.1%} of its "
        f"held-out sites, same radii as `setcalib`) contains the cell; cross-validated share of held-out "
        f"sites where that rules out the TRUE group: {wmean(true_g_blocked):.1%}",
        "",
        "## Cross-validated results (calibrate on one half of the held-out polygons, evaluate on the other)",
        "",
        "| level | true one in the set | mean set size | best one correct |",
        "|---|---|---|---|",
        f"| group | **{wmean(g_hit):.3f}** | **{wmean(g_size):.2f}** of {G} | {wmean(best_g):.3f} |",
        f"| habitat within its true group | **{wmean(h_hit):.3f}** | **{wmean(h_size):.2f}** | {wmean(best_h):.3f} |",
        "",
        "Group set sizes: " + ", ".join(f"{s} group{'s' if s > 1 else ''} {sizes[s]:.0%}"
                                        for s in range(1, G + 1) if sizes[s] >= 0.005),
        "",
        "## Per group",
        "",
        "| group | habitats | held-out polygons | group coverage | groups in set | habitat coverage in group | habitats in set |",
        "|---|---|---|---|---|---|---|",
        *[f"| {r.group} {r.group_name} | {r.n_habitats} | {r.n_test_polygons} | {r.group_coverage:.2f} | "
          f"{r.group_mean_set_size:.2f} | {r.habitat_coverage_within_group:.2f} | "
          f"{r.habitat_mean_set_size_within_group:.2f} |" for r in per.itertuples()],
        "",
        "## Reading this",
        "",
        "The habitat-within-group numbers assume the right group is in the group set, which "
        f"happens for {wmean(g_hit):.0%} of sites. On the map, habitats are listed for every "
        "plausible group, so a site's true habitat is listed whenever its group is plausible "
        "and the habitat passes its within-group cutoff.",
        "",
    ]
    write_text(cfg.dir_holdout / "group_report.md", "\n".join(lines))
    print(f"[groupcalib] groups: coverage {wmean(g_hit):.3f}, size {wmean(g_size):.2f} | "
          f"habitats in group: coverage {wmean(h_hit):.3f}, size {wmean(h_size):.2f} -> {out_csv}")


# ---------------------------------------------------------------------------
# groups - project across Denmark
# ---------------------------------------------------------------------------

def build_groups(cfg: Config, force: bool = False) -> None:
    out_dir = cfg.dir_groups
    mask_path = out_dir / "group_set_mask.tif"
    if mask_path.exists() and not force:
        print(f"[groups] exists, skipping: {mask_path}")
        return
    calib_path = cfg.dir_holdout / "group_calibration.csv"
    if not calib_path.exists():
        raise FileNotFoundError(f"{calib_path} missing - run `signatur groupcalib` first")

    class_ids, means, prec, half_logdet, df_n = _load_envelopes(cfg.dir_model / "envelopes.npz")
    means, prec = means.astype(np.float32), prec.astype(np.float32)
    half_logdet = half_logdet.astype(np.float32)
    scaler = Scaler.from_json(cfg.dir_model / "scaler.json")
    gidx, groups = _group_index(cfg, class_ids)
    C, G = len(class_ids), len(groups)

    calib = pd.read_csv(calib_path)
    cg = calib[calib["level"] == "group"]
    ch = calib[calib["level"] == "habitat"]
    g2q = dict(zip(cg["id"].astype(int), cg["log_q"]))
    log_qg = np.array([g2q.get(int(g), cg["pooled_log_q"].iloc[0]) for g in groups["group"]],
                      dtype=np.float32)
    k2q = dict(zip(ch["id"].astype(int), ch["log_q"]))
    log_qk = np.array([k2q.get(int(c), ch["pooled_log_q"].iloc[0]) for c in class_ids],
                      dtype=np.float32)
    radius = _calibrated_radii(ch.rename(columns={"id": "class_id"}), class_ids, cfg.sets)
    gbits = (np.uint8(1) << np.arange(G, dtype=np.uint8))
    kbits = (np.uint64(1) << np.arange(C, dtype=np.uint64))
    same_group = gidx[None, :] == np.arange(G)[:, None]            # (G, C)

    lut = load_lookup(cfg)
    cid2code = dict(zip(lut["class_id"], lut["naturkode"]))
    to_csv(groups.assign(bit=np.arange(G), habitats=[
        " ".join(cid2code.get(c, str(c)) for c in class_ids[gidx == g]) for g in range(G)
    ])[["bit", "group", "group_name", "habitats"]], out_dir / "group_bits.csv", index=False)

    grid = load_grid(cfg.reference_path)
    check_same_grid([*cfg.pc_paths], reference=cfg.reference_path)
    base = dict(grid.profile, count=1, compress=cfg.fuzzy.compression, tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="YES")
    d_gmask = rasterio.open(mask_path, "w", **dict(base, dtype="uint8", nodata=NODATA_GROUP))
    d_gsize = rasterio.open(out_dir / "group_set_size.tif", "w", **dict(base, dtype="int16", nodata=-1))
    d_gbest = rasterio.open(out_dir / "best_group.tif", "w", **dict(base, dtype="uint8", nodata=0))
    d_gpost = rasterio.open(out_dir / "best_group_posterior.tif", "w",
                            **dict(base, dtype="float32", nodata=NODATA_F))
    d_hmask = rasterio.open(out_dir / "habitat_set_mask.tif", "w",
                            **dict(base, dtype="uint64", nodata=int(NODATA_MASK)))
    d_hsize = rasterio.open(out_dir / "habitat_set_size.tif", "w", **dict(base, dtype="int16", nodata=-1))

    pc_srcs = [rasterio.open(p) for p in cfg.pc_paths]
    nd = None if cfg.nodata is None else np.float32(cfg.nodata)
    F = len(pc_srcs)
    batch = cfg.fuzzy.batch_px
    combo_counts = np.zeros(256, dtype=np.int64)
    windows = list(iter_block_windows(grid, cfg.processing.block_size, cfg.processing.test_bbox))
    n_land = 0
    try:
        for bi, win in enumerate(windows, 1):
            h, w = int(win.height), int(win.width)
            gmask_blk = np.full(h * w, NODATA_GROUP, dtype=np.uint8)
            gsize_blk = np.full(h * w, -1, dtype=np.int16)
            gbest_blk = np.zeros(h * w, dtype=np.uint8)
            gpost_blk = np.full(h * w, NODATA_F, dtype=np.float32)
            hmask_blk = np.full(h * w, NODATA_MASK, dtype=np.uint64)
            hsize_blk = np.full(h * w, -1, dtype=np.int16)

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
                    logp = _log_posterior(d2, half_logdet)
                    logpg = _group_log_posterior(logp, gidx, G)
                    rows = np.arange(len(logp))

                    # inside[n, k]: cell within habitat k's own calibrated envelope; a
                    # group needs >= 1 such habitat, and a cell inside no envelope
                    # gets the empty set
                    inside = d2 <= radius[None, :]
                    inside_g = _group_any(inside, gidx, G)
                    out = ~inside_g.any(1)

                    logpg_ok = np.where(inside_g, logpg, -np.inf)
                    best = np.where(out, logpg.argmax(1), logpg_ok.argmax(1))
                    g_member = (logpg >= log_qg[None, :]) & inside_g
                    g_member[rows, best] = True

                    # habitats: within-group cutoff, + the best allowed habitat of every plausible group
                    logpc = np.where(inside, _within_group(logp, logpg, gidx), -np.inf)
                    k_member = (logpc >= log_qk[None, :]) & inside
                    for g in range(G):
                        cols = np.flatnonzero(same_group[g])
                        k_member[rows, cols[logpc[:, cols].argmax(1)]] = True
                    k_member &= g_member[:, gidx] & inside

                    g_member[out] = False
                    k_member[out] = False

                    p = pos[sl]
                    gm = (g_member.astype(np.uint8) * gbits[None, :]).sum(1, dtype=np.uint8)
                    gmask_blk[p] = gm
                    gsize_blk[p] = g_member.sum(1)
                    gbest_blk[p] = groups["group"].to_numpy()[best].astype(np.uint8)
                    gpost_blk[p] = np.exp(logpg[rows, best])
                    hmask_blk[p] = (k_member.astype(np.uint64) * kbits[None, :]).sum(1, dtype=np.uint64)
                    hsize_blk[p] = k_member.sum(1)
                    combo_counts += np.bincount(gm, minlength=256)

            for dst, blk in ((d_gmask, gmask_blk), (d_gsize, gsize_blk), (d_gbest, gbest_blk),
                             (d_gpost, gpost_blk), (d_hmask, hmask_blk), (d_hsize, hsize_blk)):
                dst.write(blk.reshape(h, w), 1, window=win)
            if bi % 10 == 0 or bi == len(windows):
                print(f"[groups]   block {bi}/{len(windows)}  ({n_land:,} land cells)", flush=True)
    finally:
        for s in pc_srcs:
            s.close()
        for d in (d_gmask, d_gsize, d_gbest, d_gpost, d_hmask, d_hsize):
            d.close()

    cell_ha = abs(grid.transform.a * grid.transform.e) / 10_000
    names = groups["group_name"].tolist()
    codes = np.flatnonzero(combo_counts)
    combos = pd.DataFrame({
        "mask": codes,
        "n_groups": [bin(int(c)).count("1") for c in codes],
        "groups": [" + ".join(names[g] for g in range(G) if c >> g & 1) or "(outside every envelope)"
                   for c in codes],
        "cells": combo_counts[codes],
    }).assign(ha=lambda t: t["cells"] * cell_ha, share=lambda t: t["cells"] / t["cells"].sum())
    combos = combos.sort_values("cells", ascending=False)
    to_csv(combos, out_dir / "group_combinations.csv", index=False)
    write_group_style(combos, out_dir / "group_set_mask.qml", cfg.groups.style_top_n)
    print(f"[groups] {len(combos)} group combinations present; top 5:")
    for r in combos.head(5).itertuples():
        print(f"           {r.share:6.1%}  {r.groups}")
    print(f"[groups] wrote group / habitat set rasters -> {out_dir}")
