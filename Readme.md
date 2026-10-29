# signatur

Identify where **new nature** can be created in Denmark by learning the
geophysical conditions that existing habitat types occupy, then scoring the rest
of the country by how far it sits from each type's occupied geophysical envelope.

## Phase 1 (this scaffold): does geophysics actually separate the habitat types?

Input geophysical data: 8 principal-component rasters from the geo-ecological
sphere model (no atmosphere yet) on the standard Danish 10 m grid
(EPSG:25832) — `hydro_PC1/2`, `soil_PC1/2/3`, `toposphere_PC1/2/3`.

Input habitat data: 3 overlapping GeoParquet layers, each with a numeric-ish
`naturkode` (Natura 2000 habitat code). Overlap is resolved by priority:
`bevaringsvrdige_egekrat` > `np3__lysbne_naturtyper` > `np3__skovnaturtyper`.

### Pipeline

| step | module | output (`<output_dir>/…`) |
|------|--------|---------------------------|
| `lookup` | `lookup.py` | `00_lookup/naturkode_lookup.csv` — `naturkode` → integer `class_id` → Danish name |
| `rasterize` | `rasterize.py` | `01_merged_raster/merged_class_id.tif` (+ `overlap_report.csv`) — priority merge onto the 10 m PC grid |
| `extract` | `extract.py` | `02_training/training_pixels.parquet` / `.npz` — **every** habitat cell with its 8 PC values (block-wise, same-grid, no coordinate lookup) |
| `plots` | `plots.py` | `03_plots/*.png` — class support, per-PC violins, ridgelines, class-mean heatmap, sphere biplots, global ordination, pairwise distance |
| `stats` | `stats.py` | `04_stats/` — ANOVA + Kruskal effect sizes, cross-validated RandomForest, confusion matrix, `summary.md` |

`03_plots/class_mean_heatmap.png` and `04_stats/summary.md` are the headline
answer to "are the conditions significantly different?".

### Phase 2: the fuzzy suitability maps

| step | module | output (`<output_dir>/…`) |
|------|--------|---------------------------|
| `model` | `model.py` | `05_model/` — one multivariate-Gaussian envelope per habitat type (mean + shrunk covariance) in standardised 8-PC space, plus `scaler.json` and `model_report.csv` (per-class Gaussian-fit calibration) |
| `project` | `project.py` | `06_fuzzy/` — streams all of Denmark: `best_class.tif` (closest habitat), `best_suitability.tif`, `best_distance.tif` (Mahalanobis), and one `NN_<code>_suitability.tif` (0–1) per habitat type |

Suitability = chi-square survival function of the Mahalanobis distance to the
envelope: **1** at the centre of the occupied niche, **~0.5** at a typical
occupied cell, **~0** far outside. Run with `python -m signatur fuzzy`
(= `model` then `project`); `project` is the long step (streams the 8 rasters
over the whole grid).

| step | module | output |
|------|--------|--------|
| `fuzzyplots` | `fuzzy_plots.py` | `06_fuzzy/best_by_habitat.{png,csv}` — projected suitability sampled inside each habitat's training polygons (calibration check) |
| `richness` | `richness.py` | `07_potential/habitat_richness.tif` (count of habitats with suitability ≥ threshold), `habitat_richness_open.tif` (non-forest only), `habitat_richness_sum.tif` (Σ suitability) — type-agnostic "how much nature could this ground support", for delineating restoration / park areas |

`fuzzyplots` and `richness` only re-read the `06_fuzzy` rasters, so they are
fast to re-run with new thresholds (`richness.threshold` in `config.yaml`).

| step | module | output |
|------|--------|--------|
| `setcalib` | `sets.py` | `08_holdout/set_calibration.csv`, `set_coverage.csv`, `set_report.md` — per-habitat posterior cutoffs calibrated on the held-out polygons (conformal prediction, `sets.coverage`) |
| `sets` | `sets.py` | `10_sets/set_size.tif`, `set_mask.tif` (uint64 bitmask, see `set_bits.csv`), `best_posterior.tif`, `map_class.tif` — every habitat the geophysics cannot rule out, instead of one winner |
| `groupcalib` | `groups.py` | `08_holdout/group_calibration.csv`, `group_coverage.csv`, `group_report.md` — same calibration over the 7 habitat groups in `data/types and classes.csv` (`groups.coverage`), plus habitats within each group |
| `groups` | `groups.py` | `11_groups/group_set_mask.tif` (≤127 group combinations, see `group_combinations.csv`), `best_group.tif`, `habitat_set_mask.tif` — plausible groups, then plausible habitats inside them |

See [docs/plausible_class_sets.md](docs/plausible_class_sets.md) for the method
and how to read the results. `setcalib` needs the holdout split and takes
seconds; `sets` streams the PC rasters again (~70 min for all of Denmark).

## Setup

```bash
mamba env create -f environment.yml
mamba activate signatur
pip install -e .            # optional; or just run `python -m signatur`
```

## Run

Edit **`config.yaml`** (the only machine-specific file), then:

```bash
python -m signatur all --config config.yaml
# or a single step:
python -m signatur rasterize --force
```

For a fast end-to-end test on a small area, set `processing.test_bbox` in
`config.yaml` to `[xmin, ymin, xmax, ymax]` (CRS units), e.g.
`[520000, 6200000, 560000, 6240000]`, then `python -m signatur all`.

Each step skips itself if its output already exists; use `--force` to rebuild.

## Editing the habitat name table

`data/naturkode_lookup_seed.csv` maps each `naturkode` to a Danish name,
`hovednaturtype` (used for plot grouping/colour) and a `prioriteret` flag. It is
plain CSV — correct or extend it and re-run `python -m signatur lookup --force`.

## Later / optional

- Mask the fuzzy maps to non-habitat land only (candidate new-nature areas).
- Replace the single Gaussian with a mixture for the few non-Gaussian niches
  flagged in `model_report.csv`.
- Adding a climate / atmosphere sphere (weak nationally per the Phase 1 stats,
  but a large-scale temperature/precipitation gradient is a new axis).
