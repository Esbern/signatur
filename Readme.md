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

## Later phases (not in this scaffold)

- Fuzzy distance-to-envelope map for all of Denmark (block-wise projection over
  the full PC grid, à la `Groen_trepart/src/project_cluster_distances.py`).
- Optional per-habitat isocluster signatures.
- Adding a climate / atmosphere sphere.
