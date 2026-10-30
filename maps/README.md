# Appendix maps

Every appendix map is produced from one QGIS print layout by
[`scripts/export_maps.py`](../scripts/export_maps.py), so all maps share one
design.

## What the script uses

**The design** comes from the QGIS project:
- page and scale bar;
- legend position;
- the Bornholm inset (top right, same scale as the main map);
- the background map.

To change the design, edit the layout in QGIS and save the project.

**Per map**, the script supplies three things:
- the raster shown (the `file` column of the CSV);
- the title (the `title` column);
- the colour style (`--style` .qml).

**Fixed for all maps:**
- Denmark in light grey, with a clear coastline drawn on top of the raster;
- the style's transparent class (unsuitable land) drawn in the same grey.

## Titles

`suitability_titles.csv` has two columns, `file` and `title`, one row per map.
- `file` is relative to `output_climate/`.
- `title` is pre-filled from `data/types and classes.csv`.

To edit titles, open the CSV in Excel or a text editor. Commas or semicolons
both work, and the file must stay UTF-8. Then rerun the script.

**Title placement:** the title box is the layout label with item ID `title`. If
the layout has none, the script adds one in the top left. To move it or change
the font, add a text label with ID `title` to the layout in QGIS and save the
project. The script will then use that label.

## Run

Use the QGIS Python of the **same major version the project was saved with**. A project saved in QGIS 4 renders wrongly in QGIS 3, with a warped map and a misplaced inset. Use the conda env `qgis4` (QGIS 4.2) for the current project, `qgis` (3.44) for older ones, or the OSGeo4W shell on Windows.

```bash
mamba run -n qgis4 python scripts/export_maps.py \
    --project /mnt/d/signatur/output_climate/QGIS/suitability.qgz \
    --layout suitability \
    --csv maps/suitability_titles.csv \
    --style /mnt/d/signatur/output_climate/QGIS/suitability_layer_style_clean.qml \
    --out /mnt/d/signatur/output_climate/12_maps/suitability
```

Useful options:

| Option | Effect |
|---|---|
| `--only 9130` | export one map (for testing) |
| `--number-from N` | number of the first CSV row (default 1) |
| `--combined appendix.pdf` | also write all maps into one PDF (needs `pypdf` in the QGIS env) |
| `--legend-label` | legend heading (default "Egnethed") |
| `--land-color` | fill colour for Denmark |
| `--below-color` | colour for unsuitable land |
| `--coast-color`, `--coast-width` | coastline colour and width |

The project file is only read, never changed.

**Output names** are numbered in CSV row order: `001_01_1210_suitability.pdf`,
`002_...`. The folder therefore sorts, and merges, in report order. For a
later batch (PCs, base parameters), use `--number-from 45` to continue the
numbering. Use `--no-numbers` to switch it off.

**Style:** `suitability_layer_style_clean.qml` (in `output_climate/QGIS/`) cuts
at 0.55 (below that, grey) and has 9 classes of 0.05 up to 1.00, with Danish
decimal commas in the legend. Isabella's original `suitability_layer_style.qml`
is untouched.

## Climate: PCs and their input layers (maps 045–072)

The climate maps use the same layout. Each band gets its own data-driven
classes, because the bands have completely different ranges:
- **classes:** 10 equal-count classes (quantiles over the band's land cells);
- **colours:** ColorBrewer Spectral, with low values blue and high values red;
- **legend heading:** the band's unit (from its `units` tag), or the CSV
  `legend` column (e.g. "PC-score");
- **legend labels:** a true minus sign and Danish decimal comma.

`climate_titles.csv` has two extra columns:
- `band`, which band of the file to show;
- `legend`, an optional legend heading.

It contains:
- the 3 PCs of `climate_pca.tif`;
- the 25 bands of `climate_inputs_1km.tif` that go into the PCA, i.e. the
  seasonal bands plus the annual temperature range.

The six annual (`_ann`) bands are not PCA inputs and are left out. Add a row
to include one.

```bash
mamba run -n qgis4 python scripts/export_maps.py \
    --project /mnt/d/signatur/output_climate/QGIS/suitability.qgz \
    --layout suitability \
    --csv maps/climate_titles.csv \
    --data-root /mnt/d/klimanormal \
    --classify quantile --legend-label "{unit}" \
    --number-from 45 \
    --out /mnt/d/signatur/output_climate/12_maps/climate
```

Quantile options:

| Option | Effect |
|---|---|
| `--classes N` | number of classes (default 10) |
| `--invert-ramp` | low values red, high values blue |
| `--decimal-point` | `.` instead of `,` in the legend |

## Hydro, soil and toposphere: PCs and their input layers (maps 073–092)

Same layout and the same quantile classes as the climate maps. The data comes
from GeoILUM:
- PCs from `D:\GeoILUM\data\dk\04_principal_components`;
- raw layers from `D:\GeoILUM\data\dk\Preprossed`.

The grouping of raw layers into spheres follows `D:\GeoILUM\scripts\spheres.JSON`:

| Sphere | PCs | Raw layers |
|---|---|---|
| hydro | 2 | groundwater depth, summer and winter |
| soil | 3 | clay, silt, fine sand, coarse sand, CaCO₃, organic carbon (0–30 cm) |
| toposphere | 3 | slope, solar irradiance (day 5–175), topographic index, TPI |

`dtm_10` and `aspect_10` are not in any sphere, so they are not mapped.

The GeoTIFFs carry no unit tags, so the legend heading comes from the `legend`
column of `geophysics_titles.csv`. Where one value fills more than a class's
share, it gets its own class: CaCO₃ is 0 in 83 % of the cells, so the legend
shows a "0" class.

```bash
mamba run -n qgis4 python scripts/export_maps.py \
    --project /mnt/d/signatur/output_climate/QGIS/suitability.qgz \
    --layout suitability \
    --csv maps/geophysics_titles.csv \
    --data-root /mnt/d/GeoILUM/data/dk \
    --classify quantile \
    --number-from 73 \
    --out /mnt/d/signatur/output_climate/12_maps/geophysics
```
