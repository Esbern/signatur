"""Batch-export appendix maps from a QGIS layout, one PDF per raster.

Takes a QGIS project with a print layout (e.g. output_climate/QGIS/suitability.qgz)
and a CSV with the columns ``file,title``. For every row it swaps the raster
shown in the layout's maps for ``file``, applies a style (.qml), sets the map
title and exports the layout as a PDF. Everything else - page, Bornholm inset,
legend, scale bar, background map - comes from the layout, so all maps share
one design and the design is edited in QGIS, not here.

The project file is only read, never saved. Each map is one raster: the script
keeps the first raster layer shown in the layout's maps as the template layer
and drops other raster layers that no map shows. A broken layer path (e.g. a
background map saved on another machine) is repaired by looking for a file
with the same name under the project folder.

Denmark is drawn in light grey with a clear coastline: the fill colour of the
background layer's Denmark rule is set to ``--land-color``, and an outline-only
copy of Denmark (``--coast-color``, ``--coast-width``) is drawn ABOVE the raster
in every map, so suitable cells along the shore never hide the coastline. It is
not added to the legend. The style's transparent class (the values below the
coloured range) is drawn in ``--below-color`` instead, so unsuitable land reads
as grey Denmark rather than as a hole.

Title: a layout label with item ID ``title`` gets the CSV title. If the layout
has none, one is added in the top left; to control its position and font, add a
label with ID ``title`` to the layout in QGIS and save the project.

Run with the QGIS Python of the SAME major version the project was saved with
(a QGIS 4 project misrenders in QGIS 3): conda env ``qgis4`` (QGIS 4.2) or
``qgis`` (3.44), or the OSGeo4W shell on Windows:

    mamba run -n qgis4 python scripts/export_maps.py \\
        --project /mnt/d/signatur/output_climate/QGIS/suitability.qgz \\
        --layout suitability \\
        --csv maps/suitability_titles.csv \\
        --style /mnt/d/signatur/output_climate/QGIS/suitability_layer_style.qml \\
        --out /mnt/d/signatur/output_climate/12_maps/suitability \\
        --combined suitability_appendix.pdf

Output files are numbered in CSV row order - 001_<file>.pdf, 002_... - so the
folder sorts (and merges) in report order. --number-from lets a later batch
(e.g. the PC maps) continue the numbering; the number follows the CSV row, so
it stays the same when only some rows are exported with --only.

Multi-band rasters and data-driven classes (e.g. the climate inputs and PCs):
optional CSV columns ``band`` (1-based band of ``file``) and ``legend`` (legend
heading for that row). With ``--classify quantile`` each band gets its own
``--classes`` equal-count classes, computed from its valid cells, coloured
with the ColorBrewer Spectral ramp (low = blue, high = red; ``--invert-ramp``
flips it) and labelled with as many decimals as the band needs. The legend
heading may use ``{unit}``, read from the band's ``units`` tag.

CSV ``file`` paths are relative to ``--data-root`` (default: the folder above
the project folder, i.e. output_climate). Delimiter , or ; and UTF-8 with or
without BOM, so the file can be edited in Excel.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import qgis  # noqa: F401
except ImportError:  # conda: the bindings live under <prefix>/share/qgis/python
    sys.path.insert(0, os.path.join(sys.prefix, "share", "qgis", "python"))

from qgis.core import (  # noqa: E402
    Qgis,
    QgsApplication,
    QgsFillSymbol,
    QgsRuleBasedRenderer,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
    QgsColorRampShader,
    QgsDataProvider,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsGradientColorRamp,
    QgsGradientStop,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsTextFormat,
)
from qgis.PyQt.QtGui import QColor, QFont  # noqa: E402

# Works with QGIS 3.3x (Qt5) and QGIS 4 (Qt6): QGIS 4 dropped the old unscoped
# enum names, so only the scoped ones are used.
MM = Qgis.LayoutUnit.Millimeters
POINTS = Qgis.RenderUnit.Points
EXPORT_OK = getattr(getattr(QgsLayoutExporter, "ExportResult", QgsLayoutExporter), "Success")

# ColorBrewer Spectral (10 classes), high -> low; reversed below so low = blue
SPECTRAL = ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
            "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"]
UNIT_DISPLAY = {"degC": "°C", "mm/day": "mm/døgn", "W/m2": "W/m²", "m/s": "m/s"}

TITLE_ID = "title"
# Default title box, used only when the layout has no label with ID "title".
# Top left, clear of the Bornholm inset (which starts ~150 mm from the left).
TITLE_POS_MM = (8.0, 8.0)
TITLE_SIZE_MM = (135.0, 16.0)
TITLE_FONT_PT = 14


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;")
    rows = [r for r in csv.DictReader(text.splitlines(), dialect=dialect)]
    if not rows or not {"file", "title"} <= set(rows[0]):
        raise SystemExit(f"{path}: needs the columns 'file' and 'title'")
    return [{k: (v or "").strip() for k, v in r.items()} for r in rows if (r.get("file") or "").strip()]


def repair_broken_layers(project: QgsProject, search_root: Path) -> None:
    """Point invalid layers at a file with the same name under search_root."""
    for layer in project.mapLayers().values():
        if layer.isValid() or not layer.source():
            continue
        src = layer.source()
        path_part, sep, rest = src.partition("|")
        name = Path(path_part.replace("\\", "/")).name
        hits = sorted(search_root.rglob(name))
        if not hits:
            print(f"  ! {layer.name()}: {path_part} not found, left broken")
            continue
        layer.setDataSource(str(hits[0]) + sep + rest, layer.name(), layer.providerType(),
                            QgsDataProvider.ProviderOptions())
        print(f"  repaired {layer.name()} -> {hits[0]}")


def template_layer(project: QgsProject, maps: list[QgsLayoutItemMap]) -> QgsRasterLayer:
    shown = [lyr for m in maps for lyr in m.layers()]
    rasters = [lyr for lyr in shown if isinstance(lyr, QgsRasterLayer)]
    if not rasters:
        raise SystemExit("no raster layer is shown in the layout's maps")
    keep = rasters[0]
    unused = [lyr.id() for lyr in project.mapLayers().values()
              if isinstance(lyr, QgsRasterLayer) and lyr not in shown]
    project.removeMapLayers(unused)
    for m in maps:  # every map shows the template raster instead of any other raster
        m.setLayers([keep if isinstance(lyr, QgsRasterLayer) else lyr for lyr in m.layers()])
    return keep


def style_land_and_coast(project: QgsProject, maps: list[QgsLayoutItemMap], land_filter: str,
                         land_color: str, coast_color: str, coast_width: float) -> None:
    """Grey fill for Denmark in the background layer + a coastline drawn on top of everything."""
    shown = [lyr for m in maps for lyr in m.layers()]
    background = next((lyr for lyr in shown if isinstance(lyr, QgsVectorLayer)), None)
    if background is None:
        print("  ! no vector background layer in the maps - land/coast styling skipped")
        return
    renderer = background.renderer()
    target = land_filter.replace(" ", "")
    hits = 0
    if isinstance(renderer, QgsRuleBasedRenderer):
        for rule in renderer.rootRule().descendants():
            if rule.filterExpression().replace(" ", "") == target and rule.symbol() is not None:
                rule.symbol().setColor(QColor(land_color))
                hits += 1
    if not hits:
        print(f"  ! no rule '{land_filter}' in {background.name()} - land colour left unchanged")
    background.triggerRepaint()

    coast = QgsVectorLayer(background.source(), "coastline", background.providerType())
    coast.setSubsetString(land_filter)
    coast.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple({
        "style": "no", "outline_style": "solid",
        "outline_color": coast_color, "outline_width": str(coast_width), "outline_width_unit": "MM",
    })))
    project.addMapLayer(coast, False)  # not in the layer tree, so never in the legend
    for m in maps:
        m.setLayers([coast] + list(m.layers()))  # first in the list = drawn on top


def fill_transparent_classes(raster: QgsRasterLayer, color: str) -> None:
    """Give the style's fully transparent colour-ramp classes an opaque colour."""
    renderer = raster.renderer()
    shader = getattr(renderer, "shader", lambda: None)()
    ramp = shader.rasterShaderFunction() if shader is not None else None
    if not isinstance(ramp, QgsColorRampShader):
        return
    items = ramp.colorRampItemList()
    for item in items:
        if item.color.alpha() == 0:
            item.color = QColor(color)
    ramp.setColorRampItemList(items)


def band_info(path: Path, band: int) -> tuple[str, str]:
    """(description, display unit) of a raster band, from GDAL metadata."""
    from osgeo import gdal
    ds = gdal.Open(str(path))
    b = ds.GetRasterBand(band)
    desc = b.GetDescription() or f"b{band}"
    unit = (b.GetMetadataItem("units") or b.GetUnitType() or "").strip()
    return desc, UNIT_DISPLAY.get(unit, unit)


def quantile_classes(path: Path, band: int, n_classes: int, max_cells: int = 4_000_000) -> list[tuple[float, float]]:
    """Equal-count classes [(lower, upper), ...] over the band's valid cells (min..max)."""
    import numpy as np
    from osgeo import gdal
    ds = gdal.Open(str(path))
    b = ds.GetRasterBand(band)
    step = max(1, int((ds.RasterXSize * ds.RasterYSize / max_cells) ** 0.5))  # decimate big rasters
    a = b.ReadAsArray(buf_xsize=ds.RasterXSize // step, buf_ysize=ds.RasterYSize // step).astype("float64")
    nd = b.GetNoDataValue()
    valid = np.isfinite(a) if nd is None or np.isnan(nd) else np.isfinite(a) & (a != nd)
    v = a[valid]
    lo = float(v.min())

    def split(values, k):
        q = sorted(set(np.quantile(values, np.linspace(0, 1, k + 1)).tolist()))
        return list(zip(q[:-1], q[1:]))

    if np.mean(v == lo) > 1.0 / n_classes:
        # one value (typically 0, e.g. CaCO3) fills more than a class: it gets its own
        # class, and the remaining cells are split into n_classes - 1 quantile classes
        return [(lo, lo)] + split(v[v > lo], n_classes - 1)
    return split(v, n_classes)


def class_label_format(classes: list[tuple[float, float]], comma: bool):
    """Formatter whose decimals resolve the narrowest class (and keep all breaks distinct);
    true minus sign, decimal comma by default."""
    import math
    uppers = [hi for _, hi in classes]  # the class limits that must stay distinct
    widths = [hi - lo for lo, hi in classes if hi > lo]
    d = max(0, -math.floor(math.log10(min(widths)))) if widths else 0
    while d < 6 and len({round(x, d) for x in uppers}) < len(set(uppers)):
        d += 1

    def fmt(x: float) -> str:
        text = f"{x:.{d}f}".replace("-", "\u2212")
        return text.replace(".", ",") if comma else text
    return fmt


def quantile_renderer(raster: QgsRasterLayer, band: int, classes: list[tuple[float, float]], invert: bool, comma: bool):
    colors = SPECTRAL if invert else SPECTRAL[::-1]  # default: low values blue, high red
    ramp = QgsGradientColorRamp(QColor(colors[0]), QColor(colors[-1]), False,
                                [QgsGradientStop(i / (len(colors) - 1), QColor(c))
                                 for i, c in enumerate(colors[1:-1], 1)])
    fmt = class_label_format(classes, comma)
    n = len(classes)
    items = []
    for i, (lo, hi) in enumerate(classes):
        upper = float("inf") if i == n - 1 else hi  # discrete: a cell takes the first class with value <= upper
        color = ramp.color(i / (n - 1) if n > 1 else 0.5)
        if lo == hi:  # single-value class, e.g. "0": no trailing decimals
            label = (f"{lo:g}".replace("-", "\u2212").replace(".", "," if comma else "."))
        else:
            label = f"{fmt(lo)} – {fmt(hi)}"
        items.append(QgsColorRampShader.ColorRampItem(upper, color, label))
    breaks = [classes[0][0], classes[-1][1]]
    fn = QgsColorRampShader(breaks[0], breaks[-1])
    fn.setColorRampType(Qgis.ShaderInterpolationMethod.Discrete)
    fn.setColorRampItemList(items)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(fn)
    renderer = QgsSingleBandPseudoColorRenderer(raster.dataProvider(), band, shader)
    renderer.setClassificationMin(breaks[0])
    renderer.setClassificationMax(breaks[-1])
    raster.setRenderer(renderer)


def title_item(layout) -> QgsLayoutItemLabel:
    for item in layout.items():  # match the ID loosely: "title " typed in QGIS is common
        if isinstance(item, QgsLayoutItemLabel) and item.id().strip().lower() == TITLE_ID:
            return item
    label = QgsLayoutItemLabel(layout)
    label.setId(TITLE_ID)
    fmt = QgsTextFormat()
    font = QFont("Arial")
    font.setBold(True)
    fmt.setFont(font)
    fmt.setSize(TITLE_FONT_PT)
    fmt.setSizeUnit(POINTS)
    label.setTextFormat(fmt)
    label.setBackgroundEnabled(True)
    label.setBackgroundColor(QColor(255, 255, 255, 220))
    label.setMarginX(2)
    label.setMarginY(1.5)
    label.attemptMove(QgsLayoutPoint(*TITLE_POS_MM, MM))
    label.attemptResize(QgsLayoutSize(*TITLE_SIZE_MM, MM))
    label.setZValue(100)
    layout.addLayoutItem(label)
    return label


def combine_pdfs(paths: list[Path], target: Path) -> bool:
    try:
        from pypdf import PdfWriter
    except ImportError:
        print("  (pypdf not installed - combined PDF skipped)")
        return False
    writer = PdfWriter()
    for p in paths:
        writer.append(str(p))
    with open(target, "wb") as f:
        writer.write(f)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, type=Path, help=".qgz/.qgs with the print layout")
    ap.add_argument("--layout", required=True, help="name of the print layout")
    ap.add_argument("--csv", required=True, type=Path, help="CSV with columns file,title")
    ap.add_argument("--style", type=Path, help=".qml applied to every raster (default: keep the layer's style)")
    ap.add_argument("--out", required=True, type=Path, help="output folder for the PDFs")
    ap.add_argument("--data-root", type=Path, help="folder the CSV paths are relative to "
                                                   "(default: the project folder's parent)")
    ap.add_argument("--legend-label", default="Egnethed",
                    help="legend heading; may use {code} (file stem), {title}, {unit} (band units tag), "
                         "{band}; a CSV 'legend' column overrides it per row")
    ap.add_argument("--classify", choices=["style", "quantile"], default="style",
                    help="style: use --style as is; quantile: equal-count classes per band")
    ap.add_argument("--classes", type=int, default=10, help="number of quantile classes")
    ap.add_argument("--invert-ramp", action="store_true", help="quantile ramp: low = red, high = blue")
    ap.add_argument("--decimal-point", action="store_true", help="legend numbers with . instead of ,")
    ap.add_argument("--only", help="export only rows whose file contains this text (testing)")
    ap.add_argument("--combined", help="also write all maps into one PDF with this file name")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--number-from", type=int, default=1, help="number of the first CSV row (file prefix 001_ ...)")
    ap.add_argument("--no-numbers", action="store_true", help="no number prefix on the file names")
    ap.add_argument("--land-filter", default="\"ISO3_CODE\" = 'DNK'",
                    help="background-layer rule (and coastline subset) for Denmark")
    ap.add_argument("--land-color", default="#dcdcdc", help="fill colour for Denmark")
    ap.add_argument("--below-color", default="#dcdcdc",
                    help="colour for the style's transparent class(es), i.e. unsuitable land; 'none' keeps them transparent")
    ap.add_argument("--coast-color", default="#3c3c3c", help="coastline colour")
    ap.add_argument("--coast-width", type=float, default=0.25, help="coastline width in mm")
    args = ap.parse_args(argv)

    rows = read_rows(args.csv)
    for i, r in enumerate(rows):
        r["number"] = args.number_from + i
    if args.only:
        rows = [r for r in rows if args.only in r["file"] or args.only in r["title"]]
    data_root = (args.data_root or args.project.resolve().parent.parent).resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", sys.prefix), True)
    app = QgsApplication([], False)
    app.initQgis()
    try:
        project = QgsProject.instance()
        if not project.read(str(args.project)):
            raise SystemExit(f"could not read {args.project}")
        repair_broken_layers(project, args.project.resolve().parent)
        layout = project.layoutManager().layoutByName(args.layout)
        if layout is None:
            names = [lay.name() for lay in project.layoutManager().layouts()]
            raise SystemExit(f"layout '{args.layout}' not found; layouts: {names}")
        maps = [i for i in layout.items() if isinstance(i, QgsLayoutItemMap)]
        raster = template_layer(project, maps)
        style_land_and_coast(project, maps, args.land_filter, args.land_color,
                             args.coast_color, args.coast_width)
        title = title_item(layout)
        legends = [i for i in layout.items() if isinstance(i, QgsLayoutItemLegend)]

        done, failed = [], []
        for n, row in enumerate(rows, 1):
            src = (data_root / row["file"]).resolve()
            band = int(row.get("band") or 1)
            desc, unit = band_info(src, band) if src.exists() else ("", "")
            prefix = "" if args.no_numbers else f"{row['number']:03d}_"
            name = Path(row["file"]).stem
            if row.get("band"):  # several maps from one file: add the band name
                name += "_" + "".join(c if c.isalnum() else "_" for c in desc).strip("_")
            pdf = args.out / f"{prefix}{name}.pdf"
            if not src.exists():
                print(f"[{n}/{len(rows)}] MISSING {src}")
                failed.append(row["file"])
                continue
            raster.setDataSource(str(src), args.legend_label, "gdal", QgsDataProvider.ProviderOptions())
            if args.classify == "quantile":
                quantile_renderer(raster, band, quantile_classes(src, band, args.classes),
                                  args.invert_ramp, not args.decimal_point)
            else:
                if args.style:
                    msg, ok = raster.loadNamedStyle(str(args.style))
                    if not ok:
                        raise SystemExit(f"style {args.style}: {msg}")
                if band != 1 and hasattr(raster.renderer(), "setInputBand"):
                    raster.renderer().setInputBand(band)
                if args.below_color.lower() != "none":
                    fill_transparent_classes(raster, args.below_color)
            label = (row.get("legend") or args.legend_label).format(
                code=Path(row["file"]).stem, title=row["title"], unit=unit, band=desc)
            raster.setName(label)
            title.setText(row["title"])
            for legend in legends:
                node = legend.model().rootGroup().findLayer(raster.id())
                if node is not None:  # a label typed in the layout editor overrides the layer name
                    node.setName(label)
                    node.setCustomProperty("legend/title-label", label)
                legend.setResizeToContents(True)  # no clipped class labels
                legend.updateLegend()
                legend.adjustBoxSize()
            settings = QgsLayoutExporter.PdfExportSettings()
            settings.dpi = args.dpi
            result = QgsLayoutExporter(layout).exportToPdf(str(pdf), settings)
            if result == EXPORT_OK:
                print(f"[{n}/{len(rows)}] {pdf.name}  ({row['title']})", flush=True)
                done.append(pdf)
            else:
                print(f"[{n}/{len(rows)}] FAILED {pdf.name} (exporter result {result})", flush=True)
                failed.append(row["file"])

        if args.combined and done and combine_pdfs(done, args.out / args.combined):
            print(f"combined: {args.out / args.combined}")
        print(f"{len(done)} maps written to {args.out}" + (f"; {len(failed)} failed: {failed}" if failed else ""))
        return 1 if failed else 0
    finally:
        app.exitQgis()


if __name__ == "__main__":
    sys.exit(main())
