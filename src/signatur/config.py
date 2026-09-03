"""Load and validate ``config.yaml``.

The config file is the single place that changes between machines. Everything
downstream takes a :class:`Config` instance rather than reading paths itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HabitatLayer:
    name: str
    file: str
    priority: int


@dataclass(frozen=True)
class AnalysisCfg:
    min_pixels_per_class: int = 50
    plot_sample_per_class: int = 15000
    winsor_z: float = 5.0
    random_seed: int = 42


@dataclass(frozen=True)
class ProcessingCfg:
    block_size: int = 2048
    test_bbox: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class FuzzyCfg:
    min_cells_for_envelope: int = 150
    covariance: str = "ledoit_wolf"
    per_class_rasters: bool = True
    compression: str = "DEFLATE"
    batch_px: int = 300000


@dataclass(frozen=True)
class Config:
    crs: str
    pc_raster_dir: Path
    vector_dir: Path
    output_dir: Path
    reference_raster: str
    nodata: float
    pc_rasters: dict[str, str]
    habitat_layers: list[HabitatLayer]
    naturkode_field: str
    all_touched: bool
    lookup_seed: Path
    analysis: AnalysisCfg
    processing: ProcessingCfg
    fuzzy: FuzzyCfg
    source_path: Path = field(default=REPO_ROOT / "config.yaml")

    # -- derived helpers ---------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """PC feature names in canonical order (dict insertion order)."""
        return list(self.pc_rasters)

    def pc_path(self, name: str) -> Path:
        return self.pc_raster_dir / self.pc_rasters[name]

    @property
    def pc_paths(self) -> list[Path]:
        return [self.pc_path(n) for n in self.feature_names]

    @property
    def reference_path(self) -> Path:
        return self.pc_raster_dir / self.reference_raster

    def layer_path(self, layer: HabitatLayer) -> Path:
        return self.vector_dir / layer.file

    @property
    def layers_by_priority(self) -> list[HabitatLayer]:
        """High priority first (priority 1 before priority 3)."""
        return sorted(self.habitat_layers, key=lambda ly: ly.priority)

    # output sub-directories ---------------------------------------------

    def out(self, *parts: str) -> Path:
        p = self.output_dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def dir_lookup(self) -> Path:
        return self.out("00_lookup")

    @property
    def dir_merged(self) -> Path:
        return self.out("01_merged_raster")

    @property
    def dir_training(self) -> Path:
        return self.out("02_training")

    @property
    def dir_plots(self) -> Path:
        return self.out("03_plots")

    @property
    def dir_stats(self) -> Path:
        return self.out("04_stats")

    @property
    def dir_model(self) -> Path:
        return self.out("05_model")

    @property
    def dir_fuzzy(self) -> Path:
        return self.out("06_fuzzy")


def _resolve(base: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path).expanduser().resolve() if path else (REPO_ROOT / "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text())

    paths = raw["paths"]
    analysis = raw.get("analysis", {})
    processing = raw.get("processing", {})
    fuzzy = raw.get("fuzzy", {})

    test_bbox = processing.get("test_bbox")
    if test_bbox is not None:
        test_bbox = tuple(float(v) for v in test_bbox)
        if len(test_bbox) != 4:
            raise ValueError("processing.test_bbox must be [xmin, ymin, xmax, ymax]")

    cfg = Config(
        crs=str(raw["crs"]),
        pc_raster_dir=_resolve(REPO_ROOT, paths["pc_raster_dir"]),
        vector_dir=_resolve(REPO_ROOT, paths["vector_dir"]),
        output_dir=_resolve(REPO_ROOT, paths["output_dir"]),
        reference_raster=str(raw["reference_raster"]),
        nodata=float(raw["nodata"]),
        pc_rasters={str(k): str(v) for k, v in raw["pc_rasters"].items()},
        habitat_layers=[HabitatLayer(**ly) for ly in raw["habitat_layers"]],
        naturkode_field=str(raw.get("naturkode_field", "naturkode")),
        all_touched=bool(raw.get("all_touched", False)),
        lookup_seed=_resolve(REPO_ROOT, raw["lookup_seed"]),
        analysis=AnalysisCfg(**analysis),
        processing=ProcessingCfg(block_size=int(processing.get("block_size", 2048)), test_bbox=test_bbox),
        fuzzy=FuzzyCfg(**fuzzy),
        source_path=cfg_path,
    )
    return cfg


def check_inputs(cfg: Config) -> None:
    """Fail fast with a readable message if any required input is missing."""
    missing: list[str] = []

    if not cfg.pc_raster_dir.is_dir():
        missing.append(f"pc_raster_dir: {cfg.pc_raster_dir}")
    if not cfg.vector_dir.is_dir():
        missing.append(f"vector_dir: {cfg.vector_dir}")
    if not cfg.reference_path.exists():
        missing.append(f"reference_raster: {cfg.reference_path}")
    for name, p in zip(cfg.feature_names, cfg.pc_paths):
        if not p.exists():
            missing.append(f"pc_raster '{name}': {p}")
    for layer in cfg.habitat_layers:
        p = cfg.layer_path(layer)
        if not p.exists():
            missing.append(f"habitat_layer '{layer.name}': {p}")
    if not cfg.lookup_seed.exists():
        missing.append(f"lookup_seed: {cfg.lookup_seed}")

    if missing:
        raise FileNotFoundError(
            "Missing required inputs (edit config.yaml):\n  - " + "\n  - ".join(missing)
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
