"""Command-line entry point.

    python -m signatur <step> [--config config.yaml] [--force]

Steps: lookup, rasterize, extract, plots, stats, all
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import check_inputs, load_config

EDA_STEPS = ["lookup", "rasterize", "extract", "plots", "stats", "holdout"]
FUZZY_STEPS = ["model", "project", "fuzzyplots", "richness", "residual", "residualprofile"]
PREP_STEPS = ["climateprep"]
STEPS = PREP_STEPS + EDA_STEPS + FUZZY_STEPS


def _run_step(step: str, cfg, force: bool) -> None:
    t0 = time.time()
    print(f"\n=== {step} ===")
    if step == "lookup":
        from .lookup import build_lookup
        build_lookup(cfg, force=force)
    elif step == "rasterize":
        from .rasterize import build_merged_raster
        build_merged_raster(cfg, force=force)
    elif step == "extract":
        from .extract import build_training_store
        build_training_store(cfg, force=force)
    elif step == "plots":
        from .plots import make_all_plots
        make_all_plots(cfg, force=force)
    elif step == "stats":
        from .stats import make_stats
        make_stats(cfg, force=force)
    elif step == "holdout":
        from .holdout import run_holdout
        run_holdout(cfg, force=force)
    elif step == "model":
        from .model import build_envelopes
        build_envelopes(cfg, force=force)
    elif step == "project":
        from .project import project_fuzzy
        project_fuzzy(cfg, force=force)
    elif step == "fuzzyplots":
        from .fuzzy_plots import make_fuzzy_plots
        make_fuzzy_plots(cfg, force=force)
    elif step == "richness":
        from .richness import build_richness
        build_richness(cfg, force=force)
    elif step == "residual":
        from .residual import build_residual_category
        build_residual_category(cfg, force=force)
    elif step == "residualprofile":
        from .residual import profile_residual
        profile_residual(cfg, force=force)
    elif step == "climateprep":
        from .climate_prep import prepare_climate_pcs
        if not cfg.climate.source:
            raise ValueError("set climate.source in the config file (e.g. climate_pca.tif)")
        prepare_climate_pcs(cfg, cfg.climate.source, list(cfg.climate.out_names) or None,
                            cfg.climate.resampling, force=force)
    else:  # pragma: no cover
        raise ValueError(step)
    print(f"--- {step} done in {time.time() - t0:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signatur", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS + ["all", "fuzzy"],
                        help="a single step, 'all' (the 5 EDA steps), or 'fuzzy' (model + project)")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    steps = {"all": EDA_STEPS, "fuzzy": FUZZY_STEPS}.get(args.step, [args.step])

    # climateprep *creates* some of the pc_rasters entries, so the usual
    # existence check (which would fail on them) only applies once every
    # remaining step needs those files to already exist.
    if any(s not in PREP_STEPS for s in steps):
        check_inputs(cfg)
    print(f"config     : {cfg.source_path}")
    print(f"output_dir : {cfg.output_dir}")
    if cfg.processing.test_bbox:
        print(f"TEST BBOX  : {cfg.processing.test_bbox}  (subset run)")

    for step in steps:
        _run_step(step, cfg, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
