"""Command-line entry point.

    python -m signatur <step> [--config config.yaml] [--force]

Steps: lookup, rasterize, extract, plots, stats, all
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import check_inputs, load_config

EDA_STEPS = ["lookup", "rasterize", "extract", "plots", "stats"]
FUZZY_STEPS = ["model", "project"]
STEPS = EDA_STEPS + FUZZY_STEPS


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
    elif step == "model":
        from .model import build_envelopes
        build_envelopes(cfg, force=force)
    elif step == "project":
        from .project import project_fuzzy
        project_fuzzy(cfg, force=force)
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
    check_inputs(cfg)
    print(f"config     : {cfg.source_path}")
    print(f"output_dir : {cfg.output_dir}")
    if cfg.processing.test_bbox:
        print(f"TEST BBOX  : {cfg.processing.test_bbox}  (subset run)")

    steps = {"all": EDA_STEPS, "fuzzy": FUZZY_STEPS}.get(args.step, [args.step])
    for step in steps:
        _run_step(step, cfg, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
