"""Command-line entry point.

    python -m signatur <step> [--config config.yaml] [--force]

Steps: lookup, rasterize, extract, plots, stats, all
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import check_inputs, load_config

STEPS = ["lookup", "rasterize", "extract", "plots", "stats"]


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
    else:  # pragma: no cover
        raise ValueError(step)
    print(f"--- {step} done in {time.time() - t0:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signatur", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS + ["all"], help="pipeline step to run")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    check_inputs(cfg)
    print(f"config     : {cfg.source_path}")
    print(f"output_dir : {cfg.output_dir}")
    if cfg.processing.test_bbox:
        print(f"TEST BBOX  : {cfg.processing.test_bbox}  (subset run)")

    steps = STEPS if args.step == "all" else [args.step]
    for step in steps:
        _run_step(step, cfg, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
