"""Command line: python -m omop_etl <step> [--dataset ds] [--config config.local.yaml]

  convert   source files -> <out>/raw/<ds>/*.parquet           (per dataset)
  stage     raw parquet + mappings -> <out>/staging/<ds>/       (per dataset, validated)
  validate  check <out>/staging/<ds>/ against the contract      (per dataset)
  cdm       all staging dirs + vocabulary -> <out>/cdm/<TABLE>.parquet
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path

from omop_etl import config, db, staging

log = logging.getLogger("omop_etl")

DEFAULT_CONFIG = config.PACKAGE_ROOT / "config.local.yaml"


def dataset_module(dataset: str):
    if dataset not in config.DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; one of {config.DATASETS}")
    return importlib.import_module(f"omop_etl.{dataset}")


def run(step: str, cfg: config.Config, dataset: str | None, datasets: list[str] | None) -> None:
    job = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    db.setup_logging(cfg.log_dir / f"{step}{'_' + dataset if dataset else ''}_{job}.log")
    log.info("omop_etl %s %s, out %s", step, dataset or "", cfg.out_dir)
    con = db.connect(cfg)
    try:
        with db.step(f"{step} {dataset or ''}".strip()):
            if step == "convert":
                dataset_module(dataset).convert(cfg, con)
            elif step == "stage":
                dataset_module(dataset).stage(cfg, con)
                counts = staging.validate(cfg.staging_dir(dataset), con)
                log.info("staging %s valid: %s", dataset, counts)
            elif step == "validate":
                counts = staging.validate(cfg.staging_dir(dataset), con)
                log.info("staging %s valid: %s", dataset, counts)
            elif step == "cdm":
                from omop_etl import cdm
                cdm.build(cfg, con, datasets)
    finally:
        con.close()
        shutil.rmtree(cfg.tmp_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m omop_etl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["convert", "stage", "validate", "cdm"])
    p.add_argument("--dataset", help="mimic4, eicu or hirid (convert, stage, validate)")
    p.add_argument("--datasets", help="cdm: comma-separated subset (default: every staged dataset)")
    p.add_argument("--config", default=os.environ.get("OMOP_ETL_CONFIG", DEFAULT_CONFIG), type=Path,
                   help=f"yaml config (default $OMOP_ETL_CONFIG or {DEFAULT_CONFIG.name})")
    args = p.parse_args(argv)
    if args.step != "cdm" and not args.dataset:
        p.error(f"{args.step} needs --dataset")
    cfg = config.load(args.config)
    datasets = args.datasets.split(",") if args.datasets else None
    run(args.step, cfg, args.dataset, datasets)


if __name__ == "__main__":
    sys.exit(main())
