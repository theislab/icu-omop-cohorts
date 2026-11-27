"""Run configuration: a small yaml file (see `config.example.yaml`)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # omop_etl/
MAPPINGS = PACKAGE_ROOT / "mappings"
SCHEMA_CSV = PACKAGE_ROOT / "schema" / "OMOP_CDMv5.4_Field_Level.csv"

DATASETS = ("mimic4", "eicu", "hirid")


@dataclass(frozen=True)
class Config:
    out_dir: Path
    vocabulary_dir: Path
    sources: dict[str, Path]
    datasets: tuple[str, ...] = DATASETS
    memory_limit: str | None = None  # duckdb memory_limit, e.g. "90GB"; default 75% of the job
    threads: int | None = None  # default: SLURM_CPUS_PER_TASK, else all cores
    options: dict = field(default_factory=dict)  # free-form per-dataset switches

    def raw_dir(self, dataset: str) -> Path:
        return self.out_dir / "raw" / dataset

    def staging_dir(self, dataset: str) -> Path:
        return self.out_dir / "staging" / dataset

    @property
    def cdm_dir(self) -> Path:
        return self.out_dir / "cdm"

    @property
    def log_dir(self) -> Path:
        return self.out_dir / "_logs"

    @property
    def tmp_dir(self) -> Path:
        """Per-job scratch (duckdb spill), under `<out>/_tmp/<SLURM job id or pid>`."""
        return self.out_dir / "_tmp" / os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")

    def source(self, dataset: str) -> Path:
        if dataset not in self.sources:
            raise KeyError(f"no source path for {dataset!r} in the config (sources: {list(self.sources)})")
        return self.sources[dataset]


def load(path: Path | str) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    base = path.resolve().parent

    def resolve(p) -> Path:
        p = Path(os.path.expandvars(str(p))).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    sources = {ds: resolve(p) for ds, p in (raw.get("sources") or {}).items() if p}
    unknown = set(sources) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown dataset(s) in sources: {sorted(unknown)}; known: {DATASETS}")
    datasets = tuple(raw.get("datasets") or [ds for ds in DATASETS if ds in sources])
    db_opts = raw.get("duckdb") or {}
    return Config(
        out_dir=resolve(raw["out_dir"]),
        vocabulary_dir=resolve(raw["vocabulary_dir"]),
        sources=sources,
        datasets=datasets,
        memory_limit=db_opts.get("memory_limit"),
        threads=db_opts.get("threads"),
        options=raw.get("options") or {},
    )
