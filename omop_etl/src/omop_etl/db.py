"""DuckDB connection, parquet writes and step logging shared by all steps."""

from __future__ import annotations

import logging
import os
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

log = logging.getLogger("omop_etl")


def _job_memory_bytes() -> int | None:
    """Memory of the SLURM allocation, or None outside a job."""
    if mb := os.environ.get("SLURM_MEM_PER_NODE"):
        return int(mb) * 2**20
    if (mb := os.environ.get("SLURM_MEM_PER_CPU")) and (cpus := os.environ.get("SLURM_CPUS_PER_TASK")):
        return int(mb) * int(cpus) * 2**20
    return None


MACROS: list[str] = []


def connect(cfg=None, memory_limit: str | None = None, threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory duckdb with memory limit (~75% of the SLURM job), threads and spill dir set."""
    con = duckdb.connect()
    for macro in MACROS:
        con.execute(macro)
    memory_limit = memory_limit or (cfg.memory_limit if cfg else None)
    if memory_limit is None and (job := _job_memory_bytes()):
        memory_limit = f"{int(job * 0.75 / 2**30)}GB"
    threads = threads or (cfg.threads if cfg else None) or os.environ.get("SLURM_CPUS_PER_TASK")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if cfg is not None:
        cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{cfg.tmp_dir}'")
    # row order is never part of the contract; this lets duckdb stream and parallelise writes
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET TimeZone = 'UTC'")
    log.info("duckdb %s: memory_limit=%s threads=%s", duckdb.__version__,
             con.sql("SELECT current_setting('memory_limit')").fetchone()[0],
             con.sql("SELECT current_setting('threads')").fetchone()[0])
    return con


def connect_plain() -> duckdb.DuckDBPyConnection:
    """A small connection for reading metadata (validation, tests)."""
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    for macro in MACROS:
        con.execute(macro)
    return con


def copy(con, sql: str, target: Path | str, row_group_size: int | None = None) -> int:
    """`COPY (sql) TO target` as zstd parquet, written to a temporary name and renamed.

    Returns the number of rows written.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.partial")
    opts = "FORMAT parquet, COMPRESSION zstd"
    if row_group_size:
        opts += f", ROW_GROUP_SIZE {int(row_group_size)}"
    t0 = time.monotonic()
    con.execute(f"COPY (\n{sql}\n) TO '{tmp}' ({opts})")
    os.replace(tmp, target)
    n = con.sql(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
    log.info("wrote %s: %s rows, %.1f MB, %.1f s", target.name, f"{n:,}",
             target.stat().st_size / 2**20, time.monotonic() - t0)
    return n


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


@contextmanager
def step(name: str):
    """Log wall time and peak memory of a block."""
    t0 = time.monotonic()
    log.info("%s ...", name)
    yield
    log.info("%s done in %.1f s (peak RSS so far %.1f GB)", name, time.monotonic() - t0, peak_rss_gb())


def setup_logging(log_file: Path | None = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)
