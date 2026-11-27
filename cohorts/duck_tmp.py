"""Per-job DuckDB spill directories that clean themselves up.

DuckDB moves intermediate results to disk ("spills") when a query needs more memory than its limit.
Every job gets its own spill directory, since jobs sharing one corrupt each other's files. A job
killed for running out of memory cannot clean up, and its directory stays behind; such leftovers
once held 93 GB. Two guards therefore:
  * `make_job_tmp()` registers an `atexit` hook, which removes the directory on a normal exit or
    an exception;
  * the sbatch scripts remove `${SLURM_JOB_ID}_*` in an `EXIT` trap, which runs after the kill too.

Set `KEEP_DUCK_TMP=1` to keep the directory for debugging.
"""

from __future__ import annotations

import atexit
import os
import shutil
import time


def job_id() -> str:
    """SLURM job id when running as a job, else the pid. Used to keep spill dirs disjoint."""
    return os.environ.get("SLURM_JOB_ID") or str(os.getpid())


def warn_stale(base: str, max_age_h: float = 24.0) -> None:
    """Report spill dirs from other, older jobs -- they are almost always abandoned leaks.

    Only names and ages are printed, never contents. `du` is deliberately not called: these
    directories can hold tens of GB and walking them would slow every job start.
    """
    if not os.path.isdir(base):
        return
    mine, now, stale = job_id(), time.time(), []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path) or name.startswith(mine + "_") or name == mine:
            continue
        age_h = (now - os.path.getmtime(path)) / 3600.0
        if age_h > max_age_h and os.listdir(path):
            stale.append((name, age_h))
    if stale:
        print(f"[duck_tmp] {len(stale)} stale spill dir(s) in {base} -- likely leaked by killed "
              f"jobs; check and remove:", flush=True)
        for name, age_h in stale[:10]:
            print(f"[duck_tmp]   {name}  ({age_h / 24.0:.1f} days old)", flush=True)


def make_job_tmp(base: str, *parts: object) -> str:
    """Create `{base}/{job_id}_{parts...}` and arrange for its removal at exit.

    Per-job rather than shared on purpose: concurrent DuckDB processes pointed at one
    `temp_directory` corrupt each other's spill files (noted in the repo RUNBOOK).
    """
    tag = "_".join([job_id(), *(str(p) for p in parts if p not in (None, ""))])
    path = os.path.join(base, tag)
    os.makedirs(path, exist_ok=True)
    warn_stale(base)

    def _cleanup() -> None:
        if os.environ.get("KEEP_DUCK_TMP"):
            print(f"[duck_tmp] KEEP_DUCK_TMP set, retaining {path}", flush=True)
            return
        shutil.rmtree(path, ignore_errors=True)

    atexit.register(_cleanup)
    return path
