# tools

Checks of a CDM and of cohort builds, and the cohort chain as SLURM jobs. Every tool prints and writes aggregates
only: counts, shapes, distributions, never rows. The sbatch wrappers run the checkout that `../../env.sh` was sourced
from and take their partition and QOS from `../../config.env`.

| File | What |
|---|---|
| `fingerprint_cdm.py <cdm_dir> <out_dir>` (`run_fingerprint.sbatch`) | a duckdb pass over the CDM: per table and column counts, nulls, distributions, integrity, per-concept metrics (parquet, CSV, `meta.json`) |
| `compare_fingerprints.py <baseline_dir> <candidate_dir>` | a diff of two fingerprints, verdict first; exit 1 when anything differs beyond tolerance |
| `run_cohorts.sh <cdm_dir> <out_root> [datasets] [tasks]` (`run_cohort_cell.sbatch`) | submits the prefilters and the builds of `../../cohorts/` from a CDM into `<out_root>` |
| `compare_cohorts.py <base_root> <cand_root>` (`run_compare_cohorts.sbatch`) | per task and dataset of two builds: stays, rows, differing cells, label agreement |
| `make_vocabulary_fixture.py --vocabulary <athena_dir> --cdm <demo_cdm>` | rewrites `../tests/fixtures/vocabulary` (the concepts of the demo CDM and the mappings, names only for UCUM and OMOP's own vocabularies), which the demo test and `../../ci/demo.sh` read |

## Use

```bash
source env.sh                                   # at the repository root
python omop_etl/tools/fingerprint_cdm.py <out_dir>/cdm <out_dir>/_fingerprints/<name>
python omop_etl/tools/compare_fingerprints.py <out_dir>/_fingerprints/<earlier> <out_dir>/_fingerprints/<name>

omop_etl/tools/run_cohorts.sh <out_dir>/cdm <out_dir>/cohorts            # prefilters, then the 15 builds
sbatch -o <log> omop_etl/tools/run_compare_cohorts.sbatch <earlier>/frames <out_dir>/cohorts/frames
```

### `run_cohorts.sh`

Exports `OMOP_CDM=<cdm_dir>`, `COHORT_CACHE=<out_root>/cohort_cache` and `FRAMES_OUT=<out_root>/frames`, and copies
the modules of `cohorts/` and the sources of ehrdata and ehrapy into `<out_root>/_code/<stamp>/`, with `REVISIONS`,
which names each source's path and revision. The jobs run that copy, so edits made while they queue do not reach them.
ehrapy and ehrdata are whatever `import ehrapy` and `import ehrdata` resolve to: the environment's pins, or a checkout
first on `PYTHONPATH`. It submits one prefilter per dataset and one `run_cohort_cell.sbatch` per dataset and task with
`--dependency=afterok:<prefilter>`. `REUSE_CACHE=1` skips the prefilter of a dataset whose cache exists, `BUILD=0` the
builds. `<out_root>` must lie under `ICU_OMOP_DATA` if `../../config.env` sets it. Logs go to
`<out_root>/_logs/<name>_<jobid>.log`, the job list to `<out_root>/_logs/jobs_<stamp>.tsv`.

### `compare_cohorts.py`

Per (task, dataset) cell: stays in both builds or one, rows in both or one for sta (stay_id), dyn (stay_id, time) and
outc, duplicate join keys, and per column the joined cells whose null-ness differs or whose |a-b| > `--tol` (default
1e-6), with the largest difference; label agreement on the joined stays. A verdict per cell, IDENTICAL or DIFFERENT;
exit 1 when any cell differs. Counts under 10 are shown as `<10`.
