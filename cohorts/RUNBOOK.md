# RUNBOOK — cohorts

How to build YAIB frames from an OMOP CDM. See `README.md` for the layout.

The builder (`build_cohort.py`) computes YAIB's `dyn/sta/outc` frames from the CDM: the cohort, statics, labels and
dynamics, without ricu. YAIB trains on them with its own CLI, as on the cohorts it builds with ricu.

---

## 0. Prerequisites / environment

- **OMOP CDM input:** for example written by `../omop_etl` into `<out_dir>/cdm/`. Values are stored as recorded with
  their unit; times are each source's own clock (`../omop_etl/docs/DIFFERENCES.md`). MEASUREMENT of all three
  datasets is ~1.4 B rows.
- **Environment:** `source ../env.sh`, the environment of the whole repository (uv, python 3.13, duckdb 1.5.3,
  anndata 0.13.3). It reads `../config.env` (see `../config.example.env`), and every sbatch script sources it. It pins
  ehrdata (with the urine rate per minute and `person_visit_detail`) and ehrapy (with `ep.tl.sofa`, `ep.tl.sepsis3`,
  `ep.tl.kdigo` and `ep.tl.mark_events`) to commits on GitHub until their next releases. A local checkout goes first
  on `PYTHONPATH`; `../omop_etl/tools/run_cohorts.sh` copies whatever `import ehrapy` and `import ehrdata` resolve to
  into the jobs and names them in `_code/*/REVISIONS`.
- **SLURM:** the jobs take their partition and QOS from `SBATCH_PARTITION` and `SBATCH_QOS` in `../config.env`, and
  run the checkout `env.sh` was sourced from. Without `-o`, SLURM writes the log to `slurm-<job>.out` in the
  directory you submit from.
- **Data safety:** never commit patient data. Caches and frames live outside the repository (the `.gitignore` also
  blocks data formats).

---

## Which CDM

`OMOP_CDM` (environment, or `../config.env`) selects the CDM for the prefilter and the builder; without it they stop.
The cache and the frames go next to the CDM, into `<out_dir>/cohorts/{cohort_cache,frames}`; `COHORT_CACHE` and
`FRAMES_OUT` override them.

The whole chain for one CDM, as jobs, from a snapshot of the code taken at submission:
```bash
../omop_etl/tools/run_cohorts.sh <out_dir>/cdm <out_dir>/cohorts                    # prefilters, then builds
REUSE_CACHE=1 ../omop_etl/tools/run_cohorts.sh <cdm> <new cohorts dir>              # reuse a cache (symlink it)
EXCLUDE_NODES=<node> ../omop_etl/tools/run_cohorts.sh ...                           # keep jobs off a node
```

## 1. Pre-filter MEASUREMENT → per-dataset cache  (REQUIRED FIRST, once per dataset)

Cuts MEASUREMENT to the target concepts of one dataset. Every build below reads this cache.
```bash
source ../env.sh
OMOP_CDM=<cdm> sbatch -o <log> run_prefilter.sbatch <dataset>      # dataset ∈ miiv | eicu | hirid
OMOP_CDM=<cdm> python prefilter.py --dataset <dataset>             # or directly, on a machine with the memory
```

---

## 2. Build `dyn/sta/outc` frames

The builder extracts the 48 dynamic variables through ehrdata's OMOP IO (`ehrdata_dynamics.py`): one observation per
ICU stay, rows linked to their stay by `visit_occurrence_id`, the concepts of a variable pooled, units converted with
`UNIT_CONVERSIONS`, ricu's bounds per row, the hourly median (urine summed), sparse output.

```bash
OMOP_CDM=<cdm> sbatch -o <log> run_build_cohort.sbatch <dataset> <task> [duckmem=90GB]
# FRAMES_OUT=<dir> ...                   a check build into another directory
# COHORT_DIAG=1 ...                      also writes per-stay label ingredients (diag_*.parquet) for comparison with ricu
```
The cohort is defined on the stays' `.obs` with ehrapy: every base and task exclusion is a `filter_observations`
query tracked by one `CohortTracker` (`cohort_filters.py`). Each cell writes `attrition.csv` and
`attrition_flowchart.html` next to its frames.

---

## 3. Train with YAIB

The frames have the layout of YAIB's own cohorts, one directory per task and dataset:
```bash
icu-benchmarks train -d <cohorts>/frames/<task>/<dataset> -n <dataset> -t <YAIB task type> -m <model>
```

---

## Tasks

| task | type | YAIB gin (`-t`) |
|---|---|---|
| mortality24 | binary clf, static 24 h | BinaryClassification |
| kidney_function | regression, static 24 h | Regression |
| los | regression, dynamic 168 h | RegressionLoS |
| aki | binary clf, dynamic 168 h | BinaryClassification |
| sepsis | binary clf, dynamic 168 h | BinaryClassification |

---

## Key files

- `yaib_omop_crosswalk.py` — OMOP concepts of the 48 YAIB variables, ricu's bounds, the unit conversions, and which
  CDM is read.
- `ehrdata_dynamics.py` — the dynamics and label inputs through ehrdata's OMOP IO.
- `cohort_filters.py` — the exclusions as tracked ehrapy filters.
- `prefilter.py` — the pre-filter (stage 1).
- `build_cohort.py` — the builder (base cohort + labels + statics + dynamics).
- `label_inputs.py` — the label inputs ricu defines by rules (ventilation, vasopressors, GCS, hirid fluid balance,
  urine rate) and the antibiotic and culture rows of the suspected infection.
- `tests/ricu_labels.py`, `tests/sofa.py` — ricu's SOFA and Sepsis-3 rules re-implemented in Python, the references
  of the ehrapy equality tests; the builder uses `ep.tl.sofa` and `ep.tl.sepsis3`.

---

## Gotchas

- **Memory:** the ehrdata path aggregates only observed cells, so builds are much lighter than the dense path was
  (eicu los ~4 min instead of ~40). `../omop_etl/tools/run_cohorts.sh` still requests the old sizes (eicu up to 360G);
  they have not been re-measured.
- **Stage 1 first:** builds fail if `cohort_cache/measurement_<dataset>.parquet` is missing.
- **A unit without a conversion:** rows of a converted variable in an unlisted unit are left out of the label inputs
  (logged as `[units]`), and ehrdata raises for them in the dynamics. Add the unit to `UNIT_CONVERSIONS`.
- **The OMOP route vs ricu:** all 48 variables are mapped and the cohorts match ricu within a few dozen stays per
  cell; `../RICU.md` lists where the builder follows ricu and where it differs.
- Concurrent builds each get a **per-job duckdb temp dir** (a shared one corrupts spill files), under
  `DUCK_TMP_BASE`: by default `<out_dir>/cohorts/_duckdb_tmp` next to the frames, `<out_root>/_duckdb_tmp` in
  `run_cohorts.sh`.
