# cohorts — OMOP CDM → YAIB cohorts

Reads an OMOP CDM, such as the one `../omop_etl` writes, and builds YAIB's `dyn`, `sta` and `outc` frames for the five
YAIB tasks, using [ehrdata](https://github.com/theislab/ehrdata), [ehrapy](https://github.com/theislab/ehrapy) and
duckdb. [YAIB](https://github.com/rvandewater/YAIB) trains on these frames as on the cohorts it builds with ricu.
`../RICU.md` says where the builder follows ricu and where it differs.

**Running it:** [`RUNBOOK.md`](RUNBOOK.md).

## Layout

| Path | Purpose |
|---|---|
| `prefilter.py` | MEASUREMENT cut to one dataset's target concepts, the cache every build reads |
| `build_cohort.py` | the builder: cohort, labels, statics and dynamics of one task and dataset, computed from the CDM |
| `yaib_omop_crosswalk.py` | the YAIB↔OMOP crosswalk: concepts, units, ricu's bounds, which CDM is read |
| `ehrdata_dynamics.py` | the dynamics and label inputs through ehrdata's OMOP IO |
| `label_inputs.py`, `abx_sets.py` | the label inputs ricu defines by rules, and its antibiotic sets |
| `cohort_filters.py` | the exclusions as tracked ehrapy filters |
| `duck_tmp.py` | per-job duckdb spill directories that clean themselves up |
| `run_*.sbatch` | the jobs of the prefilter and the builder |
| `tests/` | unit tests on synthetic data (`pytest tests`) |
| `notebooks/` | from an OMOP CDM to ICU labels and a model |

## Environment

One environment serves the whole repository: `source ../env.sh` builds it from `../pyproject.toml` and `../uv.lock`
(python 3.13, duckdb 1.5.3, anndata 0.13.3, the versions the cohorts were validated with). It pins ehrdata and ehrapy
to commits on GitHub until their next releases carry what the builder needs: ehrdata's `person_visit_detail` and
interval fix, and ehrapy's `tl.sofa`, `tl.sepsis3`, `tl.kdigo` and `tl.mark_events`. A local checkout of either goes
first on `PYTHONPATH`.

## Data safety

Never commit patient or derived data. Caches and frames go next to the CDM (`<out_dir>/cohorts/`) or where
`COHORT_CACHE` and `FRAMES_OUT` say, outside the repository, and the `.gitignore` blocks data formats.
