# omop_etl

MIMIC-IV 3.1, eICU-CRD 2.0 and HiRID 1.1.1 to one OMOP CDM v5.4 in parquet (one
`<TABLE>.parquet` per table), the input of `../cohorts/`. Plain DuckDB SQL from
Python, three steps:

| step | command | reads | writes |
|---|---|---|---|
| 1 convert | `python -m omop_etl convert --dataset <ds>` | source csv.gz / parquet | `<out>/raw/<ds>/<table>.parquet` |
| 2 stage | `python -m omop_etl stage --dataset <ds>` | raw parquet, `mappings/` | `<out>/staging/<ds>/*.parquet` (checked against the contract) |
| 3 cdm | `python -m omop_etl cdm` | every staged dataset, the Athena vocabulary | `<out>/cdm/<TABLE>.parquet` |

`<ds>` is `mimic4`, `eicu` or `hirid`. Steps 1-2 run per dataset and in parallel; step 3 once.
Each step only writes its own output directory and can be re-run.

## 1. Prerequisites

- PhysioNet credentialed access and the three downloads, unpacked as PhysioNet ships them:
  `mimiciv/3.1/{hosp,icu}/*.csv.gz`, `eicu-crd/2.0/*.csv.gz`, `hirid/1.1.1/` (the
  `raw_stage` observation and pharma parquet parts and the reference tables). For a first try
  without credentials: the open MIMIC-IV demo (`mimic-iv-demo/2.2`) and eICU demo (`eicu-crd-demo/2.0.1`), which
  `../ci/demo.sh` downloads and runs through the ETL and the cohorts. The eICU demo names two tables in lower case
  (`infusiondrug`, `admissiondrug`); the convert step finds them either way.
- An OMOP vocabulary download from [Athena](https://athena.ohdsi.org) (free account): select at
  least SNOMED, LOINC, RxNorm, UCUM, Gender, Visit, Type Concept, OMOP Extension (the defaults
  Domain, Vocabulary, Concept Class, Relationship come along). Unzip it into one directory; only
  `CONCEPT.csv` and `DOMAIN.csv` are read, at run time. Concepts missing from the download are
  listed as a warning by the cdm step and left out of CONCEPT; nothing else depends on them.
- [uv](https://docs.astral.sh/uv/) and a SLURM cluster (or any Linux machine with enough memory:
  see Timings).

## 2. Install

```bash
cd omop_etl
source ../env.sh     # builds/activates the environment of the repository (python 3.13, duckdb 1.5.3)
```

`env.sh` reads `../config.env` (paths, SLURM partition and QOS, where uv keeps the environment; see
`../config.example.env`), syncs the environment and activates it. Run it once per machine or node; it is quick
after that.

## 3. Configure

```bash
cp config.example.yaml config.local.yaml    # gitignored
```

Fill in `out_dir`, `vocabulary_dir` and `sources` (leave out datasets you do not have). Every
command takes `--config <file>`, default `config.local.yaml` or `$OMOP_ETL_CONFIG`.

## 4. Run

Everything as SLURM jobs (convert + stage per dataset in parallel, then cdm when all succeeded):

```bash
slurm/submit_all.sh                         # all three datasets
slurm/submit_all.sh mimic4                  # a subset
slurm/submit_all.sh --skip-convert eicu     # raw parquet already there
```

One step by hand, as a job or on a node you are logged into:

```bash
L=<out_dir>/_logs
sbatch -J stage_eicu -c 32 --mem=160G -o $L/%x_%j.out slurm/run.sbatch stage eicu
python -m omop_etl stage --dataset eicu     # on a node, after source ../env.sh
python -m omop_etl cdm --datasets mimic4,hirid
```

Memory: duckdb takes 75% of the SLURM allocation (`duckdb.memory_limit` in the config to
override) and spills to `<out_dir>/_tmp/<job id>/`, removed at the end of the step.

## 5. Output

```
<out_dir>/raw/<ds>/<table>.parquet       typed copies of the source columns used
<out_dir>/staging/<ds>/{stays,measurement,drug,procedure,specimen,condition}.parquet
<out_dir>/cdm/<TABLE>.parquet            the 39 CDM v5.4 tables (empty ones with their columns)
<out_dir>/_logs/<step>_<ds>_<job>.log    row counts, timings, peak memory per step
```

The staging tables are the contract between a dataset module and the cdm step
(`src/omop_etl/staging.py`): times are microseconds from the stay's ICU admission, one row per
event, values as recorded with the unit they were recorded in. The cdm step adds the ids and puts
every stay at its admission on the source's own clock (MIMIC-IV's and HiRID's shifted dates, eICU
placed within its discharge year). The CDM's conventions, and how it differs from the
BlendedICU-based one, are in `docs/DIFFERENCES.md`; the OHDSI/MIMIC comparison in
`docs/OHDSI-MIMIC.md`.

## 6. Timings (full data, 2026-10-27)

`slurm/submit_all.sh` from the source files to the finished CDM: 6 min 39 s wall (submission to
the end of the cdm job, with the resources it requests; the cluster started every job at once).

| step | mimic4 | eicu | hirid | resources requested |
|---|---|---|---|---|
| convert | 2 min 0 s, 3.4 GB | 2 min 45 s, 10.4 GB | 25 s, 0.8 GB | 16 cpus, 32 GB |
| stage | 23 s, 13.9 GB | 47 s, 3.4 GB | 42 s, 3.7 GB | 16 cpus, 48 GB |
| cdm (all three) | 2 min 10 s, 15.2 GB | | | 32 cpus, 96 GB |

Time and peak memory as each step logs them (`<out_dir>/_logs/<step>_<ds>_<job>.log`; job
overhead adds ~20 s). With 64-160 GB per job the cdm step took 1 min 44 s. Output sizes: `raw/`
6.9 GB (mimic4 1.9, eicu 2.0, hirid 3.0), `staging/` 5.6 GB (0.7, 2.5, 2.4), `cdm/` 8.9 GB
(MEASUREMENT 7.9 GB, 1.38 billion rows).

## 7. Checking the output

- `python -m omop_etl validate --dataset <ds>`: staging names, types and keys (also run by `stage`).
- `python -m pytest tests`: staging, timeline rules, units, and the whole pipeline on the MIMIC-IV
  demo (`tests/test_mimic4_demo.py`, the demo at `$MIMIC_IV_DEMO`). The demo test reads the vocabulary fixture
  `tests/fixtures/vocabulary`, or an Athena download at `$OMOP_VOCABULARY`; where the BlendedICU ETL's demo CDM
  exists, it also compares the rows and relative times with it. `tools/make_vocabulary_fixture.py` rewrites the
  fixture from an Athena download and a demo CDM when the ETL starts using other concepts.
- Fingerprint a CDM (aggregates only) and compare it with the fingerprint of an earlier build:
  ```bash
  python tools/fingerprint_cdm.py <out_dir>/cdm <out_dir>/_fingerprints/<name>
  python tools/compare_fingerprints.py <out_dir>/_fingerprints/<earlier> <out_dir>/_fingerprints/<name>
  ```
  `tools/README.md` has the details; `tools/compare_cohorts.py` compares two cohort builds.

## 8. Debugging a step

Use a config for the demo (`sources.mimic4` = the demo, its own `out_dir`; `../ci/demo.sh` writes one):
the demo is open data, so rows may be printed. On the full data print shapes and counts only.

- pdb: `python -m pdb -c continue -m omop_etl stage --dataset mimic4 --config <demo config>`
  stops at the first exception; or put `breakpoint()` in the code.
- VS Code: open `omop_etl/`, `source ../env.sh` in its terminal, select
  `../.venv/bin/python`, and use the launch configurations in `.vscode/launch.json` (stage/cdm on the
  demo, any step from a prompt, the tests).
- Look at an intermediate: every staged table is a parquet file; in the debugger
  `con.sql("SELECT ... FROM stay LIMIT 5").show()` works on the step's connection (`stay`,
  `stay_window`, `item_map` are tables during `mimic4.stage`; `stay`, `span`, `stg_*` during `cdm`).
- A query is one SQL string per table (`mimic4.measurement_sql()` etc.): print it and run parts of
  it in a duckdb shell against the raw parquet.

## 9. Layout

```
src/omop_etl/
  __main__.py   CLI                       config.py  yaml config, paths
  db.py         duckdb connection, parquet writes, step logging
  convert.py    step 1 helpers            staging.py the staging contract and its validator
  mimic4.py eicu.py hirid.py              one module per dataset: convert() and stage()
  timeline.py   the CDM timeline, md5 ids
  cdm.py        step 3                    schema.py  CDM v5.4 tables (schema/*.csv, OHDSI)
  vocabulary.py the Athena CONCEPT at run time
mappings/       item, unit, drug and condition maps (provenance in mappings/README.md)
slurm/          run.sbatch (one step), submit_all.sh (everything)
tests/  tools/  docs/DIFFERENCES.md
```

To map another source item: add a row to `mappings/<ds>/measurements.csv` (the source code, the
concept id and the unit concept id it is recorded in) and re-run `stage` and `cdm`. A variable the
cohorts convert needs its unit in `cohorts/yaib_omop_crosswalk.py` `UNIT_CONVERSIONS`.
