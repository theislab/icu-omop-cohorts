"""Fingerprint an OMOP CDM parquet directory with aggregates only.

    python fingerprint_cdm.py <cdm_dir> <out_dir> [--partition-rows N] [--overwrite]

Writes to out_dir (parquet, plus a CSV copy of each):

    tables      per table: files, bytes, rows, columns, kind; vocabulary tables get a key hash
    columns     per column: type, null count, distinct count for surrogate-id columns
    integrity   per table x dataset: row count and join-integrity / consistency checks
    groups      event tables per dataset x concept_id x source value (x unit): counts, value
                quantiles, distinct persons/visits, event time relative to the row's visit
    metrics     long format (table, dataset, metric, key, value, kind): person / visit / death /
                care site / observation period summaries, categorical column counts, sig keys
    meta.json   inputs, versions, settings, per-step wall times

Only counts and distributions are written; no row of the CDM is copied. Surrogate id values
(measurement_id, person_id, ...) differ by design between ETLs and are never written or compared;
only their counts, uniqueness and join integrity are.

How the source dataset of a row is identified
---------------------------------------------
The CDM has no dataset column. The old ETL (`etl/blended_preprocessing/omop_conversion.py`)
encodes the dataset in two source values, and the consumer (`cohorts/`, `DATASET_PREFIX` in
`cohorts/yaib_omop_crosswalk.py`) selects a dataset with `visit_source_value LIKE 'mimic4-%'`:

    VISIT_OCCURRENCE.visit_source_value   = '{dataset}-{stay_id}'       e.g. 'eicu-141765'
    PERSON.person_source_value            = '{dataset}-{patient_id}'    e.g. 'hirid-1234'

with {dataset} in {mimic4, eicu, hirid}. The prefix before the first '-' or '_' is taken and
normalised to the project's names (mimic4/miiv/mimic/mimiciv -> miiv, eicu, hirid); the raw
prefixes are kept in `metrics` (metric 'source_prefix'), so a change of convention shows up there
even though the normalised groups still line up. A row of any other table gets the dataset of its
visit (`visit_occurrence_id` -> VISIT_OCCURRENCE), else of its person (`person_id` -> PERSON), else
'<unresolved>'. CARE_SITE rows get the dataset of the visits that use them ('<mixed>' if several,
'<unused>' if none). person_id ranges and CARE_SITE do not identify the dataset: ids are md5
hashes, and LOCATION maps eicu and mimic4 to the same country. Both expressions can be replaced
with --visit-dataset-sql / --person-dataset-sql if a new ETL encodes it differently.

Tables without visit_occurrence_id (SPECIMEN, DEATH) are attributed to the person's latest visit
starting at or before the event (ASOF join), else the person's first visit. That is how the
consumer resolves SPECIMEN, so the relative-time stats of those tables say whether the timeline
puts each event next to the right stay.

Memory and speed
----------------
MEASUREMENT is ~1.4 B rows. Exact quantiles keep every value of a group in memory, so the grouped
pass is split into ceil(rows / --partition-rows) passes over disjoint sets of concept_id
(hash(concept_id) % P); a group never spans two passes. duckdb gets memory_limit (default 80% of
the SLURM allocation), threads from SLURM_CPUS_PER_TASK and a private spill directory in
<out_dir>/_duckdb_tmp/ (--tmp-root) that is removed at the end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

DATASET_ALIASES = {"mimic4": "miiv", "miiv": "miiv", "mimic": "miiv", "mimiciv": "miiv",
                   "eicu": "eicu", "hirid": "hirid"}

VOCAB_KEYS = {
    "CONCEPT": ["concept_id"],
    "CONCEPT_RELATIONSHIP": ["concept_id_1", "concept_id_2", "relationship_id"],
    "CONCEPT_ANCESTOR": ["ancestor_concept_id", "descendant_concept_id"],
    "CONCEPT_SYNONYM": ["concept_id"],
    "CONCEPT_CLASS": ["concept_class_id"],
    "DOMAIN": ["domain_id"],
    "VOCABULARY": ["vocabulary_id"],
    "RELATIONSHIP": ["relationship_id"],
    "DRUG_STRENGTH": ["drug_concept_id", "ingredient_concept_id"],
    "SOURCE_TO_CONCEPT_MAP": ["source_code", "target_concept_id"],
}

QV = [0.01, 0.25, 0.5, 0.75, 0.99]   # value quantiles
QT = [0.01, 0.5, 0.99]               # time and duration quantiles


@dataclass
class EventSpec:
    concept: str
    source: str
    time: str
    unit_source: str | None = None
    unit_concept: str | None = None
    value: str | None = None
    end: str | None = None
    srcnum: str | None = None          # a text/number column summarised as numbers too
    extra: dict[str, str] = field(default_factory=dict)   # name -> SQL counted per group
    categorical: list[str] = field(default_factory=list)  # per-dataset value counts


EVENT_TABLES = {
    "MEASUREMENT": EventSpec(
        concept="measurement_concept_id", source="measurement_source_value",
        time="measurement_datetime", unit_source="unit_source_value", unit_concept="unit_concept_id",
        value="value_as_number",
        extra={"n_value_concept": "nullif(e.value_as_concept_id, 0)",
               "n_value_source": "e.value_source_value",
               "n_operator": "nullif(e.operator_concept_id, 0)",
               "n_range": "e.range_low"},
        categorical=["measurement_type_concept_id", "operator_concept_id", "value_as_concept_id"]),
    "OBSERVATION": EventSpec(
        concept="observation_concept_id", source="observation_source_value",
        time="observation_datetime", unit_source="unit_source_value", unit_concept="unit_concept_id",
        value="value_as_number", srcnum="value_source_value",
        extra={"n_value_concept": "nullif(e.value_as_concept_id, 0)",
               "n_value_source": "e.value_source_value",
               "n_value_string": "e.value_as_string"},
        categorical=["observation_type_concept_id", "value_as_concept_id", "qualifier_concept_id",
                     "value_as_string"]),
    "DRUG_EXPOSURE": EventSpec(
        concept="drug_concept_id", source="drug_source_value",
        time="drug_exposure_start_datetime", end="drug_exposure_end_datetime",
        unit_source="dose_unit_source_value", value="quantity",
        extra={"n_sig": "e.sig", "n_route_source": "e.route_source_value",
               "n_days_supply": "e.days_supply"},
        categorical=["drug_type_concept_id", "route_concept_id", "route_source_value"]),
    "PROCEDURE_OCCURRENCE": EventSpec(
        concept="procedure_concept_id", source="procedure_source_value",
        time="procedure_datetime", end="procedure_end_datetime", value="quantity",
        categorical=["procedure_type_concept_id", "modifier_concept_id"]),
    "SPECIMEN": EventSpec(
        concept="specimen_concept_id", source="specimen_source_value", time="specimen_datetime",
        unit_source="unit_source_value", unit_concept="unit_concept_id", value="quantity",
        extra={"n_source_id": "e.specimen_source_id"},
        categorical=["specimen_type_concept_id", "anatomic_site_concept_id"]),
    "CONDITION_OCCURRENCE": EventSpec(
        concept="condition_concept_id", source="condition_source_value",
        time="condition_start_datetime", end="condition_end_datetime",
        categorical=["condition_type_concept_id", "condition_status_concept_id"]),
    "DEATH": EventSpec(
        concept="cause_concept_id", source="cause_source_value", time="death_datetime",
        categorical=["death_type_concept_id"]),
}

COUNT_COLS = ["n_rows", "n_persons", "n_visits", "n_visit_null", "n_visit_orphan", "n_person_null",
              "n_person_orphan", "n_person_visit_mismatch", "n_time_null", "n_before_start",
              "n_after_end", "n_value", "n_srcnum", "n_end", "n_dur_negative"]
EXTRA_COLS = sorted({k for s in EVENT_TABLES.values() for k in s.extra})
STAT_COLS = ([f"value_p{round(q * 100):02d}" for q in QV] + ["value_mean"]
             + [f"srcnum_p{round(q * 100):02d}" for q in QV] + ["srcnum_mean"]
             + [f"rel_h_p{round(q * 100):02d}" for q in QT] + ["rel_h_mean"]
             + [f"dur_h_p{round(q * 100):02d}" for q in QT] + ["dur_h_mean"])
INTEGRITY_FROM_GROUPS = {"rows": "n_rows", "person_null": "n_person_null",
                         "person_orphan": "n_person_orphan", "visit_null": "n_visit_null",
                         "visit_orphan": "n_visit_orphan",
                         "person_visit_mismatch": "n_person_visit_mismatch"}


def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def find_tables(cdm: Path) -> dict[str, list[Path]]:
    """TABLE -> parquet files. A table is `X.parquet` or a directory `X/` of parquet parts."""
    tables: dict[str, list[Path]] = {}
    for p in sorted(cdm.iterdir()):
        if p.is_file() and p.suffix == ".parquet":
            tables.setdefault(p.stem.upper(), []).append(p)
        elif p.is_dir():
            parts = sorted(p.rglob("*.parquet"))
            if parts:
                tables.setdefault(p.name.removesuffix(".parquet").upper(), []).extend(parts)
    return tables


def scan(files: list[Path]) -> str:
    return "read_parquet([" + ", ".join(lit(str(f)) for f in files) + "], union_by_name=true)"


def memory_limit_gb() -> float:
    if os.environ.get("FP_MEMORY_GB"):
        return float(os.environ["FP_MEMORY_GB"])
    slurm_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mb:
        return 0.8 * int(slurm_mb) / 1024
    if os.environ.get("SLURM_MEM_PER_CPU") and os.environ.get("SLURM_CPUS_PER_TASK"):
        return 0.8 * int(os.environ["SLURM_MEM_PER_CPU"]) * int(os.environ["SLURM_CPUS_PER_TASK"]) / 1024
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    return min(0.5 * total, 32.0)


def dataset_sql(col: str) -> tuple[str, str]:
    """(raw prefix, normalised dataset) SQL for a '{dataset}-{id}' source value."""
    raw = f"regexp_extract(CAST({col} AS VARCHAR), '^([A-Za-z0-9]+)[-_]', 1)"
    cases = " ".join(f"WHEN {lit(k)} THEN {lit(v)}" for k, v in DATASET_ALIASES.items())
    norm = f"CASE lower({raw}) {cases} ELSE coalesce(nullif({raw}, ''), '<unknown>') END"
    return raw, norm


class Fingerprinter:
    def __init__(self, cdm: Path, out: Path, args):
        self.cdm, self.out, self.args = cdm, out, args
        self.tables = find_tables(cdm)
        self.timings: dict[str, float] = {}
        self.threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 4)
        self.mem_gb = memory_limit_gb()
        job = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
        self.tmp = Path(args.tmp_root) / f"fingerprint_{job}_{os.getpid()}"
        self.tmp.mkdir(parents=True, exist_ok=False)
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"SET memory_limit='{self.mem_gb:.1f}GB'")
        self.con.execute(f"SET threads={self.threads}")
        self.con.execute(f"SET temp_directory={lit(self.tmp)}")
        self.con.execute("SET preserve_insertion_order=false")
        self.cols: dict[str, dict[str, str]] = {}
        self.nrows: dict[str, int] = {}

    # --- helpers -------------------------------------------------------------------------------
    def sql(self, query: str):
        return self.con.execute(query)

    def step(self, name: str):
        fp = self

        class _T:
            def __enter__(self):
                self.t0 = time.time()
                log(f"{name} ...")

            def __exit__(self, *exc):
                fp.timings[name] = round(time.time() - self.t0, 1)
                if exc[0] is None:
                    log(f"{name} done in {fp.timings[name]} s")
        return _T()

    def has(self, table: str, *cols: str | None) -> bool:
        return table in self.cols and all(c is not None and c in self.cols[table] for c in cols)

    def src(self, table: str) -> str:
        return scan(self.tables[table])

    # --- 1. tables and columns -----------------------------------------------------------------
    def tables_and_columns(self) -> None:
        trows, crows = [], []
        for t, files in self.tables.items():
            meta = self.sql(f"""SELECT sum(num_rows) FROM parquet_file_metadata(
                [{", ".join(lit(str(f)) for f in files)}])""").fetchone()
            n = int(meta[0] or 0)
            schema = self.sql(f"DESCRIBE SELECT * FROM {self.src(t)}").fetchall()
            self.cols[t] = {r[0].lower(): r[1] for r in schema}
            self.nrows[t] = n
            kind = "vocabulary" if t in VOCAB_KEYS else ("event" if t in EVENT_TABLES else "other")
            vocab_hash = self.vocab_hash(t) if kind == "vocabulary" and n else None
            trows.append((t, len(files), sum(f.stat().st_size for f in files), n, len(schema),
                          kind, vocab_hash))
            nonnull = {}
            if n:
                exprs = ", ".join(f"count({q(r[0])})" for r in schema)
                nonnull = dict(zip([r[0] for r in schema], self.sql(
                    f"SELECT {exprs} FROM {self.src(t)}").fetchone()))
            for i, (name, typ, *_rest) in enumerate(schema):
                is_id = name.lower().endswith("_id") and not name.lower().endswith("concept_id")
                n_distinct = None
                if is_id and n:
                    n_distinct = self.sql(
                        f"SELECT count(DISTINCT {q(name)}) FROM {self.src(t)}").fetchone()[0]
                crows.append((t, name.lower(), i, typ, n - nonnull.get(name, 0) if n else 0,
                              n_distinct, is_id))
            log(f"  {t}: {n:,} rows, {len(schema)} columns")
        self.sql("""CREATE TABLE out_tables (table_name VARCHAR, n_files INT, file_bytes BIGINT,
                    n_rows BIGINT, n_columns INT, kind VARCHAR, key_hash VARCHAR)""")
        self.con.executemany("INSERT INTO out_tables VALUES (?, ?, ?, ?, ?, ?, ?)", trows)
        self.sql("""CREATE TABLE out_columns (table_name VARCHAR, column_name VARCHAR, position INT,
                    type VARCHAR, n_null BIGINT, n_distinct BIGINT, is_id BOOLEAN)""")
        self.con.executemany("INSERT INTO out_columns VALUES (?, ?, ?, ?, ?, ?, ?)", crows)

    def vocab_hash(self, t: str) -> str | None:
        keys = [k for k in VOCAB_KEYS[t] if k in self.cols[t]]
        if not keys:
            return None
        k = " || '|' || ".join(f"coalesce(CAST({q(c)} AS VARCHAR), '')" for c in keys)
        if self.nrows[t] <= 20_000_000:   # md5 of the sorted keys: stable across duckdb versions
            return self.sql(f"SELECT md5(string_agg(k, ',' ORDER BY k)) FROM "
                            f"(SELECT {k} AS k FROM {self.src(t)})").fetchone()[0]
        h = self.sql(f"SELECT bit_xor(hash(k)), sum(hash(k)::HUGEINT) FROM "
                     f"(SELECT {k} AS k FROM {self.src(t)})").fetchone()
        return f"xor:{h[0]}:sum:{h[1]}"

    # --- 2. dataset lookups --------------------------------------------------------------------
    def lookups(self) -> None:
        raw_p, ds_p = dataset_sql("person_source_value")
        raw_v, ds_v = dataset_sql("visit_source_value")
        ds_p = self.args.person_dataset_sql or ds_p
        ds_v = self.args.visit_dataset_sql or ds_v
        if self.nrows.get("PERSON"):
            self.sql(f"""CREATE TABLE _person AS
                SELECT person_id, min({ds_p}) AS dataset, min({raw_p}) AS raw_prefix, count(*) AS n
                FROM {self.src('PERSON')} GROUP BY person_id""")
        else:
            self.sql("CREATE TABLE _person (person_id BIGINT, dataset VARCHAR, raw_prefix VARCHAR, n BIGINT)")
        if self.nrows.get("VISIT_OCCURRENCE"):
            self.sql(f"""CREATE TABLE _visit AS
                SELECT visit_occurrence_id, min(person_id) AS person_id,
                       min({ds_v}) AS dataset, min({raw_v}) AS raw_prefix,
                       min(CAST(visit_start_datetime AS TIMESTAMP)) AS vs,
                       min(CAST(visit_end_datetime AS TIMESTAMP)) AS ve, count(*) AS n
                FROM {self.src('VISIT_OCCURRENCE')} GROUP BY visit_occurrence_id""")
        else:
            self.sql("""CREATE TABLE _visit (visit_occurrence_id BIGINT, person_id BIGINT,
                        dataset VARCHAR, raw_prefix VARCHAR, vs TIMESTAMP, ve TIMESTAMP, n BIGINT)""")
        self.sql("""CREATE TABLE _first_visit AS
            SELECT person_id, arg_min(visit_occurrence_id, (vs, visit_occurrence_id)) AS vid,
                   min(vs) AS vs, arg_min(ve, (vs, visit_occurrence_id)) AS ve
            FROM _visit WHERE person_id IS NOT NULL GROUP BY person_id""")
        n_p, n_v = (self.sql(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("_person", "_visit"))
        log(f"  lookups: {n_p:,} persons, {n_v:,} visits; datasets: "
            + ", ".join(f"{d}={c:,}" for d, c in self.sql(
                "SELECT dataset, count(*) FROM _visit GROUP BY 1 ORDER BY 1").fetchall()))

    # --- 3. event tables grouped per concept ---------------------------------------------------
    def event_source(self, t: str, s: EventSpec, pred: str = "TRUE",
                     carry: dict[str, str] | None = None) -> str:
        """Rows of `t` with dataset, the attributed visit and normalised key/value columns.
        `carry` adds further columns of `t` (alias -> SQL over `e`)."""
        c = self.cols[t]

        def col(name, cast):
            return f"TRY_CAST(e.{q(name)} AS {cast})" if name and name in c else f"NULL::{cast}"

        def num(name):
            # native float/int columns stay native: exact quantiles hold every value in memory
            if name in c and re.match(r"(FLOAT|DOUBLE|REAL|DECIMAL|(U?(TINY|SMALL|BIG|HUGE)?INT)(EGER)?)",
                                      c[name].upper()):
                return f"e.{q(name)}"
            return col(name, "DOUBLE")

        carried = "".join(f"{x} AS {k}, " for k, x in (carry or {}).items())

        has_visit = "visit_occurrence_id" in c
        person = "e.person_id" if "person_id" in c else "NULL::BIGINT"
        ev = f"""SELECT {col(s.concept, 'BIGINT')} AS concept_id,
                    {col(s.source, 'VARCHAR')} AS source_value,
                    {col(s.unit_source, 'VARCHAR')} AS unit_source_value,
                    {col(s.unit_concept, 'BIGINT')} AS unit_concept_id,
                    {num(s.value)} AS val,
                    {col(s.srcnum, 'DOUBLE')} AS srcnum,
                    {col(s.time, 'TIMESTAMP')} AS t,
                    {col(s.end, 'TIMESTAMP')} AS t_end,
                    {person} AS person_id,
                    {'e.visit_occurrence_id' if has_visit else 'NULL::BIGINT'} AS vid_raw, {carried}
                    {", ".join(f"({x}) IS NOT NULL AS {k}" for k, x in self.extras(t, s).items()) or 'NULL AS _none'}
                 FROM {self.src(t)} e WHERE {pred}"""
        if has_visit:
            return f"""SELECT ev.*, ev.vid_raw AS vid, v.visit_occurrence_id AS v_found,
                           v.person_id AS v_person, v.vs, v.ve, p.person_id AS p_found,
                           coalesce(v.dataset, p.dataset, '<unresolved>') AS dataset
                    FROM ({ev}) ev
                    LEFT JOIN _visit v ON ev.vid_raw = v.visit_occurrence_id
                    LEFT JOIN _person p ON ev.person_id = p.person_id"""
        # no visit column: latest visit of the person starting at or before t, else the first one
        return f"""SELECT a.* EXCLUDE (a_vid, a_vs, a_ve, a_person),
                       coalesce(a.a_vid, f.vid) AS vid, coalesce(a.a_vid, f.vid) AS v_found,
                       coalesce(a.a_person, f.person_id) AS v_person,
                       coalesce(a.a_vs, f.vs) AS vs, coalesce(a.a_ve, f.ve) AS ve,
                       p.person_id AS p_found,
                       coalesce(p.dataset, '<unresolved>') AS dataset
                FROM (SELECT ev.*, v.visit_occurrence_id AS a_vid, v.vs AS a_vs, v.ve AS a_ve,
                             v.person_id AS a_person
                      FROM ({ev}) ev ASOF LEFT JOIN _visit v
                        ON ev.person_id = v.person_id AND ev.t >= v.vs) a
                LEFT JOIN _first_visit f ON a.person_id = f.person_id
                LEFT JOIN _person p ON a.person_id = p.person_id"""

    def extras(self, t: str, s: EventSpec) -> dict[str, str]:
        out = {}
        for k, x in s.extra.items():
            used = re.findall(r"e\.([a-z_]+)", x)
            if all(u in self.cols[t] for u in used):
                out[k] = x
        return out

    def group_query(self, t: str, s: EventSpec, pred: str) -> str:
        has_visit = "visit_occurrence_id" in self.cols[t]
        rel = "(epoch_us(t) - epoch_us(vs)) / 3.6e9"
        dur = "(epoch_us(t_end) - epoch_us(t)) / 3.6e9"
        qv = "[" + ", ".join(map(str, QV)) + "]"
        qt = "[" + ", ".join(map(str, QT)) + "]"
        ex = self.extras(t, s)
        extra_sel = ", ".join(f"count(*) FILTER (WHERE {k}) AS {k}" if k in ex else f"NULL::BIGINT AS {k}"
                              for k in EXTRA_COLS)
        visit_null = "vid_raw IS NULL" if has_visit else "vid IS NULL"
        inner = f"""
            SELECT {lit(t)} AS table_name, dataset, concept_id, source_value, unit_source_value,
                   unit_concept_id,
                   count(*) AS n_rows,
                   count(DISTINCT person_id) AS n_persons,
                   count(DISTINCT vid) AS n_visits,
                   count(*) FILTER (WHERE {visit_null}) AS n_visit_null,
                   count(*) FILTER (WHERE vid IS NOT NULL AND v_found IS NULL) AS n_visit_orphan,
                   count(*) FILTER (WHERE person_id IS NULL) AS n_person_null,
                   count(*) FILTER (WHERE person_id IS NOT NULL AND p_found IS NULL) AS n_person_orphan,
                   count(*) FILTER (WHERE v_person IS NOT NULL AND person_id IS DISTINCT FROM v_person)
                       AS n_person_visit_mismatch,
                   count(*) FILTER (WHERE t IS NULL) AS n_time_null,
                   count(*) FILTER (WHERE t < vs) AS n_before_start,
                   count(*) FILTER (WHERE t > ve) AS n_after_end,
                   count(val) AS n_value,
                   count(srcnum) AS n_srcnum,
                   count(t_end) AS n_end,
                   count(*) FILTER (WHERE t_end < t) AS n_dur_negative,
                   {extra_sel},
                   CAST(quantile_cont(val, {qv}) AS DOUBLE[]) AS vq, avg(CAST(val AS DOUBLE)) AS value_mean,
                   quantile_cont(srcnum, {qv}) AS sq, avg(srcnum) AS srcnum_mean,
                   quantile_cont({rel}, {qt}) AS rq, avg({rel}) AS rel_h_mean,
                   quantile_cont({dur}, {qt}) AS dq, avg({dur}) AS dur_h_mean
            FROM ({self.event_source(t, s, pred)})
            GROUP BY ALL"""
        unpack = ", ".join(
            [f"vq[{i + 1}] AS value_p{round(x * 100):02d}" for i, x in enumerate(QV)] + ["value_mean"]
            + [f"sq[{i + 1}] AS srcnum_p{round(x * 100):02d}" for i, x in enumerate(QV)] + ["srcnum_mean"]
            + [f"rq[{i + 1}] AS rel_h_p{round(x * 100):02d}" for i, x in enumerate(QT)] + ["rel_h_mean"]
            + [f"dq[{i + 1}] AS dur_h_p{round(x * 100):02d}" for i, x in enumerate(QT)] + ["dur_h_mean"])
        keys = "table_name, dataset, concept_id, source_value, unit_source_value, unit_concept_id"
        return f"SELECT {keys}, {', '.join(COUNT_COLS + EXTRA_COLS)}, {unpack} FROM ({inner})"

    def event_groups(self) -> None:
        created = False
        for t, s in EVENT_TABLES.items():
            if not self.nrows.get(t):
                continue
            n_part = max(1, math.ceil(self.nrows[t] / self.args.partition_rows))
            with self.step(f"groups {t} ({self.nrows[t]:,} rows, {n_part} pass(es))"):
                concept = f"e.{q(s.concept)}" if s.concept in self.cols[t] else "NULL"
                for i in range(n_part):
                    pred = "TRUE" if n_part == 1 else f"hash({concept}) % {n_part} = {i}"
                    query = self.group_query(t, s, pred)
                    if not created:
                        self.sql(f"CREATE TABLE out_groups AS {query}")
                        created = True
                    else:
                        self.sql(f"INSERT INTO out_groups {query}")
                    if n_part > 1:
                        log(f"    pass {i + 1}/{n_part}")
            with self.step(f"categorical {t}"):
                self.categorical(t, s)
        if not created:
            raise SystemExit("no event table with rows found")

    def categorical(self, t: str, s: EventSpec) -> None:
        cats = [c for c in s.categorical if c in self.cols[t]]
        if not cats:
            return
        src = self.event_source(t, s, carry={f"cat_{c}": f"CAST(e.{q(c)} AS VARCHAR)" for c in cats})
        sets = ", ".join(f"(dataset, cat_{c})" for c in cats)
        sel = ", ".join(f"cat_{c}, grouping(cat_{c}) AS g_{c}" for c in cats)
        rows = self.sql(f"SELECT dataset, {sel}, count(*) FROM ({src}) GROUP BY GROUPING SETS ({sets})"
                        ).fetchall()
        out = []
        for r in rows:
            ds, n = r[0], r[-1]
            for j, c in enumerate(cats):
                if r[2 + 2 * j] == 0:
                    out.append((t, ds, c, "<null>" if r[1 + 2 * j] is None else r[1 + 2 * j], n, "count"))
        self.add_metrics(out)

    # --- 4. metrics ----------------------------------------------------------------------------
    def add_metrics(self, rows: list[tuple]) -> None:
        if not rows:
            return
        self.con.executemany("INSERT INTO out_metrics VALUES (?, ?, ?, ?, ?, ?)",
                             [(t, d, m, str(k), None if v is None else float(v), kind)
                              for t, d, m, k, v, kind in rows])

    def metric_sql(self, table: str, query: str) -> None:
        """`query` returns (dataset, metric, key, value, kind)."""
        self.add_metrics([(table, *r) for r in self.sql(query).fetchall()])

    def counts_sql(self, table: str, from_sql: str, cols: list[str]) -> None:
        cols = [c for c in cols if c in self.cols.get(table, {})]
        if not cols:
            return
        sets = ", ".join(f"(dataset, {q(c)})" for c in cols)
        sel = ", ".join(f"CAST({q(c)} AS VARCHAR) AS {q(c)}, grouping({q(c)}) AS {q('g_' + c)}" for c in cols)
        rows = self.sql(f"SELECT dataset, {sel}, count(*) FROM ({from_sql}) "
                        f"GROUP BY GROUPING SETS ({sets})").fetchall()
        out = []
        for r in rows:
            for j, c in enumerate(cols):
                if r[2 + 2 * j] == 0:
                    out.append((table, r[0], c, "<null>" if r[1 + 2 * j] is None else r[1 + 2 * j],
                                r[-1], "count"))
        self.add_metrics(out)

    @staticmethod
    def quantile_metrics(expr: str, prefix: str, qs=QV) -> str:
        """SQL fragment yielding (metric, key, value, kind) rows via UNNEST per dataset."""
        names = [f"{prefix}_p{round(x * 100):02d}" for x in qs] + [f"{prefix}_mean", f"{prefix}_n"]
        vals = [f"quantile_cont({expr}, {x})" for x in qs] + [f"avg({expr})", f"count({expr})"]
        kinds = ["stat"] * (len(qs) + 1) + ["count"]
        return (f"unnest([{', '.join(lit(n) for n in names)}]) AS metric, '' AS key, "
                f"unnest([{', '.join(f'CAST({v} AS DOUBLE)' for v in vals)}]) AS value, "
                f"unnest([{', '.join(lit(k) for k in kinds)}]) AS kind")

    def summaries(self) -> None:
        self.sql("""CREATE TABLE out_metrics (table_name VARCHAR, dataset VARCHAR, metric VARCHAR,
                    key VARCHAR, value DOUBLE, kind VARCHAR)""")
        if self.nrows.get("PERSON"):
            p = f"""SELECT e.*, coalesce(l.dataset, '<unresolved>') AS dataset, l.raw_prefix
                    FROM {self.src('PERSON')} e LEFT JOIN _person l USING (person_id)"""
            self.metric_sql("PERSON", f"SELECT dataset, 'n_rows', '', count(*), 'count' FROM ({p}) GROUP BY 1")
            self.metric_sql("PERSON", f"""SELECT dataset, 'source_prefix', coalesce(raw_prefix, '<null>'),
                count(*), 'count' FROM ({p}) GROUP BY 1, 3""")
            self.counts_sql("PERSON", p, ["gender_concept_id", "race_concept_id", "ethnicity_concept_id",
                                          "gender_source_value", "month_of_birth", "day_of_birth"])
            if "year_of_birth" in self.cols["PERSON"]:
                self.metric_sql("PERSON", f"""SELECT dataset, {self.quantile_metrics(
                    'CAST(year_of_birth AS DOUBLE)', 'year_of_birth')} FROM ({p}) GROUP BY dataset""")
            if "location_id" in self.cols["PERSON"]:
                self.metric_sql("PERSON", f"""SELECT dataset, 'n_locations', '', count(DISTINCT location_id),
                    'count' FROM ({p}) GROUP BY 1""")

        if self.nrows.get("VISIT_OCCURRENCE"):
            v = f"""SELECT e.*, coalesce(l.dataset, '<unresolved>') AS dataset, l.raw_prefix,
                           (epoch_us(CAST(e.visit_end_datetime AS TIMESTAMP))
                            - epoch_us(CAST(e.visit_start_datetime AS TIMESTAMP))) / 3.6e9 AS los_h
                    FROM {self.src('VISIT_OCCURRENCE')} e LEFT JOIN _visit l USING (visit_occurrence_id)"""
            self.metric_sql("VISIT_OCCURRENCE", f"""SELECT dataset, unnest(['n_rows', 'n_persons',
                    'n_care_sites', 'n_distinct_visit_start', 'n_preceding_visit']), '',
                    unnest([count(*), count(DISTINCT person_id), count(DISTINCT care_site_id),
                            count(DISTINCT visit_start_datetime),
                            count({'preceding_visit_occurrence_id' if self.has('VISIT_OCCURRENCE', 'preceding_visit_occurrence_id') else 'NULL'})]),
                    'count' FROM ({v}) GROUP BY 1""")
            self.metric_sql("VISIT_OCCURRENCE", f"""SELECT dataset, 'source_prefix',
                coalesce(raw_prefix, '<null>'), count(*), 'count' FROM ({v}) GROUP BY 1, 3""")
            self.metric_sql("VISIT_OCCURRENCE", f"SELECT dataset, {self.quantile_metrics('los_h', 'los_h')} "
                                                f"FROM ({v}) GROUP BY dataset")
            self.metric_sql("VISIT_OCCURRENCE", f"""SELECT dataset, {self.quantile_metrics(
                    'CAST(n AS DOUBLE)', 'visits_per_person', [0.5, 0.9, 0.99])} FROM (
                SELECT dataset, person_id, count(*) AS n FROM ({v}) GROUP BY 1, 2) GROUP BY dataset""")
            self.metric_sql("VISIT_OCCURRENCE", f"""SELECT dataset, unnest(['visits_per_person_max',
                    'n_persons_multi_visit']), '', unnest([max(n), count(*) FILTER (WHERE n > 1)]), 'count'
                FROM (SELECT dataset, person_id, count(*) AS n FROM ({v}) GROUP BY 1, 2) GROUP BY dataset""")
            self.counts_sql("VISIT_OCCURRENCE", v, [
                "visit_concept_id", "visit_type_concept_id", "visit_source_concept_id",
                "admitted_from_concept_id", "admitted_from_source_value",
                "discharged_to_concept_id", "discharged_to_source_value"])

        if self.nrows.get("DEATH"):
            d = f"""SELECT e.*, coalesce(l.dataset, '<unresolved>') AS dataset
                    FROM {self.src('DEATH')} e LEFT JOIN _person l USING (person_id)"""
            self.metric_sql("DEATH", f"""SELECT dataset, unnest(['n_rows', 'n_persons', 'n_datetime_null']), '',
                unnest([count(*), count(DISTINCT person_id), count(*) - count(death_datetime)]), 'count'
                FROM ({d}) GROUP BY 1""")

        if self.nrows.get("CARE_SITE"):
            c = f"""SELECT e.*, coalesce(u.dataset, '<unused>') AS dataset FROM {self.src('CARE_SITE')} e
                    LEFT JOIN (SELECT care_site_id, CASE WHEN count(DISTINCT l.dataset) > 1 THEN '<mixed>'
                                      ELSE min(l.dataset) END AS dataset
                               FROM {self.src('VISIT_OCCURRENCE')} v
                               JOIN _visit l USING (visit_occurrence_id) GROUP BY 1) u USING (care_site_id)"""
            self.metric_sql("CARE_SITE", f"""SELECT dataset, unnest(['n_rows', 'n_names', 'n_locations']), '',
                unnest([count(*), count(DISTINCT care_site_name), count(DISTINCT location_id)]), 'count'
                FROM ({c}) GROUP BY 1""")
            self.counts_sql("CARE_SITE", c, ["place_of_service_concept_id", "place_of_service_source_value"])

        if self.nrows.get("OBSERVATION_PERIOD"):
            o = f"""SELECT e.*, coalesce(l.dataset, '<unresolved>') AS dataset,
                           CAST(date_diff('day', CAST(observation_period_start_date AS DATE),
                                CAST(observation_period_end_date AS DATE)) AS DOUBLE) AS len_d
                    FROM {self.src('OBSERVATION_PERIOD')} e LEFT JOIN _person l USING (person_id)"""
            self.metric_sql("OBSERVATION_PERIOD", f"""SELECT dataset, unnest(['n_rows', 'n_persons']), '',
                unnest([count(*), count(DISTINCT person_id)]), 'count' FROM ({o}) GROUP BY 1""")
            self.metric_sql("OBSERVATION_PERIOD", f"SELECT dataset, {self.quantile_metrics('len_d', 'length_days')} "
                                                  f"FROM ({o}) GROUP BY dataset")
            self.counts_sql("OBSERVATION_PERIOD", o, ["period_type_concept_id"])

        if self.nrows.get("LOCATION"):
            loc = f"SELECT e.*, '<all>' AS dataset FROM {self.src('LOCATION')} e"
            self.metric_sql("LOCATION", f"SELECT dataset, 'n_rows', '', count(*), 'count' FROM ({loc}) GROUP BY 1")
            self.counts_sql("LOCATION", loc, ["country_source_value", "country_concept_id"])

    def sig_keys(self) -> None:
        if not self.has("DRUG_EXPOSURE", "sig") or not self.nrows.get("DRUG_EXPOSURE"):
            return
        src = self.event_source("DRUG_EXPOSURE", EVENT_TABLES["DRUG_EXPOSURE"], carry={"sig": "e.sig"})
        kv = f"""SELECT dataset, split_part(pair, '=', 1) AS k, split_part(pair, '=', 2) AS v FROM (
                    SELECT dataset, unnest(string_split(sig, ';')) AS pair FROM ({src}) WHERE sig IS NOT NULL)
                 WHERE pair <> ''"""
        self.sql(f"CREATE TEMP TABLE _sig AS {kv}")
        self.metric_sql("DRUG_EXPOSURE", """SELECT dataset, 'sig_key', k, count(*), 'count'
                                            FROM _sig GROUP BY 1, 3""")
        self.metric_sql("DRUG_EXPOSURE", """SELECT dataset, 'sig_key_distinct_values', k,
                                            count(DISTINCT v), 'count' FROM _sig GROUP BY 1, 3""")
        self.metric_sql("DRUG_EXPOSURE", """SELECT dataset, 'sig_table', v, count(*), 'count'
                                            FROM _sig WHERE k = 'table' GROUP BY 1, 3""")
        self.metric_sql("DRUG_EXPOSURE", f"""SELECT dataset, 'n_sig_distinct_keysets', '',
            count(DISTINCT regexp_replace(sig, '=[^;]*', '', 'g')), 'count'
            FROM ({src}) WHERE sig IS NOT NULL GROUP BY 1""")
        self.sql("DROP TABLE _sig")

    # --- 5. integrity --------------------------------------------------------------------------
    def integrity(self) -> None:
        self.sql("CREATE TABLE out_integrity (table_name VARCHAR, dataset VARCHAR, check_name VARCHAR, n BIGINT)")
        # event tables: rolled up from the groups
        for check, col in INTEGRITY_FROM_GROUPS.items():
            self.sql(f"""INSERT INTO out_integrity SELECT table_name, dataset, {lit(check)}, sum({col})
                         FROM out_groups GROUP BY 1, 2""")
        # other tables with person_id / visit_occurrence_id
        for t, n in self.nrows.items():
            if t in EVENT_TABLES or not n:
                continue
            c = self.cols[t]
            hp, hv = "person_id" in c, "visit_occurrence_id" in c
            if not (hp or hv):
                continue
            if t == "PERSON":
                hp = True
            vj = "LEFT JOIN _visit v ON e.visit_occurrence_id = v.visit_occurrence_id" if hv else ""
            pj = "LEFT JOIN _person p ON e.person_id = p.person_id" if hp else ""
            ds = "coalesce(" + ", ".join((["v.dataset"] if hv else []) + (["p.dataset"] if hp else [])) \
                 + ", '<unresolved>')"
            checks = {"rows": "count(*)"}
            if hp:
                checks |= {"person_null": "count(*) FILTER (WHERE e.person_id IS NULL)",
                           "person_orphan": "count(*) FILTER (WHERE e.person_id IS NOT NULL AND p.person_id IS NULL)"}
            if hv:
                checks |= {"visit_null": "count(*) FILTER (WHERE e.visit_occurrence_id IS NULL)",
                           "visit_orphan": "count(*) FILTER (WHERE e.visit_occurrence_id IS NOT NULL "
                                           "AND v.visit_occurrence_id IS NULL)"}
            if hp and hv and t != "VISIT_OCCURRENCE":
                checks["person_visit_mismatch"] = ("count(*) FILTER (WHERE v.person_id IS NOT NULL "
                                                   "AND e.person_id IS DISTINCT FROM v.person_id)")
            self.sql(f"""INSERT INTO out_integrity
                SELECT {lit(t)}, dataset, unnest(map_keys(m)), unnest(map_values(m)) FROM (
                    SELECT {ds} AS dataset, map_from_entries([{", ".join(
                        f"({lit(k)}, {x})" for k, x in checks.items())}]) AS m
                    FROM {self.src(t)} e {vj} {pj} GROUP BY 1)""")

        special: list[tuple[str, str]] = []   # (table, SELECT dataset, check, n)
        if self.nrows.get("PERSON"):
            special += [("PERSON", "SELECT dataset, 'duplicate_person_id', sum(n - 1) FROM _person GROUP BY 1"),
                        ("PERSON", """SELECT p.dataset, 'person_without_visit', count(*) FROM _person p
                            WHERE NOT EXISTS (SELECT 1 FROM _visit v WHERE v.person_id = p.person_id) GROUP BY 1"""),
                        ("PERSON", """SELECT p.dataset, 'visit_in_other_dataset', count(*) FROM _visit v
                            JOIN _person p USING (person_id) WHERE v.dataset IS DISTINCT FROM p.dataset GROUP BY 1"""),
                        ("PERSON", f"""SELECT l.dataset, 'duplicate_person_source_value', count(*) - count(DISTINCT
                            e.person_source_value) FROM {self.src('PERSON')} e JOIN _person l USING (person_id)
                            GROUP BY 1""")]
            if self.has("PERSON", "location_id") and self.nrows.get("LOCATION"):
                special.append(("PERSON", f"""SELECT l.dataset, 'location_orphan', count(*)
                    FROM {self.src('PERSON')} e JOIN _person l USING (person_id)
                    WHERE e.location_id IS NOT NULL AND e.location_id NOT IN
                          (SELECT location_id FROM {self.src('LOCATION')}) GROUP BY 1"""))
        if self.nrows.get("VISIT_OCCURRENCE"):
            vsrc = f"{self.src('VISIT_OCCURRENCE')} e JOIN _visit l USING (visit_occurrence_id)"
            special += [
                ("VISIT_OCCURRENCE", "SELECT dataset, 'duplicate_visit_occurrence_id', sum(n - 1) FROM _visit GROUP BY 1"),
                ("VISIT_OCCURRENCE", f"""SELECT l.dataset, 'duplicate_visit_source_value',
                    count(*) - count(DISTINCT e.visit_source_value) FROM {vsrc} GROUP BY 1"""),
                ("VISIT_OCCURRENCE", f"""SELECT l.dataset, 'end_before_start', count(*) FROM {vsrc}
                    WHERE l.ve < l.vs GROUP BY 1"""),
                ("VISIT_OCCURRENCE", f"""SELECT l.dataset, 'start_or_end_null', count(*) FROM {vsrc}
                    WHERE l.ve IS NULL OR l.vs IS NULL GROUP BY 1"""),
                ("VISIT_OCCURRENCE", """SELECT dataset, 'overlaps_previous_visit_of_person', count(*) FROM (
                    SELECT dataset, vs, max(ve) OVER (PARTITION BY person_id ORDER BY vs, visit_occurrence_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end FROM _visit)
                    WHERE vs < prev_end GROUP BY 1"""),
            ]
            if self.has("VISIT_OCCURRENCE", "care_site_id") and self.nrows.get("CARE_SITE"):
                special += [("VISIT_OCCURRENCE", f"""SELECT l.dataset, 'care_site_orphan', count(*) FROM {vsrc}
                    WHERE e.care_site_id IS NOT NULL AND e.care_site_id NOT IN
                          (SELECT care_site_id FROM {self.src('CARE_SITE')}) GROUP BY 1"""),
                            ("VISIT_OCCURRENCE", f"""SELECT l.dataset, 'care_site_null', count(*) FROM {vsrc}
                    WHERE e.care_site_id IS NULL GROUP BY 1""")]
            if self.has("VISIT_OCCURRENCE", "preceding_visit_occurrence_id"):
                special.append(("VISIT_OCCURRENCE", f"""SELECT l.dataset, 'preceding_visit_orphan', count(*)
                    FROM {vsrc} WHERE e.preceding_visit_occurrence_id IS NOT NULL AND
                    e.preceding_visit_occurrence_id NOT IN (SELECT visit_occurrence_id FROM _visit) GROUP BY 1"""))
            if self.nrows.get("OBSERVATION_PERIOD"):
                op = f"""(SELECT person_id, min(CAST(observation_period_start_date AS DATE)) AS s,
                                 max(CAST(observation_period_end_date AS DATE)) AS e
                          FROM {self.src('OBSERVATION_PERIOD')} GROUP BY 1)"""
                special += [("VISIT_OCCURRENCE", f"""SELECT v.dataset, 'outside_observation_period', count(*)
                                FROM _visit v JOIN {op} o USING (person_id)
                                WHERE CAST(v.vs AS DATE) < o.s OR CAST(v.ve AS DATE) > o.e GROUP BY 1"""),
                            ("VISIT_OCCURRENCE", f"""SELECT v.dataset, 'person_without_observation_period', count(*)
                                FROM _visit v WHERE v.person_id NOT IN (SELECT person_id FROM {op}) GROUP BY 1""")]
        if self.has("CARE_SITE", "location_id") and self.nrows.get("CARE_SITE") and self.nrows.get("LOCATION"):
            special.append(("CARE_SITE", f"""SELECT '<all>', 'location_orphan', count(*) FROM {self.src('CARE_SITE')}
                WHERE location_id IS NOT NULL AND location_id NOT IN (SELECT location_id FROM {self.src('LOCATION')})"""))
        if self.nrows.get("DEATH"):
            special.append(("DEATH", f"""SELECT coalesce(l.dataset, '<unresolved>'), 'multiple_per_person',
                count(*) - count(DISTINCT e.person_id) FROM {self.src('DEATH')} e LEFT JOIN _person l USING (person_id)
                GROUP BY 1"""))
        for t, query in special:
            self.sql(f"INSERT INTO out_integrity SELECT {lit(t)}, * FROM ({query})")

        # concepts used by the event tables but absent from the CDM's own CONCEPT table
        if self.nrows.get("CONCEPT"):
            self.sql(f"""INSERT INTO out_integrity
                SELECT table_name, dataset, 'concept_not_in_cdm_concept', sum(n_rows) FROM out_groups
                WHERE concept_id IS NOT NULL AND concept_id <> 0
                  AND concept_id NOT IN (SELECT CAST(concept_id AS BIGINT) FROM {self.src('CONCEPT')})
                GROUP BY 1, 2""")
            self.sql("""INSERT INTO out_integrity
                SELECT table_name, dataset, 'concept_zero_or_null', sum(n_rows) FROM out_groups
                WHERE concept_id IS NULL OR concept_id = 0 GROUP BY 1, 2""")

    # --- 6. write ------------------------------------------------------------------------------
    def write(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        order = {"tables": "table_name",
                 "columns": "table_name, position",
                 "integrity": "table_name, dataset, check_name",
                 "groups": "table_name, dataset, concept_id, source_value, unit_source_value, unit_concept_id",
                 "metrics": "table_name, dataset, metric, key"}
        for name, by in order.items():
            for fmt, opts in (("parquet", "(FORMAT parquet, COMPRESSION zstd)"), ("csv", "(HEADER)")):
                self.sql(f"COPY (SELECT * FROM out_{name} ORDER BY {by} NULLS FIRST) "
                         f"TO {lit(self.out / f'{name}.{fmt}')} {opts}")
        meta = {
            "cdm_dir": str(self.cdm),
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "tool": str(Path(__file__).resolve()),
            "duckdb": duckdb.__version__,
            "python": sys.version.split()[0],
            "threads": self.threads,
            "memory_limit_gb": round(self.mem_gb, 1),
            "partition_rows": self.args.partition_rows,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": os.uname().nodename,
            "dataset_rule": {"visit": self.args.visit_dataset_sql or "prefix of visit_source_value",
                             "person": self.args.person_dataset_sql or "prefix of person_source_value",
                             "aliases": DATASET_ALIASES},
            "value_quantiles": QV, "time_quantiles": QT,
            "timings_s": self.timings,
            "n_groups": self.sql("SELECT count(*) FROM out_groups").fetchone()[0],
        }
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def run(self) -> None:
        t0 = time.time()
        log(f"cdm {self.cdm} -> {self.out}; threads={self.threads}, memory_limit={self.mem_gb:.1f}GB, "
            f"tmp={self.tmp}")
        try:
            with self.step("tables and columns"):
                self.tables_and_columns()
            with self.step("lookups"):
                self.lookups()
            with self.step("summaries"):
                self.summaries()
            self.event_groups()
            with self.step("sig keys"):
                self.sig_keys()
            with self.step("integrity"):
                self.integrity()
            self.timings["total"] = round(time.time() - t0, 1)
            with self.step("write"):
                self.write()
        finally:
            self.con.close()
            shutil.rmtree(self.tmp, ignore_errors=True)
        log(f"total {time.time() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cdm_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--partition-rows", type=int, default=400_000_000,
                    help="rows per grouped pass over an event table (memory knob)")
    ap.add_argument("--tmp-root", default=os.environ.get("OMOP_ETL_TMP"),
                    help="duckdb's spill directory (default <out_dir>/_duckdb_tmp)")
    ap.add_argument("--visit-dataset-sql", help="SQL over VISIT_OCCURRENCE columns giving the dataset")
    ap.add_argument("--person-dataset-sql", help="SQL over PERSON columns giving the dataset")
    ap.add_argument("--overwrite", action="store_true", help="allow a non-empty out_dir")
    args = ap.parse_args()
    cdm, out = args.cdm_dir.resolve(), args.out_dir.resolve()
    if not cdm.is_dir():
        raise SystemExit(f"not a directory: {cdm}")
    if out == cdm or cdm in out.parents:
        raise SystemExit("out_dir must not be inside the CDM directory")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} is not empty; pass --overwrite to replace its files")
    if args.tmp_root is None:
        args.tmp_root = out / "_duckdb_tmp"
    Fingerprinter(cdm, out, args).run()


if __name__ == "__main__":
    main()
