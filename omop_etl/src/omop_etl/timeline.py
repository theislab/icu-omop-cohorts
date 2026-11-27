"""The timeline of the CDM, and md5 ids.

A stay's rows sit at its admission + their offset. The admission is the one the dataset module
stages in `admit_us`: MIMIC-IV's and HiRID's date-shifted times, and for eICU, which has offsets
only, a time within the hospital discharge year (see eicu.py). A stay without `admit_us` gets a
synthetic admission at a midnight on or after 2000-01-01; a person's synthetic stays are laid out
in `stay_order`, spaced so that every row is nearer to its own visit than to the person's other
visits, since SPECIMEN and DEATH carry no visit id and consumers attribute them to the nearest
visit. Stays of different persons overlap freely.
"""

from __future__ import annotations

REF_DATE = "2000-01-01"
DAY_US = 86_400 * 10**6
EPOCH_2000_US = 946_684_800 * 10**6  # 2000-01-01 in microseconds since 1970-01-01
MIN_SPACING_DAYS = 60  # smallest start-to-start step between two stays of one person
STATIC_OFFSET_H = 5  # averaged height/weight and age rows sit at admission + 5 h

MACROS = [
    # Deterministic ids: '<prefix>' followed by the md5 of the key modulo 10^12, as BIGINT. The
    # 128-bit md5 is folded in four 32-bit chunks to stay within HUGEINT.
    """CREATE OR REPLACE MACRO md5_id(prefix, key) AS CAST(prefix || CAST((
        ((((CAST(('0x' || substr(md5(key), 1, 8))::UBIGINT AS HUGEINT) % 1000000000000) * 4294967296
           + ('0x' || substr(md5(key), 9, 8))::UBIGINT) % 1000000000000 * 4294967296
           + ('0x' || substr(md5(key), 17, 8))::UBIGINT) % 1000000000000 * 4294967296
           + ('0x' || substr(md5(key), 25, 8))::UBIGINT) % 1000000000000) AS VARCHAR) AS BIGINT)""",
    f"CREATE OR REPLACE MACRO ceil_day_us(x) AS ((x + {DAY_US} - 1) // {DAY_US}) * {DAY_US}",
    f"CREATE OR REPLACE MACRO at_us(us) AS TIMESTAMP '{REF_DATE}' + to_microseconds(us)",
]


def create_macros(con) -> None:
    for m in MACROS:
        con.execute(m)


def visit_base_sql(stays: str = "stay", spans: str = "span") -> str:
    """Per stay `base_us`, microseconds from 2000-01-01 to its admission: the staged `admit_us`, or
    else a synthetic one.

    Synthetic admissions, within a person, in stay order: pre = how far the stay's rows reach before admission,
    post = how far past its end; margin = ceil_day(max(post_i, pre_{i+1})); the gap after stay i
    is ceil_day(los_i) + 2 margin + 2 days, at least 60 days. The first stay starts at 2000-01-01.
    """
    return f"""
WITH reach AS (
    SELECT s.dataset, s.stay_id, s.person_key, s.stay_order, s.admit_us,
           coalesce(s.los_us, 0) AS los_us,
           greatest(coalesce(-sp.t_min_us, 0), 0) AS pre_us,
           greatest(coalesce(sp.t_max_us - coalesce(s.los_us, 0), 0), 0) AS post_us
    FROM {stays} s LEFT JOIN {spans} sp USING (dataset, stay_id)
),
gap AS (
    SELECT *,
           ceil_day_us(los_us)
             + 2 * ceil_day_us(greatest(post_us, coalesce(lead(pre_us) OVER person_stays, 0)))
             + 2 * {DAY_US} AS need_us
    FROM reach
    WINDOW person_stays AS (PARTITION BY dataset, person_key ORDER BY stay_order)
)
SELECT dataset, stay_id,
       coalesce(admit_us - {EPOCH_2000_US}, sum(CASE WHEN need_us > {MIN_SPACING_DAYS} * CAST({DAY_US} AS BIGINT)
                         THEN need_us ELSE {MIN_SPACING_DAYS} * CAST({DAY_US} AS BIGINT) END)
                OVER (PARTITION BY dataset, person_key ORDER BY stay_order
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS base_us
FROM gap
"""
