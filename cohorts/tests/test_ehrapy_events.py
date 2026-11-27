"""ehrapy's event markers (`ep.tl.mark_events`, `ep.tl.mark_sustained_events`, `ep.tl.mark_paired_events`) and the
builder's suspected infection (`_suspected_infection`, `_first_infection`) against the builder's former `abx_cont_times`
and `si_and`, kept here as the references, stay by stay, on random administration and sample records. Synthetic data
only.

The records go on an hourly grid as the cohort puts them: a variable that holds, at the start hour of the records, the
longest of their durations floored to the hour, and a sample variable that is 1 at every sample hour. The grid spans
every record and sample of every stay, from before admission to after the last record, because abx_cont looks 72 h
ahead of every record and so sees records that a grid of the stay alone would leave out. The cohort test writes the
records into a small CDM and runs the builder from the CDM tables.

Needs an ehrapy with `ep.tl.mark_sustained_events`, as the environment pins; skipped otherwise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import ehrapy as ep  # noqa: E402

if not hasattr(ep.tl, "mark_sustained_events"):
    pytest.skip("this ehrapy has no tl.mark_sustained_events", allow_module_level=True)

import build_cohort as bn  # noqa: E402
import duckdb  # noqa: E402
import ehrdata as ed  # noqa: E402
import ehrdata_dynamics as ED  # noqa: E402

DRUGS = ["drug_0", "drug_1", "drug_2"]
DURATIONS = [0.0, 1 / 60, 0.5, 0.99, 1.0, 1.5, 6.0, 23.5, 24.0, 24.99, 25.0, 47.0, 48.0, 72.0, 100.0,
             -1 / 60, -0.5, -1.0, -2.5, -30.0]


def abx_cont_times(events, death_h=None, win=72, max_gap=24):
    """ricu-extensions `abx_cont` on administration records, the builder's until step 4d.

    `events[i]` is a list of (start_hour, duration_h) for stay i, start floored to the hour and
    possibly negative. ricu floors the duration and the death time to the hour as well
    (change_interval on the win_tbl). Records starting at the same hour keep the longest
    duration. Records after `death_h[i]` are dropped and the window end is min(death, t + win).
    Returns per stay the sorted array of start hours at which a continuous course qualifies."""
    out = []
    for i, ev in enumerate(events):
        if not ev:
            out.append(np.empty(0))
            continue
        d = {}
        for t, dur in ev:
            d[t] = max(d.get(t, -np.inf), np.floor(dur))
        starts = np.array(sorted(d), dtype=float)
        durs = np.array([d[t] for t in sorted(d)], dtype=float)
        death = np.nan if death_h is None else np.floor(death_h[i])
        if np.isfinite(death):
            keep = starts <= death
            starts, durs = starts[keep], durs[keep]
        q = []
        for t0 in starts:
            w = (starts >= t0) & (starts <= t0 + win)
            st, du = starts[w], durs[w]
            end = t0 + win if not np.isfinite(death) else min(death, t0 + win)
            nxt = np.append(st[1:], end)
            gap = np.max(nxt - np.maximum.accumulate(st + du))
            if gap <= max_gap:
                q.append(t0)
        out.append(np.array(q))
    return out


def si_and(abx_times, samp_times, abx_win=24, samp_win=72):
    """ricu si_and, the builder's until step 4d: an antibiotic time counts if a sample follows within `abx_win` hours,
    a sample time if an antibiotic follows within `samp_win` hours. Per stay, sorted times."""
    out = []
    for a, s in zip(abx_times, samp_times):
        a, s = np.asarray(a, float), np.asarray(s, float)
        if not a.size or not s.size:
            out.append(np.empty(0))
            continue
        ja = np.searchsorted(s, a, side="left")
        ok_a = (ja < s.size) & (s[np.minimum(ja, s.size - 1)] - a <= abx_win)
        sa = np.sort(a)
        js = np.searchsorted(sa, s, side="left")
        ok_s = (js < sa.size) & (sa[np.minimum(js, sa.size - 1)] - s <= samp_win)
        out.append(np.unique(np.concatenate([a[ok_a], s[ok_s]])))
    return out


def _records(rng, n):
    """Per stay a list of (start hour, duration, drug), with courses of regular doses, scattered doses and same-hour
    duplicates; the sample hours; and the death hours, NaN for half of the stays."""
    records, samples = [], []
    for _ in range(n):
        rec = []
        for _ in range(rng.integers(0, 3)):  # courses of regular doses
            t, every = int(rng.integers(-120, 150)), int(rng.choice([4, 8, 12, 23, 24, 25, 30, 48]))
            for k in range(rng.integers(1, 8)):
                rec.append(t + k * every)
        rec += list(rng.integers(-120, 200, size=rng.integers(0, 5)))  # scattered doses
        rec += [rec[j] for j in rng.integers(0, len(rec), size=rng.integers(0, 3))] if rec else []
        records.append([(int(t), float(rng.choice(DURATIONS) if rng.random() < 0.8 else rng.uniform(-5, 80)),
                         int(rng.integers(0, len(DRUGS)))) for t in rec])
        near = [t + int(rng.choice([-73, -72, -30, -1, 0, 1, 23, 24, 25, 71, 72, 73])) for t, _, _ in records[-1][:3]]
        samples.append(sorted(set(near + list(rng.integers(-120, 200, size=rng.integers(0, 3))))))
    death = np.array([np.nan if rng.random() < 0.5 or not r else rng.uniform(min(t for t, _, _ in r) - 10,
                      max(t for t, _, _ in r) + 100) for r in records])
    return records, samples, death


def _grid(records, samples, death):
    """The records and samples on an hourly grid that spans all of them, with the death in obs as a position on it."""
    hours = [t for r in records for t, _, _ in r] + [h for s in samples for h in s]
    h_lo, h_hi = min(hours), max(hours) + 1
    X = np.full((len(records), len(DRUGS) + 1, h_hi - h_lo), np.nan)
    for i, (rec, samp) in enumerate(zip(records, samples, strict=True)):
        for t, dur, drug in rec:
            X[i, drug, t - h_lo] = np.fmax(X[i, drug, t - h_lo], np.floor(dur))
        X[i, len(DRUGS), np.asarray(samp, dtype=np.int64) - h_lo] = 1.0
    edata = ed.EHRData(shape=(len(records), len(DRUGS) + 1),
                       obs=pd.DataFrame({"death": death - h_lo}, index=[f"s{i}" for i in range(len(records))]),
                       var=pd.DataFrame(index=[*DRUGS, "sample"]), layers={"tem_data": X})
    edata.tem = pd.DataFrame({"time_value": np.arange(h_lo, h_hi, dtype=np.float64)})
    return edata


def _qualifying(edata, key):
    times = edata.tem["time_value"].to_numpy()
    return [times[row] for row in edata.obsm[key]]


def _differing(ours, reference):
    return sum(not np.array_equal(a, np.asarray(b, dtype=np.float64)) for a, b in zip(ours, reference, strict=True))


def _first(times):
    return np.array([q.min() if len(q) else np.nan for q in times])


def _first_hour(edata, key):
    """The first position of `key` in obs as the hour of the grid of `_grid`."""
    return edata.obs[f"{key}_first"].to_numpy() + edata.tem["time_value"].iloc[0]


@pytest.mark.parametrize(("seed", "win", "max_gap"), [(0, 72, 24), (1, 72, 24), (2, 24, 6), (3, 10, 0)])
def test_mark_sustained_events_equal_abx_cont_times(seed, win, max_gap):
    rng = np.random.default_rng(seed)
    records, samples, death = _records(rng, 800)
    edata = _grid(records, samples, death)
    ep.tl.mark_events(edata, DRUGS, layer="tem_data", key_added="abx")
    ep.tl.mark_events(edata, DRUGS, how="max", layer="tem_data", key_added="abx_hours")
    ep.tl.mark_sustained_events(edata, "abx", duration=win, max_gap=max_gap, lengths="abx_hours",
                                end_key="death", layer="tem_data", key_added="abx_cont")

    reference = abx_cont_times([[(t, d) for t, d, _ in r] for r in records], death, win=win, max_gap=max_gap)
    assert _differing(_qualifying(edata, "abx_cont"), reference) == 0
    np.testing.assert_array_equal(_first_hour(edata, "abx_cont"), _first(reference))
    # the records exercise qualifying and failing stays, and stays that qualify only through their death
    n_qualifying = sum(len(q) > 0 for q in reference)
    assert 0.1 * len(records) < n_qualifying < 0.9 * len(records)
    alive = abx_cont_times([[(t, d) for t, d, _ in r] for r in records], None, win=win, max_gap=max_gap)
    assert sum(len(q) > 0 and len(a) == 0 for q, a in zip(reference, alive, strict=True)) > 0


@pytest.mark.parametrize("seed", range(3))
def test_mark_paired_events_equal_si_and(seed):
    rng = np.random.default_rng(seed)
    records, samples, death = _records(rng, 800)
    edata = _grid(records, samples, death)
    ep.tl.mark_events(edata, DRUGS, layer="tem_data", key_added="abx")
    ep.tl.mark_events(edata, DRUGS, how="max", layer="tem_data", key_added="abx_hours")
    ep.tl.mark_events(edata, "sample", layer="tem_data", key_added="samples")
    ep.tl.mark_sustained_events(edata, "abx", duration=72, max_gap=24, lengths="abx_hours", end_key="death",
                                layer="tem_data", key_added="abx_cont")
    abx_cont = abx_cont_times([[(t, d) for t, d, _ in r] for r in records], death)
    abx_all = [np.unique([t for t, _, _ in r]).astype(np.float64) for r in records]

    for a, abx_times, a_to_b, b_to_a in (("abx_cont", abx_cont, 24, 72), ("abx", abx_all, 24, 72),
                                         ("abx", abx_all, 0, 1), ("abx", abx_all, 72, 24)):
        ep.tl.mark_paired_events(edata, a, "samples", a_to_b=a_to_b, b_to_a=b_to_a, key_added="si")
        reference = si_and(abx_times, samples, abx_win=a_to_b, samp_win=b_to_a)
        assert _differing(_qualifying(edata, "si"), reference) == 0, (a, a_to_b, b_to_a)
        np.testing.assert_array_equal(_first_hour(edata, "si"), _first(reference))
        assert 0 < sum(len(q) > 0 for q in reference) < len(records)


def _infection_grid(records, samples, death):
    """The object of `bn.infection_object` for the records, without a CDM: `abx` and `sample` on hours from the
    first record, the death in obs as a position on them; and the first hour."""
    hours = [t for r in records for t, _ in r] + [h for s in samples for h in s] + [0, 1]
    h_lo, h_hi = min(hours), max(hours) + 1
    X = np.full((len(records), 2, h_hi - h_lo), np.nan)
    for i, (rec, samp) in enumerate(zip(records, samples, strict=True)):
        for t, dur in rec:
            X[i, 0, t - h_lo] = np.fmax(X[i, 0, t - h_lo], np.floor(dur))
        X[i, 1, np.asarray(samp, dtype=np.int64) - h_lo] = 1.0
    obs = pd.DataFrame({"death": np.floor(np.asarray(death, dtype=float)) - h_lo},
                       index=[f"s{i}" for i in range(len(records))])
    return ed.EHRData(obs=obs, var=pd.DataFrame(index=["abx", "sample"]), layers={ED.LAYER: X}), h_lo


def _first_infection(records, samples, death, ds):
    """`bn._first_infection` on the grid of the records, in hours."""
    grid, h_lo = _infection_grid(records, samples, death)
    return bn._first_infection(grid, ds) + h_lo


@pytest.mark.parametrize("seed", range(3))
def test_first_infection_equals_the_references(seed):
    rng = np.random.default_rng(seed)
    records, samples, death = _records(rng, 500)
    records = [[(t, d) for t, d, _ in r] for r in records]
    abx_cont = abx_cont_times(records, death)
    np.testing.assert_array_equal(_first_infection(records, samples, death, "eicu"), _first(abx_cont))
    np.testing.assert_array_equal(_first_infection(records, samples, death, "miiv"), _first(si_and(abx_cont, samples)))


def test_first_infection_rules():
    """ricu-extensions' abx_cont and ricu's si_and on hand-made records: (start hour, duration in hours)."""
    nan = np.nan
    cases = [  # records, death, first suspected infection
        ([(0, 0.0), (24, 0.0), (48, 0.0)], nan, 0),            # daily doses: every gap, also to the window end, 24 h
        ([(0, 0.0), (24, 0.0)], nan, nan),                     # from 24 h the gap to the window end is 48 h
        ([(10, 4.0)], nan, nan),                               # a single 4 h infusion is no course
        ([(10, 72.0)], nan, 10),                               # an order of 72 h spans the window
        ([(10, 4.0), (58, 4.0)], nan, nan),                    # a 44 h gap is too long, and 58 is a single dose
        ([(0, 1.0)], 10.0, 0),                                 # the window ends at death
        ([(20, 1.0)], 10.0, nan),                              # a dose after death is left out
        ([(-30, 1.0), (-10, 1.0), (10, 1.0), (30, 1.0)], nan, -30),   # before admission, kept
        ([(-30, -5.0), (-10, 1.0), (10, 1.0), (30, 1.0)], nan, nan),  # a negative duration shortens the course
        ([], nan, nan),
    ]
    records = [r for r, _, _ in cases]
    death = np.array([d for _, d, _ in cases])
    got = _first_infection(records, [[] for _ in cases], death, "eicu")
    np.testing.assert_array_equal(got, [f for *_, f in cases])
    # miiv: a sample within 24 h after the course start, or the course within 72 h after a sample
    samples = [[20], [], [], [100], [], [-73], [], [-100], [], [5]]
    got = _first_infection(records, samples, death, "miiv")
    np.testing.assert_array_equal(got, [0, nan, nan, nan, nan, nan, nan, -100, nan, nan])


def _cdm(tmp_path, ds, records, samples):
    """A small CDM of the records: stays 1..n, two per person 300 h apart, each with its antibiotic rows (start at
    the record's hour plus some minutes) and, on miiv, its samples linked by FACT_RELATIONSHIP. A visit outside the
    cohort (9999) of the first person has rows of its own, and every stay a self-reported and a non-antibiotic row.
    Returns the connection scoped to the stays and the stays."""
    rng = np.random.default_rng(7)
    t0 = pd.Timestamp("2100-01-01 10:23:00")
    n = len(records)
    vois = np.arange(1, n + 1)
    starts = [t0 + pd.Timedelta(hours=300 * (v % 2)) + pd.Timedelta(days=60 * ((v - 1) // 2)) for v in vois]
    visits = pd.DataFrame({"visit_occurrence_id": [*vois, 9999], "person_id": [*((vois + 1) // 2), 1],
                           "visit_start_datetime": [*starts, starts[0]],
                           "visit_end_datetime": [s + pd.Timedelta(hours=200) for s in [*starts, starts[0]]]})
    visits["visit_start_date"] = visits["visit_start_datetime"].dt.date
    visits["visit_end_date"] = visits["visit_end_datetime"].dt.date
    visits["visit_source_value"] = [f"x-{v}" for v in visits["visit_occurrence_id"]]
    visits.to_parquet(tmp_path / "VISIT_OCCURRENCE.parquet")
    pd.DataFrame({"person_id": np.arange(1, (n + 1) // 2 + 1), "gender_concept_id": 8507, "year_of_birth": 2050,
                  "birth_datetime": pd.Timestamp("2050-01-01")}).to_parquet(tmp_path / "PERSON.parquet")
    name, table = ("Vancomycin", "") if ds == "miiv" else ("vancomycin", "table=medication")
    drugs = []
    for v, start, rec in zip([*vois, 9999], [*starts, starts[0]], [*records, [(5, 100.0)] * 3]):
        for t, dur in rec:
            begin = start + pd.Timedelta(hours=t, minutes=int(rng.integers(0, 60)))
            if dur < 0:  # eicu: a source stop before the start, which the CDM keeps in sig
                drugs.append((v, name, begin, begin, f"{table};source_stop_min={dur * 60!r}", 0))
            else:
                drugs.append((v, name, begin, begin + pd.Timedelta(seconds=round(dur * 3600)), table, 0))
        drugs.append((v, name, start, start + pd.Timedelta(hours=200), table, 32865))     # self-reported
        drugs.append((v, "Norepinephrine", start, start + pd.Timedelta(hours=200), table, 0))
    pd.DataFrame(drugs, columns=["visit_occurrence_id", "drug_source_value", "drug_exposure_start_datetime",
                                 "drug_exposure_end_datetime", "sig", "drug_type_concept_id"]
                 ).to_parquet(tmp_path / "DRUG_EXPOSURE.parquet")
    spec = [(v, start + pd.Timedelta(hours=h, minutes=int(rng.integers(0, 60))))
            for v, start, samp in zip([*vois, 9999], [*starts, starts[0]], [*samples, [6, 50]]) for h in samp]
    pd.DataFrame({"specimen_id": np.arange(len(spec)), "person_id": [(v + 1) // 2 if v != 9999 else 1 for v, _ in spec],
                  "specimen_datetime": [t for _, t in spec]}).to_parquet(tmp_path / "SPECIMEN.parquet")
    pd.DataFrame({"domain_concept_id_1": 36, "fact_id_1": np.arange(len(spec)), "domain_concept_id_2": 8,
                  "fact_id_2": [v for v, _ in spec], "relationship_concept_id": 33136}
                 ).to_parquet(tmp_path / "FACT_RELATIONSHIP.parquet")
    con = duckdb.connect()
    con.execute(f"CREATE TABLE _cohort_visits AS SELECT * FROM parquet_scan('{tmp_path}/VISIT_OCCURRENCE.parquet') "
                f"WHERE visit_occurrence_id <= {n}")
    con.execute("CREATE VIEW visit_occurrence AS SELECT * FROM _cohort_visits")
    con.execute(f"CREATE VIEW person AS SELECT * FROM parquet_scan('{tmp_path}/PERSON.parquet')")
    return con, ED.setup_stays(con)


@pytest.mark.parametrize("ds", ["miiv", "eicu"])
def test_suspected_infection_from_the_cdm_equals_the_references(tmp_path, monkeypatch, ds):
    """The builder from the CDM tables, in chunks of stays with moved starts, against the references on the records."""
    rng = np.random.default_rng(11)
    records, samples, death = _records(rng, 120)
    records = [[(t, d if ds == "eicu" else max(d, 0.0)) for t, d, _ in r] for r in records]
    con, stays = _cdm(tmp_path, ds, records, samples)
    monkeypatch.setattr(bn, "OMOP", str(tmp_path))
    monkeypatch.setattr(bn, "SI_MAX_CELLS", 3000)   # several chunks
    bn._suspected_infection(con, ds, stays, death)
    out = stays.obs

    # the reference reads the durations as the CDM holds them: whole seconds, or the source stop in minutes
    rows = [[(t, (d * 60) / 60 if d < 0 else round(d * 3600) / 3600) for t, d in r] for r in records]
    reference = abx_cont_times(rows, death)
    if ds == "miiv":
        reference = si_and(reference, samples)
    assert list(stays.obs["voi"]) == list(range(1, len(records) + 1))
    np.testing.assert_array_equal(bn._obs_hours(out, "si_first"), _first(reference))
    np.testing.assert_array_equal(out["n_abx"].to_numpy(), [len(r) for r in rows])
    np.testing.assert_array_equal(out["abx_first"].to_numpy(), [min((t for t, _ in r), default=np.nan) for r in rows])
    assert 0 < out["si_first"].notna().sum() < len(records)
    assert con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name IN ('drug_exposure', 'specimen')"
                       ).fetchone()[0] == 0, "the tables of the rows are dropped"
