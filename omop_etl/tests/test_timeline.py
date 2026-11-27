import hashlib

import pytest

from omop_etl import db, timeline


@pytest.fixture
def con():
    c = db.connect_plain()
    timeline.create_macros(c)
    return c


def old_id(key, prefix):
    return int(f"{prefix}{int(hashlib.md5(key.encode()).hexdigest(), 16) % 10**12}")


# mock keys, of the form <dataset>-<source id>
@pytest.mark.parametrize("key", ["mimic4-11111111", "mimic4-22222222", "eicu-333333", "hirid-1", "x"])
def test_md5_id_is_the_folded_md5(con, key):
    assert con.execute("SELECT md5_id('2', ?)", [key]).fetchone()[0] == old_id(key, 2)


def test_md5_id_many(con):
    keys = [f"eicu-{i}" for i in range(2000)]
    got = con.execute("SELECT md5_id('1', k) FROM unnest(?) t(k)", [keys]).fetchall()
    assert [g[0] for g in got] == [old_id(k, 1) for k in keys]


def test_ceil_day(con):
    d = timeline.DAY_US
    assert con.execute(f"SELECT ceil_day_us(0), ceil_day_us(1), ceil_day_us({d}), ceil_day_us({d}+1)").fetchone() == (0, d, d, 2 * d)
