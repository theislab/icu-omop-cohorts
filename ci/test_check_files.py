"""The guard of ci/check_files.py, on synthetic files: no data enters the repository."""

import json
import os

import check_files
import pytest

DEMO = "   icu_stays  persons\n0        140      100"


def notebook(path, counts=DEMO, extra=""):
    cells = [{"cell_type": "code", "execution_count": 1, "metadata": {}, "source": [], "outputs": [
        {"output_type": "execute_result", "execution_count": 1, "metadata": {}, "data": {"text/plain": counts}},
        {"output_type": "stream", "name": "stdout", "text": extra}]}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    return str(path)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_demo_outputs_allowed(repo):
    assert check_files.problems(notebook(repo / check_files.WITH_OUTPUTS).replace(f"{repo}{os.sep}", "")) is None


def test_outputs_of_another_cdm_rejected(repo):
    path = notebook(repo / check_files.WITH_OUTPUTS, counts="   icu_stays  persons\n0      94458    65366")
    assert "MIMIC-IV demo" in check_files.problems(path.replace(f"{repo}{os.sep}", ""))


def test_local_path_rejected(repo):
    path = notebook(repo / check_files.WITH_OUTPUTS, extra="/home/someone/data\n")
    assert "local path" in check_files.problems(path.replace(f"{repo}{os.sep}", ""))


def test_other_notebook_with_outputs_rejected(repo):
    assert "store it without them" in check_files.problems(notebook(repo / "x.ipynb").replace(f"{repo}{os.sep}", ""))


@pytest.mark.parametrize("name", ["cohorts/frames/dyn.parquet", "x.npy", "measurement.csv", "a/b.csv.gz"])
def test_data_rejected(name):
    assert check_files.problems(name) is not None


def test_mapping_tables_allowed():
    assert check_files.problems("omop_etl/mappings/units.csv") is None
