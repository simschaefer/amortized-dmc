
from pandas.testing import assert_frame_equal
import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import numpy as np
import pandas as pd
import pytest

from dmc_helpers import format_sim_data


def test_format_sim_data_returns_expected_long_format():
    sim_data = {
        "rt": np.array(
            [
                [[0.5], [0.6], [0.7]],
                [[0.8], [0.9], [1.0]],
            ]
        ),
        "accuracy": np.array(
            [
                [[1.0], [0.0], [1.0]],
                [[0.0], [1.0], [1.0]],
            ]
        ),
        "conditions": np.array(
            [
                [[0.0], [1.0], [0.0]],
                [[1.0], [0.0], [1.0]],
            ]
        ),
    }

    result = format_sim_data(sim_data, congruency_coding=0, only_convergents=True, id_name="id")

    assert len(result) == 6
    assert set(result.columns) == {"rt", "accuracy", "conditions", "id", "congruency", "accuracy_name"}

    assert set(result["id"]) == {0, 1}
    assert set(result["congruency"]) == {"congruent", "incongruent"}
    assert set(result["accuracy_name"]) == {"correct", "incorrect"}

    # Check a few exact values
    first_row = result.iloc[0]
    assert first_row["rt"] == 0.5
    assert first_row["accuracy"] == 1.0
    assert first_row["conditions"] == 0.0
    assert first_row["id"] == 0
    assert first_row["congruency"] == "congruent"
    assert first_row["accuracy_name"] == "correct"


def test_format_sim_data_excludes_nonconvergents_by_default():
    sim_data = {
        "rt": np.array(
            [
                [[0.5], [-1.0], [0.7]],
            ]
        ),
        "accuracy": np.array(
            [
                [[1.0], [0.0], [1.0]],
            ]
        ),
        "conditions": np.array(
            [
                [[0.0], [1.0], [0.0]],
            ]
        ),
    }

    result = format_sim_data(sim_data, only_convergents=True)

    assert len(result) == 2
    assert (-1.0 not in result["rt"].values)


def test_format_sim_data_keeps_nonconvergents_when_requested():
    sim_data = {
        "rt": np.array(
            [
                [[0.5], [-1.0], [0.7]],
            ]
        ),
        "accuracy": np.array(
            [
                [[1.0], [0.0], [1.0]],
            ]
        ),
        "conditions": np.array(
            [
                [[0.0], [1.0], [0.0]],
            ]
        ),
    }

    result = format_sim_data(sim_data, only_convergents=False)

    assert len(result) == 3
    assert -1.0 in result["rt"].values


def test_format_sim_data_supports_custom_id_name():
    sim_data = {
        "rt": np.array(
            [
                [[0.5], [0.6]],
                [[0.7], [0.8]],
            ]
        ),
        "accuracy": np.array(
            [
                [[1.0], [0.0]],
                [[1.0], [1.0]],
            ]
        ),
        "conditions": np.array(
            [
                [[0.0], [1.0]],
                [[1.0], [0.0]],
            ]
        ),
    }

    result = format_sim_data(sim_data, id_name="subject")

    assert "subject" in result.columns
    assert "id" not in result.columns
    assert set(result["subject"]) == {0, 1}


def test_format_sim_data_supports_custom_congruency_coding():
    sim_data = {
        "rt": np.array(
            [
                [[0.5], [0.6]],
            ]
        ),
        "accuracy": np.array(
            [
                [[1.0], [0.0]],
            ]
        ),
        "conditions": np.array(
            [
                [[2.0], [3.0]],
            ]
        ),
    }

    result = format_sim_data(sim_data, congruency_coding=2)

    assert list(result["congruency"]) == ["congruent", "incongruent"]