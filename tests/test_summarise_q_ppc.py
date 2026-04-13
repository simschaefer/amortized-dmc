
from pandas.testing import assert_frame_equal
import os
import sys

import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import summarise_q_ppc

def test_summarise_q_ppc_basic_grouping():
    data = pd.DataFrame(
        {
            "id": ["s1"] * 8,
            "congruency": [
                "congruent", "congruent", "congruent", "congruent",
                "incongruent", "incongruent", "incongruent", "incongruent",
            ],
            "rt": [100, 200, 300, 400, 200, 300, 400, 500],
            "accuracy": [1, 1, 0, 1, 1, 0, 1, 1],
        }
    )

    result = summarise_q_ppc(
        data=data,
        grouping_vars=["id", "congruency"],
        id_name="id",
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    result = result.sort_values(["id", "congruency"]).reset_index(drop=True)
    result.columns.name = None

    expected = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "congruency": ["congruent", "incongruent"],
            "mean_rt": [250.0, 350.0],
            "mean_acc": [0.75, 0.75],
            "rt_q25": [175.0, 275.0],
            "rt_q50": [250.0, 350.0],
            "rt_q75": [325.0, 425.0],
        }
    )

    assert_frame_equal(result, expected)


def test_summarise_q_ppc_with_accuracy_in_grouping_vars():
    data = pd.DataFrame(
        {
            "id": ["s1"] * 8,
            "congruency": [
                "congruent", "congruent", "congruent", "congruent",
                "incongruent", "incongruent", "incongruent", "incongruent",
            ],
            "rt": [100, 200, 300, 400, 200, 300, 400, 500],
            "accuracy": [1, 1, 0, 1, 1, 0, 1, 1],
        }
    )

    result = summarise_q_ppc(
        data=data,
        grouping_vars=["id", "congruency", "accuracy"],
        id_name="id",
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    result = result.sort_values(["id", "congruency", "accuracy"]).reset_index(drop=True)
    result.columns.name = None

    expected = pd.DataFrame(
        {
            "id": ["s1", "s1", "s1", "s1"],
            "congruency": ["congruent", "congruent", "incongruent", "incongruent"],
            "accuracy": [0, 1, 0, 1],
            "mean_rt": [300.0, 233.33333333333334, 300.0, 366.6666666666667],
            "mean_acc": [0.75, 0.75, 0.75, 0.75],
            "rt_q25": [300.0, 150.0, 300.0, 300.0],
            "rt_q50": [300.0, 200.0, 300.0, 400.0],
            "rt_q75": [300.0, 300.0, 300.0, 450.0],
        }
    )

    assert_frame_equal(result, expected)


def test_summarise_q_ppc_recodes_numeric_congruency():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1", "s1", "s1"],
            "congruency": [0, 0, 1, 1],
            "rt": [100, 200, 300, 400],
            "accuracy": [1, 1, 1, 1],
        }
    )

    with pytest.warns(
        UserWarning,
        match=r"has been recoded to 0 -> congruent / 1 -> incongruent"
    ):
        result = summarise_q_ppc(
            data=data,
            grouping_vars=["id", "congruency"],
            id_name="id",
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )

    assert set(result["congruency"]) == {"congruent", "incongruent"}

def test_summarise_q_ppc_raises_for_missing_required_column():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "congruency": ["congruent", "incongruent"],
            "accuracy": [1, 0],
        }
    )

    with pytest.raises(ValueError, match=r"Variable 'rt' does not exist in data"):
        summarise_q_ppc(
            data=data,
            grouping_vars=["id", "congruency"],
            id_name="id",
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )