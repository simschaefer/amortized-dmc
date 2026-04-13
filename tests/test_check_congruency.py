
from pandas.testing import assert_frame_equal
import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import pandas as pd
import pytest

from dmc_helpers import check_congruency


def test_returns_data_unchanged_when_congruency_is_none():
    data = pd.DataFrame(
        {
            "rt": [0.4, 0.5],
            "condition": ["con", "inc"],
        }
    )

    result = check_congruency(data.copy(), rt="rt", congruency=None)

    pd.testing.assert_frame_equal(result, data)


def test_raises_when_congruency_column_is_missing():
    data = pd.DataFrame(
        {
            "rt": [0.4, 0.5],
        }
    )

    with pytest.raises(ValueError, match=r"Variable 'condition' does not exist in data"):
        check_congruency(data, rt="rt", congruency="condition")


def test_raises_for_invalid_congruency_coding():
    data = pd.DataFrame(
        {
            "rt": [0.4, 0.5],
            "condition": ["foo", "bar"],
        }
    )

    with pytest.raises(ValueError, match=r"Congruency variable is coded as"):
        check_congruency(data, rt="rt", congruency="condition")


def test_recodes_con_inc_and_warns():
    data = pd.DataFrame(
        {
            "rt": [400, 500],
            "condition": ["con", "inc"],
        }
    )

    with pytest.warns(UserWarning, match=r"has been recoded to con -> congruent / inc -> incongruent"):
        result = check_congruency(data.copy(), rt="rt", congruency="condition")

    assert set(result["condition"]) == {"congruent", "incongruent"}
    assert result.loc[0, "condition"] == "congruent"
    assert result.loc[1, "condition"] == "incongruent"


def test_recodes_zero_one_and_warns():
    data = pd.DataFrame(
        {
            "rt": [400, 500],
            "condition": [0, 1],
        }
    )

    with pytest.warns(UserWarning, match=r"has been recoded to 0 -> congruent / 1 -> incongruent"):
        result = check_congruency(data.copy(), rt="rt", congruency="condition")

    assert set(result["condition"]) == {"congruent", "incongruent"}
    assert result.loc[0, "condition"] == "congruent"
    assert result.loc[1, "condition"] == "incongruent"


def test_recodes_default_labels_to_custom_output_and_warns():
    data = pd.DataFrame(
        {
            "rt": [400, 500],
            "condition": ["congruent", "incongruent"],
        }
    )

    with pytest.warns(
        UserWarning,
        match=r"has been recoded to congruent -> CON / incongruent -> INC"
    ):
        result = check_congruency(
            data.copy(),
            rt="rt",
            congruency="condition",
            output_coding_con="CON",
            output_coding_inc="INC",
        )

    assert set(result["condition"]) == {"CON", "INC"}
    assert result.loc[0, "condition"] == "CON"
    assert result.loc[1, "condition"] == "INC"


def test_default_congruent_incongruent_coding_does_not_warn_when_difference_positive():
    data = pd.DataFrame(
        {
            "rt": [400, 500],
            "condition": ["congruent", "incongruent"],
        }
    )

    result = check_congruency(data.copy(), rt="rt", congruency="condition")

    assert set(result["condition"]) == {"congruent", "incongruent"}
    assert result.loc[0, "condition"] == "congruent"
    assert result.loc[1, "condition"] == "incongruent"


def test_warns_when_rt_difference_is_negative():
    data = pd.DataFrame(
        {
            "rt": [500, 400],
            "condition": ["congruent", "incongruent"],
        }
    )

    with pytest.warns(
        UserWarning,
        match=r"RT Difference between incongruent - congruent conditions is negative"
    ):
        result = check_congruency(data.copy(), rt="rt", congruency="condition")

    assert set(result["condition"]) == {"congruent", "incongruent"}