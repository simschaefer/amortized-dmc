
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
from numpy.testing import assert_array_equal

from dmc_helpers import format_empirical_data


def test_format_empirical_data_returns_expected_dict_and_shapes():
    data = pd.DataFrame(
        {
            "rt": [0.40, 0.55, 0.62],
            "accuracy": [1, 0, 1],
            "congruency": [0, 1, 0],
        }
    )

    result = format_empirical_data(
        data=data,
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    assert set(result.keys()) == {"rt", "accuracy", "conditions", "num_obs"}

    assert result["rt"].shape == (1, 3, 1)
    assert result["accuracy"].shape == (1, 3, 1)
    assert result["conditions"].shape == (1, 3, 1)
    assert result["num_obs"].shape == (1, 1)

    assert_array_equal(result["rt"], np.array([[[0.40], [0.55], [0.62]]]))
    assert_array_equal(result["accuracy"], np.array([[[1], [0], [1]]]))
    assert_array_equal(result["conditions"], np.array([[[0], [1], [0]]]))
    assert_array_equal(result["num_obs"], np.array([[3]]))


def test_format_empirical_data_recodes_congruency_strings_to_zero_one():
    data = pd.DataFrame(
        {
            "rt": [0.40, 0.55],
            "accuracy": [1, 0],
            "congruency": ["congruent", "incongruent"],
        }
    )

    with pytest.warns(UserWarning, match=r"has been recoded to congruent -> 0 / incongruent -> 1"):
        result = format_empirical_data(
            data=data,
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )

    assert_array_equal(result["conditions"], np.array([[[0], [1]]]))
    assert_array_equal(result["num_obs"], np.array([[2]]))


def test_format_empirical_data_recodes_con_inc_to_zero_one():
    data = pd.DataFrame(
        {
            "rt": [0.40, 0.55],
            "accuracy": [1, 1],
            "congruency": ["con", "inc"],
        }
    )

    with pytest.warns(UserWarning, match=r"has been recoded to con -> 0 / inc -> 1"):
        result = format_empirical_data(
            data=data,
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )
        
    assert_array_equal(result["conditions"], np.array([[[0], [1]]]))


def test_format_empirical_data_raises_for_invalid_congruency_coding():
    data = pd.DataFrame(
        {
            "rt": [0.40, 0.55],
            "accuracy": [1, 0],
            "congruency": ["foo", "bar"],
        }
    )

    with pytest.raises(ValueError, match=r"Congruency variable is coded as"):
        format_empirical_data(
            data=data,
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )


def test_format_empirical_data_raises_for_missing_required_column():
    data = pd.DataFrame(
        {
            "rt": [0.40, 0.55],
            "congruency": [0, 1],
        }
    )

    with pytest.raises(KeyError, match="accuracy"):
        format_empirical_data(
            data=data,
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )