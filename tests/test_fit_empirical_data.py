
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

from dmc_helpers import fit_empirical_data


class FakeApproximator:
    def __init__(self, include_sd_r=True):
        self.include_sd_r = include_sd_r
        self.calls = []

    def sample(self, conditions, num_samples):
        self.calls.append(
            {
                "conditions": conditions,
                "num_samples": num_samples,
            }
        )

        n = num_samples

        samples = {
            "A": np.full((n, 1), 100.0),
            "tau": np.full((n, 1), 80.0),
            "mu_c": np.full((n, 1), 0.5),
            "mu_r": np.full((n, 1), 300.0),
            "b": np.full((n, 1), 120.0),
        }

        if self.include_sd_r:
            samples["sd_r"] = np.full((n, 1), 30.0)

        return samples


def test_fit_empirical_data_returns_samples_for_each_id():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2"],
            "rt": [0.40, 0.50, 0.45, 0.55],
            "accuracy": [1, 0, 1, 1],
            "congruency": [0, 1, 0, 1],
        }
    )

    approximator = FakeApproximator()

    result = fit_empirical_data(
        data=data,
        approximator=approximator,
        num_samples=3,
        id_name="id",
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    assert set(result.columns) == {
        "A", "tau", "mu_c", "mu_r", "b", "sd_r", "id", "sampling_time"
    }

    # 2 subjects × 3 posterior samples each
    assert len(result) == 6

    assert set(result["id"]) == {"s1", "s2"}
    assert (result["A"] == 100.0).all()
    assert (result["tau"] == 80.0).all()
    assert (result["mu_c"] == 0.5).all()
    assert (result["mu_r"] == 300.0).all()
    assert (result["b"] == 120.0).all()
    assert (result["sd_r"] == 30.0).all()
    assert (result["sampling_time"] >= 0).all()


def test_fit_empirical_data_calls_approximator_once_per_id():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "rt": [0.40, 0.50, 0.45, 0.55, 0.60, 0.70],
            "accuracy": [1, 0, 1, 1, 0, 1],
            "congruency": [0, 1, 0, 1, 0, 1],
        }
    )

    approximator = FakeApproximator()

    result = fit_empirical_data(
        data=data,
        approximator=approximator,
        num_samples=4,
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    assert len(approximator.calls) == 3
    assert all(call["num_samples"] == 4 for call in approximator.calls)
    assert len(result) == 12


def test_fit_empirical_data_passes_correctly_shaped_conditions():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "rt": [0.40, 0.50],
            "accuracy": [1, 0],
            "congruency": [0, 1],
        }
    )

    approximator = FakeApproximator()

    fit_empirical_data(
        data=data,
        approximator=approximator,
        num_samples=2,
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    assert len(approximator.calls) == 1

    conditions = approximator.calls[0]["conditions"]

    assert set(conditions.keys()) == {"rt", "accuracy", "conditions", "num_obs"}
    assert conditions["rt"].shape == (1, 2, 1)
    assert conditions["accuracy"].shape == (1, 2, 1)
    assert conditions["conditions"].shape == (1, 2, 1)
    assert conditions["num_obs"].shape == (1, 1)


def test_fit_empirical_data_respects_num_samples():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "rt": [0.40, 0.50],
            "accuracy": [1, 0],
            "congruency": [0, 1],
        }
    )

    approximator = FakeApproximator()

    result = fit_empirical_data(
        data=data,
        approximator=approximator,
        num_samples=5,
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    assert len(result) == 5


def test_fit_empirical_data_raises_for_missing_column():
    data = pd.DataFrame(
        {
            "id": ["s1"],
            "rt": [0.40],
            "congruency": [0],
        }
    )

    approximator = FakeApproximator()

    with pytest.raises(ValueError, match=r"Variable 'accuracy' does not exist in data"):
        fit_empirical_data(
            data=data,
            approximator=approximator,
            num_samples=3,
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )