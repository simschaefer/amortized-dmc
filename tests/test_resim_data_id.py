import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

import numpy as np
import pandas as pd
import pytest

from dmc_helpers import resim_data_id
from dmc_simulator import DMC


class FakeSimulator:
    def __init__(self):
        self.calls = []

    def experiment(self, **kwargs):
        self.calls.append(kwargs.copy())
        num_obs = kwargs["num_obs"]
        return {
            "rt": np.full(num_obs, 0.5),
            "accuracy": np.ones(num_obs, dtype=int),
            "conditions": np.zeros(num_obs, dtype=int),
        }


@pytest.fixture
def posterior_samples_df():
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "tau": [10.0, 20.0, 30.0],
            "mu_c": [0.1, 0.2, 0.3],
            "mu_r": [100.0, 110.0, 120.0],
            "b": [50.0, 60.0, 70.0],
            "sd_r": [5.0, 6.0, 7.0],
        }
    )


@pytest.fixture
def real_simulator():
    rng = np.random.default_rng(123)

    return DMC(
        prior_means=np.array([100.0, 80.0, 0.5, 300.0, 120.0, 30.0]),
        prior_sds=np.array([10.0, 10.0, 0.1, 20.0, 10.0, 5.0]),
        param_names=("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
        fixed_num_obs=10,
        rng=rng,
    )


# -----------------------------
# Unit tests with fake simulator
# -----------------------------

def test_resim_data_id_calls_simulator_with_expected_parameters(
    monkeypatch, posterior_samples_df
):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    simulator = FakeSimulator()

    _, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=3,
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18  # 6 parameters × 3 samples each

    assert len(simulator.calls) == 3
    assert simulator.calls == [
        {"A": 1.0, "tau": 10.0, "mu_c": 0.1, "mu_r": 100.0, "b": 50.0, "sd_r": 5.0, "num_obs": 2},
        {"A": 2.0, "tau": 20.0, "mu_c": 0.2, "mu_r": 110.0, "b": 60.0, "sd_r": 6.0, "num_obs": 2},
        {"A": 3.0, "tau": 30.0, "mu_c": 0.3, "mu_r": 120.0, "b": 70.0, "sd_r": 7.0, "num_obs": 2},
    ]


def test_resim_data_id_filters_samples_below_lower_bound_with_fake_simulator(monkeypatch):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    simulator = FakeSimulator()

    post_sample_data = pd.DataFrame(
        {
            "A": [-1.0, 2.0, 3.0],
            "tau": [10.0, -20.0, 30.0],
            "mu_c": [0.1, 0.2, 0.3],
            "mu_r": [100.0, 110.0, 120.0],
            "b": [50.0, 60.0, 70.0],
            "sd_r": [5.0, 6.0, 7.0],
        }
    )

    _, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=post_sample_data,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=2,
        lower_bound=0,
    )

    assert n_excluded_samples == 2
    assert n_all_samples == 18

    assert len(simulator.calls) == 2
    for call in simulator.calls:
        assert call["A"] >= 0
        assert call["tau"] >= 0
        assert call["mu_c"] >= 0
        assert call["mu_r"] >= 0
        assert call["b"] >= 0
        assert call["sd_r"] >= 0


def test_resim_data_id_supports_custom_id_name_with_fake_simulator(
    monkeypatch, posterior_samples_df
):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    simulator = FakeSimulator()

    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=3,
        simulator=simulator,
        id=42,
        id_name="subject",
        num_resims=2,
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18

    assert "subject" in result.columns
    assert set(result["subject"]) == {42}
    assert "id" not in result.columns


def test_resim_data_id_respects_param_names_subset_with_fake_simulator(
    monkeypatch, posterior_samples_df
):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    simulator = FakeSimulator()

    _, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=2,
        param_names=("A", "tau"),
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 6  # 2 parameters × 3 samples each

    assert len(simulator.calls) == 2
    for call in simulator.calls:
        assert set(call.keys()) == {"A", "tau", "num_obs"}


# --------------------------------
# Integration tests with real DMC
# --------------------------------

def test_resim_data_id_returns_expected_shape_and_columns_with_real_simulator(
    monkeypatch, real_simulator, posterior_samples_df
):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=6,
        simulator=real_simulator,
        id="s1",
        id_name="id",
        num_resims=3,
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18

    assert set(result.columns) == {"rt", "accuracy", "conditions", "num_obs", "num_resim", "id"}
    assert (result["num_obs"] == 6).all()
    assert len(result) == 18
    assert set(result["num_resim"]) == {0, 1, 2}
    assert set(result["id"]) == {"s1"}


def test_resim_data_id_returns_valid_trial_level_output_with_real_simulator(
    monkeypatch, real_simulator, posterior_samples_df
):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=8,
        simulator=real_simulator,
        id="p01",
        num_resims=2,
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18

    counts_per_resim = result.groupby("num_resim").size()
    assert (counts_per_resim == 8).all()

    assert result["rt"].shape[0] == 16
    assert result["accuracy"].shape[0] == 16
    assert result["conditions"].shape[0] == 16

    assert result["accuracy"].isin([-1, 0, 1]).all()
    assert result["conditions"].isin([0, 1]).all()


def test_resim_data_id_filters_negative_samples_with_real_simulator(
    monkeypatch, real_simulator
):
    monkeypatch.setattr(np.random, "shuffle", lambda x: None)

    post_sample_data = pd.DataFrame(
        {
            "A": [-100.0, 110.0, 120.0],
            "tau": [80.0, -85.0, 90.0],
            "mu_c": [0.4, 0.5, 0.6],
            "mu_r": [300.0, 310.0, 320.0],
            "b": [100.0, 110.0, 120.0],
            "sd_r": [20.0, 25.0, 30.0],
        }
    )

    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=post_sample_data,
        num_obs=5,
        simulator=real_simulator,
        id="s1",
        num_resims=2,
        lower_bound=0,
    )

    assert n_excluded_samples == 2
    assert n_all_samples == 18

    assert len(result) == 10
    assert set(result["num_resim"]) == {0, 1}