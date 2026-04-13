import os
import sys

import pandas as pd
import pytest

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import resim_data


class DummySimulator:
    pass


@pytest.fixture
def empirical_data():
    return pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2", "s2"],
            "rt": [0.4, 0.5, 0.6, 0.7, 0.8],
            "accuracy": [1, 0, 1, 1, 0],
            "congruency": ["congruent", "incongruent", "congruent", "incongruent", "congruent"],
        }
    )


@pytest.fixture
def post_samples():
    return pd.DataFrame(
        {
            "id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "A": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "tau": [10.0, 20.0, 30.0, 12.0, 22.0, 32.0],
            "mu_c": [0.1, 0.2, 0.3, 0.15, 0.25, 0.35],
            "mu_r": [100.0, 110.0, 120.0, 105.0, 115.0, 125.0],
            "b": [50.0, 60.0, 70.0, 55.0, 65.0, 75.0],
            "sd_r": [5.0, 6.0, 7.0, 5.5, 6.5, 7.5],
        }
    )


def test_resim_data_concatenates_results_and_recodes_congruency(monkeypatch, empirical_data, post_samples):
    calls = []

    def fake_resim_data_id(post_sample_data, num_obs, simulator, id, param_names, lower_bound, num_resims=50):
        calls.append(
            {
                "id": id,
                "num_obs": num_obs,
                "num_resims": num_resims,
                "n_post_rows": len(post_sample_data),
            }
        )
        return (
            pd.DataFrame(
                {
                    "rt": [0.51, 0.62],
                    "accuracy": [1, 0],
                    "conditions": [0.0, 1.0],
                    "num_resim": [0, 1],
                    "id": [id, id],
                }
            ),
            0,
            len(post_sample_data),
        )

    monkeypatch.setattr("dmc_helpers.resim_data_id", fake_resim_data_id)

    result = resim_data(
        empirical_data=empirical_data,
        post_samples=post_samples,
        simulator=DummySimulator(),
        num_resims=2,
        rt="rt",
        id_name="id",
        congruency="congruency",
        simulator_congruency="conditions",
    )

    assert len(result) == 4
    assert set(result["id"]) == {"s1", "s2"}
    assert set(result["congruency"]) == {"congruent", "incongruent"}
    assert set(result["conditions"]) == {0.0, 1.0}

    assert calls == [
        {"id": "s1", "num_obs": 2, "num_resims": 2, "n_post_rows": 3},
        {"id": "s2", "num_obs": 3, "num_resims": 2, "n_post_rows": 3},
    ]


def test_resim_data_excludes_nonconvergents_by_default(monkeypatch, empirical_data, post_samples):
    def fake_resim_data_id(post_sample_data, num_obs, simulator, id, param_names, lower_bound, num_resims=50):
        return (
            pd.DataFrame(
                {
                    "rt": [-1.0, 0.55, -1.0, 0.70],
                    "accuracy": [-1, 1, -1, 0],
                    "conditions": [0.0, 0.0, 1.0, 1.0],
                    "num_resim": [0, 0, 1, 1],
                    "id": [id, id, id, id],
                }
            ),
            1,
            len(post_sample_data),
        )

    monkeypatch.setattr("dmc_helpers.resim_data_id", fake_resim_data_id)

    result = resim_data(
        empirical_data=empirical_data,
        post_samples=post_samples,
        simulator=DummySimulator(),
        num_resims=2,
        rt="rt",
        id_name="id",
        congruency="congruency",
        simulator_congruency="conditions",
        exclude_nonconvergents=True,
    )

    assert (result["rt"] == -1).sum() == 0
    assert set(result["congruency"]) == {"congruent", "incongruent"}


def test_resim_data_keeps_nonconvergents_when_requested(monkeypatch, empirical_data, post_samples):
    def fake_resim_data_id(post_sample_data, num_obs, simulator, id, param_names, lower_bound, num_resims=50):
        return (
            pd.DataFrame(
                {
                    "rt": [-1.0, 0.55],
                    "accuracy": [-1, 1],
                    "conditions": [0.0, 1.0],
                    "num_resim": [0, 1],
                    "id": [id, id],
                }
            ),
            1,
            len(post_sample_data),
        )

    monkeypatch.setattr("dmc_helpers.resim_data_id", fake_resim_data_id)

    result = resim_data(
        empirical_data=empirical_data,
        post_samples=post_samples,
        simulator=DummySimulator(),
        num_resims=2,
        rt="rt",
        id_name="id",
        congruency="congruency",
        simulator_congruency="conditions",
        exclude_nonconvergents=False,
    )

    assert (result["rt"] == -1).sum() == 2


def test_resim_data_supports_custom_condition_coding(monkeypatch, empirical_data, post_samples):
    def fake_resim_data_id(post_sample_data, num_obs, simulator, id, param_names, lower_bound, num_resims=50):
        return (
            pd.DataFrame(
                {
                    "rt": [0.5, 0.6],
                    "accuracy": [1, 0],
                    "cond_code": [10.0, 20.0],
                    "num_resim": [0, 1],
                    "id": [id, id],
                }
            ),
            0,
            len(post_sample_data),
        )

    monkeypatch.setattr("dmc_helpers.resim_data_id", fake_resim_data_id)

    result = resim_data(
        empirical_data=empirical_data,
        post_samples=post_samples,
        simulator=DummySimulator(),
        num_resims=2,
        rt="rt",
        id_name="id",
        congruency="congruency",
        simulator_congruency="cond_code",
        simulator_congruency_coding=10.0,
        simulator_incongruency_coding=20.0,
    )

    assert set(result["congruency"]) == {"congruent", "incongruent"}
    assert set(result["cond_code"]) == {10.0, 20.0}


def test_resim_data_raises_when_required_empirical_column_is_missing(post_samples):
    empirical_data = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "accuracy": [1, 0],
            "congruency": ["congruent", "incongruent"],
        }
    )

    with pytest.raises(ValueError, match=r"Variable 'rt' does not exist in data"):
        resim_data(
            empirical_data=empirical_data,
            post_samples=post_samples,
            simulator=DummySimulator(),
            rt="rt",
            id_name="id",
            congruency="congruency",
        )


def test_resim_data_passes_param_names_and_lower_bound_to_resim_data_id(monkeypatch, empirical_data, post_samples):
    seen = []

    def fake_resim_data_id(post_sample_data, num_obs, simulator, id, param_names, lower_bound, num_resims=50):
        seen.append(
            {
                "id": id,
                "param_names": tuple(param_names),
                "lower_bound": lower_bound,
                "num_resims": num_resims,
            }
        )
        return (
            pd.DataFrame(
                {
                    "rt": [0.5],
                    "accuracy": [1],
                    "conditions": [0.0],
                    "num_resim": [0],
                    "id": [id],
                }
            ),
            0,
            len(post_sample_data),
        )

    monkeypatch.setattr("dmc_helpers.resim_data_id", fake_resim_data_id)

    _ = resim_data(
        empirical_data=empirical_data,
        post_samples=post_samples,
        simulator=DummySimulator(),
        num_resims=7,
        param_names=("A", "tau"),
        lower_bound=0.25,
        rt="rt",
        id_name="id",
        congruency="congruency",
        simulator_congruency="conditions",
    )

    assert seen == [
        {"id": "s1", "param_names": ("A", "tau"), "lower_bound": 0.25, "num_resims": 7},
        {"id": "s2", "param_names": ("A", "tau"), "lower_bound": 0.25, "num_resims": 7},
    ]