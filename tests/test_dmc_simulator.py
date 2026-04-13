import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import pandas as pd
import pytest
import numpy as np


scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)


scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_simulator import DMC


@pytest.fixture
def valid_prior_means():
    return np.array([100.0, 80.0, 0.5, 300.0, 120.0, 30.0])


@pytest.fixture
def valid_prior_sds():
    return np.array([10.0, 10.0, 0.1, 20.0, 10.0, 5.0])


@pytest.fixture
def simulator(valid_prior_means, valid_prior_sds):
    return DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_names=("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
        fixed_num_obs=20,
        rng=np.random.default_rng(123),
    )


def test_dmc_init_stores_basic_attributes(simulator):
    assert simulator.fixed_num_obs == 20
    assert simulator.param_names == ("A", "tau", "mu_c", "mu_r", "b", "sd_r")
    assert simulator.num_conditions == 2
    assert simulator.dt == 1.0
    assert simulator.tmax == 1200


def test_dmc_init_raises_for_mismatched_prior_lengths(valid_prior_means):
    prior_sds = np.array([10.0, 10.0, 0.1])

    with pytest.raises(ValueError, match="must have the same length"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_init_raises_for_non_1d_priors(valid_prior_sds):
    prior_means = np.array([[100.0, 80.0, 0.5, 300.0, 120.0, 30.0]])

    with pytest.raises(ValueError, match="must be 1D arrays"):
        DMC(
            prior_means=prior_means,
            prior_sds=valid_prior_sds,
        )


def test_dmc_init_raises_for_invalid_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains invalid entries"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A", "tau", "mu_c", "mu_r", "b", "not_a_param"),
        )


def test_dmc_init_raises_for_duplicate_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains duplicates"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A", "tau", "mu_c", "mu_r", "b", "b"),
        )


def test_dmc_init_raises_for_nonpositive_prior_sds(valid_prior_means):
    prior_sds = np.array([10.0, 10.0, 0.1, 20.0, 10.0, 0.0])

    with pytest.raises(ValueError, match="strictly positive"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_prior_returns_expected_keys(simulator):
    params = simulator.prior(rng=np.random.default_rng(1))

    assert set(params.keys()) == {"A", "tau", "mu_c", "mu_r", "b", "sd_r"}
    assert all(np.isscalar(v) for v in params.values())


def test_prior_respects_lower_bound(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_lower_bound=0,
        rng=np.random.default_rng(1),
    )

    params = sim.prior(rng=np.random.default_rng(2))

    assert all(v >= 0 for v in params.values())


def test_experiment_returns_expected_structure(simulator):
    result = simulator.experiment(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=30.0,
        num_obs=10,
        rng=np.random.default_rng(42),
    )

    assert set(result.keys()) == {"rt", "accuracy", "conditions", "num_obs"}
    assert result["rt"].shape == (10,)
    assert result["accuracy"].shape == (10,)
    assert result["conditions"].shape == (10,)
    assert result["num_obs"] == 10

    assert set(np.unique(result["conditions"])).issubset({0, 1})
    assert set(np.unique(result["accuracy"])).issubset({-1, 0, 1})


def test_call_returns_parameters_and_data(simulator):
    result = simulator(num_obs=10, rng=np.random.default_rng(42))

    expected_keys = {"A", "tau", "mu_c", "mu_r", "b", "sd_r", "rt", "accuracy", "conditions", "num_obs"}
    assert set(result.keys()) == expected_keys

    assert np.isscalar(result["A"])
    assert np.isscalar(result["tau"])
    assert result["rt"].shape == (10,)
    assert result["accuracy"].shape == (10,)
    assert result["conditions"].shape == (10,)
    assert result["num_obs"] == 10


def test_sample_returns_expected_shapes(simulator):
    sims = simulator.sample(batch_size=4, num_obs=12, seed=123)

    assert sims["A"].shape == (4, 1)
    assert sims["tau"].shape == (4, 1)
    assert sims["mu_c"].shape == (4, 1)
    assert sims["mu_r"].shape == (4, 1)
    assert sims["b"].shape == (4, 1)
    assert sims["sd_r"].shape == (4, 1)

    assert sims["rt"].shape == (4, 12, 1)
    assert sims["accuracy"].shape == (4, 12, 1)
    assert sims["conditions"].shape == (4, 12, 1)
    assert sims["num_obs"].shape == (4, 1)


def test_sample_is_reproducible_with_seed(simulator):
    sims1 = simulator.sample(batch_size=3, num_obs=8, seed=999)
    sims2 = simulator.sample(batch_size=3, num_obs=8, seed=999)

    for key in sims1:
        assert np.array_equal(sims1[key], sims2[key])


def test_sample_uses_fixed_num_obs_when_num_obs_is_none(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=15,
        rng=np.random.default_rng(123),
    )

    sims = sim.sample(batch_size=2, num_obs=None, seed=123)

    assert sims["rt"].shape == (2, 15, 1)
    assert sims["accuracy"].shape == (2, 15, 1)
    assert sims["conditions"].shape == (2, 15, 1)
    assert np.all(sims["num_obs"] == 15)


def test_sample_accepts_tuple_batch_size(simulator):
    sims = simulator.sample(batch_size=(3,), num_obs=7, seed=123)

    assert sims["rt"].shape == (3, 7, 1)
    assert sims["accuracy"].shape == (3, 7, 1)
    assert sims["conditions"].shape == (3, 7, 1)


def test_trial_returns_expected_shape(simulator):
    rng = np.random.default_rng(42)
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    noise = rng.normal(size=(5, len(t)))
    non_decision_ts = rng.normal(loc=300.0, scale=30.0, size=5)

    out = simulator.trial(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        b=120.0,
        t=t,
        noise=noise,
        non_decision_ts=non_decision_ts,
        rng=rng,
    )

    assert out.shape == (5, 2)
    assert set(np.unique(out[:, 1])).issubset({-1, 0, 1})


def test_dmc_with_sdr_fixed_requires_param_names_without_sd_r():
    prior_means = np.array([100.0, 80.0, 0.5, 300.0, 120.0])
    prior_sds = np.array([10.0, 10.0, 0.1, 20.0, 10.0])

    sim = DMC(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A", "tau", "mu_c", "mu_r", "b"),
        sdr_fixed=30.0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    params = sim.prior(rng=np.random.default_rng(1))
    assert set(params.keys()) == {"A", "tau", "mu_c", "mu_r", "b"}

    result = sim(num_obs=8, rng=np.random.default_rng(42))
    assert set(result.keys()) == {"A", "tau", "mu_c", "mu_r", "b", "rt", "accuracy", "conditions", "num_obs"}
    assert "sd_r" not in result
    assert result["rt"].shape == (8,)
    assert result["accuracy"].shape == (8,)
    assert result["conditions"].shape == (8,)
    assert result["num_obs"] == 8


def test_dmc_raises_if_sdr_fixed_but_param_names_still_include_sd_r(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
            sdr_fixed=30.0,
            fixed_num_obs=10,
            rng=np.random.default_rng(123),
        )


def test_sample_uses_random_num_obs_when_fixed_num_obs_is_none(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=None,
        min_num_obs=5,
        max_num_obs=9,
        rng=np.random.default_rng(123),
    )

    sims = sim.sample(batch_size=3, num_obs=None, seed=42)

    sampled_num_obs = int(sims["num_obs"][0, 0])
    assert 5 <= sampled_num_obs <= 9

    # all datasets in the batch should share the same sampled num_obs for that call
    assert np.all(sims["num_obs"] == sampled_num_obs)
    assert sims["rt"].shape == (3, sampled_num_obs, 1)
    assert sims["accuracy"].shape == (3, sampled_num_obs, 1)
    assert sims["conditions"].shape == (3, sampled_num_obs, 1)


def test_trial_runs_for_a_value_not_equal_to_2(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        a_value=3,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    rng = np.random.default_rng(42)
    t = np.arange(sim.dt, sim.tmax + sim.dt, sim.dt)
    noise = rng.normal(size=(4, len(t)))
    non_decision_ts = rng.normal(loc=300.0, scale=30.0, size=4)

    out = sim.trial(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        b=120.0,
        t=t,
        noise=noise,
        non_decision_ts=non_decision_ts,
        rng=rng,
    )

    assert out.shape == (4, 2)
    assert np.isfinite(out[:, 0]).all()
    assert set(np.unique(out[:, 1])).issubset({-1, 0, 1})


def test_experiment_uses_sdr_fixed_instead_of_passed_sd_r():
    prior_means = np.array([100.0, 80.0, 0.5, 300.0, 120.0])
    prior_sds = np.array([10.0, 10.0, 0.1, 20.0, 10.0])

    sim = DMC(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A", "tau", "mu_c", "mu_r", "b"),
        sdr_fixed=0.0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    # passed sd_r should be ignored because sdr_fixed=0.0
    out1 = sim.experiment(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=0.0,
        num_obs=8,
        rng=rng1,
    )

    out2 = sim.experiment(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=999.0,
        num_obs=8,
        rng=rng2,
    )

    assert np.array_equal(out1["rt"], out2["rt"])
    assert np.array_equal(out1["accuracy"], out2["accuracy"])
    assert np.array_equal(out1["conditions"], out2["conditions"])


def test_experiment_returns_valid_condition_split(simulator):
    out = simulator.experiment(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=30.0,
        num_obs=7,
        rng=np.random.default_rng(42),
    )

    assert out["conditions"].shape == (7,)
    assert set(np.unique(out["conditions"])).issubset({0, 1})

    # for num_conditions=2 and ceil split, first condition should appear at least as often
    n_con = np.sum(out["conditions"] == 0)
    n_inc = np.sum(out["conditions"] == 1)
    assert n_con >= n_inc
    assert n_con + n_inc == 7


def test_experiment_with_contamination_probability_one_replaces_rts_and_responses(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        contamination_probability=1.0,
        contamination_uniform_lower=0.25,
        contamination_uniform_upper=0.5,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    out = sim.experiment(
        A=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=30.0,
        num_obs=12,
        rng=np.random.default_rng(42),
    )

    # all RTs should be from the contamination uniform range
    assert np.all(out["rt"] >= 0.25)
    assert np.all(out["rt"] <= 0.5)

    # all responses should be binary after contamination replacement
    assert set(np.unique(out["accuracy"])).issubset({0, 1})


def test_random_num_obs_is_reproducible_with_seed(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=None,
        min_num_obs=5,
        max_num_obs=9,
        rng=np.random.default_rng(123),
    )

    sims1 = sim.sample(batch_size=2, num_obs=None, seed=42)
    sims2 = sim.sample(batch_size=2, num_obs=None, seed=42)

    assert np.array_equal(sims1["num_obs"], sims2["num_obs"])
    assert sims1["rt"].shape == sims2["rt"].shape


def test_random_num_obs_can_vary_across_seeds(valid_prior_means, valid_prior_sds):
    sim = DMC(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=None,
        min_num_obs=5,
        max_num_obs=9,
        rng=np.random.default_rng(123),
    )

    observed = set()
    for seed in range(10):
        sims = sim.sample(batch_size=1, num_obs=None, seed=seed)
        observed.add(int(sims["num_obs"][0, 0]))

    assert all(5 <= n <= 9 for n in observed)
    assert len(observed) >= 2


def test_dmc_init_raises_for_invalid_dt(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="dt must be > 0"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            dt=0,
        )


def test_dmc_init_raises_for_invalid_tmax(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="tmax must be > 0"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            tmax=0,
        )


def test_dmc_init_raises_for_invalid_a_value(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Please choose a value larger than 1"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            a_value=1,
        )

def test_dmc_init_raises_for_invalid_min_max_num_obs(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Require 0 < min_num_obs <= max_num_obs"):
        DMC(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            fixed_num_obs=None,
            min_num_obs=10,
            max_num_obs=5,
        )