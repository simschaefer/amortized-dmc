import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import pytest
import numpy as np

from dmc_simulator_asym import DMCasym


@pytest.fixture
def valid_prior_means():
    return np.array([100.0, 100.0, 80.0, 0.5, 300.0, 120.0, 30.0])


@pytest.fixture
def valid_prior_sds():
    return np.array([10.0, 10.0, 10.0, 0.1, 20.0, 10.0, 5.0])


@pytest.fixture
def simulator(valid_prior_means, valid_prior_sds):
    return DMCasym(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
        fixed_num_obs=20,
        rng=np.random.default_rng(123),
    )


def test_dmc_asym_init_stores_basic_attributes(simulator):
    assert simulator.fixed_num_obs == 20
    assert simulator.param_names == ("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r")
    assert simulator.num_conditions == 2
    assert simulator.dt == 1.0
    assert simulator.tmax == 1200


def test_dmc_asym_init_raises_for_mismatched_prior_lengths(valid_prior_means):
    prior_sds = np.array([10.0, 10.0, 0.1])

    with pytest.raises(ValueError, match="must have the same length"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_asym_init_raises_for_non_1d_priors(valid_prior_sds):
    prior_means = np.array([[100.0, 100.0, 80.0, 0.5, 300.0, 120.0, 30.0]])

    with pytest.raises(ValueError, match="must be 1D arrays"):
        DMCasym(
            prior_means=prior_means,
            prior_sds=valid_prior_sds,
        )


def test_dmc_asym_init_raises_for_invalid_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains invalid entries"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "not_a_param"),
        )


def test_dmc_asym_init_raises_for_duplicate_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains duplicates"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "b"),
        )


def test_dmc_asym_init_raises_for_missing_a_con_or_a_inc(valid_prior_sds):
    prior_means = np.array([100.0, 80.0, 0.5, 300.0, 120.0, 30.0])
    prior_sds = np.array([10.0, 10.0, 0.1, 20.0, 10.0, 5.0])

    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMCasym(
            prior_means=prior_means,
            prior_sds=prior_sds,
            param_names=("A_con", "tau", "mu_c", "mu_r", "b", "sd_r"),
        )


def test_dmc_asym_init_raises_for_nonpositive_prior_sds(valid_prior_means):
    prior_sds = np.array([10.0, 10.0, 10.0, 0.1, 20.0, 10.0, 0.0])

    with pytest.raises(ValueError, match="strictly positive"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_asym_init_raises_for_invalid_dt(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="dt must be > 0"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            dt=0,
        )


def test_dmc_asym_init_raises_for_invalid_tmax(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="tmax must be > 0"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            tmax=0,
        )


def test_dmc_asym_init_raises_for_invalid_a_value(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Please choose a value larger than 1"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            a_value=1,
        )


def test_dmc_asym_init_raises_for_invalid_min_max_num_obs(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Require 0 < min_num_obs <= max_num_obs"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            fixed_num_obs=None,
            min_num_obs=10,
            max_num_obs=5,
        )


def test_prior_returns_expected_keys(simulator):
    params = simulator.prior(rng=np.random.default_rng(1))

    assert set(params.keys()) == {"A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"}
    assert all(np.isscalar(v) for v in params.values())


def test_prior_respects_lower_bound(valid_prior_means, valid_prior_sds):
    sim = DMCasym(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_lower_bound=0,
        rng=np.random.default_rng(1),
    )

    params = sim.prior(rng=np.random.default_rng(2))

    assert all(v >= 0 for v in params.values())


def test_prior_respects_custom_param_names_order(valid_prior_sds):
    # A_inc listed before A_con: prior() must label draws by position in
    # param_names, not by a hardcoded assumption about ordering.
    prior_means = np.array([100.0, 100.0, 80.0, 0.5, 300.0, 120.0, 30.0])

    sim = DMCasym(
        prior_means=prior_means,
        prior_sds=valid_prior_sds,
        param_names=("A_inc", "A_con", "tau", "mu_c", "mu_r", "b", "sd_r"),
        param_lower_bound=None,
        rng=np.random.default_rng(1),
    )

    params = sim.prior(rng=np.random.default_rng(0))
    rng_check = np.random.default_rng(0)
    expected = rng_check.normal(prior_means, valid_prior_sds)

    assert params["A_inc"] == expected[0]
    assert params["A_con"] == expected[1]


def test_experiment_returns_expected_structure(simulator):
    result = simulator.experiment(
        A_con=100.0,
        A_inc=100.0,
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


def test_experiment_incongruent_trials_use_negated_a_inc(simulator):
    # Core contract of DMCasym: A_inc is estimated as a positive magnitude
    # (same prior/truncation as A_con), and experiment() must negate it before
    # simulating incongruent trials, so its automatic activation opposes A_con's.
    A_con, A_inc = 40.0, 55.0
    tau, mu_c, mu_r, b, sd_r = 80.0, 0.5, 300.0, 120.0, 30.0
    num_obs = 10

    result = simulator.experiment(
        A_con=A_con, A_inc=A_inc, tau=tau, mu_c=mu_c, mu_r=mu_r, b=b, sd_r=sd_r,
        num_obs=num_obs, rng=np.random.default_rng(7),
    )

    # Reproduce experiment()'s internals by hand with a fresh, identically
    # seeded rng, calling trial() directly with A_con and -A_inc.
    rng_manual = np.random.default_rng(7)
    obs_per_condition = int(np.ceil(num_obs / simulator.num_conditions))
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    T = len(t)

    noise = rng_manual.normal(size=(num_obs, T))
    non_decision_ts = rng_manual.normal(size=num_obs, loc=mu_r, scale=sd_r)

    data = np.zeros((num_obs, 2))
    data[:obs_per_condition] = simulator.trial(
        A=A_con, tau=tau, mu_c=mu_c, b=b, t=t,
        noise=noise[:obs_per_condition],
        non_decision_ts=non_decision_ts[:obs_per_condition],
        rng=rng_manual,
    )
    data[obs_per_condition:] = simulator.trial(
        A=-A_inc, tau=tau, mu_c=mu_c, b=b, t=t,
        noise=noise[obs_per_condition:],
        non_decision_ts=non_decision_ts[obs_per_condition:],
        rng=rng_manual,
    )

    assert np.array_equal(result["rt"], data[:, 0])
    assert np.array_equal(result["accuracy"], data[:, 1])


def test_congruency_effect_direction_with_matched_a_con_a_inc(valid_prior_sds):
    # With A_con and A_inc drawn from the same positive-only prior, incongruent
    # trials should still be systematically slower than congruent trials on
    # average, because A_inc is negated internally (mirroring the classic
    # sign-flipped-A DMC), not because the two amplitudes differ in magnitude.
    prior_means = np.array([32.4, 32.4, 93.88, 0.49, 387.53, 88.21, 48.25])
    prior_sds = np.array([9.05, 9.05, 29.67, 0.15, 57.65, 16.56, 10.41])

    sim = DMCasym(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
        param_lower_bound=[0, 0, 0, 0, 0, 0, 0],
        fixed_num_obs=500,
        rng=np.random.default_rng(0),
    )

    test_data = sim.sample(batch_size=200, seed=0)

    rt = test_data["rt"][:, :, 0]
    cond = test_data["conditions"][:, :, 0]
    valid = rt > 0

    n = rt.shape[0]
    rt_con = np.array([rt[i][valid[i] & (cond[i] == 0)].mean() for i in range(n)])
    rt_inc = np.array([rt[i][valid[i] & (cond[i] == 1)].mean() for i in range(n)])
    diff = rt_inc - rt_con

    assert np.nanmean(diff) > 0
    assert np.mean(diff < 0) < 0.2


def test_call_returns_parameters_and_data(simulator):
    result = simulator(num_obs=10, rng=np.random.default_rng(42))

    expected_keys = {
        "A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r",
        "rt", "accuracy", "conditions", "num_obs",
    }
    assert set(result.keys()) == expected_keys

    assert np.isscalar(result["A_con"])
    assert np.isscalar(result["A_inc"])
    assert result["rt"].shape == (10,)
    assert result["accuracy"].shape == (10,)
    assert result["conditions"].shape == (10,)
    assert result["num_obs"] == 10


def test_sample_returns_expected_shapes(simulator):
    sims = simulator.sample(batch_size=4, num_obs=12, seed=123)

    for key in ("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"):
        assert sims[key].shape == (4, 1)

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
    sim = DMCasym(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=15,
        rng=np.random.default_rng(123),
    )

    sims = sim.sample(batch_size=2, num_obs=None, seed=123)

    assert sims["rt"].shape == (2, 15, 1)
    assert np.all(sims["num_obs"] == 15)


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


def test_trial_runs_for_a_value_not_equal_to_2(valid_prior_means, valid_prior_sds):
    sim = DMCasym(
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


def test_dmc_asym_with_sdr_fixed_requires_param_names_without_sd_r():
    prior_means = np.array([100.0, 100.0, 80.0, 0.5, 300.0, 120.0])
    prior_sds = np.array([10.0, 10.0, 10.0, 0.1, 20.0, 10.0])

    sim = DMCasym(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b"),
        sdr_fixed=30.0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    params = sim.prior(rng=np.random.default_rng(1))
    assert set(params.keys()) == {"A_con", "A_inc", "tau", "mu_c", "mu_r", "b"}

    result = sim(num_obs=8, rng=np.random.default_rng(42))
    assert "sd_r" not in result
    assert result["rt"].shape == (8,)
    assert result["num_obs"] == 8


def test_dmc_asym_raises_if_sdr_fixed_but_param_names_still_include_sd_r(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMCasym(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
            sdr_fixed=30.0,
            fixed_num_obs=10,
            rng=np.random.default_rng(123),
        )


def test_experiment_with_contamination_probability_one_replaces_rts_and_responses(valid_prior_means, valid_prior_sds):
    sim = DMCasym(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        contamination_probability=1.0,
        contamination_uniform_lower=0.25,
        contamination_uniform_upper=0.5,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    out = sim.experiment(
        A_con=100.0,
        A_inc=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=30.0,
        num_obs=12,
        rng=np.random.default_rng(42),
    )

    assert np.all(out["rt"] >= 0.25)
    assert np.all(out["rt"] <= 0.5)
    assert set(np.unique(out["accuracy"])).issubset({0, 1})


def test_time_grid_has_exact_dt_spacing(simulator):
    # Regression test: an earlier version of this simulator used np.linspace,
    # which does not reproduce an exact dt-spaced grid. experiment() must use
    # np.arange so the SDE discretization step matches the recorded times.
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    assert np.allclose(np.diff(t), simulator.dt)


def test_experiment_runs_for_fractional_dt(valid_prior_means, valid_prior_sds):
    # Regression test: an earlier version crashed for dt != 1 because the
    # noise array was sized to self.tmax instead of len(t).
    sim = DMCasym(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        dt=0.5,
        fixed_num_obs=10,
        rng=np.random.default_rng(1),
    )

    result = sim(num_obs=10, rng=np.random.default_rng(2))
    assert result["rt"].shape == (10,)
