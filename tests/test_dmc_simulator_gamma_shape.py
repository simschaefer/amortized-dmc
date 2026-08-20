import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import pytest
import numpy as np

from dmc_simulator_gamma_shape import DMCgamma


@pytest.fixture
def valid_prior_means():
    return np.array([100.0, 3.0, 0.5, 300.0, 120.0, 30.0])


@pytest.fixture
def valid_prior_sds():
    return np.array([10.0, 0.5, 0.1, 20.0, 10.0, 5.0])


@pytest.fixture
def simulator(valid_prior_means, valid_prior_sds):
    return DMCgamma(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_names=("A", "a", "mu_c", "mu_r", "b", "sd_r"),
        fixed_num_obs=20,
        rng=np.random.default_rng(123),
    )


def test_dmc_gamma_init_stores_basic_attributes(simulator):
    assert simulator.fixed_num_obs == 20
    assert simulator.param_names == ("A", "a", "mu_c", "mu_r", "b", "sd_r")
    assert simulator.num_conditions == 2
    assert simulator.dt == 1.0
    assert simulator.tmax == 1200
    assert simulator.time_to_peak == 100
    assert simulator.a_value == 2


def test_dmc_gamma_init_raises_for_mismatched_prior_lengths(valid_prior_means):
    prior_sds = np.array([10.0, 0.5, 0.1])

    with pytest.raises(ValueError, match="must have the same length"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_gamma_init_raises_for_non_1d_priors(valid_prior_sds):
    prior_means = np.array([[100.0, 3.0, 0.5, 300.0, 120.0, 30.0]])

    with pytest.raises(ValueError, match="must be 1D arrays"):
        DMCgamma(
            prior_means=prior_means,
            prior_sds=valid_prior_sds,
        )


def test_dmc_gamma_init_raises_for_invalid_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains invalid entries"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A", "a", "mu_c", "mu_r", "b", "not_a_param"),
        )


def test_dmc_gamma_init_raises_for_duplicate_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains duplicates"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A", "a", "mu_c", "mu_r", "b", "b"),
        )


def test_dmc_gamma_init_raises_for_missing_a(valid_prior_sds):
    prior_means = np.array([100.0, 0.5, 300.0, 120.0, 30.0])
    prior_sds = np.array([10.0, 0.1, 20.0, 10.0, 5.0])

    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMCgamma(
            prior_means=prior_means,
            prior_sds=prior_sds,
            param_names=("A", "mu_c", "mu_r", "b", "sd_r"),
        )


def test_dmc_gamma_init_raises_for_nonpositive_prior_sds(valid_prior_means):
    prior_sds = np.array([10.0, 0.5, 0.1, 20.0, 10.0, 0.0])

    with pytest.raises(ValueError, match="strictly positive"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_gamma_init_raises_for_invalid_dt(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="dt must be > 0"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            dt=0,
        )


def test_dmc_gamma_init_raises_for_invalid_tmax(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="tmax must be > 0"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            tmax=0,
        )


def test_dmc_gamma_init_raises_for_invalid_a_value(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Please choose a value larger than 1"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            a_value=1,
        )


def test_dmc_gamma_init_raises_for_invalid_min_max_num_obs(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Require 0 < min_num_obs <= max_num_obs"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            fixed_num_obs=None,
            min_num_obs=10,
            max_num_obs=5,
        )


def test_prior_returns_expected_keys(simulator):
    params = simulator.prior(rng=np.random.default_rng(1))

    assert set(params.keys()) == {"A", "a", "mu_c", "mu_r", "b", "sd_r"}
    assert all(np.isscalar(v) for v in params.values())


def test_prior_default_lower_bound_keeps_a_strictly_above_one(simulator):
    # a <= 1 makes tau = time_to_peak / (a - 1) undefined/negative, so the
    # default param_lower_bound=(0, 1.5, 0, 0, 0, 0) must keep every sampled
    # `a` comfortably above 1.
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = simulator.prior(rng=rng)
        assert params["a"] > 1.0


def test_prior_respects_custom_lower_bound(valid_prior_means, valid_prior_sds):
    sim = DMCgamma(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_lower_bound=(0, 1.5, 0, 0, 0, 0),
        rng=np.random.default_rng(1),
    )

    params = sim.prior(rng=np.random.default_rng(2))

    assert params["A"] >= 0
    assert params["a"] >= 1.5
    assert params["mu_c"] >= 0
    assert params["mu_r"] >= 0
    assert params["b"] >= 0
    assert params["sd_r"] >= 0


def test_trial_returns_expected_shape(simulator):
    rng = np.random.default_rng(42)
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    noise = rng.normal(size=(5, len(t)))
    non_decision_ts = rng.normal(loc=300.0, scale=30.0, size=5)

    out = simulator.trial(
        A=100.0,
        a=3.0,
        mu_c=0.5,
        b=120.0,
        t=t,
        noise=noise,
        non_decision_ts=non_decision_ts,
        rng=rng,
    )

    assert out.shape == (5, 2)
    assert set(np.unique(out[:, 1])).issubset({-1, 0, 1})


def test_trial_matches_reference_drift_formula(simulator):
    # Regression test for a historical bug: the drift's power_term exponent
    # must use the sampled/free `a` throughout (together with tau and
    # deriv_term), not the fixed constructor constant `self.a_value`. Using
    # a != simulator.a_value here means the bug (if reintroduced) would make
    # this test fail, since a == a_value would otherwise hide it.
    A, a, mu_c, b = 1.5, 4.0, 0.2, 60.0
    assert a != simulator.a_value

    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    num_trials = 6
    rng_inputs = np.random.default_rng(11)
    noise = rng_inputs.normal(size=(num_trials, len(t)))
    non_decision_ts = rng_inputs.normal(loc=300.0, scale=20.0, size=num_trials)

    actual = simulator.trial(
        A=A, a=a, mu_c=mu_c, b=b, t=t, noise=noise,
        non_decision_ts=non_decision_ts, rng=np.random.default_rng(99),
    )

    # Independent reference implementation of the intended math.
    rng_ref = np.random.default_rng(99)
    X0 = rng_ref.beta(
        simulator.X0_beta_shape_fixed, simulator.X0_beta_shape_fixed, size=num_trials
    ) * (2 * b) - b

    tau = simulator.time_to_peak / (a - 1)
    t_div_tau = t / tau
    exponent_term = np.exp(-t_div_tau)
    power_term = (np.exp(1) * t_div_tau / (a - 1)) ** (a - 1)
    deriv_term = ((a - 1) / t) - (1 / tau)
    mu_t = A * exponent_term * power_term * deriv_term + mu_c

    sqrt_dt_sigma = simulator.sigma * np.sqrt(simulator.dt)
    dX = mu_t[None, :] * simulator.dt + sqrt_dt_sigma * noise
    X_shift = np.cumsum(dX, axis=1) + X0[:, None]

    crossed_any = (X_shift >= b) | (X_shift <= -b)
    first_crossing = np.argmax(crossed_any, axis=1)
    has_crossed = np.any(crossed_any, axis=1)

    rts_ref = np.full(num_trials, -1.0)
    resps_ref = np.full(num_trials, -1)
    idx = np.where(has_crossed)[0]
    rts_ref[idx] = (t[first_crossing[idx]] + non_decision_ts[idx]) / 1000
    resp_hit = X_shift[idx, first_crossing[idx]]
    resps_ref[idx] = (resp_hit >= b).astype(int)

    assert np.allclose(actual[:, 0], rts_ref)
    assert np.array_equal(actual[:, 1], resps_ref)


def test_time_to_peak_invariant_holds_across_sampled_a(simulator):
    # Documents the model's core design property: the automatic activation's
    # peak (where the drift crosses zero) stays at simulator.time_to_peak
    # regardless of the sampled gamma shape `a`, because tau is derived as
    # time_to_peak / (a - 1) for each simulation.
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)

    for a_val in (1.8, 2.0, 2.5, 4.0):
        tau = simulator.time_to_peak / (a_val - 1)
        t_div_tau = t / tau
        exponent_term = np.exp(-t_div_tau)
        power_term = (np.exp(1) * t_div_tau / (a_val - 1)) ** (a_val - 1)
        deriv_term = ((a_val - 1) / t) - (1 / tau)
        mu_t = exponent_term * power_term * deriv_term

        sign_changes = np.where(np.diff(np.sign(mu_t)) != 0)[0]
        assert len(sign_changes) > 0
        zero_cross_t = t[sign_changes[0]]
        assert abs(zero_cross_t - simulator.time_to_peak) <= simulator.dt


def test_experiment_returns_expected_structure(simulator):
    result = simulator.experiment(
        A=100.0,
        a=3.0,
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

    expected_keys = {"A", "a", "mu_c", "mu_r", "b", "sd_r", "rt", "accuracy", "conditions", "num_obs"}
    assert set(result.keys()) == expected_keys

    assert np.isscalar(result["A"])
    assert np.isscalar(result["a"])
    assert result["a"] > 1.0
    assert result["rt"].shape == (10,)
    assert result["accuracy"].shape == (10,)
    assert result["conditions"].shape == (10,)
    assert result["num_obs"] == 10


def test_sample_returns_expected_shapes(simulator):
    sims = simulator.sample(batch_size=4, num_obs=12, seed=123)

    for key in ("A", "a", "mu_c", "mu_r", "b", "sd_r"):
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
    sim = DMCgamma(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=15,
        rng=np.random.default_rng(123),
    )

    sims = sim.sample(batch_size=2, num_obs=None, seed=123)

    assert sims["rt"].shape == (2, 15, 1)
    assert np.all(sims["num_obs"] == 15)


def test_dmc_gamma_with_sdr_fixed_requires_param_names_without_sd_r():
    prior_means = np.array([100.0, 3.0, 0.5, 300.0, 120.0])
    prior_sds = np.array([10.0, 0.5, 0.1, 20.0, 10.0])

    sim = DMCgamma(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A", "a", "mu_c", "mu_r", "b"),
        param_lower_bound=(0, 1.5, 0, 0, 0),
        sdr_fixed=30.0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    params = sim.prior(rng=np.random.default_rng(1))
    assert set(params.keys()) == {"A", "a", "mu_c", "mu_r", "b"}

    result = sim(num_obs=8, rng=np.random.default_rng(42))
    assert "sd_r" not in result
    assert result["rt"].shape == (8,)
    assert result["num_obs"] == 8


def test_dmc_gamma_raises_if_sdr_fixed_but_param_names_still_include_sd_r(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMCgamma(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A", "a", "mu_c", "mu_r", "b", "sd_r"),
            sdr_fixed=30.0,
            fixed_num_obs=10,
            rng=np.random.default_rng(123),
        )


def test_experiment_with_contamination_probability_one_replaces_rts_and_responses(valid_prior_means, valid_prior_sds):
    sim = DMCgamma(
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
        a=3.0,
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
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    assert np.allclose(np.diff(t), simulator.dt)


def test_experiment_runs_for_fractional_dt(valid_prior_means, valid_prior_sds):
    sim = DMCgamma(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        dt=0.5,
        fixed_num_obs=10,
        rng=np.random.default_rng(1),
    )

    result = sim(num_obs=10, rng=np.random.default_rng(2))
    assert result["rt"].shape == (10,)
