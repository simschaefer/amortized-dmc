import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import pytest
import numpy as np

from dmc_simulator_neutral import DMCneutral


@pytest.fixture
def valid_prior_means():
    # A_con, A_inc, tau, mu_c, mu_r, b, sd_r
    return np.array([100.0, 100.0, 80.0, 0.5, 300.0, 120.0, 30.0])


@pytest.fixture
def valid_prior_sds():
    return np.array([10.0, 10.0, 10.0, 0.1, 20.0, 10.0, 5.0])


@pytest.fixture
def simulator(valid_prior_means, valid_prior_sds):
    # A_neutral fixed at 0 (the default), matching the "normal" configuration
    return DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
        A_neutral=0,
        fixed_num_obs=20,
        rng=np.random.default_rng(123),
    )


@pytest.fixture
def simulator_estimated_neutral():
    # A_neutral estimated as a free parameter, with its own unbounded prior
    prior_means = np.array([100.0, 100.0, -1.0, 80.0, 0.5, 300.0, 120.0, 30.0])
    prior_sds = np.array([10.0, 10.0, 5.0, 10.0, 0.1, 20.0, 10.0, 5.0])

    return DMCneutral(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A_con", "A_inc", "A_neutral", "tau", "mu_c", "mu_r", "b", "sd_r"),
        A_neutral=None,
        param_lower_bound=(0, 0, -np.inf, 0, 0, 0, 0, 0),
        fixed_num_obs=20,
        rng=np.random.default_rng(123),
    )


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------
def test_dmc_neutral_init_stores_basic_attributes(simulator):
    assert simulator.fixed_num_obs == 20
    assert simulator.param_names == ("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r")
    assert simulator.num_conditions == 3
    assert simulator.A_neutral == 0
    assert simulator.dt == 1.0
    assert simulator.tmax == 1200


def test_dmc_neutral_init_raises_for_mismatched_prior_lengths(valid_prior_means):
    prior_sds = np.array([10.0, 10.0, 0.1])

    with pytest.raises(ValueError, match="must have the same length"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_neutral_init_raises_for_non_1d_priors(valid_prior_sds):
    prior_means = np.array([[100.0, 100.0, 80.0, 0.5, 300.0, 120.0, 30.0]])

    with pytest.raises(ValueError, match="must be 1D arrays"):
        DMCneutral(
            prior_means=prior_means,
            prior_sds=valid_prior_sds,
        )


def test_dmc_neutral_init_raises_for_invalid_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains invalid entries"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "not_a_param"),
        )


def test_dmc_neutral_init_raises_for_duplicate_param_names(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="contains duplicates"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "b"),
        )


def test_dmc_neutral_init_raises_for_missing_a_con_or_a_inc(valid_prior_sds):
    prior_means = np.array([100.0, 80.0, 0.5, 300.0, 120.0, 30.0])
    prior_sds = np.array([10.0, 10.0, 0.1, 20.0, 10.0, 5.0])

    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMCneutral(
            prior_means=prior_means,
            prior_sds=prior_sds,
            param_names=("A_con", "tau", "mu_c", "mu_r", "b", "sd_r"),
        )


def test_dmc_neutral_init_raises_for_nonpositive_prior_sds(valid_prior_means):
    prior_sds = np.array([10.0, 10.0, 10.0, 0.1, 20.0, 10.0, 0.0])

    with pytest.raises(ValueError, match="strictly positive"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=prior_sds,
        )


def test_dmc_neutral_init_raises_for_invalid_dt(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="dt must be > 0"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            dt=0,
        )


def test_dmc_neutral_init_raises_for_invalid_tmax(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="tmax must be > 0"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            tmax=0,
        )


def test_dmc_neutral_init_raises_for_invalid_a_value(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Please choose a value larger than 1"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            a_value=1,
        )


def test_dmc_neutral_init_raises_for_invalid_min_max_num_obs(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Require 0 < min_num_obs <= max_num_obs"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            fixed_num_obs=None,
            min_num_obs=10,
            max_num_obs=5,
        )


def test_dmc_neutral_init_raises_for_num_conditions_other_than_three(valid_prior_means, valid_prior_sds):
    with pytest.raises(ValueError, match="Number of conditions must be 3"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            num_conditions=2,
        )


# --- A_neutral fixed-vs-estimated contract ----------------------------------
def test_dmc_neutral_init_raises_when_a_neutral_is_both_fixed_and_in_param_names(
    valid_prior_sds,
):
    prior_means = np.array([100.0, 100.0, -1.0, 80.0, 0.5, 300.0, 120.0, 30.0])
    prior_sds = np.array([10.0, 10.0, 5.0, 10.0, 0.1, 20.0, 10.0, 5.0])

    with pytest.raises(ValueError, match="A_neutral"):
        DMCneutral(
            prior_means=prior_means,
            prior_sds=prior_sds,
            param_names=("A_con", "A_inc", "A_neutral", "tau", "mu_c", "mu_r", "b", "sd_r"),
            A_neutral=0,  # fixed AND estimated -- contradictory
        )


def test_dmc_neutral_init_raises_when_a_neutral_is_neither_fixed_nor_in_param_names(
    valid_prior_means, valid_prior_sds,
):
    with pytest.raises(ValueError, match="A_neutral"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
            A_neutral=None,  # not fixed, and not in param_names either
        )


def test_dmc_neutral_accepts_a_neutral_fixed_at_a_nonzero_value(valid_prior_means, valid_prior_sds):
    sim = DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
        A_neutral=-3.5,
        rng=np.random.default_rng(0),
    )
    assert sim.A_neutral == -3.5

    result = sim(num_obs=12, rng=np.random.default_rng(1))
    assert "A_neutral" not in result  # it is a constant, not a sampled parameter
    assert result["rt"].shape == (12,)


def test_dmc_neutral_accepts_a_neutral_estimated(simulator_estimated_neutral):
    sim = simulator_estimated_neutral
    assert sim.A_neutral is None
    assert "A_neutral" in sim.param_names

    params = sim.prior(rng=np.random.default_rng(1))
    assert "A_neutral" in params


# ---------------------------------------------------------------------------
# prior()
# ---------------------------------------------------------------------------
def test_prior_returns_expected_keys(simulator):
    params = simulator.prior(rng=np.random.default_rng(1))

    assert set(params.keys()) == {"A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"}
    assert all(np.isscalar(v) for v in params.values())


def test_prior_does_not_return_none(simulator):
    # Regression test: an earlier version had a return statement nested one
    # level too deep inside prior(), so every branch except
    # param_lower_bound=None fell through and returned None instead of a dict.
    params = simulator.prior(rng=np.random.default_rng(1))
    assert params is not None
    assert isinstance(params, dict)


def test_prior_respects_scalar_lower_bound(valid_prior_means, valid_prior_sds):
    sim = DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        param_lower_bound=0,
        rng=np.random.default_rng(1),
    )

    params = sim.prior(rng=np.random.default_rng(2))

    assert all(v >= 0 for v in params.values())


def test_prior_respects_per_parameter_lower_bound(simulator_estimated_neutral):
    # A_neutral's bound is -inf while every other parameter is bounded at 0;
    # this must hold across many draws, not just by construction.
    sim = simulator_estimated_neutral
    draws = [sim.prior(rng=np.random.default_rng(seed)) for seed in range(300)]

    a_neutral_vals = np.array([d["A_neutral"] for d in draws])
    assert np.any(a_neutral_vals < 0), "A_neutral should be able to draw negative values"

    for name in ("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"):
        vals = np.array([d[name] for d in draws])
        assert np.all(vals >= 0), f"{name} violated its lower bound of 0"


def test_prior_respects_custom_param_names_order(valid_prior_sds):
    # A_inc listed before A_con: prior() must label draws by position in
    # param_names, not by a hardcoded assumption about ordering.
    prior_means = np.array([100.0, 100.0, 80.0, 0.5, 300.0, 120.0, 30.0])

    sim = DMCneutral(
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


# ---------------------------------------------------------------------------
# experiment()
# ---------------------------------------------------------------------------
def test_experiment_returns_expected_structure(simulator):
    result = simulator.experiment(
        A_con=100.0,
        A_inc=100.0,
        tau=80.0,
        mu_c=0.5,
        mu_r=300.0,
        b=120.0,
        sd_r=30.0,
        num_obs=30,
        rng=np.random.default_rng(42),
    )

    assert set(result.keys()) == {"rt", "accuracy", "conditions", "num_obs"}
    assert result["rt"].shape == (30,)
    assert result["accuracy"].shape == (30,)
    assert result["conditions"].shape == (30,)
    assert result["num_obs"] == 30

    assert set(np.unique(result["conditions"])).issubset({0, 1, 2})
    assert set(np.unique(result["accuracy"])).issubset({-1, 0, 1})


def test_experiment_produces_all_three_conditions(simulator):
    result = simulator.experiment(
        A_con=100.0, A_inc=100.0, tau=80.0, mu_c=0.5, mu_r=300.0, b=120.0, sd_r=30.0,
        num_obs=90, rng=np.random.default_rng(1),
    )
    assert set(np.unique(result["conditions"])) == {0, 1, 2}


def test_experiment_conditions_are_balanced(simulator):
    # Regression test: an earlier version drew conditions independently via
    # rng.choice, which does not guarantee a balanced split. experiment() must
    # use a deterministic ceil-split (then shuffle) across the three conditions.
    #
    # With 3 conditions, a ceil-split-then-truncate can leave the counts up to
    # 2 apart when num_obs is not a multiple of 3 (e.g. 301 -> 101/101/99), so
    # the tolerance here is num_conditions - 1 rather than a fixed 1.
    for num_obs in (9, 90, 301):
        result = simulator.experiment(
            A_con=100.0, A_inc=100.0, tau=80.0, mu_c=0.5, mu_r=300.0, b=120.0,
            sd_r=30.0, num_obs=num_obs, rng=np.random.default_rng(1),
        )
        counts = np.bincount(result["conditions"].astype(int), minlength=3)
        assert counts.sum() == num_obs
        assert counts.max() - counts.min() <= simulator.num_conditions - 1, (
            f"unbalanced split for num_obs={num_obs}: {counts}"
        )


def test_experiment_condition_order_is_shuffled_not_blocked(simulator):
    # A ceil-split without shuffling would put all of condition 0 first, all of
    # condition 1 next, etc. The first and last several trials should not all
    # share the same condition.
    result = simulator.experiment(
        A_con=100.0, A_inc=100.0, tau=80.0, mu_c=0.5, mu_r=300.0, b=120.0,
        sd_r=30.0, num_obs=90, rng=np.random.default_rng(3),
    )
    conditions = result["conditions"]
    assert len(set(conditions[:10].tolist())) > 1
    assert len(set(conditions[-10:].tolist())) > 1


def test_experiment_incongruent_trials_use_negated_a_inc(simulator):
    # Core contract shared with DMCasym: A_inc is estimated as a positive
    # magnitude and experiment() must negate it before simulating incongruent
    # trials, so its automatic activation opposes A_con's.
    A_con, A_inc, A_neutral = 40.0, 55.0, 0.0
    tau, mu_c, mu_r, b, sd_r = 80.0, 0.5, 300.0, 120.0, 30.0
    num_obs = 30

    result = simulator.experiment(
        A_con=A_con, A_inc=A_inc, tau=tau, mu_c=mu_c, mu_r=mu_r, b=b, sd_r=sd_r,
        num_obs=num_obs, rng=np.random.default_rng(7),
    )

    # Reproduce experiment()'s internals by hand with a fresh, identically
    # seeded rng and the exact same balanced-and-shuffled condition assignment.
    rng_manual = np.random.default_rng(7)
    obs_per_condition = int(np.ceil(num_obs / simulator.num_conditions))
    conditions = np.repeat(np.arange(simulator.num_conditions), obs_per_condition)[:num_obs]
    rng_manual.shuffle(conditions)

    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    T = len(t)
    noise = rng_manual.normal(size=(num_obs, T))
    non_decision_ts = rng_manual.normal(size=num_obs, loc=mu_r, scale=sd_r)

    data = np.zeros((num_obs, 2))
    for code, A in ((0, A_con), (1, -A_inc), (2, A_neutral)):
        mask = conditions == code
        data[mask, :] = simulator.trial(
            A=A, tau=tau, mu_c=mu_c, b=b, t=t,
            noise=noise[mask], non_decision_ts=non_decision_ts[mask], rng=rng_manual,
        )

    assert np.array_equal(result["conditions"], conditions)
    assert np.array_equal(result["rt"], data[:, 0])
    assert np.array_equal(result["accuracy"], data[:, 1])


def test_experiment_neutral_trials_use_the_resolved_a_neutral(simulator_estimated_neutral):
    # Regression test: an earlier version called trial() for the neutral
    # condition with A=self.A_neutral directly, instead of the local A_neutral
    # variable resolved from the passed-in / fixed value. That crashed with
    # TypeError whenever A_neutral was estimated (self.A_neutral is None).
    sim = simulator_estimated_neutral

    result = sim.experiment(
        A_con=40.0, A_inc=55.0, A_neutral=-2.0,
        tau=80.0, mu_c=0.5, mu_r=300.0, b=120.0, sd_r=30.0,
        num_obs=30, rng=np.random.default_rng(5),
    )
    assert result["rt"].shape == (30,)


def test_experiment_raises_rather_than_silently_using_none_for_neutral(
    simulator_estimated_neutral,
):
    # If A_neutral is neither passed in nor fixed, there is no valid value to
    # simulate neutral trials with; this must fail loudly, not silently run
    # trial() with A=None.
    sim = simulator_estimated_neutral
    with pytest.raises((TypeError, ValueError)):
        sim.experiment(
            A_con=40.0, A_inc=55.0,  # A_neutral omitted
            tau=80.0, mu_c=0.5, mu_r=300.0, b=120.0, sd_r=30.0,
            num_obs=10, rng=np.random.default_rng(5),
        )


def test_congruency_effect_direction_with_matched_a_con_a_inc(valid_prior_sds):
    # With A_con and A_inc drawn from the same positive-only prior, incongruent
    # trials should still be systematically slower than congruent trials on
    # average, because A_inc is negated internally -- not because the two
    # amplitudes differ in magnitude.
    prior_means = np.array([32.4, 32.4, 93.88, 0.49, 387.53, 88.21, 48.25])
    prior_sds = np.array([9.05, 9.05, 29.67, 0.15, 57.65, 16.56, 10.41])

    sim = DMCneutral(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
        param_lower_bound=[0, 0, 0, 0, 0, 0, 0],
        A_neutral=0,
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


def test_experiment_with_contamination_probability_one_replaces_rts_and_responses(
    valid_prior_means, valid_prior_sds,
):
    sim = DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        contamination_probability=1.0,
        contamination_uniform_lower=0.25,
        contamination_uniform_upper=0.5,
        A_neutral=0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    out = sim.experiment(
        A_con=100.0, A_inc=100.0, tau=80.0, mu_c=0.5, mu_r=300.0, b=120.0, sd_r=30.0,
        num_obs=30, rng=np.random.default_rng(42),
    )

    assert np.all(out["rt"] >= 0.25)
    assert np.all(out["rt"] <= 0.5)
    assert set(np.unique(out["accuracy"])).issubset({0, 1})


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------
def test_call_returns_parameters_and_data(simulator):
    result = simulator(num_obs=12, rng=np.random.default_rng(42))

    expected_keys = {
        "A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r",
        "rt", "accuracy", "conditions", "num_obs",
    }
    assert set(result.keys()) == expected_keys

    assert np.isscalar(result["A_con"])
    assert np.isscalar(result["A_inc"])
    assert result["rt"].shape == (12,)
    assert result["accuracy"].shape == (12,)
    assert result["conditions"].shape == (12,)
    assert result["num_obs"] == 12


def test_call_includes_a_neutral_when_estimated(simulator_estimated_neutral):
    result = simulator_estimated_neutral(num_obs=12, rng=np.random.default_rng(42))
    assert "A_neutral" in result
    assert np.isscalar(result["A_neutral"])


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------
def test_sample_does_not_raise_typeerror_from_none_prior(simulator):
    # End-to-end regression test for the prior()-returns-None bug: sample()
    # must not raise "argument after ** must be a mapping, not NoneType".
    sims = simulator.sample(batch_size=50, num_obs=10, seed=1)
    assert sims["rt"].shape == (50, 10, 1)


def test_sample_returns_expected_shapes(simulator):
    sims = simulator.sample(batch_size=4, num_obs=12, seed=123)

    for key in ("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"):
        assert sims[key].shape == (4, 1)

    assert sims["rt"].shape == (4, 12, 1)
    assert sims["accuracy"].shape == (4, 12, 1)
    assert sims["conditions"].shape == (4, 12, 1)
    assert sims["num_obs"].shape == (4, 1)


def test_sample_returns_expected_shapes_with_estimated_a_neutral(
    simulator_estimated_neutral,
):
    sims = simulator_estimated_neutral.sample(batch_size=4, num_obs=12, seed=123)
    assert sims["A_neutral"].shape == (4, 1)
    assert sims["rt"].shape == (4, 12, 1)


def test_sample_is_reproducible_with_seed(simulator):
    sims1 = simulator.sample(batch_size=3, num_obs=8, seed=999)
    sims2 = simulator.sample(batch_size=3, num_obs=8, seed=999)

    for key in sims1:
        assert np.array_equal(sims1[key], sims2[key])


def test_sample_uses_fixed_num_obs_when_num_obs_is_none(valid_prior_means, valid_prior_sds):
    sim = DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        fixed_num_obs=15,
        A_neutral=0,
        rng=np.random.default_rng(123),
    )

    sims = sim.sample(batch_size=2, num_obs=None, seed=123)

    assert sims["rt"].shape == (2, 15, 1)
    assert np.all(sims["num_obs"] == 15)


def test_sample_accepts_tuple_batch_size(simulator):
    sims = simulator.sample(batch_size=(3,), num_obs=9, seed=123)
    assert sims["rt"].shape == (3, 9, 1)


# ---------------------------------------------------------------------------
# trial()
# ---------------------------------------------------------------------------
def test_trial_returns_expected_shape(simulator):
    rng = np.random.default_rng(42)
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    noise = rng.normal(size=(5, len(t)))
    non_decision_ts = rng.normal(loc=300.0, scale=30.0, size=5)

    out = simulator.trial(
        A=100.0, tau=80.0, mu_c=0.5, b=120.0, t=t,
        noise=noise, non_decision_ts=non_decision_ts, rng=rng,
    )

    assert out.shape == (5, 2)
    assert set(np.unique(out[:, 1])).issubset({-1, 0, 1})


def test_trial_accepts_a_negative_amplitude(simulator):
    # The neutral condition may use a negative A_neutral (e.g. bound = -inf);
    # trial() must handle a negative amplitude without special-casing it.
    rng = np.random.default_rng(42)
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    noise = rng.normal(size=(5, len(t)))
    non_decision_ts = rng.normal(loc=300.0, scale=30.0, size=5)

    out = simulator.trial(
        A=-2.5, tau=80.0, mu_c=0.5, b=120.0, t=t,
        noise=noise, non_decision_ts=non_decision_ts, rng=rng,
    )

    assert out.shape == (5, 2)
    assert np.isfinite(out[:, 0]).all()


def test_trial_runs_for_a_value_not_equal_to_2(valid_prior_means, valid_prior_sds):
    sim = DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        a_value=3,
        A_neutral=0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    rng = np.random.default_rng(42)
    t = np.arange(sim.dt, sim.tmax + sim.dt, sim.dt)
    noise = rng.normal(size=(4, len(t)))
    non_decision_ts = rng.normal(loc=300.0, scale=30.0, size=4)

    out = sim.trial(
        A=100.0, tau=80.0, mu_c=0.5, b=120.0, t=t,
        noise=noise, non_decision_ts=non_decision_ts, rng=rng,
    )

    assert out.shape == (4, 2)
    assert np.isfinite(out[:, 0]).all()
    assert set(np.unique(out[:, 1])).issubset({-1, 0, 1})


def test_time_grid_has_exact_dt_spacing(simulator):
    t = np.arange(simulator.dt, simulator.tmax + simulator.dt, simulator.dt)
    assert np.allclose(np.diff(t), simulator.dt)


def test_experiment_runs_for_fractional_dt(valid_prior_means, valid_prior_sds):
    sim = DMCneutral(
        prior_means=valid_prior_means,
        prior_sds=valid_prior_sds,
        dt=0.5,
        A_neutral=0,
        fixed_num_obs=10,
        rng=np.random.default_rng(1),
    )

    result = sim(num_obs=10, rng=np.random.default_rng(2))
    assert result["rt"].shape == (10,)


# ---------------------------------------------------------------------------
# sd_r fixed
# ---------------------------------------------------------------------------
def test_dmc_neutral_with_sdr_fixed_requires_param_names_without_sd_r():
    prior_means = np.array([100.0, 100.0, 80.0, 0.5, 300.0, 120.0])
    prior_sds = np.array([10.0, 10.0, 10.0, 0.1, 20.0, 10.0])

    sim = DMCneutral(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b"),
        A_neutral=0,
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


def test_dmc_neutral_raises_if_sdr_fixed_but_param_names_still_include_sd_r(
    valid_prior_means, valid_prior_sds,
):
    with pytest.raises(ValueError, match="param_names must contain exactly"):
        DMCneutral(
            prior_means=valid_prior_means,
            prior_sds=valid_prior_sds,
            param_names=("A_con", "A_inc", "tau", "mu_c", "mu_r", "b", "sd_r"),
            A_neutral=0,
            sdr_fixed=30.0,
            fixed_num_obs=10,
            rng=np.random.default_rng(123),
        )


def test_dmc_neutral_supports_a_neutral_estimated_and_sdr_fixed_together():
    # The four-way cross of {A_neutral fixed/estimated} x {sd_r fixed/estimated}
    # must all be constructible; this is the remaining untested quadrant.
    prior_means = np.array([100.0, 100.0, -1.0, 80.0, 0.5, 300.0, 120.0])
    prior_sds = np.array([10.0, 10.0, 5.0, 10.0, 0.1, 20.0, 10.0])

    sim = DMCneutral(
        prior_means=prior_means,
        prior_sds=prior_sds,
        param_names=("A_con", "A_inc", "A_neutral", "tau", "mu_c", "mu_r", "b"),
        A_neutral=None,
        param_lower_bound=(0, 0, -np.inf, 0, 0, 0, 0),
        sdr_fixed=20.0,
        fixed_num_obs=10,
        rng=np.random.default_rng(123),
    )

    result = sim(num_obs=8, rng=np.random.default_rng(1))
    assert "A_neutral" in result
    assert "sd_r" not in result
