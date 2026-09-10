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


PARAMS = ("A", "tau", "mu_c", "mu_r", "b", "sd_r")
SEED = 20240101


class FakeSimulator:
    def __init__(self):
        self.calls = []
        self.param_names = ("A", "tau", "mu_c", "mu_r", "b", "sd_r")
        self.param_lower_bound = 0

    def experiment(self, **kwargs):
        self.calls.append(kwargs.copy())
        num_obs = kwargs["num_obs"]
        return {
            "rt": np.full(num_obs, 0.5),
            "accuracy": np.ones(num_obs, dtype=int),
            "conditions": np.zeros(num_obs, dtype=int),
        }

class StrictFakeSimulator(FakeSimulator):
    """
    Mirrors a real experiment() signature, so a param_names that omits a
    parameter is caught instead of silently falling back to a default.

    FakeSimulator.experiment(**kwargs) accepts anything and therefore cannot
    detect that class of mistake; use this one wherever the calling contract
    itself is what is under test.
    """

    def experiment(self, A, tau, mu_c, mu_r, b, sd_r, num_obs, rng=None):
        return super().experiment(A=A, tau=tau, mu_c=mu_c, mu_r=mu_r, b=b,
                                  sd_r=sd_r, num_obs=num_obs, rng=rng)


def param_rows(df, param_names=PARAMS):
    """The set of parameter combinations that actually occur in the posterior."""
    return {tuple(float(v) for v in row) for row in df[list(param_names)].to_numpy()}


def call_combo(call, param_names=PARAMS):
    return tuple(float(call[p]) for p in param_names)


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
def posterior_with_invalid_rows():
    """Rows 0 and 1 each violate the lower bound in one parameter; rows 2 and 3 are valid."""
    return pd.DataFrame(
        {
            "A":    [-1.0,   2.0,  3.0,  4.0, 1],
            "tau":  [10.0, -20.0, 30.0, 40.0, 1],
            "mu_c": [0.1,    0.2,  0.3,  0.4, 1],
            "mu_r": [100.0, 110.0, 120.0, 130.0, 1],
            "b":    [50.0,  60.0, 70.0, 80.0, 1],
            "sd_r": [5.0,    6.0,  7.0,  8.0, 1],
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

def test_resim_data_id_calls_simulator_with_posterior_rows(posterior_samples_df):
    """Every simulated parameter set must be a row that exists in the posterior."""
    simulator = FakeSimulator()

    _, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=3,
        rng=np.random.default_rng(SEED),
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18  # 3 draws x 6 parameters

    assert len(simulator.calls) == 3
    assert all(call["num_obs"] == 2 for call in simulator.calls)

    rows = param_rows(posterior_samples_df)
    combos = [call_combo(c) for c in simulator.calls]

    # each call is an actual posterior draw, and with num_resims == n_draws
    # every draw is used exactly once (sampling is without replacement)
    assert all(combo in rows for combo in combos)
    assert set(combos) == rows
    assert len(set(combos)) == 3


def test_resim_data_id_preserves_joint_posterior_structure():
    """
    Regression test: parameters must stay paired within a draw.

    Filtering or shuffling each parameter independently would produce
    combinations that never occur in the posterior.
    """
    post = pd.DataFrame(
        {
            "A":    [1.0, 2.0, 3.0, 4.0, 5.0],
            "tau":  [10.0, 20.0, 30.0, 40.0, 50.0],
            "mu_c": [0.1, 0.2, 0.3, 0.4, 0.5],
            "mu_r": [100.0, 200.0, 300.0, 400.0, 500.0],
            "b":    [50.0, 60.0, 70.0, 80.0, 90.0],
            "sd_r": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    simulator = FakeSimulator()

    resim_data_id(
        post_sample_data=post,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=4,
        rng=np.random.default_rng(SEED),
    )

    rows = param_rows(post)
    for call in simulator.calls:
        assert call_combo(call) in rows

    # A and tau move together in this fixture (tau == 10 * A); an independent
    # shuffle would break that relationship
    for call in simulator.calls:
        assert call["tau"] == pytest.approx(10.0 * call["A"])

def test_resim_data_id_excludes_whole_draws_below_lower_bound(posterior_with_invalid_rows):
    simulator = FakeSimulator()

    _, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_with_invalid_rows,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=2,
        lower_bound=0,
        rng=np.random.default_rng(SEED),
    )

    # counts are draws x parameters, derived from the fixture so that adding a
    # row to it does not require editing this test
    n_rows = len(posterior_with_invalid_rows)
    n_invalid = int((posterior_with_invalid_rows[list(PARAMS)] < 0).any(axis=1).sum())

    assert n_all_samples == n_rows * len(PARAMS)
    assert n_excluded_samples == n_invalid * len(PARAMS)

    assert len(simulator.calls) == 2
    for call in simulator.calls:
        for p in PARAMS:
            assert call[p] >= 0

    # only fully valid rows may be used -- which of them get drawn depends on
    # the seed, so this is a subset check rather than an equality
    valid = posterior_with_invalid_rows[
        (posterior_with_invalid_rows[list(PARAMS)] >= 0).all(axis=1)
    ]
    used = {call_combo(c) for c in simulator.calls}
    assert used <= param_rows(valid)
    assert len(used) == 2


def test_resim_data_id_raises_when_too_few_valid_draws(posterior_with_invalid_rows):
    simulator = FakeSimulator()

    with pytest.raises(ValueError, match="only 3 valid posterior draws remain"):
        resim_data_id(
            post_sample_data=posterior_with_invalid_rows,
            num_obs=2,
            simulator=simulator,
            id="s1",
            num_resims=4,  # only 3 valid draws remain
            lower_bound=0,
            rng=np.random.default_rng(SEED),
        )

    assert simulator.calls == []


def test_resim_data_id_supports_custom_id_name_with_fake_simulator(posterior_samples_df):
    simulator = FakeSimulator()

    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=3,
        simulator=simulator,
        id=42,
        id_name="subject",
        num_resims=2,
        rng=np.random.default_rng(SEED),
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18

    assert "subject" in result.columns
    assert set(result["subject"]) == {42}
    assert "id" not in result.columns


def test_resim_data_id_respects_param_names_subset_with_fake_simulator(posterior_samples_df):
    simulator = FakeSimulator()

    _, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=2,
        param_names=("A", "tau"),
        rng=np.random.default_rng(SEED),
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 6  # 2 parameters x 3 draws

    assert len(simulator.calls) == 2
    for call in simulator.calls:
        # rng is forwarded to the simulator so that data generation is seeded too
        assert set(call.keys()) == {"A", "tau", "num_obs", "rng"}


def test_resim_data_id_is_reproducible_with_a_seeded_rng(posterior_samples_df):
    sims = []
    for _ in range(2):
        simulator = FakeSimulator()
        resim_data_id(
            post_sample_data=posterior_samples_df,
            num_obs=2,
            simulator=simulator,
            id="s1",
            num_resims=3,
            rng=np.random.default_rng(SEED),
        )
        sims.append([call_combo(c) for c in simulator.calls])

    assert sims[0] == sims[1]


# --------------------------------
# Integration tests with real DMC
# --------------------------------

def test_resim_data_id_returns_expected_shape_and_columns_with_real_simulator(
    real_simulator, posterior_samples_df
):
    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=6,
        simulator=real_simulator,
        id="s1",
        id_name="id",
        num_resims=3,
        rng=np.random.default_rng(SEED),
    )

    assert n_excluded_samples == 0
    assert n_all_samples == 18

    assert set(result.columns) == {"rt", "accuracy", "conditions", "num_obs", "num_resim", "id"}
    assert (result["num_obs"] == 6).all()
    assert len(result) == 18
    assert set(result["num_resim"]) == {0, 1, 2}
    assert set(result["id"]) == {"s1"}


def test_resim_data_id_returns_valid_trial_level_output_with_real_simulator(
    real_simulator, posterior_samples_df
):
    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=8,
        simulator=real_simulator,
        id="p01",
        num_resims=2,
        rng=np.random.default_rng(SEED),
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


def test_resim_data_id_filters_negative_samples_with_real_simulator(real_simulator):
    post_sample_data = pd.DataFrame(
        {
            "A":    [-100.0, 110.0, 120.0, 130.0],
            "tau":  [80.0,   -85.0,  90.0,  95.0],
            "mu_c": [0.4,      0.5,   0.6,   0.7],
            "mu_r": [300.0,  310.0, 320.0, 330.0],
            "b":    [100.0,  110.0, 120.0, 130.0],
            "sd_r": [20.0,    25.0,  30.0,  35.0],
        }
    )

    result, n_excluded_samples, n_all_samples = resim_data_id(
        post_sample_data=post_sample_data,
        num_obs=5,
        simulator=real_simulator,
        id="s1",
        num_resims=2,
        lower_bound=0,
        rng=np.random.default_rng(SEED),
    )

    assert n_all_samples == 24       # 4 draws x 6 parameters
    assert n_excluded_samples == 12  # 2 invalid draws x 6 parameters

    assert len(result) == 10
    assert set(result["num_resim"]) == {0, 1}


def test_resim_data_id_forwards_rng_to_real_simulator(real_simulator, posterior_samples_df):
    """Same seed -> identical resimulated data, including the simulated trials."""
    results = [
        resim_data_id(
            post_sample_data=posterior_samples_df,
            num_obs=8,
            simulator=real_simulator,
            id="s1",
            num_resims=2,
            rng=np.random.default_rng(SEED),
        )[0].reset_index(drop=True)
        for _ in range(2)
    ]

    assert results[0].equals(results[1])


def test_resim_data_id_passes_every_parameter_to_a_strict_signature(posterior_samples_df):
    """A simulator with an explicit signature must receive all of its parameters."""
    simulator = StrictFakeSimulator()

    resim_data_id(
        post_sample_data=posterior_samples_df,
        num_obs=2,
        simulator=simulator,
        id="s1",
        num_resims=2,
        rng=np.random.default_rng(SEED),
    )

    assert len(simulator.calls) == 2
    for call in simulator.calls:
        assert set(call.keys()) == set(PARAMS) | {"num_obs", "rng"}


def test_resim_data_id_rejects_param_names_that_drop_a_simulator_parameter(
    posterior_samples_df,
):
    """
    Guards the silent-default hazard: the posterior has the parameter, the
    simulator accepts it, but param_names omits it -- so experiment() would fall
    back to its default and quietly simulate a different model.
    """
    simulator = StrictFakeSimulator()

    with pytest.raises((ValueError, TypeError)) as excinfo:
        resim_data_id(
            post_sample_data=posterior_samples_df,
            num_obs=2,
            simulator=simulator,
            id="s1",
            num_resims=2,
            param_names=("A", "tau"),
            rng=np.random.default_rng(SEED),
        )

    assert "sd_r" in str(excinfo.value)
    assert simulator.calls == []