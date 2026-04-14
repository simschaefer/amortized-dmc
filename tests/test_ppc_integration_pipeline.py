import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import resim_data, compute_stats_ppc, plot_fit_ppc
from dmc_simulator import DMC


@pytest.fixture
def empirical_data():
    return pd.DataFrame(
        {
            "id": ["s1"] * 8 + ["s2"] * 8,
            "rt": [
                0.35, 0.42, 0.51, 0.60, 0.40, 0.48, 0.58, 0.68,
                0.36, 0.43, 0.52, 0.61, 0.41, 0.49, 0.59, 0.69,
            ],
            "accuracy": [
                1, 1, 1, 0, 1, 1, 0, 1,
                1, 1, 1, 0, 1, 1, 0, 1,
            ],
            "congruency": [
                "congruent", "congruent", "congruent", "congruent",
                "incongruent", "incongruent", "incongruent", "incongruent",
                "congruent", "congruent", "congruent", "congruent",
                "incongruent", "incongruent", "incongruent", "incongruent",
            ],
        }
    )


@pytest.fixture
def post_samples():
    return pd.DataFrame(
        {
            "id": ["s1"] * 4 + ["s2"] * 4,
            "A": [100.0, 105.0, 110.0, 115.0, 98.0, 103.0, 108.0, 113.0],
            "tau": [80.0, 82.0, 84.0, 86.0, 79.0, 81.0, 83.0, 85.0],
            "mu_c": [0.50, 0.52, 0.54, 0.56, 0.49, 0.51, 0.53, 0.55],
            "mu_r": [300.0, 305.0, 310.0, 315.0, 295.0, 300.0, 305.0, 310.0],
            "b": [120.0, 122.0, 124.0, 126.0, 118.0, 120.0, 122.0, 124.0],
            "sd_r": [30.0, 31.0, 32.0, 33.0, 29.0, 30.0, 31.0, 32.0],
        }
    )


@pytest.fixture
def real_simulator():
    return DMC(
        prior_means=np.array([100.0, 80.0, 0.5, 300.0, 120.0, 30.0]),
        prior_sds=np.array([10.0, 10.0, 0.1, 20.0, 10.0, 5.0]),
        param_names=("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
        fixed_num_obs=20,
        rng=np.random.default_rng(123),
    )


@pytest.mark.integration
def test_resimulate_compute_stats_and_plot_ppc(monkeypatch, empirical_data, post_samples, real_simulator):
    def deterministic_resim_data_id(
        post_sample_data,
        num_obs,
        simulator,
        id,
        id_name="id",
        num_resims=50,
        param_names=("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
        lower_bound=0,
    ):
        part_samples = post_sample_data.reset_index(drop=True).copy()

        dfs = []
        n_excluded = 0
        n_all = 0

        for p in param_names:
            if p in part_samples.columns:
                vals = part_samples[p].to_numpy()
                n_all += len(vals)
                n_excluded += int(np.sum(vals < lower_bound))

        for i in range(num_resims):
            row = part_samples.iloc[i % len(part_samples)]
            kwargs = {p: row[p] for p in param_names if p in row.index}
            sim = simulator.experiment(**kwargs, num_obs=num_obs)
            df = pd.DataFrame(sim)
            df["num_resim"] = i
            df[id_name] = id
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True), n_excluded, n_all

    monkeypatch.setattr("dmc_helpers.resim_data_id", deterministic_resim_data_id)

    ppc_data = resim_data(
        empirical_data=empirical_data,
        post_samples=post_samples,
        simulator=real_simulator,
        num_resims=3,
        param_names=("A", "tau", "mu_c", "mu_r", "b", "sd_r"),
        rt="rt",
        id_name="id",
        congruency="congruency",
        simulator_congruency="conditions",
        simulator_congruency_coding=0.0,
        simulator_incongruency_coding=1.0,
        exclude_nonconvergents=True,
        lower_bound=0,
    )

    assert not ppc_data.empty
    assert {"rt", "accuracy", "conditions", "num_resim", "id", "congruency"}.issubset(ppc_data.columns)
    assert set(ppc_data["id"]) == {"s1", "s2"}
    assert set(ppc_data["num_resim"]) == {0, 1, 2}

    caf_emp, cdf_emp, delta_emp = compute_stats_ppc(
        data=empirical_data,
        id_name="id",
        draw_name=None,
        n_rt_bins=2,
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
        quantiles=[0.25, 0.5, 0.75],
    )

    caf_ppc, cdf_ppc, delta_ppc = compute_stats_ppc(
        data=ppc_data,
        id_name="id",
        draw_name="num_resim",
        n_rt_bins=2,
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
        quantiles=[0.25, 0.5, 0.75],
    )

    assert not caf_emp.empty
    assert not cdf_emp.empty
    assert not delta_emp.empty

    assert not caf_ppc.empty
    assert not cdf_ppc.empty
    assert not delta_ppc.empty

    assert {"congruency", "rt_bin", "accuracy"}.issubset(caf_emp.columns)
    assert {"num_resim", "congruency", "rt_bin", "accuracy"}.issubset(caf_ppc.columns)
    assert {"quantile", "congruency", "rt"}.issubset(cdf_emp.columns)
    assert {"num_resim", "quantile", "congruency", "rt"}.issubset(cdf_ppc.columns)
    assert {"quantile", "mean_qu", "delta"}.issubset(delta_emp.columns)
    assert {"num_resim", "quantile", "mean_qu", "delta"}.issubset(delta_ppc.columns)

    fig, axes = plot_fit_ppc(
        caf_data=caf_ppc,
        cdf_data=cdf_ppc,
        delta_data=delta_ppc,
        caf_data_emp=caf_emp,
        cdf_data_emp=cdf_emp,
        delta_data_emp=delta_emp,
        show_draws_caf=True,
        show_draws_cdf=True,
        show_draws_delta=True,
        show_draws_mean=True,
        draw_name="num_resim",
        congruency="congruency",
        congruency_emp="congruency",
    )

    assert fig is not None
    assert len(axes) == 3
    assert axes[0].get_title() == "CAF"
    assert axes[1].get_title() == "CDF"
    assert axes[2].get_title() == "$\\Delta$-Function"

    plt.close(fig)