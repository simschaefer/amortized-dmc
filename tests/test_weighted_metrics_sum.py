
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
from dmc_helpers import weighted_metric_sum


def test_weighted_metric_sum_basic_equal_weights():
    metrics_table = pd.DataFrame(
        {
            "A": [7.0, 0.0, 0.4],
            "tau": [13.0, 0.0, 0.5],
            "mu_c": [2.0, 0.0, 0.2],
        },
        index=["NRMSE", "Posterior Contraction", "Calibration Error"],
    )

    result = weighted_metric_sum(metrics_table)

    nrmse_mean = (7.0 + 13.0 + 2.0) / 3
    pc_mean = (1.0 + 1.0 + 1.0) / 3
    cal_mean = (0.4 + 0.5 + 0.2) / 3

    expected = nrmse_mean + pc_mean + cal_mean

    assert result == pytest.approx(expected)


def test_weighted_metric_sum_with_custom_weights():
    metrics_table = pd.DataFrame(
        {
            "A": [7.0, 0.0, 0.4],
            "tau": [13.0, 0.0, 0.5],
            "mu_c": [2.0, 0.0, 0.2],
        },
        index=["NRMSE", "Posterior Contraction", "Calibration Error"],
    )

    result = weighted_metric_sum(
        metrics_table,
        weight_recovery=0.5,
        weight_pc=2.0,
        weight_sbc=1.5,
    )

    nrmse_mean = (7.0 + 13.0 + 2.0) / 3
    pc_mean = (1.0 + 1.0 + 1.0) / 3
    cal_mean = (0.4 + 0.5 + 0.2) / 3

    expected = 0.5 * nrmse_mean + 2.0 * pc_mean + 1.5 * cal_mean

    assert result == pytest.approx(expected)


def test_weighted_metric_sum_inverts_posterior_contraction_row():
    metrics_table = pd.DataFrame(
        {
            "A": [10.0, 0.25, 0.5],
            "tau": [20.0, 0.75, 0.5],
        },
        index=["NRMSE", "Posterior Contraction", "Calibration Error"],
    )

    result = weighted_metric_sum(metrics_table)

    nrmse_mean = (10.0 + 20.0) / 2
    pc_mean = ((1 - 0.25) + (1 - 0.75)) / 2
    cal_mean = (0.5 + 0.5) / 2

    expected = nrmse_mean + pc_mean + cal_mean

    assert result == pytest.approx(expected)


def test_weighted_metric_sum_does_not_modify_input():
    metrics_table = pd.DataFrame(
        {
            "A": [7.0, 0.0, 0.4],
            "tau": [13.0, 0.0, 0.5],
            "mu_c": [2.0, 0.0, 0.2],
        },
        index=["NRMSE", "Posterior Contraction", "Calibration Error"],
    )

    original = metrics_table.copy(deep=True)

    _ = weighted_metric_sum(metrics_table)

    pd.testing.assert_frame_equal(metrics_table, original)


def test_weighted_metric_sum_with_realistic_example():
    metrics_table = pd.DataFrame(
        {
            "A": [7.353421, 0.0, 0.447895],
            "tau": [13.715958, 0.0, 0.500000],
            "mu_c": [2.506501, 0.0, 0.184474],
            "mu_r": [0.896055, 0.0, 0.447895],
            "b": [1.663133, 0.0, 0.447895],
            "sd_r": [3.120877, 0.0, 0.239474],
        },
        index=["NRMSE", "Posterior Contraction", "Calibration Error"],
    )

    result = weighted_metric_sum(metrics_table)

    nrmse_mean = metrics_table.loc["NRMSE"].mean()
    pc_mean = (1 - metrics_table.loc["Posterior Contraction"]).mean()
    cal_mean = metrics_table.loc["Calibration Error"].mean()

    expected = nrmse_mean + pc_mean + cal_mean

    assert result == pytest.approx(expected)