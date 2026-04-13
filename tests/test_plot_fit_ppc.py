
import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt


scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import plot_fit_ppc


@pytest.fixture
def example_ppc_data():
    caf_data = pd.DataFrame(
        {
            "num_resim": [0, 0, 0, 0, 1, 1, 1, 1],
            "congruency": [
                "congruent", "congruent", "incongruent", "incongruent",
                "congruent", "congruent", "incongruent", "incongruent",
            ],
            "rt_bin": [0, 1, 0, 1, 0, 1, 0, 1],
            "accuracy": [0.95, 0.90, 0.85, 0.80, 0.96, 0.91, 0.84, 0.79],
        }
    )

    cdf_data = pd.DataFrame(
        {
            "num_resim": [0, 0, 0, 0, 1, 1, 1, 1],
            "quantile": [0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.75, 0.75],
            "congruency": [
                "congruent", "incongruent", "congruent", "incongruent",
                "congruent", "incongruent", "congruent", "incongruent",
            ],
            "rt": [0.35, 0.45, 0.55, 0.70, 0.36, 0.46, 0.56, 0.71],
        }
    )

    delta_data = pd.DataFrame(
        {
            "num_resim": [0, 0, 1, 1],
            "quantile": [0.25, 0.75, 0.25, 0.75],
            "mean_qu": [0.40, 0.62, 0.41, 0.63],
            "delta": [0.10, 0.15, 0.11, 0.14],
        }
    )

    caf_data_emp = pd.DataFrame(
        {
            "congruency": ["congruent", "congruent", "incongruent", "incongruent"],
            "rt_bin": [0, 1, 0, 1],
            "accuracy": [0.955, 0.905, 0.845, 0.795],
        }
    )

    cdf_data_emp = pd.DataFrame(
        {
            "quantile": [0.25, 0.25, 0.75, 0.75],
            "congruency": ["congruent", "incongruent", "congruent", "incongruent"],
            "rt": [0.355, 0.455, 0.555, 0.705],
        }
    )

    delta_data_emp = pd.DataFrame(
        {
            "quantile": [0.25, 0.75],
            "mean_qu": [0.405, 0.625],
            "delta": [0.10, 0.15],
        }
    )

    return caf_data, cdf_data, delta_data, caf_data_emp, cdf_data_emp, delta_data_emp


def test_plot_fit_ppc_returns_figure_and_axes(example_ppc_data):
    caf_data, cdf_data, delta_data, caf_data_emp, cdf_data_emp, delta_data_emp = example_ppc_data

    fig, axes = plot_fit_ppc(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        caf_data_emp=caf_data_emp,
        cdf_data_emp=cdf_data_emp,
        delta_data_emp=delta_data_emp,
    )

    assert fig is not None
    assert len(axes) == 3

    assert axes[0].get_title() == "CAF"
    assert axes[1].get_title() == "CDF"
    assert axes[2].get_title() == "$\\Delta$-Function"

    assert axes[0].get_ylabel() == "CAF"
    assert axes[1].get_ylabel() == "Cumulative Density"
    assert axes[2].get_ylabel() == "$\\Delta$"

    plt.close(fig)


def test_plot_fit_ppc_applies_axis_limits(example_ppc_data):
    caf_data, cdf_data, delta_data, caf_data_emp, cdf_data_emp, delta_data_emp = example_ppc_data

    fig, axes = plot_fit_ppc(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        caf_data_emp=caf_data_emp,
        cdf_data_emp=cdf_data_emp,
        delta_data_emp=delta_data_emp,
        cdf_xlim=(0.2, 0.8),
        caf_ylim=(0.7, 1.0),
        delta_xlim=(0.3, 0.8),
        delta_ylim=(-0.05, 0.25),
    )

    assert axes[1].get_xlim() == pytest.approx((0.2, 0.8))
    assert axes[0].get_ylim() == pytest.approx((0.7, 1.0))
    assert axes[2].get_xlim() == pytest.approx((0.3, 0.8))
    assert axes[2].get_ylim() == pytest.approx((-0.05, 0.25))

    plt.close(fig)


def test_plot_fit_ppc_can_draw_on_existing_axes(example_ppc_data):
    caf_data, cdf_data, delta_data, caf_data_emp, cdf_data_emp, delta_data_emp = example_ppc_data

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    returned_fig, returned_axes = plot_fit_ppc(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        caf_data_emp=caf_data_emp,
        cdf_data_emp=cdf_data_emp,
        delta_data_emp=delta_data_emp,
        new_plot=False,
        fig=fig,
        axes=axes,
    )

    assert returned_fig is fig
    assert returned_axes is axes
    assert len(returned_axes) == 3

    plt.close(fig)


def test_plot_fit_ppc_with_mean_and_draw_lines(example_ppc_data):
    caf_data, cdf_data, delta_data, caf_data_emp, cdf_data_emp, delta_data_emp = example_ppc_data

    fig, axes = plot_fit_ppc(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        caf_data_emp=caf_data_emp,
        cdf_data_emp=cdf_data_emp,
        delta_data_emp=delta_data_emp,
        show_draws_caf=True,
        show_draws_cdf=True,
        show_draws_delta=True,
        show_draws_mean=True,
    )

    assert fig is not None
    assert len(axes) == 3

    plt.close(fig)