
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
from numpy.testing import assert_array_equal
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import plot_stats


@pytest.fixture
def example_stats_data():
    caf_data = pd.DataFrame(
        {
            "id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
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
            "id": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
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
            "id": ["s1", "s1", "s2", "s2"],
            "quantile": [0.25, 0.75, 0.25, 0.75],
            "congruent": [0.35, 0.55, 0.36, 0.56],
            "incongruent": [0.45, 0.70, 0.46, 0.71],
            "delta": [0.10, 0.15, 0.10, 0.15],
            "mean_qu": [0.40, 0.625, 0.41, 0.635],
        }
    )

    return caf_data, cdf_data, delta_data


def test_plot_stats_returns_figure_and_axes(example_stats_data):
    caf_data, cdf_data, delta_data = example_stats_data

    fig, axes = plot_stats(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
    )

    assert fig is not None
    assert len(axes) == 3

    assert axes[0].get_title() == "CAF"
    assert axes[1].get_title() == "CDF"
    assert axes[2].get_title() == r"$\Delta$-Function"

    assert axes[0].get_ylabel() == "CAF"
    assert axes[1].get_ylabel() == "Cumulative Density"
    assert axes[2].get_ylabel() == r"$\Delta$"

    plt.close(fig)


def test_plot_stats_applies_axis_limits(example_stats_data):
    caf_data, cdf_data, delta_data = example_stats_data

    fig, axes = plot_stats(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        cdf_xlim=(0.2, 0.8),
        delta_xlim=(0.3, 0.8),
        delta_ylim=(-0.05, 0.25),
    )

    assert axes[1].get_xlim() == pytest.approx((0.2, 0.8))
    assert axes[2].get_xlim() == pytest.approx((0.3, 0.8))
    assert axes[2].get_ylim() == pytest.approx((-0.05, 0.25))

    plt.close(fig)


def test_plot_stats_can_draw_on_existing_axes(example_stats_data):
    caf_data, cdf_data, delta_data = example_stats_data

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    returned_fig, returned_axes = plot_stats(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        new_plot=False,
        fig=fig,
        axes=axes,
    )

    assert returned_fig is fig
    assert returned_axes is axes
    assert len(returned_axes) == 3

    plt.close(fig)


def test_plot_stats_with_individual_curves(example_stats_data):
    caf_data, cdf_data, delta_data = example_stats_data

    fig, axes = plot_stats(
        caf_data=caf_data,
        cdf_data=cdf_data,
        delta_data=delta_data,
        individual_cafs=True,
        individual_cdfs=True,
        individual_deltas=True,
    )

    assert fig is not None
    assert len(axes) == 3

    plt.close(fig)