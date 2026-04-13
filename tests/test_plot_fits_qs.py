import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pytest

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import plot_fit_qs


@pytest.fixture
def fit_qs_point_data():
    return pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2"],
            "congruency": ["congruent", "incongruent", "congruent", "incongruent"],
            "mean_rt_emp": [0.40, 0.52, 0.42, 0.54],
            "mean_rt_resim": [0.41, 0.50, 0.43, 0.55],
            "mean_acc_emp": [0.92, 0.86, 0.94, 0.88],
            "mean_acc_resim": [0.91, 0.85, 0.95, 0.87],
            "rt_q25_emp": [0.30, 0.40, 0.31, 0.41],
            "rt_q25_resim": [0.32, 0.39, 0.33, 0.42],
            "rt_q50_emp": [0.40, 0.52, 0.42, 0.54],
            "rt_q50_resim": [0.41, 0.50, 0.43, 0.55],
            "rt_q75_emp": [0.50, 0.64, 0.52, 0.66],
            "rt_q75_resim": [0.51, 0.62, 0.53, 0.67],
        }
    )


@pytest.fixture
def fit_qs_uncertainty_data():
    return pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2"],
            "congruency": ["congruent", "incongruent", "congruent", "incongruent"],

            "mean_rt_emp": [0.40, 0.52, 0.42, 0.54],
            "mean_rt_resim_median": [0.41, 0.50, 0.43, 0.55],
            "mean_rt_resim_q05": [0.37, 0.46, 0.39, 0.50],
            "mean_rt_resim_q95": [0.45, 0.54, 0.47, 0.59],

            "mean_acc_emp": [0.92, 0.86, 0.94, 0.88],
            "mean_acc_resim_median": [0.91, 0.85, 0.95, 0.87],
            "mean_acc_resim_q05": [0.88, 0.81, 0.92, 0.84],
            "mean_acc_resim_q95": [0.95, 0.89, 0.98, 0.91],

            "rt_q25_emp": [0.30, 0.40, 0.31, 0.41],
            "rt_q25_resim_median": [0.32, 0.39, 0.33, 0.42],
            "rt_q25_resim_q05": [0.28, 0.35, 0.29, 0.38],
            "rt_q25_resim_q95": [0.36, 0.43, 0.37, 0.46],

            "rt_q50_emp": [0.40, 0.52, 0.42, 0.54],
            "rt_q50_resim_median": [0.41, 0.50, 0.43, 0.55],
            "rt_q50_resim_q05": [0.37, 0.46, 0.39, 0.50],
            "rt_q50_resim_q95": [0.45, 0.54, 0.47, 0.59],

            "rt_q75_emp": [0.50, 0.64, 0.52, 0.66],
            "rt_q75_resim_median": [0.51, 0.62, 0.53, 0.67],
            "rt_q75_resim_q05": [0.47, 0.58, 0.49, 0.62],
            "rt_q75_resim_q95": [0.55, 0.66, 0.57, 0.71],
        }
    )


def test_plot_fit_qs_returns_figure_and_axes_in_point_mode(fit_qs_point_data):
    fig, axes = plot_fit_qs(fit_qs_point_data, plot_uncertainty=False)

    assert fig is not None
    assert isinstance(axes, list)
    assert len(axes) == 5

    expected_titles = [
        "Mean RT",
        "Mean Accuracy",
        "25% Quantile RT",
        "Median RT",
        "75% Quantile RT",
    ]
    assert [ax.get_title() for ax in axes] == expected_titles

    plt.close(fig)


def test_plot_fit_qs_returns_figure_and_axes_in_uncertainty_mode(fit_qs_uncertainty_data):
    fig, axes = plot_fit_qs(fit_qs_uncertainty_data, plot_uncertainty=True)

    assert fig is not None
    assert isinstance(axes, list)
    assert len(axes) == 5

    expected_titles = [
        "Mean RT",
        "Mean Accuracy",
        "25% Quantile RT",
        "Median RT",
        "75% Quantile RT",
    ]
    assert [ax.get_title() for ax in axes] == expected_titles

    plt.close(fig)


def test_plot_fit_qs_applies_accuracy_limits(fit_qs_point_data):
    fig, axes = plot_fit_qs(
        fit_qs_point_data,
        plot_uncertainty=False,
        accuracy_lims=(0.7, 1.0),
    )

    assert axes[1].get_xlim() == pytest.approx((0.7, 1.0))
    assert axes[1].get_ylim() == pytest.approx((0.7, 1.0))

    plt.close(fig)


def test_plot_fit_qs_raises_when_uncertainty_columns_are_missing(fit_qs_point_data):
    with pytest.raises(ValueError, match="data does not include the columns required for uncertainty plotting"):
        plot_fit_qs(fit_qs_point_data, plot_uncertainty=True)


def test_plot_fit_qs_raises_when_aggregated_columns_are_passed_in_point_mode(fit_qs_uncertainty_data):
    with pytest.raises(ValueError, match="data includes aggregated values across resimulations"):
        plot_fit_qs(fit_qs_uncertainty_data, plot_uncertainty=False)


def test_plot_fit_qs_accepts_additional_scatter_kwargs(fit_qs_point_data):
    fig, axes = plot_fit_qs(
        fit_qs_point_data,
        plot_uncertainty=False,
        alpha=0.8,
        s=30,
    )

    assert fig is not None
    assert len(axes) == 5

    plt.close(fig)