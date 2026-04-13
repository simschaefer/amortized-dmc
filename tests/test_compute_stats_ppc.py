
from pandas.testing import assert_frame_equal
import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dmc_helpers import compute_stats_ppc


def test_compute_stats_ppc_missing_required_column_raises_keyerror():
    data = pd.DataFrame(
        {
            "id": ["s1"],
            "congruency": ["congruent"],
            "accuracy": [1],
        }
    )

    with pytest.raises(KeyError, match="Missing required columns"):
        compute_stats_ppc(data, draw_name=None)


def test_compute_stats_ppc_no_draw_returns_expected_outputs():
    data = pd.DataFrame(
        {
            "id": ["s1"] * 6,
            "congruency": [
                "congruent", "congruent", "congruent",
                "incongruent", "incongruent", "incongruent",
            ],
            "rt": [100, 200, 1000, 300, 400, 2000],
            "accuracy": [1, 1, 0, 1, 1, 0],
        }
    )

    caf, cdf, delta = compute_stats_ppc(
        data,
        draw_name=None,
        n_rt_bins=2,
        quantiles=[0.5],
    )

    expected_delta = pd.DataFrame(
        {
            "id": ["s1"],
            "quantile": [0.5],
            "congruent": [150.0],
            "incongruent": [350.0],
            "delta": [200.0],
            "mean_qu": [250.0],
        }
    )

    delta = delta.sort_values(["id", "quantile"]).reset_index(drop=True)
    delta.columns.name = None
    assert_frame_equal(delta, expected_delta)

    expected_cdf = pd.DataFrame(
        {
            "id": ["s1", "s1"],
            "quantile": [0.5, 0.5],
            "congruency": ["congruent", "incongruent"],
            "rt": [150.0, 350.0],
        }
    )

    cdf = cdf.sort_values(["id", "quantile", "congruency"]).reset_index(drop=True)
    expected_cdf = expected_cdf.sort_values(["id", "quantile", "congruency"]).reset_index(drop=True)
    assert_frame_equal(cdf, expected_cdf)

    assert set(caf.columns) == {"id", "congruency", "rt_bin", "accuracy"}
    assert len(caf) == 4
    assert caf["accuracy"].between(0, 1).all()


def test_compute_stats_ppc_draw_level_aggregation():
    data = pd.DataFrame(
        {
            "num_resim": [1] * 8 + [1] * 8,
            "id": ["s1"] * 8 + ["s2"] * 8,
            "congruency": (
                ["congruent"] * 4 + ["incongruent"] * 4
                + ["congruent"] * 4 + ["incongruent"] * 4
            ),
            "rt": [
                100, 200, 300, 400,
                200, 300, 400, 500,
                110, 210, 310, 410,
                210, 310, 410, 510,
            ],
            "accuracy": [1] * 16,
        }
    )

    caf, cdf, delta = compute_stats_ppc(
        data,
        draw_name="num_resim",
        n_rt_bins=2,
        quantiles=[0.5],
    )

    expected_delta = pd.DataFrame(
        {
            "num_resim": [1],
            "quantile": [0.5],
            "congruent": [255.0],
            "incongruent": [355.0],
            "delta": [100.0],
            "mean_qu": [305.0],
        }
    )

    delta = delta.sort_values(["num_resim", "quantile"]).reset_index(drop=True)
    delta.columns.name = None
    assert_frame_equal(delta, expected_delta)

    assert set(cdf.columns) == {"num_resim", "congruency", "quantile", "rt"}
    assert set(caf.columns) == {"num_resim", "congruency", "rt_bin", "accuracy"}


def test_compute_stats_ppc_raises_for_invalid_congruency_levels():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1", "s1"],
            "congruency": ["only_one_level"] * 3,
            "rt": [100, 200, 300],
            "accuracy": [1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="Congruency variable is coded as"):
        compute_stats_ppc(data, draw_name=None, quantiles=[0.5])


def test_compute_stats_ppc_drops_missing_rows():
    data = pd.DataFrame(
        {
            "id": ["s1", "s1", None, "s1"],
            "congruency": ["congruent", "incongruent", "congruent", "incongruent"],
            "rt": [100, 200, 300, None],
            "accuracy": [1, 1, 1, 1],
        }
    )

    caf, cdf, delta = compute_stats_ppc(
        data,
        draw_name=None,
        n_rt_bins=2,
        quantiles=[0.5],
    )

    expected_delta = pd.DataFrame(
        {
            "id": ["s1"],
            "quantile": [0.5],
            "congruent": [100.0],
            "incongruent": [200.0],
            "delta": [100.0],
            "mean_qu": [150.0],
        }
    )

    delta = delta.sort_values(["id", "quantile"]).reset_index(drop=True)
    delta.columns.name = None
    assert_frame_equal(delta, expected_delta)