import os
import sys

import pandas as pd
import pytest

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
dmc_module_dir = parent_dir + "/dmc"
sys.path.append(dmc_module_dir)

from dmc_helpers import compute_fit_qs


@pytest.fixture
def empirical_data():
    return pd.DataFrame(
        {
            "id": ["s1"] * 8,
            "congruency": [
                "congruent", "congruent", "congruent", "congruent",
                "incongruent", "incongruent", "incongruent", "incongruent",
            ],
            "rt": [100, 200, 300, 400, 200, 300, 400, 500],
            "accuracy": [1, 1, 0, 1, 1, 0, 1, 1],
        }
    )


@pytest.fixture
def resimulated_data():
    return pd.DataFrame(
        {
            "id": ["s1"] * 16,
            "num_resim": [0] * 8 + [1] * 8,
            "congruency": (
                ["congruent"] * 4 + ["incongruent"] * 4
                + ["congruent"] * 4 + ["incongruent"] * 4
            ),
            "rt": [
                110, 210, 310, 410, 210, 310, 410, 510,
                120, 220, 320, 420, 220, 320, 420, 520,
            ],
            "accuracy": [
                1, 1, 0, 1, 1, 0, 1, 1,
                1, 1, 0, 1, 1, 0, 1, 1,
            ],
        }
    )


def test_compute_fit_qs_summarises_draws_and_merges(empirical_data, resimulated_data):
    result = compute_fit_qs(
        resimulated_data=resimulated_data,
        empirical_data=empirical_data,
        grouping_vars=["id", "congruency"],
        draw_name="num_resim",
        summarise_draws=True,
        id_name="id",
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    result.columns.name = None
    result = result.sort_values(["id", "congruency"]).reset_index(drop=True)

    assert len(result) == 2
    assert set(result["id"]) == {"s1"}
    assert set(result["congruency"]) == {"congruent", "incongruent"}

    expected_columns = {
        "id",
        "congruency",
        "mean_rt_resim_median",
        "mean_rt_resim_q05",
        "mean_rt_resim_q95",
        "mean_acc_resim_median",
        "mean_acc_resim_q05",
        "mean_acc_resim_q95",
        "rt_q25_resim_median",
        "rt_q25_resim_q05",
        "rt_q25_resim_q95",
        "rt_q50_resim_median",
        "rt_q50_resim_q05",
        "rt_q50_resim_q95",
        "rt_q75_resim_median",
        "rt_q75_resim_q05",
        "rt_q75_resim_q95",
        "mean_rt_emp",
        "mean_acc_emp",
        "rt_q25_emp",
        "rt_q50_emp",
        "rt_q75_emp",
    }
    assert expected_columns.issubset(result.columns)

    congruent_row = result[result["congruency"] == "congruent"].iloc[0]
    incongruent_row = result[result["congruency"] == "incongruent"].iloc[0]

    assert congruent_row["mean_rt_emp"] == pytest.approx(250.0)
    assert congruent_row["mean_acc_emp"] == pytest.approx(0.75)
    assert congruent_row["rt_q25_emp"] == pytest.approx(175.0)
    assert congruent_row["rt_q50_emp"] == pytest.approx(250.0)
    assert congruent_row["rt_q75_emp"] == pytest.approx(325.0)

    assert incongruent_row["mean_rt_emp"] == pytest.approx(350.0)
    assert incongruent_row["mean_acc_emp"] == pytest.approx(0.75)
    assert incongruent_row["rt_q25_emp"] == pytest.approx(275.0)
    assert incongruent_row["rt_q50_emp"] == pytest.approx(350.0)
    assert incongruent_row["rt_q75_emp"] == pytest.approx(425.0)

    # congruent resim mean RTs across draws: 260 and 270 -> median 265
    assert congruent_row["mean_rt_resim_median"] == pytest.approx(265.0)
    # incongruent resim mean RTs across draws: 360 and 370 -> median 365
    assert incongruent_row["mean_rt_resim_median"] == pytest.approx(365.0)


def test_compute_fit_qs_without_draw_name_returns_unsummarised_merge(empirical_data, resimulated_data):
    result = compute_fit_qs(
        resimulated_data=resimulated_data.drop(columns=["num_resim"]),
        empirical_data=empirical_data,
        grouping_vars=["id", "congruency"],
        draw_name=None,
        summarise_draws=True,  # ignored internally
        id_name="id",
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    result.columns.name = None
    result = result.sort_values(["id", "congruency"]).reset_index(drop=True)

    assert len(result) == 2
    assert set(result["id"]) == {"s1"}
    assert set(result["congruency"]) == {"congruent", "incongruent"}

    assert "mean_rt_resim" in result.columns
    assert "mean_acc_resim" in result.columns
    assert "rt_q25_resim" in result.columns
    assert "rt_q50_resim" in result.columns
    assert "rt_q75_resim" in result.columns

    assert "mean_rt_emp" in result.columns
    assert "mean_acc_emp" in result.columns
    assert "rt_q25_emp" in result.columns
    assert "rt_q50_emp" in result.columns
    assert "rt_q75_emp" in result.columns

    assert "mean_rt_resim_median" not in result.columns

    congruent_row = result[result["congruency"] == "congruent"].iloc[0]
    assert congruent_row["mean_rt_resim"] == pytest.approx(265.0)
    assert congruent_row["mean_rt_emp"] == pytest.approx(250.0)


def test_compute_fit_qs_keeps_draw_level_summaries_when_requested(empirical_data, resimulated_data):
    result = compute_fit_qs(
        resimulated_data=resimulated_data,
        empirical_data=empirical_data,
        grouping_vars=["id", "congruency"],
        draw_name="num_resim",
        summarise_draws=False,
        id_name="id",
        rt="rt",
        accuracy="accuracy",
        congruency="congruency",
    )

    result.columns.name = None
    result = result.sort_values(["id", "congruency", "num_resim"]).reset_index(drop=True)

    assert len(result) == 4
    assert set(result["num_resim"]) == {0, 1}
    assert set(result["congruency"]) == {"congruent", "incongruent"}

    assert "mean_rt_resim" in result.columns
    assert "mean_acc_resim" in result.columns
    assert "rt_q25_resim" in result.columns
    assert "rt_q50_resim" in result.columns
    assert "rt_q75_resim" in result.columns

    assert "mean_rt_emp" in result.columns
    assert "mean_acc_emp" in result.columns
    assert "rt_q25_emp" in result.columns
    assert "rt_q50_emp" in result.columns
    assert "rt_q75_emp" in result.columns

    assert "mean_rt_resim_median" not in result.columns

    row_g0 = result[(result["congruency"] == "congruent") & (result["num_resim"] == 0)].iloc[0]
    row_g1 = result[(result["congruency"] == "congruent") & (result["num_resim"] == 1)].iloc[0]

    assert row_g0["mean_rt_resim"] == pytest.approx(260.0)
    assert row_g1["mean_rt_resim"] == pytest.approx(270.0)
    assert row_g0["mean_rt_emp"] == pytest.approx(250.0)
    assert row_g1["mean_rt_emp"] == pytest.approx(250.0)


def test_compute_fit_qs_raises_when_draw_name_missing(empirical_data, resimulated_data):
    bad_resim = resimulated_data.drop(columns=["num_resim"])

    with pytest.raises(ValueError, match="draw_name 'num_resim' not present in resimulated_data"):
        compute_fit_qs(
            resimulated_data=bad_resim,
            empirical_data=empirical_data,
            grouping_vars=["id", "congruency"],
            draw_name="num_resim",
            summarise_draws=True,
            id_name="id",
            rt="rt",
            accuracy="accuracy",
            congruency="congruency",
        )