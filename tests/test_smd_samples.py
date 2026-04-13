
from pandas.testing import assert_frame_equal
import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from dmc_helpers import smd_samples

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from dmc_helpers import smd_samples


def test_smd_samples_computes_expected_cohens_d(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace=False: np.array([0, 1, 2]))

    samples1 = pd.DataFrame(
        {
            "id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "A":  [3.0, 4.0, 6.0, 5.0, 7.0, 10.0],
        }
    )

    samples2 = pd.DataFrame(
        {
            "id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "A":  [1.0, 2.0, 3.0, 2.0, 3.5, 4.0],
        }
    )

    data_d, fig = smd_samples(
        samples1=samples1,
        samples2=samples2,
        param_names=["A"],
        num_samples=3,
        id_name="id",
    )

    assert list(data_d.columns) == ["A"]
    assert data_d.shape == (3, 1)
    assert np.isfinite(data_d["A"]).all()
    assert data_d["A"].nunique() > 1

    plt.close(fig)


def test_smd_samples_warns_when_participants_do_not_match(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace=False: np.array([0, 1]))

    samples1 = pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2"],
            "A": [3.0, 4.0, 5.0, 6.0],
        }
    )

    samples2 = pd.DataFrame(
        {
            "id": ["s1", "s1", "s2"],
            "A": [1.0, 2.0, 3.0],
        }
    )

    with pytest.warns(UserWarning):
        data_d, fig = smd_samples(
            samples1=samples1,
            samples2=samples2,
            param_names=["A"],
            num_samples=2,
            id_name="id",
        )

    assert list(data_d.columns) == ["A"]
    assert data_d.shape == (2, 1)

    plt.close(fig)

def test_smd_samples_warns_with_specific_participant_mismatch_message(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace=False: np.array([0, 1]))

    samples1 = pd.DataFrame(
        {
            "id": ["s1", "s1", "s2", "s2"],
            "A": [3.0, 4.0, 5.0, 6.0],
        }
    )

    samples2 = pd.DataFrame(
        {
            "id": ["s1", "s1", "s2"],
            "A": [1.0, 2.0, 3.0],
        }
    )

    with pytest.warns(UserWarning) as record:
        data_d, fig = smd_samples(
            samples1=samples1,
            samples2=samples2,
            param_names=["A"],
            num_samples=2,
            id_name="id",
        )

    messages = [str(w.message) for w in record]
    assert any(
        "Participants in sub sample 2 and sample id 1 are not identical to all participants!"
        in msg
        for msg in messages
    )

    plt.close(fig)