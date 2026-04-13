
from pandas.testing import assert_frame_equal
import os
import sys

scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

dmc_module_dir = parent_dir + '/dmc'

sys.path.append(dmc_module_dir)

import pandas as pd
import pytest

from dmc_helpers import check_vars


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"rt": "rt"}, r"Variable 'rt' does not exist in data"),
        ({"accuracy": "accuracy"}, r"Variable 'accuracy' does not exist in data"),
        ({"congruency": "congruency"}, r"Variable 'congruency' does not exist in data"),
        ({"id_name": "id"}, r"Variable 'id' does not exist in data"),
    ],
)
def test_check_vars_missing_columns(kwargs, expected_message):
    data = pd.DataFrame()

    with pytest.raises(ValueError, match=expected_message):
        check_vars(data, **kwargs)