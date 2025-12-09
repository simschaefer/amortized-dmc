import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle

import keras

import bayesflow as bf

from dmc import DMC, dmc_helpers

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

network_names = [
    'updated_priors_sdr_fixed',
    'updated_priors_sdr_estimated',
    'initial_priors_sdr_fixed',
    'initial_priors_sdr_estimated',
]

host = 'local'

parent_dir = os.path.dirname(os.getcwd())

included_parts = np.array([
    275, 808, 810, 833, 837, 845, 916, 985, 1108, 1430, 1507, 1538, 1582, 1583, 1597, 1601,
    1614, 1638, 1657, 1663, 1761, 1768, 1813, 1821, 1824, 3286, 3292, 3487, 3580, 3625, 3754, 3910,
    3988, 4222, 4264, 5281, 5332, 5575, 5731, 5761, 5803, 5815, 6055, 6109, 6214, 6253, 6262, 6361,
    6427, 6583, 6634, 6844, 7624, 7756, 7768, 7807, 7813, 7828, 7840, 7924, 7939, 8026, 8308, 8311,
    8446, 8521, 8704, 8755, 8785, 8788, 161753, 337788
])

train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 6585, 3487, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214, 8521])


# Load Empirical Data

narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')

narrow_data = narrow_data[narrow_data['participant'].isin(included_parts)]

wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

wide_data = wide_data[wide_data['participant'].isin(included_parts)]


for network_name in network_names:

    approximator = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name + '.keras')

    samples_narrow = dmc_helpers.fit_empirical_data(narrow_data, approximator)

    samples_narrow["spacing"]="narrow"

    samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator)

    samples_wide["spacing"]="wide"

    samples_complete = pd.concat((samples_wide, samples_narrow))

    data_path = parent_dir + '/data_complete/empirical_estimates/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    samples_complete.to_csv(data_path + network_name + '.csv')

    print(network_name, 'estimation completed')





