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


param_names =  ["A", "tau", "mu_c", "mu_r", "b"]

network_name = "sims_crazy_net_test_metrics.keras"


# inintialize simulator
simulator = DMC(
    prior_means=np.array([16., 111., 0.5, 322., 75.]), 
    prior_sds=np.array([10., 47., 0.13, 40., 23.]),
    tmax=1500
)

# Load checkpoints
approximator = keras.saving.load_model("../checkpoints/" + network_name)


narrow_data = pd.read_csv('../data/model_data/experiment_data_narrow.csv')
wide_data = pd.read_csv('../data/model_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


narrow_data_even=narrow_data.iloc[::2]

narrow_data_odd=narrow_data.iloc[1::2]

post_samples_narrow_even = dmc_helpers.fit_empirical_data(narrow_data_even, approximator)

post_samples_narrow_odd = dmc_helpers.fit_empirical_data(narrow_data_odd, approximator)


post_median_odd = post_samples_narrow_odd.groupby("participant").median()

post_median_even = post_samples_narrow_even.groupby("participant").median()

fig, axes = plt.subplots(1,5, figsize=(15,3))
for p, ax in zip(param_names, axes):
    ax.plot(post_median_odd[p], post_median_even[p], "o")
    ax.plot([])
    
    ax.set_title(p)