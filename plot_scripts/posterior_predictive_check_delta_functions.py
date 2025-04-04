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


samples_narrow=dmc_helpers.fit_empirical_data(narrow_data, approximator)

samples_narrow["spacing"]="narrow"

samples_wide=dmc_helpers.fit_empirical_data(wide_data, approximator)

samples_wide["spacing"]="wide"

samples_complete=pd.concat((samples_wide, samples_narrow))


parts=samples_complete["participant"].unique()


fig, axes = plt.subplots(6, 10, figsize=(20,12))

axes = axes.flatten()

for part, ax in zip(parts, axes):
    
    # filter sample data for given participant and narrow spacing
    part_data_samples = samples_complete[samples_complete["participant"]==part]

    part_data_samples = part_data_samples[part_data_samples["spacing"] == "narrow"]

    # filter empirical data for given participant and narrow spacing
    part_data = empirical_data[empirical_data["participant"] == part]

    part_data = part_data[part_data["spacing_num"] == 1]

    part_data["condition_label"] = part_data["congruency_num"].map({0.0: "congruent", 1.0: "incongruent"})

    # resimulate data
    data_resimulated = dmc_helpers.resim_data(part_data_samples, num_obs = part_data.shape[0])

    # exclude non-convergents
    data_resimulated = data_resimulated[data_resimulated["rt"] != -1]

    # recode congruency
    data_resimulated["condition_label"] = data_resimulated["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

    quantiles = np.arange(0,1, 0.1)

    quantile_data_wide_resim = dmc_helpers.delta_functions(data_resimulated, quantiles = np.arange(0, 1, 0.1))

    quantile_data_wide_empirical = dmc_helpers.delta_functions(part_data, quantiles = np.arange(0, 1, 0.1))

    ax.plot(quantile_data_wide_resim["mean_qu"] ,quantile_data_wide_resim["delta"] ,"o")
    ax.plot(quantile_data_wide_empirical["mean_qu"] ,quantile_data_wide_empirical["delta"] )

fig.tight_layout()
    