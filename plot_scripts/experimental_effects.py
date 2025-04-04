import sys
sys.path.append("../../BayesFlow")
sys.path.append("../dmc")

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


network_name = "simons_crazy_net_test_metrics"


# inintialize simulator
# simulator = DMC(
#     prior_means=np.array([16., 111., 0.5, 322., 75.]), 
#     prior_sds=np.array([10., 47., 0.13, 40., 23.]),
#     tmax=1500)

file_path = 'simulators/simulator_' + network_name + '.pickle'

with open(file_path, 'wb') as file:
    simulator = pickle.load(file)

# Load checkpoints
approximator = keras.saving.load_model("checkpoints/" + network_name + ".keras")


narrow_data = pd.read_csv('data/model_data/experiment_data_narrow.csv')
wide_data = pd.read_csv('data/model_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


samples_narrow=dmc_helpers.fit_empirical_data(narrow_data, approximator)

samples_narrow["spacing"]="narrow"

samples_wide=dmc_helpers.fit_empirical_data(wide_data, approximator)

samples_wide["spacing"]="wide"

samples_complete=pd.concat((samples_wide, samples_narrow))


parts=samples_complete["participant"].unique()


param_names = ["A", "tau", "mu_c", "mu_r", "b"]

network_plot_folder = "plots/experimental_effects/" + network_name

if not os.path.existis(network_plot_folder):
    os.makedirs(network_plot_folder)

for i, part in enumerate(parts):
    
    fig, axes = plt.subplots(1,5, figsize=(10,3))
    
    axes = axes.flatten()

    for p, ax in zip(param_names, axes):
        
        part_data = samples_complete[samples_complete["participant"]==part]
        
        sns.boxplot(part_data, ax=ax, x="spacing", y=p)
        ax.set_ylabel("")

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        label = suff + p + "$"

        ax.set_title(label)
            
    fig.suptitle(str(part))    
    fig.tight_layout()
    fig.savefig(network_plot_folder + "experimental_effects_" + network_name + str(part) + ".png")
