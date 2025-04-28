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
import seaborn as sns

import matplotlib.pyplot as plt


parent_dir = '/home/administrator/Documents/bf_dmc'


network_name_fixed = 'dmc_optimized_winsim_priors_sdr_fixed_200_795737'
network_name_estimated = 'dmc_optimized_winsim_priors_sdr_estimated_200_795738'
plot_name = 'prior_predictive_check_winsim'

import bayesflow as bf
from dmc import DMC, dmc_helpers

import pandas as pd


narrow_data = pd.read_csv('../data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv('../data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])

empirical_accuracies = empirical_data.groupby('participant').mean('accuracy')

num_obs_empirical = int(round(empirical_data.groupby('participant').count().mean())['rt'])

# load model_specs

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'


with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

#model_specs['simulation_settings']['param_names'] = ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r')


simulator_fixed, adapter_fixed, inference_net_fixed, summary_net_fixed, workflow_fixed = dmc_helpers.load_model_specs(model_specs, network_name_fixed)

simulator_estimated, adapter_estimated, inference_net_estimated, summary_net_estimated, workflow_estimated = dmc_helpers.load_model_specs(model_specs, network_name_estimated)

## Load Approximator

models = [simulator_estimated, simulator_fixed]


n_sims = 200

fig, axes = plt.subplots(2,2)

model_titles = ['$sd_r \\ estimated$', '$sd_r = 0$']

alpha = 0.05

# rt sdr vs. empirical - rt sdr fixed vs. empirical
# acc sdr vs. empirical - acc sdr fixed vs. empirical

# Share x-axis within each row
for row_axes in axes:
    # Share all x-axes in this row with the first one in the row
    for ax in row_axes[1:]:
        ax.sharex(row_axes[0])
        ax.sharey(row_axes[0])


for ax in axes[0]:
    ax.set_ylim(0, 12)

for ax in axes[1]:
    ax.set_ylim(0, 18)

for j, model in enumerate(models):

    lst_acc_means = list()

    for i in range(0, n_sims):
        sim_iteration = model.experiment(**model.prior(), num_obs=350)
        sim_iteration['accuracy'] = sim_iteration['accuracy'][sim_iteration['accuracy'] != -1.0]
        lst_acc_means.append(np.mean(sim_iteration['accuracy']))

        if i == n_sims-1:
            label = 'Simulated'
        else:
            label = ''

        sim_iteration['rt'] = sim_iteration['rt'][sim_iteration['rt'] != -1.0]
        sns.kdeplot(sim_iteration['rt'], ax=axes[0,j], alpha=alpha, color = 'steelblue', label=label)
        axes[0,j].set_title(model_titles[j])

        sns.kdeplot(np.array(lst_acc_means), ax =axes[1, j], alpha=alpha, color = 'steelblue', label=label)
    sns.kdeplot(empirical_accuracies['accuracy'], ax =  axes[1, j], label = 'Empirical', color='maroon')
    sns.kdeplot(empirical_accuracies['rt'], ax =  axes[0, j], label = 'Empirical', color='maroon')

plt.legend()
#fig.suptitle('Prior Predictive Check')
fig.tight_layout()


fig.savefig(parent_dir + '/plots/prior_predictive_check/' + plot_name + '.png')

