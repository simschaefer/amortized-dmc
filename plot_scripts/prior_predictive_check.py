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


import bayesflow as bf
from dmc import DMC

import pandas as pd


narrow_data = pd.read_csv('../data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv('../data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])

empirical_accuracies = empirical_data.groupby('participant').mean('accuracy')

num_obs_empirical = int(round(empirical_data.groupby('participant').count().mean())['rt'])




# 322.

settings_sdr = {'prior_means': np.array([16., 111., 0.5, 322., 75., 40.91]),
                'prior_sds': np.array([10., 47., 0.13, 40., 23., 11.74]),
                'sdr_fixed': None,
                "tmax": 1500,
                "contamination_probability": None,
                'param_names': ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'),
                "fixed_num_obs": num_obs_empirical}

settings = {"prior_means": np.array([16., 111., 0.5, 322., 75.]),
                                       "prior_sds": np.array([10., 47., 0.13, 40., 23.]),
                                       'sdr_fixed': 0,
                                       "tmax": 1500,
                                       "contamination_probability": None,
                                       'param_names': ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'),
                                       "fixed_num_obs": num_obs_empirical}

dmc_sdr = DMC(**settings_sdr)

dmc = DMC(**settings)

models = [dmc_sdr, dmc]



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
        sim_iteration = model.experiment(**model.prior(), num_obs=model.fixed_num_obs)
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
fig.suptitle('Prior Predictive Check Updated Priors')
fig.tight_layout()


fig.savefig(parent_dir + '/plots/prior_predictive_check/prior_predictive_check_updated.png')

settings_sdr_winsim = {"prior_means": np.array([70.8, 114.71, 0.71, 332.34, 98.36, 43.36]),
                       "prior_sds": np.array([19.42, 40.08, 0.14, 52.07, 30.05, 9.19]),
                       "tmax": 1500,
                       'sdr_fixed': None,
                       "contamination_probability": None,
                       'param_names': ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'),
                       "fixed_num_obs": num_obs_empirical}

settings_winsim = {"prior_means": np.array([70.8, 114.71, 0.71, 332.34, 98.36]),
                       "prior_sds": np.array([19.42, 40.08, 0.14, 52.07, 30.05]),
                   'sdr_fixed': 0,
                   "tmax": 1500,
                   "contamination_probability": None,
                   'param_names': ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r'),
                   "fixed_num_obs": num_obs_empirical}

dmc_sdr_winsim = DMC(**settings_sdr_winsim)

dmc_winsim = DMC(**settings_winsim)

models = [dmc_sdr_winsim, dmc_winsim]


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
        sim_iteration = model.experiment(**model.prior(), num_obs=model.fixed_num_obs)
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
fig.suptitle('Prior Predictive Check WinSim Priors')
fig.tight_layout()


fig.savefig(parent_dir + '/plots/prior_predictive_check/prior_predictive_check_winsim.png')