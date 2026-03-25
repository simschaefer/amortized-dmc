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
import bayesflow as bf
from dmc import DMC, dmc_helpers
import pandas as pd
from matplotlib.lines import Line2D

arguments = sys.argv[1:]

if 'executed_from_bash' in arguments:
    network_name_fixed = str(arguments[0])
    network_name_estimated = str(arguments[3])
    host = str(arguments[1])
    fixed_n_obs = int(arguments[2])
    num_resims = int(arguments[6])

else:

    network_names = [
        'updated_priors_sdr_fixed',
        'updated_priors_sdr_estimated',
        'initial_priors_sdr_fixed',
        'initial_priors_sdr_estimated',
    ]

    fixed_n_obs = 300
    num_resims = 100
    host = 'local'


if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

plot_name = 'prior_predictive_check'

# load empirical data
narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')[['participant', 'rt', 'accuracy', 'congruency_num']]
wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')[['participant', 'rt', 'accuracy', 'congruency_num']]

# these ids were used for prior updating (randomly sampled):
train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214])

# exclude training ids
narrow_data = narrow_data[~narrow_data['participant'].isin(train_idx)]
wide_data = wide_data[~wide_data['participant'].isin(train_idx)]

# recode congruency
narrow_data['congruency_num'] = ['congruent' if x == 0 else 'incongruent' for x in narrow_data['congruency_num']]
wide_data['congruency_num'] = ['congruent' if x == 0 else 'incongruent' for x in wide_data['congruency_num']]
    
# compute stats for empirical data sets
caf_data_emp, cdf_data_emp,  delta_data_emp = dmc_helpers.compute_stats(narrow_data, id_name='participant', congruency='congruency_num', n_rt_bins=5)

# load model_specs

simulators = {}

for i in range(0, len(network_names)):

    model_specs_path_updated_fixed = parent_dir + '/model_specs/model_specs_' + network_names[i] + '.pickle'

    with open(model_specs_path_updated_fixed, 'rb') as file:
        model_specs = pickle.load(file)

    model_specs['simulation_settings']['fixed_num_obs'] = 350

    simulators[network_names[i]] = DMC(**model_specs['simulation_settings'])


id_name='participant'
congruency_name='congruency_num'
quantiles = np.arange(0.1, 1, 0.1)
legend=True
num_samples = 1000

for priors in ['initial', 'updated']:

    if priors == 'updated':
        legend=False
        ylim=(0,0.08)
    else:
        ylim=(0,0.18)

    sim_data_estimated = simulators[priors + '_priors_sdr_estimated'].sample(num_samples)

    sim_data_fixed = simulators[priors + '_priors_sdr_fixed'].sample(num_samples)

    sim_data_estimated = dmc_helpers.format_sim_data(sim_data_estimated, congruency_coding=0)

    sim_data_fixed = dmc_helpers.format_sim_data(sim_data_fixed, congruency_coding=0)

    caf_data_fixed, cdf_data_fixed,  delta_data_fixed = dmc_helpers.compute_stats(sim_data_fixed, id_name='id', congruency="congruency", n_rt_bins=5)

    caf_data_estimated, cdf_data_estimated,  delta_data_estimated = dmc_helpers.compute_stats(sim_data_estimated, id_name='id', congruency="congruency", n_rt_bins=5)

    linewidth = 1.5
    fontsize=14
    fontsize_axes=14
    fontsize_ticklabels=12
    fontsize_legend=12

    fig, axes = dmc_helpers.plot_fit(delta_data=delta_data_estimated,
                        delta_data_emp=delta_data_emp,
                        caf_data=caf_data_estimated,
                        caf_data_emp=caf_data_emp,
                        cdf_data=cdf_data_estimated,
                        cdf_data_emp=cdf_data_emp,
                        fontsize=fontsize,
                        congruency='congruency',
                        congruency_emp='congruency_num',
                        legend=legend,
                        delta_ylim=ylim,
                        fontsize_axes=fontsize_axes,
                        fontsize_ticklabels=fontsize_ticklabels,
                        fontsize_legend=fontsize_legend,
                        linewidth=linewidth)

    fig, axes = dmc_helpers.plot_fit(delta_data=delta_data_fixed,
                        delta_data_emp=delta_data_emp,
                        caf_data=caf_data_fixed,
                        caf_data_emp=caf_data_emp,
                        cdf_data=cdf_data_fixed,
                        cdf_data_emp=cdf_data_emp,
                        legend=False,
                        fontsize=fontsize,
                        congruency='congruency',
                        congruency_emp='congruency_num',
                        new_plot=False,
                        fig=fig, 
                        axes=axes,
                        delta_ylim=ylim,
                        fontsize_axes=fontsize_axes,
                        fontsize_ticklabels=fontsize_ticklabels,
                        fontsize_legend=fontsize_legend,
                        #palette_model={"congruent": '#FFBD00', "incongruent": '#9E0059'},
                        delta_linestyle_model=':',
                        caf_linestyle_model=':',
                        cdf_linestyle_model=':',
                        linewidth=linewidth)
    
    if priors == 'updated':
        custom_legend = [
            Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="$\\text{sd}_r$ estimated"),
            Line2D([0], [0], color="black", lw=1.5, linestyle=":", label="$\\text{sd}_r$ fixed"),
            Line2D([0], [0], color="black", lw=1.5, linestyle="--", marker="o", label="Empirical"),
        ]

        axes[0].legend(
            handles=custom_legend,
            loc="lower right",
            fontsize=fontsize_legend,
            frameon=False
        )
        
    #fig.savefig(parent_dir + '/plots/prior_predictive_check/prior_predictive_'+priors + '.png', dpi=600)

