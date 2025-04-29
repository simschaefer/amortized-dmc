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

#network_name_fixed = 'dmc_optimized_updated_priors_sdr_fixed_200_797801'
#network_name_estimated = 'dmc_optimized_updated_priors_sdr_estimated_200_797802'
#plot_name = 'prior_predictive_check_updated'

test = True
train = True



import bayesflow as bf
from dmc import DMC, dmc_helpers

import pandas as pd


narrow_data = pd.read_csv('../data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv('../data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])
empirical_data["condition_label"] = empirical_data["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"})


train_idx = np.array([6361, 5281, 6214, 1108, 1538,  833, 4222,  275, 8755, 5281, 4222,
        985, 1601, 8788,  845, 4222, 8785, 3286, 1761, 3625, 3625, 1583,
    6844, 7768, 3754,  833, 1821, 7828,  275, 3754, 1657, 5815, 1583])

if not test:

    empirical_data = empirical_data['train_test'] = empirical_data['participant'].isin(train_idx)

if not train:
    
    empirical_data = empirical_data[~empirical_data['participant'].isin(train_idx)]



min_acc = empirical_data.groupby(['participant']).mean('accuracy')['accuracy'].min()

plt.hist(empirical_data.groupby(['participant']).mean('accuracy')['accuracy'])

empirical_accuracies = empirical_data.groupby('participant').mean('accuracy')

num_obs_empirical = int(round(empirical_data.groupby('participant').count().mean())['rt'])

# load model_specs

model_specs_path_fixed = parent_dir + '/model_specs/model_specs_' + network_name_fixed + '.pickle'

with open(model_specs_path_fixed, 'rb') as file:
    model_specs_fixed = pickle.load(file)

model_specs_path_estimated = parent_dir + '/model_specs/model_specs_' + network_name_estimated + '.pickle'

with open(model_specs_path_estimated, 'rb') as file:
    model_specs_estimated = pickle.load(file)


#model_specs['simulation_settings']['param_names'] = ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r')

simulator_fixed, adapter_fixed, inference_net_fixed, summary_net_fixed, workflow_fixed = dmc_helpers.load_model_specs(model_specs_fixed, network_name_fixed)

simulator_estimated, adapter_estimated, inference_net_estimated, summary_net_estimated, workflow_estimated = dmc_helpers.load_model_specs(model_specs_estimated, network_name_estimated)

## Load Approximator

models = [simulator_estimated, simulator_fixed]

con_color = '#132a70'
inc_color = "maroon"

hue_order = ["Congruent", "Incongruent"]
palette = {"Congruent": con_color, "Incongruent": inc_color}


n_sims = 200

fig, axes = plt.subplots(2,2)

model_titles = ['$sd_r \\ estimated$', '$sd_r = 0$']

alpha = 0.03

# rt sdr vs. empirical - rt sdr fixed vs. empirical
# acc sdr vs. empirical - acc sdr fixed vs. empirical

# Share x-axis within each row
for row_axes in axes:
    # Share all x-axes in this row with the first one in the row
    for ax in row_axes[1:]:
        ax.sharex(row_axes[0])
        ax.sharey(row_axes[0])


for ax in axes[0]:
    ax.set_ylim(0, 4)
    ax.set_xlim(0.15, 1.2)

for ax in axes[1]:
    ax.set_ylim(0, 17)
    ax.set_xlim(min_acc, 1.02)

for j, model in enumerate(models):

    lst_acc = list()

    for i in range(0, n_sims):

        sim_iteration = model.experiment(**model.prior(), num_obs=350)

        sim_iteration = {k: sim_iteration[k] for k in ['rt', 'accuracy', 'conditions']}
        sim_id_df = pd.DataFrame(sim_iteration)

        sim_id_df['n_sim'] = i

        sim_id_df["condition_label"] = sim_id_df["conditions"].map({0.0: "Congruent", 1.0: "Incongruent"})

        accuracy = sim_id_df['accuracy'][sim_id_df['accuracy'] != -1.0]

        lst_acc.append(sim_id_df)

        sim_id_df = sim_id_df[sim_id_df['rt'] != -1.0]
        sim_id_df = sim_id_df[sim_id_df['accuracy'] != -1.0]

        sns.kdeplot(sim_id_df, x='rt', ax=axes[0,j], alpha=alpha, hue='condition_label', hue_order=hue_order, palette=palette)
        axes[0,j].set_title(model_titles[j])

    acc_df = pd.concat(lst_acc)
    acc_df = acc_df.groupby(['n_sim', 'condition_label']).mean('accuracy')
    
    sns.kdeplot(acc_df, x='accuracy', ax =axes[1, j], hue='condition_label', hue_order=hue_order, palette=palette, alpha=0.5, legend=False)
    emp_acc = empirical_data.groupby(['participant', 'congruency_num']).mean('accuracy')
    emp_acc.reset_index(inplace=True)

    emp_acc['condition_label'] = emp_acc["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"})

    sns.kdeplot(emp_acc, x='accuracy', hue='condition_label', ax=axes[1, j], hue_order=hue_order, palette=palette)

    sns.kdeplot(empirical_data.reset_index(), x='rt', ax=axes[0, j], hue='condition_label', hue_order=hue_order, palette=palette, alpha=1)

    axes[1, j].set_xlabel('Accuracy')
    axes[0, j].set_xlabel('RT')



axes[0,0].legend_.remove()
axes[1,0].legend_.remove()


plt.legend()

axes[1,1].legend_.remove()

#fig.suptitle('Prior Predictive Check')
fig.tight_layout()
if axes[0,1].legend_ is not None:
    axes[0,1].legend_.set_title("")

suffix = ''

if train:
    suffix = '_train'

if test:
    suffix = suffix + '_test'


fig.savefig(parent_dir + '/plots/prior_predictive_check/' + plot_name + suffix + '.png', dpi=600)
