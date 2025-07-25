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


#arguments = sys.argv[1:]
#network_name_fixed = str(arguments[0])
#host = str(arguments[1])
#num_resims = int(arguments[6])
#network_name_estimated = str(arguments[3])

network_name_fixed = 'initial_priors_sdr_fixed'
network_name_estimated = 'initial_priors_sdr_estimated'
num_resims = 100
host = 'local'


if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

plot_name = 'prior_predictive_check'


test = True
train = False


import bayesflow as bf
from dmc import DMC

import pandas as pd


narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])
empirical_data["condition_label"] = empirical_data["congruency_num"].map({0.0: "Congruent", 1.0: "Incongruent"})

train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214, 8521])


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

# specify simulators:

simulator_fixed = DMC(**model_specs_fixed['simulation_settings'])

simulator_estimated  = DMC(**model_specs_estimated['simulation_settings'])

## Load Approximators

models = [simulator_estimated, simulator_fixed]

# Define plot colors

con_color = '#10225e'
inc_color = '#FF6361'

hue_order = ["Congruent", "Incongruent"]
palette = {"Congruent": con_color, "Incongruent": inc_color}


n_sims = 200

fig, axes = plt.subplots(2,2)

model_titles = ['$sd_r \\ estimated$', '$sd_r \\ fixed$']

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
    ax.set_xlim(0.8, 1.02)

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


fig.savefig(parent_dir + '/plots/prior_predictive_check/' + plot_name + suffix + network_name_fixed + '_' + network_name_fixed +'.png', dpi=600)
