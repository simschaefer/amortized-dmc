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


parent_dir = '/home/administrator/Documents/BF-LIGHT'

network_name = 'dmc_optimized_updated_priors'


rel_plot_folder = parent_dir + "/plots/plots_reliability/" + network_name

if not os.path.exists(rel_plot_folder):
    os.makedirs(rel_plot_folder)


model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")
approximator.compile()


narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])

# Split Data
narrow_data_even=narrow_data.iloc[::2]

narrow_data_odd=narrow_data.iloc[1::2]


# Sample Individual Posterior Samples
post_samples_narrow_even = dmc_helpers.fit_empirical_data(narrow_data_even, approximator)

post_samples_narrow_odd = dmc_helpers.fit_empirical_data(narrow_data_odd, approximator)

# Index Samples per participant ODD
df_sorted_odd = post_samples_narrow_odd.sort_values(by=['participant']).reset_index(drop=True)
df_sorted_odd['row_within_part'] = df_sorted_odd.groupby('participant').cumcount()

# Index Samples per participant EVEN
df_sorted_even = post_samples_narrow_even.sort_values(by=['participant']).reset_index(drop=True)
df_sorted_even['row_within_part'] = df_sorted_even.groupby('participant').cumcount()

# Name Params (ODD/EVEN)
df_sorted_odd.columns = df_sorted_odd.columns + '_odd'
df_sorted_even.columns = df_sorted_even.columns + '_even'

# Combine two data sets
post_samples_combined = pd.concat((df_sorted_even, df_sorted_odd), axis=1)

# Create empty array for results (Correlations)
corr_arr = np.zeros((1000, len(model_specs['param_names'])))

# Compute Correlations for each sample across all participants
for idx in np.arange(0,1000):

    for i, param in enumerate(model_specs['param_names']):
        
        post_samples_idx = post_samples_combined[post_samples_combined['row_within_part_even'] == idx] 
        corr_arr[idx,i] = np.corrcoef(post_samples_idx[param + '_odd'], post_samples_idx[param + '_even'])[0, 1]

corr_data = pd.DataFrame(corr_arr, columns=model_specs['param_names'])


#%%
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for p, ax in zip(model_specs['param_names'], axes):
    ax.plot(post_samples_narrow_odd[p], post_samples_narrow_even[p], "o")
    ax.plot(
        np.linspace(min(post_samples_narrow_odd[p]), max(post_samples_narrow_odd[p]), 100),
        np.linspace(min(post_samples_narrow_odd[p]), max(post_samples_narrow_odd[p]), 100),
        color='black'
    )
    ax.set_title(dmc_helpers.param_labels([p]))

fig.savefig(rel_plot_folder + '/plot_reliability_' + network_name + '.png')


# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for p, ax in zip(model_specs['param_names'], axes):
    ax.hist(corr_data[p])
    ax.set_title(dmc_helpers.param_labels([p]))

fig.tight_layout()

fig.savefig(rel_plot_folder + '/plot_reliability_hist_' + network_name + '.png')


# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for p, ax in zip(model_specs['param_names'], axes):
    sns.kdeplot(data=corr_data, x=p, ax=ax)
    ax.set_title(dmc_helpers.param_labels([p]))

fig.tight_layout() 

fig.savefig(rel_plot_folder + '/plot_reliability_kde_' + network_name + '.png')
# %%
