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

sys.path.append('/Users/simonschaefer/Documents/bf_dmc')
from dmc import dmc_helpers
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
parent_dir = '/home/administrator/Documents/bf_dmc'

#network_name = 'dmc_optimized_winsim_priors_sdr_estimated_200_795738'
arguments = sys.argv[1:]
network_name = str(arguments[0])

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)


simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

param_names = model_specs['simulation_settings']['param_names']

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")


narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow_reliability.csv')
wide_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_wide_reliability.csv')

empirical_data = pd.concat([narrow_data, wide_data])

parts = empirical_data['participant'].unique()

# Split narrow Data
narrow_data_even=narrow_data[narrow_data['n_trial_experiment'] % 2 == 0]

narrow_data_odd=narrow_data[narrow_data['n_trial_experiment'] % 2 != 0]

# trials per participant
sns.histplot(narrow_data_even.groupby('participant').count(), x='accuracy', label = 'even narrow')

# trials per participant
sns.histplot(narrow_data_odd.groupby('participant').count(), x='accuracy', label = 'odd narrow')

# split wide data
wide_data_even=wide_data[wide_data['n_trial_experiment'] % 2 == 0]

wide_data_odd=wide_data[wide_data['n_trial_experiment'] % 2 != 0]

# trials per participant
sns.histplot(wide_data_even.groupby('participant').count(), x='accuracy', label = 'even wide')

# trials per participant
sns.histplot(wide_data_odd.groupby('participant').count(), x='accuracy', label = 'odd wide')

plt.legend()


# Sample Individual Posterior Samples NARROW
post_samples_narrow_even = dmc_helpers.fit_empirical_data(narrow_data_even, approximator)

post_samples_narrow_odd = dmc_helpers.fit_empirical_data(narrow_data_odd, approximator)

# Sample Individual Posterior Samples WIDE
post_samples_wide_even = dmc_helpers.fit_empirical_data(wide_data_even, approximator)

post_samples_wide_odd = dmc_helpers.fit_empirical_data(wide_data_odd, approximator)

# inspect samples


fig, axes = plt.subplots(1, len(param_names), figsize=(15,3))

axes.flatten()
for p in parts:

    for i, param in enumerate(param_names):

        sns.kdeplot(post_samples_narrow_even.reset_index(), hue='participant', x=param, ax=axes[i], legend=False)
    
fig.suptitle('Posterior Samples Narrow Even')
fig.tight_layout()



fig, axes = plt.subplots(1, len(param_names), figsize=(15,3))

axes.flatten()
for p in parts:

    for i, param in enumerate(param_names):

        sns.kdeplot(post_samples_narrow_odd.reset_index(), hue='participant', x=param, ax=axes[i], legend=False)
    
fig.suptitle('Posterior Samples Narrow Odd')
fig.tight_layout()



fig, axes = plt.subplots(1, len(param_names), figsize=(15,3))

axes.flatten()
for p in parts:

    for i, param in enumerate(param_names):

        sns.kdeplot(post_samples_wide_even.reset_index(), hue='participant', x=param, ax=axes[i], legend=False)
    
fig.suptitle('Posterior Samples Wide Even')
fig.tight_layout()



fig, axes = plt.subplots(1, len(param_names), figsize=(15,3))

axes.flatten()
for p in parts:

    for i, param in enumerate(param_names):

        sns.kdeplot(post_samples_wide_odd.reset_index(), hue='participant', x=param, ax=axes[i], legend=False)
    
fig.suptitle('Posterior Samples Wide Odd')
fig.tight_layout()






# Index Samples per participant ODD
post_samples_narrow_even_sorted = post_samples_narrow_even.sort_values(by=['participant']).reset_index(drop=True)
post_samples_narrow_even_sorted['row_within_part'] = post_samples_narrow_even_sorted.groupby('participant').cumcount()

# Index Samples per participant ODD
post_samples_narrow_odd_sorted = post_samples_narrow_odd.sort_values(by=['participant']).reset_index(drop=True)
post_samples_narrow_odd_sorted['row_within_part'] = post_samples_narrow_odd_sorted.groupby('participant').cumcount()



# Index Samples per participant ODD
post_samples_wide_even_sorted = post_samples_wide_even.sort_values(by=['participant']).reset_index(drop=True)
post_samples_wide_even_sorted['row_within_part'] = post_samples_wide_even_sorted.groupby('participant').cumcount()

# Index Samples per participant ODD
post_samples_wide_odd_sorted = post_samples_wide_odd.sort_values(by=['participant']).reset_index(drop=True)
post_samples_wide_odd_sorted['row_within_part'] = post_samples_wide_odd_sorted.groupby('participant').cumcount()



# Name Params (ODD/EVEN)
post_samples_narrow_even_sorted.columns = post_samples_narrow_even_sorted.columns + '_even'
post_samples_narrow_odd_sorted.columns = post_samples_narrow_odd_sorted.columns + '_odd'

post_samples_wide_even_sorted.columns = post_samples_wide_even_sorted.columns + '_even'
post_samples_wide_odd_sorted.columns = post_samples_wide_odd_sorted.columns + '_odd'

# Combine two data sets
post_samples_combined_narrow = pd.concat((post_samples_narrow_even_sorted, post_samples_narrow_odd_sorted), axis=1)


post_samples_combined_wide = pd.concat((post_samples_wide_even_sorted, post_samples_wide_odd_sorted), axis=1)

# Create empty array for results (Correlations)
corr_arr = np.zeros((1000, len(param_names)))


# Compute Correlations for each sample across all participants
for idx in np.arange(0,1000):

    for i, param in enumerate(param_names):
        
        post_samples_idx = post_samples_combined_narrow[post_samples_combined_narrow['row_within_part_even'] == idx].dropna()

        # exclude NaNs

        #post_samples_idx[~np.isnan(post_samples_idx)]
        corr_arr[idx,i] = post_samples_idx[param + '_odd'].corr(post_samples_idx[param + '_even'], method='pearson')
        #corr_arr[idx,i] = pearsonr(post_samples_idx[param + '_odd'], post_samples_idx[param + '_even'])[0]

corr_data_narrow = pd.DataFrame(corr_arr, columns=param_names)

post_means_odd_narrow = post_samples_narrow_odd.groupby('participant').mean()
post_means_even_narrow = post_samples_narrow_even.groupby('participant').mean()



# Compute Correlations for each sample across all participants
for idx in np.arange(0,1000):

    for i, param in enumerate(param_names):
        
        post_samples_idx = post_samples_combined_wide[post_samples_combined_wide['row_within_part_even'] == idx].dropna()

        # exclude NaNs

        #post_samples_idx[~np.isnan(post_samples_idx)]
        corr_arr[idx,i] = post_samples_idx[param + '_odd'].corr(post_samples_idx[param + '_even'], method='pearson')
        #corr_arr[idx,i] = pearsonr(post_samples_idx[param + '_odd'], post_samples_idx[param + '_even'])[0]

corr_data_wide = pd.DataFrame(corr_arr, columns=param_names)

post_means_odd_wide = post_samples_wide_odd.groupby('participant').mean()
post_means_even_wide = post_samples_wide_even.groupby('participant').mean()


network_plot_folder = "../plots/plots_reliability/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)


#%%
fig, axes = plt.subplots(1, len(param_names), figsize=(15, 3))

for p, ax in zip(param_names, axes):
    #ax.plot(post_samples_narrow_odd[p], post_samples_narrow_even[p], "o")

    ax.plot(post_means_odd_narrow[p], post_means_even_narrow[p], "o", color='#132a70', alpha=0.7, label = 'Narrow')

    ax.plot(post_means_odd_wide[p], post_means_even_wide[p], "o", color='maroon', alpha=0.7, label = 'Wide')

    ax.plot(
        np.linspace(min(post_means_odd_narrow[p]), max(post_means_odd_narrow[p]), 100),
        np.linspace(min(post_means_odd_narrow[p]), max(post_means_odd_narrow[p]), 100),
        color='black'
    )

    corr_narrow = post_means_odd_narrow[p].corr(post_means_even_narrow[p])

    ax.text(0.98, 0.09, 'r = ' +  str(round(corr_narrow,2)),
         transform=ax.transAxes,  # use axes coordinates
         fontsize=12,
         verticalalignment='bottom',  # align top of text at y=0.99
         horizontalalignment='right',
         color='#132a70')
    
    corr_wide = post_means_odd_wide[p].corr(post_means_even_wide[p])

    ax.text(0.987, 0.01, 'r = ' +  str(round(corr_wide,2)),
         transform=ax.transAxes,  # use axes coordinates
         fontsize=12,
         verticalalignment='bottom',  # align top of text at y=0.99
         horizontalalignment='right',
         color='maroon')
    
    ax.set_title(dmc_helpers.param_labels([p]))

    if p == 'sd_r':
        ax.legend()

fig.tight_layout

fig.savefig(network_plot_folder + '/plot_reliability_' + network_name + '_scatterplot.png')


# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for p, ax in zip(param_names, axes):
    ax.hist(corr_data[p])
    ax.set_title(dmc_helpers.param_labels([p]))

fig.tight_layout()

fig.savefig(network_plot_folder + '/plot_reliability_hist_' + network_name + '.png')


# %%
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for p, ax in zip(param_names, axes):
    sns.kdeplot(data=corr_data, x=p, ax=ax)
    ax.set_title(dmc_helpers.param_labels([p]))

fig.tight_layout() 

fig.savefig(network_plot_folder + '/plot_reliability_kde_' + network_name + '.png')
# %%
