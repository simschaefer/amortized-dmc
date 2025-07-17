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
import time

import bayesflow as bf

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# get arguments
arguments = sys.argv[1:]
network_name = str(arguments[0])
host = str(arguments[1])


if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

print(f'parent_dir: {parent_dir}', flush=True)

from dmc import DMC, param_labels, format_empirical_data, fit_empirical_data

# get model specifications
model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# set simulator
simulator = DMC(**model_specs['simulation_settings'])

## Load Approximator

param_names = model_specs['simulation_settings']['param_names']

approximator = keras.saving.load_model(parent_dir +"/bf_dmc/data/training_checkpoints/" + network_name + ".keras")


included_parts = np.array([
    275, 808, 810, 833, 837, 845, 916, 985, 1108, 1430, 1507, 1538, 1582, 1583, 1597, 1601,
    1614, 1638, 1657, 1663, 1761, 1768, 1813, 1821, 1824, 3286, 3292, 3487, 3580, 3625, 3754, 3910,
    3988, 4222, 4264, 5281, 5332, 5575, 5731, 5761, 5803, 5815, 6055, 6109, 6214, 6253, 6262, 6361,
    6427, 6583, 6634, 6844, 7624, 7756, 7768, 7807, 7813, 7828, 7840, 7924, 7939, 8026, 8308, 8311,
    8446, 8521, 8704, 8755, 8785, 8788, 161753, 337788
])


narrow_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/clean_experiment_data_narrow_complete.csv')
#narrow_data = narrow_data[narrow_data['participant'].isin(included_parts)]

wide_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_wide_reliability.csv')
#wide_data = wide_data[wide_data['participant'].isin(included_parts)]

empirical_data = pd.concat([narrow_data, wide_data])

data_count = empirical_data.groupby('participant').count()



empirical_data = empirical_data[empirical_data['participant'].isin(included_parts)]

parts = empirical_data['participant'].unique()

# Split narrow Data
narrow_data_even=narrow_data[narrow_data['n_trial_experiment'] % 2 == 0]

narrow_data_odd=narrow_data[narrow_data['n_trial_experiment'] % 2 != 0]

# split wide data
wide_data_even=wide_data[wide_data['n_trial_experiment'] % 2 == 0]

wide_data_odd=wide_data[wide_data['n_trial_experiment'] % 2 != 0]


# Sample Individual Posterior Samples NARROW
post_samples_narrow_even = fit_empirical_data(narrow_data_even, approximator)

post_samples_narrow_odd = fit_empirical_data(narrow_data_odd, approximator)

# Sample Individual Posterior Samples WIDE
post_samples_wide_even = fit_empirical_data(wide_data_even, approximator)

post_samples_wide_odd = fit_empirical_data(wide_data_odd, approximator)


# Index Samples per participant Narrow EVEN
post_samples_narrow_even_sorted = post_samples_narrow_even.sort_values(by=['participant']).reset_index(drop=True)
post_samples_narrow_even_sorted['row_within_part'] = post_samples_narrow_even_sorted.groupby('participant').cumcount()

# Index Samples per participant Narrow ODD
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

post_means_odd_narrow = post_samples_narrow_odd.groupby('participant').mean().reset_index()
post_means_even_narrow = post_samples_narrow_even.groupby('participant').mean().reset_index()



# Compute Correlations for each sample across all participants
for idx in np.arange(0,1000):

    for i, param in enumerate(param_names):
        
        post_samples_idx = post_samples_combined_wide[post_samples_combined_wide['row_within_part_even'] == idx].dropna()

        # exclude NaNs

        #post_samples_idx[~np.isnan(post_samples_idx)]
        corr_arr[idx,i] = post_samples_idx[param + '_odd'].corr(post_samples_idx[param + '_even'], method='pearson')
        #corr_arr[idx,i] = pearsonr(post_samples_idx[param + '_odd'], post_samples_idx[param + '_even'])[0]

corr_data_wide = pd.DataFrame(corr_arr, columns=param_names)

post_means_odd_wide = post_samples_wide_odd.groupby('participant').mean().reset_index()
post_means_even_wide = post_samples_wide_even.groupby('participant').mean().reset_index()


network_plot_folder = parent_dir + "/bf_dmc/plots/plots_reliability/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)


def spearman(r, k=2):

    return (k*r)/(1+r)

rel_table = pd.DataFrame(np.ones((2, len(param_names))))

rel_table.columns = param_names

if np.sum(post_means_even_wide['participant'] != post_means_odd_wide['participant']) != 0:
    print('Part ID does not correspond between odd and even trials (wide)!', flush = True)

elif np.sum(post_means_even_narrow['participant'] != post_means_odd_narrow['participant']) != 0:
    print('Part ID does not correspond between odd and even trials (wide)!', flush = True)

else:
    print('Part ID correspond between odd and even trials.', flush = True)


#%%
fig, axes = plt.subplots(1, len(param_names), figsize=(15, 3))

for p, ax in zip(param_names, axes):

    ax.plot(post_means_odd_narrow[p], post_means_even_narrow[p], "o", color='#132a70', alpha=0.7, label = 'Narrow')

    ax.plot(post_means_odd_wide[p], post_means_even_wide[p], "o", color='maroon', alpha=0.7, label = 'Wide')

    ax.plot(
        np.linspace(min(post_means_odd_narrow[p]), max(post_means_odd_narrow[p]), 100),
        np.linspace(min(post_means_odd_narrow[p]), max(post_means_odd_narrow[p]), 100),
        color='black'
    )

    corr_narrow = post_means_odd_narrow[p].corr(post_means_even_narrow[p])

    rel_table[p][0] = corr_narrow

    ax.text(0.98, 0.09, '$r_c$ = ' +  str(round(spearman(corr_narrow),2)),
         transform=ax.transAxes,  # use axes coordinates
         fontsize=12,
         verticalalignment='bottom',  # align top of text at y=0.99
         horizontalalignment='right',
         color='#132a70')
    
    corr_wide = post_means_odd_wide[p].corr(post_means_even_wide[p])

    rel_table[p][1] = corr_wide

    ax.text(0.987, 0.01, '$r_c$ = ' +  str(round(spearman(corr_wide),2)),
         transform=ax.transAxes,  # use axes coordinates
         fontsize=12,
         verticalalignment='bottom',  # align top of text at y=0.99
         horizontalalignment='right',
         color='maroon')
    
    ax.set_title(param_labels([p]))

    if p == 'sd_r':
        ax.legend(loc='upper left')


rel_table['network_name'] = network_name
rel_table['data_set'] = 'empirical_study'

rel_table['spacing'] = np.array(['narrow', 'wide'])

rel_table.to_csv(parent_dir + '/bf_dmc/data/reliability/reliabilities_uncorrected_' + network_name + '.csv')

fig.tight_layout

fig.savefig(network_plot_folder + '/plot_reliability_' + network_name + '_scatterplot.png')


## ACDC data sets

if False:

    data_sets = ['model_data_hedge_hedge1', 'model_data_hedge_hedge2', 'model_data_hedge_hedge3', 'model_data_hedge_hedge4', 'model_data_hedge_hedge5', 'model_data_hedge_whitehead1', 'model_data_hedge_whitehead2', 'model_data_hedge_whitehead3'] 

    for ds in data_sets:

        data = pd.read_csv(parent_dir + '/bf_dmc/data/acdc/' + ds + '.csv')

        data_even = data[data.iloc[:,0] % 2 == 0]
        data_even.rename(columns={'RT': 'rt', 'corr_resp': 'accuracy'}, inplace=True)

        data_odd = data[data.iloc[:,0] % 2 != 0]
        data_odd.rename(columns={'RT': 'rt', 'corr_resp': 'accuracy'}, inplace=True)


        post_samples_even = fit_empirical_data(data_even, approximator, id_label="participant")

        post_samples_odd = fit_empirical_data(data_odd, approximator, id_label="participant")


        post_means_even = post_samples_even.groupby('participant').mean().reset_index()

        post_means_odd = post_samples_odd.groupby('participant').mean().reset_index()

        rel_table = pd.DataFrame(np.ones((1, len(param_names))))

        rel_table.columns = param_names

        fig, axes = plt.subplots(1, len(param_names), figsize=(15, 3))

        for p, ax in zip(param_names, axes):

            ax.plot(post_means_even[p], post_means_odd[p], "o", color='#132a70', alpha=0.7)

            ax.plot(
                np.linspace(min(post_means_even[p]), max(post_means_even[p]), 100),
                np.linspace(min(post_means_odd[p]), max(post_means_odd[p]), 100),
                color='black'
            )

            corr = post_means_even[p].corr(post_means_odd[p])

            rel_table[p][0] = corr

            ax.text(0.98, 0.09, '$r_c$ = ' +  str(round(corr,2)),
                transform=ax.transAxes,  # use axes coordinates
                fontsize=12,
                verticalalignment='bottom',  # align top of text at y=0.99
                horizontalalignment='right',
                color='#132a70')

            ax.set_title(param_labels([p]))

        fig.tight_layout()
        fig.suptitle(ds)

        fig.savefig(network_plot_folder + '/plot_reliability_' + network_name + '_' + ds + '_scatterplot.png')


        rel_table['data_set'] = ds

        rel_table['network_name'] = network_name

        rel_table.to_csv(parent_dir + '/bf_dmc/data/reliability/reliabilities_uncorrected_' + network_name + '_' + ds + '.csv')