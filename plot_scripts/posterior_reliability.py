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

parent_dir = os.getcwd()

parent_dir = '/home/administrator/Documents'

print(f'parent_dir: {parent_dir}', flush=True)

dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)

from dmc import DMC


arguments = sys.argv[1:]
network_name = str(arguments[0])

network_name = 'dmc_optimized_winsim_priors_sdr_estimated_200_810183'
#network_name = 'dmc_optimized_winsim_priors_sdr_estimated_200_805375'

model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)


def param_labels(param_names):

    param_labels = []

    for p in param_names:

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        param_labels.append(suff + p + "$")

    if len(param_labels) <= 1:
        param_labels = param_labels[0]
        
    return param_labels


def format_empirical_data(data, var_names=['rt', 'accuracy', "congruency_num"]):
    
    # extract relveant variables
    data_np = data[var_names].values

    # convert to dictionary
    inference_data = dict(rt=data_np[:,0],
                          accuracy=data_np[:,1],
                          conditions=data_np[:,2])

    # add dimensions so it fits training data
    inference_data = {k: v[np.newaxis,..., np.newaxis] for k, v in inference_data.items()}

    # adjust dimensions of num_obs
    inference_data["num_obs"] = np.array([data_np.shape[0]])[:,np.newaxis]
    
    return inference_data


def fit_empirical_data(data, approximator, id_label="participant"):

    ids=data[id_label].unique()

    list_data_samples=[]

    for i, id in enumerate(ids):
        
        part_data = data[data[id_label]==id]
        
        part_data = format_empirical_data(part_data)
        
        start_time=time.time()
        samples = approximator.sample(conditions=part_data, num_samples=1000)
        end_time=time.time()
        
        sampling_time=end_time-start_time

        samples_2d={k: v.flatten() for k, v in samples.items()}
        
        data_samples=pd.DataFrame(samples_2d)
        
        data_samples[id_label]=id
        data_samples["sampling_time"]=sampling_time
        
        list_data_samples.append(data_samples)

    data_samples_complete=pd.concat(list_data_samples)

    return data_samples_complete


def load_model_specs(model_specs, network_name):

    simulator = DMC(**model_specs['simulation_settings'])

    
    if simulator.sdr_fixed == 0:

        adapter = (
            bf.adapters.Adapter()
            .drop('sd_r')
            .convert_dtype("float64", "float32")
            .sqrt("num_obs")
            .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
            .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
            .standardize(include="inference_variables")
            .rename("num_obs", "inference_conditions")
        )
    else:
        adapter = (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .sqrt("num_obs")
            .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
            .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
            .standardize(include="inference_variables")
            .rename("num_obs", "inference_conditions")
        )

    # Create inference net 
    #inference_net = bf.networks.CouplingFlow(**model_specs['inference_network_settings'])

    inference_net = bf.networks.FlowMatching()

    summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        initial_learning_rate=model_specs['learning_rate'],
        inference_network=inference_net,
        summary_network=summary_net,
        checkpoint_filepath='../data/training_checkpoints',
        checkpoint_name=network_name,
        inference_variables=model_specs['simulation_settings']['param_names']
    )

    return simulator, adapter, inference_net, summary_net, workflow


simulator, adapter, inference_net, summary_net, workflow = load_model_specs(model_specs, network_name)
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


narrow_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_narrow_reliability.csv')
narrow_data = narrow_data[narrow_data['participant'].isin(included_parts)]

wide_data = pd.read_csv(parent_dir + '/bf_dmc/data/empirical_data/experiment_data_wide_reliability.csv')
wide_data = wide_data[wide_data['participant'].isin(included_parts)]

empirical_data = pd.concat([narrow_data, wide_data])

data_count = empirical_data.groupby('participant').count()



empirical_data = empirical_data[empirical_data['participant'].isin(included_parts)]

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
post_samples_narrow_even = fit_empirical_data(narrow_data_even, approximator)

post_samples_narrow_odd = fit_empirical_data(narrow_data_odd, approximator)

# Sample Individual Posterior Samples WIDE
post_samples_wide_even = fit_empirical_data(wide_data_even, approximator)

post_samples_wide_odd = fit_empirical_data(wide_data_odd, approximator)

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


network_plot_folder = parent_dir + "/bf_dmc/plots/plots_reliability/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)


def spearman(r, k=2):

    return (k*r)/(1+r)

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

    ax.text(0.98, 0.09, '$r_c$ = ' +  str(round(spearman(corr_narrow),2)),
         transform=ax.transAxes,  # use axes coordinates
         fontsize=12,
         verticalalignment='bottom',  # align top of text at y=0.99
         horizontalalignment='right',
         color='#132a70')
    
    corr_wide = post_means_odd_wide[p].corr(post_means_even_wide[p])

    ax.text(0.987, 0.01, '$r_c$ = ' +  str(round(spearman(corr_wide),2)),
         transform=ax.transAxes,  # use axes coordinates
         fontsize=12,
         verticalalignment='bottom',  # align top of text at y=0.99
         horizontalalignment='right',
         color='maroon')
    
    ax.set_title(param_labels([p]))

    if p == 'sd_r':
        ax.legend(loc='upper left')

fig.tight_layout

fig.savefig(network_plot_folder + '/plot_reliability_' + network_name + '_scatterplot.png')


# %%
#fig, axes = plt.subplots(1, 5, figsize=(15, 3))#

#for p, ax in zip(param_names, axes):
#    ax.hist(corr_data[p])
#    ax.set_title(param_labels([p]))

#fig.tight_layout()

#fig.savefig(network_plot_folder + '/plot_reliability_hist_' + network_name + '.png')


# %%
#fig, axes = plt.subplots(1, 5, figsize=(15, 3))

#for p, ax in zip(param_names, axes):
#    sns.kdeplot(data=corr_data, x=p, ax=ax)
#    ax.set_title(param_labels([p]))

#fig.tight_layout() 

#fig.savefig(network_plot_folder + '/plot_reliability_kde_' + network_name + '.png')
# %%
