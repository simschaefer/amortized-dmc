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
from dmc import DMC, dmc_helpers

import pandas as pd



parent_dir = '/home/administrator/Documents/bf_dmc'

network_name = 'dmc_optimized_winsim_priors_sdr_estimated_200_810183'

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")



narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir +'/data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


all_parts = empirical_data['participant'].unique()

#train_idx = np.random.choice(all_parts, size = all_parts.shape[0]//2, replace=False)

#train_idx = np.array([6361, 5281, 6214, 1108, 1538,  833, 4222,  275, 8755, 5281, 4222,
#        985, 1601, 8788,  845, 4222, 8785, 3286, 1761, 3625, 3625, 1583,
#       6844, 7768, 3754,  833, 1821, 7828,  275, 3754, 1657, 5815, 1583])

train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214, 8521])


train_data_narrow = narrow_data[narrow_data['participant'].isin(train_idx)]

train_data_narrow['participant'].unique().shape

train_data_wide = wide_data[wide_data['participant'].isin(train_idx)]

train_data_wide['participant'].unique().shape

## check rts and accuracies


plt.figure()
sns.kdeplot(train_data_narrow, x='rt', hue='congruency_num')

plt.figure()
sns.kdeplot(train_data_wide, x='rt', hue='congruency_num')

train_data_narrow.groupby(['participant', 'congruency_num']).mean('accuracy').reset_index()

plt.figure()
sns.histplot(train_data_narrow.groupby(['participant', 'congruency_num']).mean('accuracy').reset_index(), x='accuracy', hue='congruency_num')

plt.figure()
sns.histplot(train_data_wide.groupby(['participant', 'congruency_num']).mean('accuracy').reset_index(), x='accuracy', hue='congruency_num')


empirical_samples_narrow = dmc_helpers.fit_empirical_data(train_data_narrow, approximator)

#empirical_samples_wide = dmc_helpers.fit_empirical_data(train_data_wide, approximator)

#empirical_samples_complete = dmc_helpers.fit_empirical_data(empirical_data[empirical_data['participant'].isin(train_idx)], approximator)




for part in empirical_samples_narrow['participant'].unique():
    fig, axes = plt.subplots(1, 5, figsize=(15,3))
    for p, ax in zip(model_specs['simulation_settings']['param_names'], axes): 
        sns.kdeplot(empirical_samples_narrow[empirical_samples_narrow['participant'] == part], x=p, ax=ax)
    fig.suptitle(part)

plt.figure()
sns.kdeplot(train_data_narrow[train_data_narrow['participant'] ==1797], x='rt', hue='congruency_num')

# extract weird samples

updated_priors_narrow = empirical_samples_narrow.agg(['mean', 'std'])

updated_priors_narrow

updated_priors_narrow.to_csv(parent_dir + '/data/updated_priors/updated_priors_' + network_name + '.csv')


