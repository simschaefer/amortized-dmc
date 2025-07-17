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
import os


parent_dir =  os.path.dirname(os.getcwd())


import bayesflow as bf
from dmc import DMC, dmc_helpers

import pandas as pd


network_name = 'initial_priors_sdr_estimated'

# load model specs
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")

# load empirical data

narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir +'/data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


all_parts = empirical_data['participant'].unique()

# participant IDs that should be used for prior updating (randomly sampled):
train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214])


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

# posterior samples 
empirical_samples_narrow = dmc_helpers.fit_empirical_data(train_data_narrow, approximator)


for part in empirical_samples_narrow['participant'].unique():
    fig, axes = plt.subplots(1, 5, figsize=(15,3))
    for p, ax in zip(model_specs['simulation_settings']['param_names'], axes): 
        sns.kdeplot(empirical_samples_narrow[empirical_samples_narrow['participant'] == part], x=p, ax=ax)
    fig.suptitle(part)

plt.figure()
#sns.kdeplot(train_data_narrow[train_data_narrow['participant'] ==1797], x='rt', hue='congruency_num')


updated_priors_narrow = empirical_samples_narrow.agg(['mean', 'std'])

# save mean and sds:
updated_priors_narrow.to_csv(parent_dir + '/data/updated_priors/updated_priors_' + network_name + '.csv')


