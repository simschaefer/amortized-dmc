import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle

import bayesflow as bf
import keras

import pandas as pd

import matplotlib.pyplot as plt

# define network:
# network_name = 'initial_priors_sdr_fixed'
# network_name = 'initial_priors_sdr_estimated'
# network_name = 'updated_priors_sdr_fixed'
network_name = 'updated_priors_sdr_estimated'

# get parent directory:
scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
# check parent directory (should be '.../amortized-dmc/')
print(f'parent_dir: {parent_dir}', flush=True)

from dmc import DMC, param_labels, fit_empirical_data

# get model specifications for the given network
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# extract parameter names:
param_names = model_specs['simulation_settings']['param_names']

# set simulator
simulator = DMC(**model_specs['simulation_settings'])

## Load Approximator
approximator = keras.saving.load_model(parent_dir +"/training_checkpoints/" + network_name + ".keras")

# load empirical data (narrow)
# clean_experiment_data_narrow_complete.csv includes the trial number (exact order from the experiment) and all narrow spacing trials
narrow_data = pd.read_csv(parent_dir + '/empirical_data/clean_experiment_data_narrow_complete.csv')

# Split narrow Data by trial number
# EVEN
narrow_data_even=narrow_data[narrow_data['n_trial_experiment'] % 2 == 0]
# ODD
narrow_data_odd=narrow_data[narrow_data['n_trial_experiment'] % 2 != 0]

# Sample Posteriors seperately (ODD/EVEN)
post_samples_narrow_even = fit_empirical_data(narrow_data_even, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

post_samples_narrow_odd = fit_empirical_data(narrow_data_odd, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

# Compute posterior means
post_means_odd_narrow = post_samples_narrow_odd.groupby('participant').mean().reset_index()
post_means_even_narrow = post_samples_narrow_even.groupby('participant').mean().reset_index()

# prepare correlation table:
rel_table = pd.DataFrame(np.ones((1, len(param_names))))
rel_table.columns = param_names

# sanity check: do participants match between odd/even data sets?
if np.sum(post_means_even_narrow['participant'] != post_means_odd_narrow['participant']) != 0:
    print('Part IDs does not correspond between odd and even trials (wide)!', flush = True)

else:
    print('Part IDs correspond between odd and even trials.', flush = True)

# Compute Odd-Even Correlations
for p in param_names:

    corr_narrow = post_means_odd_narrow[p].corr(post_means_even_narrow[p])

    rel_table.loc[0, p] = corr_narrow

# Save with network name and spacing condition label
rel_table['network_name'] = network_name
rel_table['data_set'] = 'empirical_study'
rel_table['spacing'] = 'narrow'

rel_table.to_csv(parent_dir + '/data/reliability/reliabilities_uncorrected_' + network_name + '.csv')


# Hedge Data Sets

data_sets = ['model_data_hedge_hedge1', 'model_data_hedge_hedge2', 'model_data_hedge_hedge3', 'model_data_hedge_hedge4', 'model_data_hedge_hedge5', 'model_data_hedge_whitehead1', 'model_data_hedge_whitehead2', 'model_data_hedge_whitehead3'] 

for ds in data_sets:

    data = pd.read_csv(parent_dir + '/data_complete/acdc/' + ds + '.csv')

    data_even = data.loc[data.iloc[:, 0] % 2 == 0].copy()
    data_even = data_even.rename(columns={'RT': 'rt', 'corr_resp': 'accuracy'})

    data_odd = data.loc[data.iloc[:, 0] % 2 != 0].copy()
    data_odd = data_odd.rename(columns={'RT': 'rt', 'corr_resp': 'accuracy'})

    # post samples EVEN trials
    post_samples_even = fit_empirical_data(
        data_even,
        approximator,
        id_name='participant',
        rt='rt',
        accuracy='accuracy',
        congruency='congruency_num'
    )

    # post samples ODD trials
    post_samples_odd = fit_empirical_data(
        data_odd,
        approximator,
        id_name='participant',
        rt='rt',
        accuracy='accuracy',
        congruency='congruency_num'
    )

    # compute posterior means
    post_means_even = post_samples_even.groupby('participant').mean().reset_index()
    post_means_odd = post_samples_odd.groupby('participant').mean().reset_index()

    rel_table = pd.DataFrame(np.ones((1, len(param_names))), columns=param_names)

    rel_table['data_set'] = ds
    rel_table['network_name'] = network_name

    rel_table.to_csv(
        parent_dir + '/data/reliability/reliabilities_uncorrected_' + network_name + '_' + ds + '.csv',
        index=False
    )
