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

network_name = 'dmc_optimized_winsim_priors_sdr_fixed_200_795737'

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")



narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir +'/data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


#all_parts = empirical_data['participant'].unique()

#random_idx = np.random.choice(all_parts, size = all_parts.shape[0]//2)

train_idx = np.array([6361, 5281, 6214, 1108, 1538,  833, 4222,  275, 8755, 5281, 4222,
        985, 1601, 8788,  845, 4222, 8785, 3286, 1761, 3625, 3625, 1583,
       6844, 7768, 3754,  833, 1821, 7828,  275, 3754, 1657, 5815, 1583])

empirical_samples_narrow = dmc_helpers.fit_empirical_data(narrow_data[narrow_data['participant'].isin(train_idx)], approximator)

empirical_samples_wide = dmc_helpers.fit_empirical_data(wide_data[wide_data['participant'].isin(train_idx)], approximator)

empirical_samples_complete = dmc_helpers.fit_empirical_data(empirical_data[empirical_data['participant'].isin(train_idx)], approximator)

updated_priors_narrow = empirical_samples_narrow.agg(['mean', 'std'])

updated_priors_narrow

updated_priors_narrow.to_csv(parent_dir + '/data/updated_priors/updated_priors_' + network_name + '.csv')

empirical_samples_wide.agg(['mean', 'std'])

empirical_samples_complete.agg(['mean', 'std'])

