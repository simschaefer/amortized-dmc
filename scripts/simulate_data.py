import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle

import matplotlib.pyplot as plt
import time
import os
import keras
import bayesflow as bf
from dmc import dmc_helpers
import pandas as pd

parent_dir = os.path.dirname(os.getcwd())

network_name = "updated_priors_sdr_estimated"

# fixed number of trials (100, 200, 300, 400, 500):
n_trials = 500

# load model specifications
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'

with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# extract parameter names
param_names = model_specs['simulation_settings']['param_names']

# define simulator based on model specifications
simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
approximator = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name + '.keras')

# fix number of trials
simulator.fixed_num_obs = n_trials

df_list = []
df_samples_lst = []

# number of data sets
num_sims = 500

for sim_idx in range(0, num_sims):

    data_keys = ('rt', 'accuracy', 'conditions') + param_names

    single_sim = simulator.sample(1)

    start_time=time.time()
    samples = approximator.sample(conditions=single_sim, num_samples=1000)
    end_time=time.time()

    df_samples = pd.DataFrame()

    for k, j in samples.items():
        #print(f'{k}, {j.shape}')

        df_samples[k] = j.flatten()

    df_samples['sampling_time'] = end_time-start_time
    df_samples['sim_idx'] = sim_idx
    df_samples['n_obs'] = single_sim['num_obs'][0, 0]

    df_samples['network_name'] = network_name

    df_samples_lst.append(df_samples)


    data_only = {k: single_sim[k] for k in data_keys}

    df = pd.DataFrame()

    for k, dat in data_only.items():
        #i = i.reshape(data_shape)
        #print(f'{k}, {dat.shape}')
        
        if k in param_names:
            df[k] = dat.flatten()[0]

        else:
            df[k] = dat.flatten()

    df['sim_idx'] = sim_idx
    df['n_obs'] = single_sim['num_obs'][0, 0]
    df['network_name'] = network_name

    print(f'{sim_idx}')

    df_list.append(df)

df_complete = pd.concat(df_list)
df_samples_complete = pd.concat(df_samples_lst)

simulated_data_dir = parent_dir + '/data_complete/simulated_data/'

if not os.path.exists(simulated_data_dir):
    os.makedirs(simulated_data_dir)

df_samples_complete.to_csv(simulated_data_dir + network_name+ '_' + str(n_trials) + '_samples.csv')
df_complete.to_csv(simulated_data_dir + network_name + '_' + str(n_trials) + '_trials_data.csv')

