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
import time


parent_dir = '/home/administrator/Documents/bf_dmc'

#'dmc_optimized_updated_priors_sdr_fixed_200_797801',
#'dmc_optimized_updated_priors_sdr_estimated_200_797802'

network_name = 'dmc_optimized_updated_priors_sdr_fixed_200_821685'

import bayesflow as bf
from dmc import DMC, dmc_helpers

import pandas as pd

n_trials = 200

simulators = []
approximators = []

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'

with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

param_names = model_specs['simulation_settings']['param_names']

simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
approximator = keras.saving.load_model(parent_dir + "/data/training_checkpoints/" + network_name + '.keras')

simulator.fixed_num_obs = n_trials

df_list = []
df_samples_lst = []

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


df_samples_complete.to_csv(parent_dir + '/data/simulated_data/' + network_name+ '_' + str(n_trials) + '_samples.csv')
df_complete.to_csv(parent_dir + '/data/simulated_data/' + network_name + '_' + str(n_trials) + '_trials_data.csv')

