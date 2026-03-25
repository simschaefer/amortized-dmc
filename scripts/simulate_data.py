import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import pickle
import time
import os
import bayesflow as bf
import keras
from dmc import DMC
import pandas as pd

# get parent directory:
scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
# check parent directory (should be '.../amortized-dmc/')
print(f'parent_dir: {parent_dir}', flush=True)

# define network:
# network_name = 'initial_priors_sdr_fixed'
# network_name = 'initial_priors_sdr_estimated'
# network_name = 'updated_priors_sdr_fixed'
network_name = 'updated_priors_sdr_estimated'

# fixed number of trials (100, 200, 300, 400, 500):
n_trials = 300

# number of data sets
num_sims = 500

# load model specifications
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'

with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# extract parameter names
param_names = model_specs['simulation_settings']['param_names']

# load approximator
approximator = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name + '.keras')

# define simulator based on model specifications
simulator = DMC(**model_specs['simulation_settings'])

# fix number of trials
simulator.fixed_num_obs = n_trials

df_list = []
df_samples_lst = []

for sim_idx in range(0, num_sims):

    data_keys = ('rt', 'accuracy', 'conditions') + param_names

    # simulate data set
    single_sim = simulator.sample(1)

    # posterior samples
    start_time=time.time()
    samples = approximator.sample(conditions=single_sim, num_samples=1000)
    end_time=time.time()

    # convert samples to data frame and save in list:
    df_samples = pd.DataFrame()

    for k, j in samples.items():
        df_samples[k] = j.flatten()

    df_samples['sampling_time'] = end_time-start_time
    df_samples['sim_idx'] = sim_idx
    df_samples['n_obs'] = single_sim['num_obs'][0, 0]
    df_samples['network_name'] = network_name
    df_samples_lst.append(df_samples)

    # extract RT and Accuracy
    data_only = {k: single_sim[k] for k in data_keys}

    # convert to data frame and save in list
    df = pd.DataFrame()

    for k, dat in data_only.items():

        if k in param_names:
            df[k] = dat.flatten()[0]
        else:
            df[k] = dat.flatten()

    df['sim_idx'] = sim_idx
    df['n_obs'] = single_sim['num_obs'][0, 0]
    df['network_name'] = network_name
    df_list.append(df)

    print(f'{sim_idx}')

# combine data set lists:
df_complete = pd.concat(df_list)
df_samples_complete = pd.concat(df_samples_lst)

# save data
simulated_data_dir = parent_dir + '/data_complete/simulated_data/'

if not os.path.exists(simulated_data_dir):
    os.makedirs(simulated_data_dir)

# save samples data:
df_samples_complete.to_csv(simulated_data_dir + network_name+ '_' + str(n_trials) + '_samples.csv')

# save RT and Accuracy data:
#df_complete.to_csv(simulated_data_dir + network_name + '_' + str(n_trials) + '_trials_data.csv')

