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

import optuna


import bayesflow as bf

## add functions direction
parent_dir = os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(dmc_module_dir)

sys.path.append(dmc_module_dir)

from dmc import DMC


network_name = "oos500trials_noco"


model_specs = {'prior_means': np.array([16., 111., 0.5, 322., 75.]),
               'prior_sds': np.array([10., 47., 0.13, 40., 23.]),
               'tmax': 1500,
               'num_obs': 500,
               'network_name': network_name}


simulator = DMC(
    prior_means=model_specs['prior_means'], 
    prior_sds=model_specs['prior_sds'],
    tmax=model_specs['tmax'],
    # contamination_probability=.05,
    num_obs=model_specs['num_obs']
)

# file_path = '../model_specs/model_specs_' + network_name + '.pickle'

# with open(file_path, 'wb') as file:
#     pickle.dump(model_specs, file)


## simulate Training data

training_file_path = parent_dir + '/data/data_offline_training/data_offline_training_' + network_name + '.pickle'

train_data = simulator.sample(50000)

with open(training_file_path, 'wb') as file:
    pickle.dump(train_data, file)


## simulate validation data

val_file_path = parent_dir +  '/data/data_offline_training/data_offline_training_' + network_name + '_validation.pickle'

val_data = simulator.sample(1000)

with open(val_file_path, 'wb') as file:
    pickle.dump(val_data, file)