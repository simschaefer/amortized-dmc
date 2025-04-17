import sys

sys.path.append("../../BayesFlow")
sys.path.append("../")

import os

import torch 

print("CUDA available:", torch.cuda.is_available(), flush=True)
print(torch.cuda.device_count(), flush=True)
print("Using device:", torch.cuda.get_device_name(0))


if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle

import keras

import bayesflow as bf

parent_dir = os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)

from dmc import DMC

#########
network_name = "dmc_optimized_winsim_priors_sdr"
######### 

epochs = 100
num_batches_per_epoch = 1000

model_specs = {"simulation_settings": {"prior_means": np.array([70.8, 114.71, 0.71, 332.34, 98.36, 43.36]),
                                       "prior_sds": np.array([19.42, 40.08, 0.14, 52.07, 30.05, 9.19]),
                                       'sdr_fixed': None,
                                       "tmax": 1500,
                                       "contamination_probability": None,
                                       "min_num_obs": 50,
                                       "max_num_obs": 800,
                                       "fixed_num_obs": None},
"inference_network_settings": {"coupling_kwargs": {"subnet_kwargs": {"dropout":0.0100967297}}, "depth":10},
"summary_network_settings": {"dropout": 0.0100967297,
                             "num_seeds": 2,
                             "summary_dim": 32,
                             "embed_dim": (128, 128)},
                             'batch_size': 16,
                             'learning_rate': 0.0004916,
                             'param_names': ["A", "tau", "mu_c", "mu_r", "b"]}


file_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'


with open(file_path, 'wb') as file:
    pickle.dump(model_specs, file)

adapter = (
    bf.adapters.Adapter()
    .drop('sd_r')
    .convert_dtype("float64", "float32")
    .sqrt("num_obs")
    .concatenate(model_specs["param_names"], into="inference_variables")
    .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
    .standardize(include="inference_variables")
    .rename("num_obs", "inference_conditions")
)

simulator = DMC(**model_specs['simulation_settings'])

inference_net = bf.networks.CouplingFlow(**model_specs['inference_network_settings'])

# inference_net = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))

summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=model_specs["learning_rate"],
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath=parent_dir + '/bf_dmc/data/training_ceckpoints',
    checkpoint_name=network_name,
    inference_variables=model_specs["param_names"]
)

# file_path = '../data/data_offline_training/data_offline_training_' + network_name + '.pickle'

# train_data = simulator.sample(50000)

# with open(file_path, 'wb') as file:
#     pickle.dump(train_data, file)

# with open(file_path, 'rb') as file:
#     train_data = pickle.load(file)
    

val_file_path = parent_dir + '/bf_dmc/data/data_offline_training/data_offline_validation_online_training_' + network_name + '.pickle'

val_data = simulator.sample(200)

with open(val_file_path, 'wb') as file:
    pickle.dump(val_data, file)

# with open(val_file_path, 'rb') as file:
#     val_data = pickle.load(file)


_ = adapter(val_data, strict=True, stage="inference")


history = workflow.fit_online(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch, batch_size=model_specs["batch_size"], validation_data=val_data)

# approximator = keras.saving.load_model("../checkpoints/" + network_name)
