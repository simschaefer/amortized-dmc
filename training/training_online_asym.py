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

from datetime import datetime

import bayesflow as bf
import keras


arguments = sys.argv[1:]
slurm_id = str(arguments[0])
epochs = int(arguments[1])

parent_dir =  os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)

from dmc import DMCasym

num_batches_per_epoch = 250

#########
network_name = "dmc_sdr_estimated_4afc_asym_settrans_" + str(epochs) + '_' + slurm_id
#########

print(network_name, flush=True)

model_specs = {"simulation_settings": {"prior_means": np.array([32.40, 32.40, 93.88, 0.49, 387.53, 48.25, 88.21]),
                                       "prior_sds": np.array([9.05, 9.05, 29.67, 0.15, 57.65, 10.41, 16.56]),
                                       "tmax": 2000,
                                       "contamination_probability": None,
                                       'param_lower_bound' : np.array([0, 0, 0, 0, 0, 0, 0]),
                                       "fixed_num_obs": None,
                                       'param_names': ("A_con", "A_inc", "tau", "mu_c", "mu_r", "sd_r", "b")},
"inference_network_settings": {"network_type": 'FlowMatching',
                               "dropout": 0.01070354852467715},
"summary_network_settings": {"dropout": 0.01070354852467715,
                             "num_seeds": 7,
                             "summary_dim": 22,
                             "embed_dims": (128, 128)},
                             'batch_size': 64,
                             'learning_rate': 0.0005721790353631461,
                             'epochs': epochs,
                             'num_batches_per_epoch': num_batches_per_epoch,
                             'start_time': datetime.now(),
                             'network_name': network_name}

print(model_specs, flush=True)

#file_path = '../../amortized-dmc/model_specs/model_specs_' + network_name + '.pickle'

#with open(file_path, 'wb') as file:
#    pickle.dump(model_specs, file)

simulator = DMCasym(**model_specs['simulation_settings'])


adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .sqrt("num_obs")
    .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
    .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
    .log("inference_variables")
    .rename("num_obs", "inference_conditions")
)


inference_net = bf.networks.FlowMatching(
    subnet_kwargs={"dropout": model_specs["inference_network_settings"]["dropout"]}
)

summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=model_specs["learning_rate"],
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath=parent_dir + '/bf_dmc/data/training_checkpoints',
    checkpoint_name=network_name,
    inference_variables=model_specs['simulation_settings']["param_names"],
    save_best_only=True
)


val_data = simulator.sample(200, seed=23)

history = workflow.fit_online(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch, batch_size=model_specs["batch_size"], validation_data=val_data)

file_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'

model_specs['end_time'] = datetime.now()

with open(file_path, 'wb') as file:
    pickle.dump(model_specs, file)

# approximator = keras.saving.load_model("../checkpoints/" + network_name)

def param_labels(param_names):

    param_labels = []

    for p in param_names:

        suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

        param_labels.append(suff + p + "$")

    if len(param_labels) <= 1:
        param_labels = param_labels[0]

    return param_labels


figs = workflow.plot_default_diagnostics(test_data=val_data, variable_names=param_labels(model_specs['simulation_settings']['param_names']), calibration_ecdf_kwargs={'difference': True})


plots_dir = parent_dir + '/bf_dmc/plots/diagnostics/' + network_name
os.makedirs(plots_dir, exist_ok=True)


for k, i in figs.items():
    figs[k].savefig(plots_dir + '/' + network_name + '_' + k + '_posttraining.png')
