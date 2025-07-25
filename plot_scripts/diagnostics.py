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

# get arguments:
#arguments = sys.argv[1:]
#network_name = str(arguments[0])
#host = str(arguments[1])
#fixed_n_obs = int(arguments[2])

network_name = 'updated_priors_sdr_estimated'

host = 'local'

fixed_n_obs = 300

# set working directory (local/mogon)
if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

# check working directory
print(f'parent_dir: {parent_dir}', flush=True)


import bayesflow as bf
from dmc import DMC, dmc_helpers

# load model specifications
network_dir = parent_dir + "/training_checkpoints/" + network_name + '.keras'

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# set simulator
simulator = DMC(**model_specs['simulation_settings'])

# set adapter (sdr fixed /estimated)
if simulator.sdr_fixed == 0:

    adapter = (
        bf.adapters.Adapter()
        .drop('sd_r')
        .convert_dtype("float64", "float32")
        .sqrt("num_obs")
        .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
        .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
        .standardize(include="inference_variables")
        .rename("num_obs", "inference_conditions")
    )
else:
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .sqrt("num_obs")
        .concatenate(model_specs['simulation_settings']['param_names'], into="inference_variables")
        .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
        .standardize(include="inference_variables")
        .rename("num_obs", "inference_conditions")
    )


# Create inference net 
if model_specs['inference_network_settings']['network_type'] == 'FlowMatching':

    inference_net = bf.networks.FlowMatching(**model_specs['inference_network_settings'])

else:

    inference_net = bf.networks.CouplingFlow(**model_specs['inference_network_settings'])

# inference_net = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))

summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=model_specs['learning_rate'],
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath= parent_dir + '/data/training_checkpoints',
    checkpoint_name=network_name,
    inference_variables=model_specs['simulation_settings']['param_names']
)

approximator = keras.saving.load_model(network_dir)

workflow.approximator = approximator

simulator.fixed_num_obs = fixed_n_obs

val_data = simulator.sample(500)
n_obs = val_data['rt'].shape[1]

print(f' {n_obs}')

#_ = workflow.sample(conditions=val_data, num_samples=100, strict=True)

figs = workflow.plot_default_diagnostics(test_data=val_data, variable_names=dmc_helpers.param_labels(model_specs['simulation_settings']['param_names']), calibration_ecdf_kwargs={'difference': True})

plots_dir = parent_dir + '/plots/diagnostics/' + network_name
os.makedirs(plots_dir, exist_ok=True)


for k, i in figs.items():
    figs[k].savefig(plots_dir + '/' + network_name + '_' + k + '_' + str(n_obs) + 'trials.png')
