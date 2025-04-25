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


network_name = 'dmc_optimized_updated_priors_sdr_fixed'

fixed_n_obs = 800

network_dir = parent_dir + "/data/training_checkpoints/" + network_name + '.keras'


model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

#model_specs['simulation_settings']['param_names'] = ('A', 'tau', 'mu_c', 'mu_r', 'b')

#model_specs_path = parent_dir + '/model_specs/model_specs_dmc_optimized_updated_priors_sdr_fixed.pickle'
#with open(model_specs_path, 'rb') as file:
#    model_specs_updated = pickle.load(file)

simulator = DMC(**model_specs['simulation_settings'])


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
#approximator.compile()

workflow.approximator = approximator

simulator.fixed_num_obs = fixed_n_obs

val_data = simulator.sample(1000)

#_ = workflow.sample(conditions=val_data, num_samples=100, strict=True)

figs = workflow.plot_default_diagnostics(test_data=val_data, variable_names=dmc_helpers.param_labels(model_specs['simulation_settings']['param_names']), calibration_ecdf_kwargs={'difference': True})

plots_dir = parent_dir + '/plots/diagnostics/' + network_name
os.makedirs(plots_dir, exist_ok=True)

for k, i in figs.items():
    figs[k].savefig(plots_dir + '/' + network_name + '_' + k + '_' + str(fixed_n_obs) + 'trials.png')