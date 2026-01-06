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

import bayesflow as bf
from dmc import DMC, dmc_helpers


# get arguments 
arguments = sys.argv[1:]

if 'executed_from_bash' in arguments:
    network_name = str(arguments[0])
    host = str(arguments[1])
    fixed_n_obs = int(arguments[2])

else:
    # set host (local / mogon)
    host = 'local'

    # choose network 
    network_name = 'initial_priors_sdr_fixed'

    # number of observations in data set
    fixed_n_obs = 300

# set working directory (local/mogon)
if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

# check working directory
print(f'parent_dir: {parent_dir}', flush=True)

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
        .drop('sd_r') # <- sd_r is dropped if fixed
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


# Specify Inferene Network
inference_net = bf.networks.FlowMatching(**model_specs['inference_network_settings'])


# Specify Summary Network
summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

# Put everything together in workflow object
workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=model_specs['learning_rate'],
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath= parent_dir + '/training_checkpoints',
    checkpoint_name=network_name,
    inference_variables=model_specs['simulation_settings']['param_names']
)

# load training checkpoints
approximator = keras.saving.load_model(network_dir)

# add loaded approximator to workflow object
workflow.approximator = approximator

# fix number of simulated observations:
simulator.fixed_num_obs = fixed_n_obs

# simulate 500 data set:
#val_data = simulator.sample(500)

val_data_path = parent_dir + '/data_complete/data_offline_training/data_offline_validation_online_training_dmc_optimized_winsim_priors_sdr_fixed_200_818801.pickle'

with open(val_data_path, 'rb') as file:
    val_data = pickle.load(file)

val_data = simulator.sample(500)

# Check if number of observations match fixed_n_obs:
n_obs = val_data['rt'].shape[1]
print(f' {n_obs}')

# Plot Recovery, SBC and PC
#figs = workflow.plot_default_diagnostics(test_data=val_data, variable_names=dmc_helpers.param_labels(model_specs['simulation_settings']['param_names']), 
#                                         calibration_ecdf_kwargs={'difference': True,
#                                                                  'title_fontsize': 15})

post_samples = approximator.sample(conditions=val_data, num_samples=1000)

title_fontsize = 40
label_fontsize = 25
legend_fontsize = 15

fic_sbc = bf.diagnostics.calibration_ecdf(targets=val_data, estimates=post_samples, variable_names=dmc_helpers.param_labels(model_specs['simulation_settings']['param_names']), difference=True, title_fontsize=title_fontsize, label_fontsize=label_fontsize, legend_fontsize=legend_fontsize)

fig_rec = bf.diagnostics.recovery(targets=val_data, estimates=post_samples, metric_fontsize=25, variable_names=dmc_helpers.param_labels(model_specs['simulation_settings']['param_names']), title_fontsize=title_fontsize, label_fontsize=label_fontsize)

fig_pc = bf.diagnostics.z_score_contraction(targets=val_data, estimates=post_samples, variable_names=dmc_helpers.param_labels(model_specs['simulation_settings']['param_names']), title_fontsize=title_fontsize, label_fontsize=label_fontsize)

# create plot folder if necessary
plots_dir = parent_dir + '/plots/diagnostics/' + network_name
os.makedirs(plots_dir, exist_ok=True)

fic_sbc.savefig(plots_dir + '/' + network_name + '_calibration_ecdf_' + str(n_obs) + 'trials.png')

fig_rec.savefig(plots_dir + '/' + network_name + '_recovery_' + str(n_obs) + 'trials.png')

fig_pc.savefig(plots_dir + '/' + network_name + '_z_score_contraction_' + str(n_obs) + 'trials.png')

# save all figures
#for k, i in figs.items():
#    figs[k].savefig(plots_dir + '/' + network_name + '_' + k + '_' + str(n_obs) + 'trials.png')
