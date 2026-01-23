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

# load training checkpoints
approximator = keras.saving.load_model(network_dir)

# fix number of simulated observations:
simulator.fixed_num_obs = fixed_n_obs

# simulate 500 data set:
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
