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
import time 

import bayesflow as bf
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


#arguments = sys.argv[1:]
#network_name_fixed = str(arguments[0])
#host = str(arguments[1])
#fixed_n_obs = int(arguments[2])
#num_resims = int(arguments[6])
#network_name_estimated = str(arguments[3])

network_name_fixed = 'updated_priors_sdr_fixed'
network_name_estimated = 'updated_priors_sdr_estimated'
fixed_n_obs = 300
num_resims = 100
host = 'local'

if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()



print(f'parent_dir: {parent_dir}', flush=True)


from dmc import DMC, dmc_helpers


model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name_fixed + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs_fixed = pickle.load(file)


model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name_estimated + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs_estimated = pickle.load(file)

# Define simulatores based on model_specs
simulator_fixed = DMC(**model_specs_fixed['simulation_settings'])
simulator_estimated = DMC(**model_specs_estimated['simulation_settings'])

# Load checkpoints
approximator_fixed = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name_fixed + '.keras')
approximator_estimated = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name_estimated + '.keras')

# create plots folder
ppc_plot_folder = parent_dir + "/plots/ppc/" + network_name_fixed

if not os.path.exists(ppc_plot_folder):
    os.makedirs(ppc_plot_folder)

# laod empirical data
narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


## Estimate parameters for empirical data FIXED 

samples_narrow_fixed = dmc_helpers.fit_empirical_data(narrow_data, approximator_fixed)

samples_narrow_fixed["spacing"]="narrow"

samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator_fixed)

samples_wide["spacing"]="wide"

samples_complete_fixed=pd.concat((samples_wide, samples_narrow_fixed))


## Estimate parameters for empirical data ESTIMATED 

samples_narrow_estimated = dmc_helpers.fit_empirical_data(narrow_data, approximator_estimated)

samples_narrow_estimated["spacing"]="narrow"

samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator_estimated)

samples_wide["spacing"]="wide"

samples_complete_estimated=pd.concat((samples_wide, samples_narrow_estimated))


parts=samples_complete_estimated["participant"].unique()

# Plots delta Functions

fig, axes = plt.subplots(6, 10, figsize=(20,12))

axes = axes.flatten()

for part, ax in zip(parts, axes):
    
    # filter sample data for given participant and narrow spacing
    part_data_samples_fixed = samples_complete_fixed[samples_complete_fixed["participant"]==part]

    part_data_samples_fixed = part_data_samples_fixed[part_data_samples_fixed["spacing"] == "narrow"]

    # filter empirical data for given participant and narrow spacing
    part_data = empirical_data[empirical_data["participant"] == part]

    part_data = part_data[part_data["spacing_num"] == 1]

    part_data["condition_label"] = part_data["congruency_num"].map({0.0: "congruent", 1.0: "incongruent"})

    # resimulate data
    data_resimulated_fixed = dmc_helpers.resim_data(part_data_samples_fixed, num_obs=part_data.shape[0], simulator=simulator_fixed, part=part, param_names=model_specs_fixed['simulation_settings']['param_names'] )
    
    # exclude non-convergents
    data_resimulated_fixed = data_resimulated_fixed[data_resimulated_fixed["rt"] != -1]

    # recode congruency
    data_resimulated_fixed["condition_label"] = data_resimulated_fixed["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

    quantiles = np.arange(0,1, 0.1)

    quantile_data_wide_resim_fixed = dmc_helpers.delta_functions(data_resimulated_fixed, quantiles = np.arange(0, 1, 0.1))

    quantile_data_wide_empirical_fixed = dmc_helpers.delta_functions(part_data, quantiles = np.arange(0, 1,0.1))

    ax.plot(quantile_data_wide_resim_fixed["mean_qu"] ,quantile_data_wide_resim_fixed["delta"] ,"-", color='#132a70', label = '$sd_r$ fixed')




        # filter sample data for given participant and narrow spacing
    part_data_samples_estimated = samples_complete_estimated[samples_complete_estimated["participant"]==part]

    part_data_samples_estimated = part_data_samples_estimated[part_data_samples_estimated["spacing"] == "narrow"]

    # filter empirical data for given participant and narrow spacing
    part_data = empirical_data[empirical_data["participant"] == part]

    part_data = part_data[part_data["spacing_num"] == 1]

    part_data["condition_label"] = part_data["congruency_num"].map({0.0: "congruent", 1.0: "incongruent"})

    # resimulate data
    data_resimulated_estimated = dmc_helpers.resim_data(part_data_samples_estimated, num_obs=part_data.shape[0], simulator=simulator_estimated, part=part, param_names=model_specs_estimated['simulation_settings']['param_names'] )
    
    # exclude non-convergents
    data_resimulated_estimated = data_resimulated_estimated[data_resimulated_estimated["rt"] != -1]

    # recode congruency
    data_resimulated_estimated["condition_label"] = data_resimulated_estimated["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

    quantiles = np.arange(0,1, 0.1)

    quantile_data_wide_resim_estimated = dmc_helpers.delta_functions(data_resimulated_estimated)

    quantile_data_wide_empirical = dmc_helpers.delta_functions(part_data, quantiles = np.arange(0, 1,0.1))

    ax.plot(quantile_data_wide_resim_estimated["mean_qu"] ,quantile_data_wide_resim_estimated["delta"] ,color = 'maroon', label = '$sd_r$ estimated')

    ax.plot(quantile_data_wide_empirical["mean_qu"] ,quantile_data_wide_empirical["delta"],"o",  color='black')

fig.tight_layout()
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Mean RT')
fig.savefig(ppc_plot_folder + '/'  + network_name_estimated + '_delta_functions.png')
fig.savefig(ppc_plot_folder + '/'  + network_name_fixed + '_delta_functions.png')
    