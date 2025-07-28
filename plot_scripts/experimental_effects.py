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

import bayesflow as bf

# Get arguments (from bash script)
arguments = sys.argv[1:]

if 'executed_from_bash' in arguments:
    network_name = str(arguments[0])
    host = str(arguments[1])
    fixed_n_obs = int(arguments[2])

else:
    # set host (local / mogon)
    host = 'local'

    # choose network 
    network_name = 'updated_priors_sdr_estimated'

    # number of observations in data set
    fixed_n_obs = 300

# Plot individual effects? 
plot_individual_effects = False

# define parent directory
if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()


from dmc import dmc_helpers
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# load mode specifications
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# Load Checkpoints
approximator = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name + '.keras')

# load narrow and wide data
narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

# IDs that were used to update the priors
train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214])


# exclude training IDs
narrow_data = narrow_data[~narrow_data['participant'].isin(train_idx)]
wide_data = wide_data[~wide_data['participant'].isin(train_idx)]

# Estimate Parameters for narrow data
samples_narrow = dmc_helpers.fit_empirical_data(narrow_data, approximator)

samples_narrow["spacing"] = "narrow"

# Estimate Parameters for wide data
samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator)

samples_wide["spacing"] = "wide"

# concatenate sample data from both spacing conditions
samples_complete = pd.concat((samples_wide, samples_narrow))

# extract list of all participants
parts = samples_complete["participant"].unique()

# get parameter names
param_names = model_specs['simulation_settings']['param_names']

# define directory
network_plot_folder = parent_dir + "/plots/experimental_effects/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)

# define plot colors:
hue_order = ["narrow", "wide"]
palette = {"narrow": '#132a70', "wide": '#FF6361'}

if plot_individual_effects:
    for i, part in enumerate(parts):
        
        fig, axes = plt.subplots(1,len(param_names), figsize=(10,3))
        
        axes = axes.flatten()


        for p, ax in zip(param_names, axes):
            
            part_data = samples_complete[samples_complete["participant"]==part]
            part_data = pd.DataFrame(part_data.reset_index(drop=True))

            #dat = pd.DataFrame(part_data.groupby(['participant', 'spacing']).agg(['mean', 'std']))
            
            sns.pointplot(data=part_data, ax=ax, x="spacing", y=p, color='#132a70',errorbar="sd")

            for patch in ax.patches:
                facecolor = patch.get_facecolor()
                patch.set_facecolor((*facecolor[:3], 0.3)) 
            ax.set_ylabel("")

            suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

            label = suff + p + "$"

            ax.set_title(label)
                
        fig.suptitle(str(part))    
        fig.tight_layout()
        fig.savefig(network_plot_folder + "/experimental_effects_" + network_name + str(part) + ".png")


data, fig = dmc_helpers.cohens_d_samples(samples_narrow, samples_wide, param_names, num_samples=1000, subj_id='participant', hdi_color='white', sharex=False)

fig.savefig(network_plot_folder + "/experimental_effects_" + network_name + "_post_samples_difference.png")
