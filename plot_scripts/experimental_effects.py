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


arguments = sys.argv[1:]
network_name = str(arguments[0])
host = str(arguments[1])
fixed_n_obs = int(arguments[2])

if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'


print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)


from dmc import dmc_helpers
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# Load Checkpoints
approximator = keras.saving.load_model(parent_dir + "/bf_dmc/training_checkpoints/" + network_name + '.keras')


# load narrow and wide data
narrow_data = pd.read_csv(parent_dir + '/bf_dmc/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/bf_dmc/empirical_data/experiment_data_wide.csv')


train_idx = np.array([1761, 5281,  845, 1824, 5575, 8755, 8026, 8704, 7813, 1597, 7756,
       7624, 1108,  837, 7828, 6055,  833, 1821,  985, 1582, 8311, 8785,
       3286, 4264, 6583, 3487, 6565, 6427, 1430, 6361, 5815, 6262, 5332,
       1614, 7939, 6214])


narrow_data = narrow_data[~narrow_data['participant'].isin(train_idx)]
wide_data = wide_data[~wide_data['participant'].isin(train_idx)]

# concatenate data from both spacing conditions
empirical_data = pd.concat([narrow_data, wide_data])

# fit narrow data
samples_narrow=dmc_helpers.fit_empirical_data(narrow_data, approximator)

samples_narrow["spacing"]="narrow"

# fit wide data
samples_wide=dmc_helpers.fit_empirical_data(wide_data, approximator)

samples_wide["spacing"]="wide"

# concatenate sample data from both spacings
samples_complete=pd.concat((samples_wide, samples_narrow))


parts=samples_complete["participant"].unique()


param_names = model_specs['simulation_settings']['param_names']

network_plot_folder = parent_dir + "/bf_dmc/plots/experimental_effects/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)

hue_order = ["narrow", "wide"]
palette = {"narrow": '#132a70', "wide": 'maroon'}

#parts = [275]

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
