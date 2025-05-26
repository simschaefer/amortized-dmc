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


parent_dir = os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'


print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)


from dmc import dmc_helpers
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


arguments = sys.argv[1:]
network_name = str(arguments[0])
#network_name = 'dmc_optimized_winsim_priors_sdr_fixed_200_795737'

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)


# Load Checkpoints
approximator = keras.saving.load_model(parent_dir + "/data/training_checkpoints/" + network_name + '.keras')


# load narrow and wide data
narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_wide.csv')

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

network_plot_folder = "../plots/experimental_effects/" + network_name

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


fig, axes = plt.subplots(1, 5, figsize=(15,3))

#samples_complete_filtered = samples_complete[samples_complete["participant"].isin([275])].reset_index()

for i, ax in enumerate(axes):

    sns.kdeplot(data=samples_complete.reset_index(), x=param_names[i], ax=ax, hue='spacing')
    if i < 4:
        ax.legend_.remove()

fig.tight_layout()

pd.DataFrame(samples_complete.reset_index().pivot(index=['participant', 'sampling_time'], columns=['A', 'tau']))


samples_narrow.columns = [name + '_narrow' for name in samples_narrow.columns]
samples_wide.columns = [name + '_wide' for name in samples_wide.columns]

sample_com_wide = pd.concat([samples_narrow, samples_wide], axis=1)

for p in param_names:

    sample_com_wide['d_'+ p] = sample_com_wide[p + '_narrow'] - sample_com_wide[p + '_wide']

    sample_com_wide['d_'+ p] = sample_com_wide['d_'+ p] / np.std(sample_com_wide['d_'+ p])



fig, axes = plt.subplots(1, len(param_names), sharey=True, sharex=True, figsize=(15,3))

for i, ax in enumerate(axes):

    ax.set_xlim(-5,5)

    sns.kdeplot(data=sample_com_wide.reset_index(), x= 'd_' + param_names[i], ax=ax, color='#132a70', fill=True)

    post_mean = np.mean(sample_com_wide['d_' + param_names[i]])
    ax.axvline(x=post_mean, color='black', linestyle='--', linewidth=1)

    suff = "$\\" if param_names[i] in ["tau", "mu_c", "mu_r"] else "$"

    label = suff + param_names[i] + "$"

    ax.set_title(label)
    ax.set_xlabel('Post Samples Difference')
 

    ax.text(post_mean + 1.1, 0.4, '$d = $' + str(round(post_mean, 2)), fontsize=12, color='black')


fig.tight_layout()


fig.savefig(network_plot_folder + "/experimental_effects_" + network_name + "_post_samples_difference.png")
