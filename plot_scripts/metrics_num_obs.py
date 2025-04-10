import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle
import time
import keras

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import bayesflow as bf
from dmc import DMC, subset_data

parent_dir = os.getcwd()

network_name = 'testrun'

step_size = 50

num_max_obs = 800




file_path = 'model_specs/model_specs_' + network_name + '.pickle'

with open(file_path, 'rb') as file:
    model_specs = pickle.load(file)

simulator = DMC(**model_specs['simulation_settings'])


## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")
approximator.compile()


simulator.fixed_num_obs = num_max_obs
## Simulate Validation Data Set
val_data = simulator.sample(1000)


val_file_path = parent_dir + '/data/validation_data_metrics/validation_data_metrics_' + network_name + '_validation.pickle'

val_data = simulator.sample(1000)

with open(val_file_path, 'wb') as file:
    pickle.dump(val_data, file)

for key, values in val_data.items():
    print(f'{key}: {values.shape}')

list_metrics = []

for n_obs in np.arange(50, num_max_obs+1, step_size):
    
    print(f'num_obs: {n_obs}')
    # simulator.num_obs = n_obs

    data_subset = subset_data(val_data.copy(), num_obs=n_obs)

    start_time = time.time()
    samples = approximator.sample(conditions=data_subset, num_samples=1000)
    end_time = time.time()


    pc_df = pd.DataFrame(bf.diagnostics.metrics.posterior_contraction(samples, data_subset))

    # pc_df['values'] = 1 - pc_df['values']

    ce_df = pd.DataFrame(bf.diagnostics.metrics.calibration_error(samples, data_subset))

    nrmse_df = pd.DataFrame(bf.diagnostics.metrics.root_mean_squared_error(samples, data_subset))

    results_single = pd.concat([ce_df, pc_df, nrmse_df])
    
    
    results_single["num_obs"] = n_obs
    results_single["sampling_time"] = end_time - start_time
    
    list_metrics.append(results_single)
    
data_set_metrics = pd.concat(list_metrics)


fig, axes = plt.subplots(1,5,sharey=True, figsize=(15,3))

for p, ax in zip(model_specs['param_names'], axes):
    
    suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

    label = suff + p + "$"
    
    sns.lineplot(data_set_metrics[data_set_metrics["variable_names"] == p], x="num_obs", y="values", hue="metric_name", ax=ax, palette="colorblind")
    ax.set_title(label)
    ax.legend(title="")
    if p != "b":
        ax.get_legend().remove()
    
    ax.set_xlabel("")

    plt.ylim(0, 1)

fig.tight_layout()
fig.supxlabel("Number of Observations", fontsize=12) 

fig.savefig(parent_dir + '/plots/metrics_num_obs_' + network_name + '.png')


data_set_metrics_time = data_set_metrics[(data_set_metrics["metric_name"] == 'Calibration Error')]
data_set_metrics_time = data_set_metrics_time[(data_set_metrics_time["variable_names"] == 'A')]


plt.figure()

time_plot = sns.lineplot(data_set_metrics_time, x="num_obs", y="sampling_time")
    # ax.set_title(label)
    # ax.legend(title="")
    # if p != "b":
    #     ax.get_legend().remove()
    
    # ax.set_xlabel("")

    # plt.ylim(0, 1)

# fig.tight_layout()
# fig.supxlabel("Number of Observations", fontsize=12) 
time_plot_fig = time_plot.get_figure()
time_plot_fig.savefig(parent_dir + '/plots/metrics_num_obs_sampling_time_' + network_name + '.png')