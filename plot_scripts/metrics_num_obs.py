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


arguments = sys.argv[1:]
network_name = str(arguments[0])
host = str(arguments[1])

num_reptitions = 1000
num_data_sets = 100

network_name = 'updated_priors_sdr_estimated'

host = 'local'

fixed_n_obs = 300


if host == 'local':
    parent_dir = os.path.dirname(os.getcwd())
else:
    parent_dir = os.getcwd()

fontsize = 18


from dmc import DMC
import copy

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)


simulator = DMC(**model_specs['simulation_settings'])
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/training_checkpoints/" + network_name + ".keras")

network_plot_folder = parent_dir + "/plots/metrics_num_obs/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)

list_metrics = []


for rep in range(0, num_reptitions):


    data_subset = simulator.sample(num_data_sets)

    n_obs = data_subset['rt'].shape[1]

    print(f"Repetition #{rep+1} of {num_reptitions}, {n_obs} trials", flush=True)

    start_time = time.time()
    samples = approximator.sample(conditions=data_subset, num_samples=1000)
    end_time = time.time()


    pc_df = pd.DataFrame(bf.diagnostics.metrics.posterior_contraction(samples, data_subset))

    pc_df['values'] = 1 - pc_df['values']

    ce_df = pd.DataFrame(bf.diagnostics.metrics.calibration_error(samples, data_subset))

    nrmse_df = pd.DataFrame(bf.diagnostics.metrics.root_mean_squared_error(samples, data_subset))

    results_single = pd.concat([ce_df, pc_df, nrmse_df])
    
    
    results_single["num_obs"] = n_obs
    results_single["sampling_time"] = end_time - start_time
    
    list_metrics.append(results_single)
    
data_set_metrics = pd.concat(list_metrics)

data_set_metrics.reset_index(inplace=True)

bin_width = 100

data_set_metrics['num_obs_bin'] = pd.cut(data_set_metrics['num_obs'], bins=list(range(50, 1000 + bin_width, bin_width)),labels=list(range(50 + bin_width//2, 1000 + bin_width//2, bin_width)), right=False)

data_path = parent_dir + '/data/insilico/' + network_name

if not os.path.exists(data_path):
    os.makedirs(data_path)

data_set_metrics['metric_name'] = data_set_metrics['metric_name'].replace('Posterior Contraction', 'Contraction Factor')

data_set_metrics.to_csv(data_path + '/' + network_name + '_metrics_num_obs.csv')
### PLOT n trials - metrics ###

fig, axes = plt.subplots(1,len(model_specs['simulation_settings']['param_names']),sharey=True, figsize=(18,4))

hue_order = ["Calibration Error", "Contraction Factor", "NRMSE"]
palette = {"Calibration Error": "#8a90a0", "Contraction Factor": "#f28c38", "NRMSE": "#132a70"}


for p, ax in zip(model_specs['simulation_settings']['param_names'], axes):
    
    suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

    label = suff + p + "$"

    
    sns.lineplot(data_set_metrics[data_set_metrics["variable_names"] == p], x="num_obs_bin", y="values", hue="metric_name", ax=ax, hue_order=hue_order, palette=palette, errorbar='sd')
    sns.scatterplot(data_set_metrics[data_set_metrics["variable_names"] == p], 
                    s=4, 
                    alpha=0.3, 
                    x="num_obs", 
                    y="values", 
                    hue="metric_name", 
                    ax=ax, 
                    hue_order=hue_order, 
                    palette=palette,
                    legend=False)
    
    ax.set_title(label, fontsize=fontsize)
    legend = ax.legend(title="", fontsize=fontsize - 4) 
    if p != model_specs['simulation_settings']['param_names'][-1]:
        ax.get_legend().remove()
    
    ax.set_ylabel("Metric Value", fontsize=fontsize)
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=fontsize - 4)  
    ax.tick_params(axis='y', labelsize=fontsize - 4)



fig.supxlabel("Number of Observations", fontsize=fontsize) 
fig.tight_layout()

fig.savefig(network_plot_folder + '/metrics_num_obs_' + network_name + '_random.png')


data_set_metrics_time = data_set_metrics[(data_set_metrics["metric_name"] == 'Calibration Error')]
data_set_metrics_time = data_set_metrics_time[(data_set_metrics_time["variable_names"] == 'A')]


plt.figure()

# convert time in ms per data set
data_set_metrics_time['sampling_time'] = (data_set_metrics_time['sampling_time'] / num_data_sets)*1000

time_plot = sns.scatterplot(data_set_metrics_time, x="num_obs", y="sampling_time", color= "#132a70")
time_plot.set_xlabel('Number Of Observations')
time_plot.set_ylabel('Sampling time per data set [ms]')
    # ax.set_title(label)
    # ax.legend(title="")
    # if p != "b":
    #     ax.get_legend().remove()
    
    # ax.set_xlabel("")

    # plt.ylim(0, 1)

# fig.tight_layout()
# fig.supxlabel("Number of Observations", fontsize=12) 
time_plot_fig = time_plot.get_figure()
time_plot_fig.savefig(network_plot_folder + '/metrics_num_obs_sampling_time_' + network_name + '_random.png')