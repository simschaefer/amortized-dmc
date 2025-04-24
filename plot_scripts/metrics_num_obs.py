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
from dmc import DMC, dmc_helpers
import copy


network_name = 'dmc_optimized_winsim_priors_sdr_fixed_150_795633'


step_size = 25
num_reptitions = 20
num_data_sets = 100

min_num_obs = 50
max_num_obs = 800


parent_dir = '/home/administrator/Documents/bf_dmc'

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'


with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

#model_specs['simulation_settings']['param_names'] = ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r')


simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")


simulator.fixed_num_obs = max_num_obs

## Simulate Validation Data Set
#val_data = simulator.sample(1000)


#val_file_path = parent_dir + '/data/validation_data_metrics/validation_data_metrics_' + network_name + '.pickle'

#with open(val_file_path, 'wb') as file:
#    pickle.dump(val_data, file)

#for key, values in val_data.items():
#    print(f'{key}: {values.shape}')


network_plot_folder = parent_dir + "/plots/metrics_num_obs/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)

list_metrics = []

random_idx = np.random.choice(np.arange(0,max_num_obs), size=max_num_obs, replace=False)

for rep in range(0, num_reptitions):
    
    print(f"Repitition #{rep+1} of {num_reptitions}")

    for n_obs in np.arange(min_num_obs, max_num_obs+1, step_size):
        
        #print(f'num_obs: {n_obs}')
        # simulator.num_obs = n_obs
    #
        #data_subset = dmc_helpers.subset_data(copy.deepcopy(val_data), idx=random_idx[:n_obs])

        simulator.fixed_num_obs = n_obs
        data_subset = simulator.sample(num_data_sets)

            

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

### PLOT n trials - metrics ###

fig, axes = plt.subplots(1,len(model_specs['simulation_settings']['param_names']),sharey=True, figsize=(15,3))

hue_order = ["Calibration Error", "Posterior Contraction", "NRMSE"]
palette = {"Calibration Error": "#8a90a0", "Posterior Contraction": "#f28c38", "NRMSE": "#132a70"}


for p, ax in zip(model_specs['simulation_settings']['param_names'], axes):
    
    suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

    label = suff + p + "$"

    #data_set_metrics = data_set_metrics[data_set_metrics['metric_name'] != 'Calibration Error']
    
    sns.lineplot(data_set_metrics[data_set_metrics["variable_names"] == p], x="num_obs", y="values", hue="metric_name", ax=ax, hue_order=hue_order, palette=palette, errorbar='sd')
    ax.set_title(label)
    ax.legend(title="")
    if p != model_specs['simulation_settings']['param_names'][-1]:
        ax.get_legend().remove()
    
    ax.set_xlabel("")

    # plt.ylim(0, 1)

fig.tight_layout()
fig.supxlabel("Number of Observations", fontsize=12) 

fig.savefig(network_plot_folder + '/metrics_num_obs_' + network_name + '.png')


data_set_metrics_time = data_set_metrics[(data_set_metrics["metric_name"] == 'Calibration Error')]
data_set_metrics_time = data_set_metrics_time[(data_set_metrics_time["variable_names"] == 'A')]


plt.figure()

# convert time in ms per data set
data_set_metrics_time['sampling_time'] = (data_set_metrics_time['sampling_time'] / num_data_sets)*1000

time_plot = sns.lineplot(data_set_metrics_time, x="num_obs", y="sampling_time", color= "#132a70")
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
time_plot_fig.savefig(network_plot_folder + '/metrics_num_obs_sampling_time_' + network_name + '.png')