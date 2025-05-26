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


parent_dir = os.getcwd()

dmc_module_dir = parent_dir + '/bf_dmc/dmc'


print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)

from dmc import DMC
import copy


arguments = sys.argv[1:]
network_name = str(arguments[0])


#step_size = 25
num_reptitions = 1000
num_data_sets = 100

#min_num_obs = 50
#max_num_obs = 800

model_specs_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'


with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

#model_specs['simulation_settings']['param_names'] = ('A', 'tau', 'mu_c', 'mu_r', 'b', 'sd_r')


def load_model_specs(model_specs, network_name):

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
        checkpoint_filepath='../data/training_checkpoints',
        checkpoint_name=network_name,
        inference_variables=model_specs['simulation_settings']['param_names']
    )

    return simulator, adapter, inference_net, summary_net, workflow


simulator, adapter, inference_net, summary_net, workflow = load_model_specs(model_specs, network_name)
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/bf_dmc/data/training_checkpoints/" + network_name + ".keras")

network_plot_folder = parent_dir + "/bf_dmc/plots/metrics_num_obs/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)

list_metrics = []


for rep in range(0, num_reptitions):
    
    #print(f'num_obs: {n_obs}')
    # simulator.num_obs = n_obs
#
    #data_subset = dmc_helpers.subset_data(copy.deepcopy(val_data), idx=random_idx[:n_obs])

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

data_set_metrics.to_csv(data_path + '/' + network_name + '_metrics_num_obs.csv')
### PLOT n trials - metrics ###

fig, axes = plt.subplots(1,len(model_specs['simulation_settings']['param_names']),sharey=True, figsize=(18,4))

hue_order = ["Calibration Error", "Posterior Contraction", "NRMSE"]
palette = {"Calibration Error": "#8a90a0", "Posterior Contraction": "#f28c38", "NRMSE": "#132a70"}


for p, ax in zip(model_specs['simulation_settings']['param_names'], axes):
    
    suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

    label = suff + p + "$"

    #data_set_metrics = data_set_metrics[data_set_metrics['metric_name'] != 'Calibration Error']
    
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
    
    ax.set_title(label)
    ax.legend(title="")
    if p != model_specs['simulation_settings']['param_names'][-1]:
        ax.get_legend().remove()
    
    ax.set_xlabel("")

    # plt.ylim(0, 1)


fig.supxlabel("Number of Observations", fontsize=12) 
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