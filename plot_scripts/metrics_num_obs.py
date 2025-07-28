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
from dmc import DMC

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

# load model specifications
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# Define simulator
simulator = DMC(**model_specs['simulation_settings'])

# Load Approximator
approximator = keras.saving.load_model(parent_dir +"/training_checkpoints/" + network_name + ".keras")

# Define plot directory
network_plot_folder = parent_dir + "/plots/metrics_num_obs/" + network_name

if not os.path.exists(network_plot_folder):
    os.makedirs(network_plot_folder)


list_metrics = []

# loop over n repetitions
for rep in range(0, num_reptitions):

    # simulate data sets
    data_subset = simulator.sample(num_data_sets)

    n_obs = data_subset['rt'].shape[1]

    print(f"Repetition #{rep+1} of {num_reptitions}, {n_obs} trials", flush=True)

    # Estimate Parameters
    start_time = time.time()
    samples = approximator.sample(conditions=data_subset, num_samples=1000)
    end_time = time.time()

    # Compute posterior contraction
    pc_df = pd.DataFrame(bf.diagnostics.metrics.posterior_contraction(samples, data_subset))

    # Convert to Contraction Factor
    pc_df['values'] = 1 - pc_df['values']

    # Compute Calibration Error
    ce_df = pd.DataFrame(bf.diagnostics.metrics.calibration_error(samples, data_subset))

    # Compute Recovery
    nrmse_df = pd.DataFrame(bf.diagnostics.metrics.root_mean_squared_error(samples, data_subset))

    # Concatenate Metrics
    results_single = pd.concat([ce_df, pc_df, nrmse_df])
    
    
    results_single["num_obs"] = n_obs
    results_single["sampling_time"] = end_time - start_time
    
    list_metrics.append(results_single)

# combine results from all repetitions:
data_set_metrics = pd.concat(list_metrics)

data_set_metrics.reset_index(inplace=True)

# Compute binwise mean
bin_width = 100

data_set_metrics['num_obs_bin'] = pd.cut(data_set_metrics['num_obs'], bins=list(range(50, 1000 + bin_width, bin_width)),labels=list(range(50 + bin_width//2, 1000 + bin_width//2, bin_width)), right=False)

# Define data path
data_path = parent_dir + '/data_complete/insilico/' + network_name

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Rename PC to Contraction Factor
data_set_metrics['metric_name'] = data_set_metrics['metric_name'].replace('Posterior Contraction', 'Contraction Factor')

# Save results
data_set_metrics.to_csv(data_path + '/' + network_name + '_metrics_num_obs.csv')
