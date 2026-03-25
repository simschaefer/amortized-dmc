import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os
if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import pickle
import time
import pandas as pd
import bayesflow as bf
import keras
from dmc import DMC

# get parent directory:
scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)
# check parent directory (should be '.../amortized-dmc/')
print(f'parent_dir: {parent_dir}', flush=True)

# define network:
# network_name = 'initial_priors_sdr_fixed'
# network_name = 'initial_priors_sdr_estimated'
# network_name = 'updated_priors_sdr_fixed'
network_name = 'updated_priors_sdr_estimated'

# how many times should data sets be simulated?
num_repetitions = 1000

# how many data sets per repetition?
num_data_sets = 100

# load model specifications
model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

# Define simulator
simulator = DMC(**model_specs['simulation_settings'])

# Load Approximator
approximator = keras.saving.load_model(parent_dir +"/training_checkpoints/" + network_name + ".keras")

# loop over n repetitions:

list_metrics = []

for rep in range(0, num_repetitions):

    # simulate data sets
    data_subset = simulator.sample(num_data_sets)
    n_obs = data_subset['rt'].shape[1]

    # Estimate Parameters
    start_time = time.time()
    samples = approximator.sample(conditions=data_subset, num_samples=1000)
    end_time = time.time()

    print(f"Repetition #{rep+1} of {num_repetitions}, {n_obs} trials - Sampling Completed", flush=True)

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
    
    # save trial number and sampling time
    results_single["num_obs"] = n_obs
    results_single["sampling_time"] = end_time - start_time
    
    list_metrics.append(results_single)

# combine results from all repetitions:
data_set_metrics = pd.concat(list_metrics)
data_set_metrics.reset_index(inplace=True)

# Compute binwise mean
bin_width = 100
data_set_metrics['num_obs_bin'] = pd.cut(data_set_metrics['num_obs'], bins=list(range(50, 1000 + bin_width, bin_width)),labels=list(range(50 + bin_width//2, 1000 + bin_width//2, bin_width)), right=False)

# Rename PC to Contraction Factor
data_set_metrics['metric_name'] = data_set_metrics['metric_name'].replace('Posterior Contraction', 'Contraction Factor')

# Define data path
data_path = parent_dir + '/data_complete/insilico/' + network_name

if not os.path.exists(data_path):
    os.makedirs(data_path)

# Save results
data_set_metrics.to_csv(data_path + '/' + network_name + '_metrics_num_obs.csv')
