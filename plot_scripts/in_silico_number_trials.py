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
import seaborn as sns

import matplotlib.pyplot as plt

import bayesflow as bf
from dmc import DMC, dmc_helpers


network_name = "oos500trials_noco"


# file_path = '../model_specs/model_specs_' + network_name + '.pickle'

# with open(file_path, 'wb') as file:
#     pickle.load(file)

model_specs = {'prior_means': np.array([16., 111., 0.5, 322., 75.]),
               'prior_sds': np.array([10., 47., 0.13, 40., 23.]),
               'tmax': 1500,
               'num_obs': 500,
               'network_name': network_name}


simulator = DMC(
    prior_means=model_specs['prior_means'], 
    prior_sds=model_specs['prior_sds'],
    tmax=model_specs['tmax'],
    # contamination_probability=.05,
    num_obs=model_specs['num_obs']
)

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .sqrt("num_obs")
    .concatenate(["A", "tau", "mu_c", "mu_r", "b"], into="inference_variables")
    .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
    .standardize(include="inference_variables")
    .rename("num_obs", "inference_conditions")
)
# Create inference net 
inference_net = bf.networks.CouplingFlow(coupling_kwargs=dict(subnet_kwargs=dict(dropout=0.1)))

# inference_net = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))

summary_net = bf.networks.SetTransformer(summary_dim=32, num_seeds=2, dropout=0.1)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=1e-4,
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath='../data/training_checkpoints',
    checkpoint_name= "optuna_run1",
    inference_variables=["A", "tau", "mu_c", "mu_r", "b"]
)


def subset_data(data, num_obs):

    data = data.copy()

    keys = ['rt', 'accuracy', 'conditions']

    max_obs = data['rt'].shape[1]

    random_idx = np.random.choice(np.arange(0, max_obs), size=num_obs, replace=False)

    for k in keys:
        # print(f'{data[k].shape}')
        data[k] = data[k][:, random_idx, :]
        # print(f'{data[k].shape}')


    data['num_obs'] = np.array([num_obs]*1000).reshape(1000,1)

    return data

val_data = simulator.sample(1000)

# val_data['t0'] = val_data.pop('mu_r')

# val_data['rt'].shape[1]


import pandas as pd

list_metrics = []

step_size = 50

num_max_obs = 500


for n_obs in np.arange(50, num_max_obs, step_size):
    
    print(f'num_obs: {n_obs}')
    # simulator.num_obs = n_obs

    data_subset = subset_data(val_data, num_obs=n_obs)

    samples = approximator.sample(conditions=data_subset, num_samples=1000)
    
    # metrics_table=workflow.compute_default_diagnostics(test_data=data_subset)
    results_single = pd.concat([pd.DataFrame(bf.diagnostics.metrics.calibration_error(samples, data_subset)),
                                 pd.DataFrame(bf.diagnostics.metrics.posterior_contraction(samples, data_subset)),
                                 pd.DataFrame(bf.diagnostics.metrics.root_mean_squared_error(samples, data_subset))])
    
    
    results_single["num_obs"] = n_obs
    
    list_metrics.append(results_single)
    
data_set_metrics = pd.concat(list_metrics)


import matplotlib.pyplot as plt
import seaborn as sns

param_names =  ["A", "tau", "mu_c", "t0", "b"]

fig, axes = plt.subplots(1,5,sharey=True, figsize=(15,3))

for p, ax in zip(param_names, axes):
    
    suff = "$\\" if p in ["tau", "mu_c", "mu_r"] else "$"

    label = suff + p + "$"
    
    sns.lineplot(data_set_metrics[data_set_metrics["variable_names"] == p], x="num_obs", y="values", hue="metric_name", ax=ax, palette="colorblind")
    ax.set_title(label)
    ax.legend(title="")
    if p != "b":
        ax.get_legend().remove()

    plt.ylim(0, 1)

fig.tight_layout()
