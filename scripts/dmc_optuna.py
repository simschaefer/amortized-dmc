import sys
sys.path.append("../../BayesFlow")
sys.path.append("../")

import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

import torch 

print("CUDA available:", torch.cuda.is_available(), flush=True)
print(torch.cuda.device_count(), flush=True)

# torch.cuda.device_count.cache_clear()


print("Using device:", torch.cuda.get_device_name(0))


if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pickle

import keras

import optuna


import bayesflow as bf

parent_dir = os.getcwd()
dmc_module_dir = parent_dir + '/bf_dmc/dmc'

print(f'parent_dir: {parent_dir}', flush=True)
print(f'dmc_module_dir: {dmc_module_dir}')

sys.path.append(dmc_module_dir)

from dmc import DMC, weighted_metric_sum



network_name = "oos500trials_noco"
n_trials = 20
n_epochs = 50


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

file_path = parent_dir + '/bf_dmc/model_specs/model_specs_' + network_name + '.pickle'

with open(file_path, 'wb') as file:
    pickle.dump(model_specs, file)


adapter = (
bf.adapters.Adapter()
.convert_dtype("float64", "float32")
.sqrt("num_obs")
.concatenate(["A", "tau", "mu_c", "mu_r", "b"], into="inference_variables")
.concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
.standardize(include="inference_variables")
.rename("num_obs", "inference_conditions")
)


training_file_path = parent_dir + '/bf_dmc/data/data_offline_training/data_offline_training_' + network_name + '.pickle'

# train_data = simulator.sample(50000)

# with open(file_path, 'wb') as file:
#     pickle.dump(train_data, file)

with open(training_file_path, 'rb') as file:
    train_data = pickle.load(file)


# val_data = simulator.sample(1000)

val_file_path = parent_dir + '/bf_dmc/data/data_offline_training/data_offline_training_' + network_name + '_validation.pickle'

# with open(val_file_path, 'wb') as file:
#     pickle.dump(val_data, file)

with open(val_file_path, 'rb') as file:
    val_data = pickle.load(file)


def objective(trial, epochs=n_epochs):

    # Optimize hyperparameters
    dropout = trial.suggest_float("dropout", 0.01, 0.3)
    initial_learning_rate = trial.suggest_float("lr", 1e-4, 1e-3) 
    num_seeds = trial.suggest_int("num_seeds", 1, 8)
    depth = trial.suggest_int("depth", 5, 12)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    embed_dim=trial.suggest_categorical("embed_dim", [64, 128])

    # Create inference net 
    inference_net = bf.networks.CouplingFlow(coupling_kwargs=dict(subnet_kwargs=dict(dropout=dropout)), depth=depth)

    summary_net = bf.networks.SetTransformer(summary_dim=32, num_seeds=num_seeds, dropout=dropout, embed_dim=(embed_dim, embed_dim))
    
    
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        initial_learning_rate=initial_learning_rate,
        inference_network=inference_net,
        summary_network=summary_net,
        checkpoint_filepath= parent_dir + '/bf_dmc/data/optuna_checkpoints',
        checkpoint_name= f'network_{round(dropout, 2)}_{round(initial_learning_rate, 2)}_{num_seeds}_{depth}_{batch_size}_{embed_dim}',
        inference_variables=["A", "tau", "mu_c", "mu_r", "b"])
    
    history = workflow.fit_offline(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, verbose=0)
    
    metrics_table=workflow.compute_default_diagnostics(test_data=val_data)

    # compute weighted sum
    weighted_sum = weighted_metric_sum(metrics_table)
    
    # loss=np.mean(history.history["val_loss"][-5:])
        
    return weighted_sum

study = optuna.create_study(direction="minimize")

with open(parent_dir + '/bf_dmc/optuna_results/' + network_name + '_optuna_results.pickle', 'wb') as file:
    pickle.dump(study, file)

study.enqueue_trial({"dropout": 0.0100967297,
                     "lr": 0.0004916,
                     "num_seeds": 2,
                     "batch_size": 16,
                     "embed_dim": 128})

study.optimize(objective, n_trials=n_trials)

trial = study.best_trial
print("Outcome Metric: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

with open(parent_dir + '/bf_dmc/optuna_results/' + network_name + '_optuna_results.pickle', 'wb') as file:
    pickle.dump(study, file)

print(f'Study saved successfully.')

with open(parent_dir + '/bf_dmc/optuna_results/' + network_name + '_optuna_results.pickle', 'rb') as file:
    study_reloaded = pickle.load(file)

print(f'Study loaded successfully.')

trial = study_reloaded.best_trial
print("Outcome Metric: {}".format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
