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
from dmc import DMC


#########
network_name = "contamination500trials"
contamination_probability = .05
#########


simulator = DMC(
    prior_means=np.array([16., 111., 0.5, 322., 75.]), 
    prior_sds=np.array([10., 47., 0.13, 40., 23.]),
    tmax=1500,
    contamination_probability=contamination_probability
)

file_path = '../simulators/simulator_' + network_name + '.pickle'

with open(file_path, 'rb') as file:
    pickle.dump(simulator, file)
    
    

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .sqrt("num_obs")
    .concatenate(["A", "tau", "mu_c", "mu_r", "b"], into="inference_variables")
    .concatenate(["rt", "accuracy", "conditions"], into="summary_variables")
    .standardize(include="inference_variables")
    .rename("num_obs", "inference_conditions")
)

inference_net = bf.networks.CouplingFlow(coupling_kwargs=dict(subnet_kwargs=dict(dropout=0.011529)), depth=7)

# inference_net = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))

summary_net = bf.networks.SetTransformer(summary_dim=32, num_seeds=2, dropout=0.015278248983667964)

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=0.0009994591668317414,
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath='../checkpoints',
    checkpoint_name= network_name,
    inference_variables=["A", "tau", "mu_c", "mu_r", "b"]
)

file_path = '../data/data_offline_training/data_offline_training_' + network_name + '.pickle'

# train_data = simulator.sample(50000)

# with open(file_path, 'wb') as file:
#     pickle.dump(train_data, file)

with open(file_path, 'rb') as file:
    train_data = pickle.load(file)
    

val_file_path = '../data/data_offline_training/data_offline_validation_' + network_name + '.pickle'
    
# val_data = simulator.sample(1000)

# with open(val_file_path, 'wb') as file:
#     pickle.dump(val_data, file)

with open(val_file_path, 'rb') as file:
    val_data = pickle.load(file)
    
    
%%time
history = workflow.fit_offline(train_data, epochs=100, batch_size=32, validation_data=val_data)


approximator = keras.saving.load_model("../checkpoints/" + network_name)

figs = workflow.plot_default_diagnostics(test_data=val_data, calibration_ecdf_kwargs=dict(difference=True))