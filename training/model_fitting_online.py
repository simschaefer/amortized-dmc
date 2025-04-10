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

os.getcwd()

#########
network_name = "testrun"
#########

epochs = 1
num_batches_per_epoch = 100


model_specs = {"simulation_settings": {"prior_means": np.array([16., 111., 0.5, 322., 75.]),
                                       "prior_sds": np.array([10., 47., 0.13, 40., 23.]),
                                       "tmax": 1500,
                                       "contamination_probability": None,
                                       "min_num_obs": 50,
                                       "max_num_obs": 800,
                                       "fixed_num_obs": None},
"inference_network_settings": {"coupling_kwargs": {"subnet_kwargs": {"dropout":0.011529815885353391}}, "depth":7},
"summary_network_settings": {"dropout": 0.011529815885353391,
                             "num_seeds": 2,
                             "summary_dim": 32,
                             "embed_dim": (128, 128)},
                             'batch_size': 16,
                             'learning_rate': 0.00083,
                             'param_names': ["A", "tau", "mu_c", "mu_r", "b"]}


file_path = 'model_specs/model_specs_' + network_name + '.pickle'

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

simulator = DMC(**model_specs['simulation_settings'])

inference_net = bf.networks.CouplingFlow(**model_specs['inference_network_settings'])

# inference_net = bf.networks.FlowMatching(subnet_kwargs=dict(dropout=0.1))

summary_net = bf.networks.SetTransformer(**model_specs['summary_network_settings'])

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    initial_learning_rate=model_specs["learning_rate"],
    inference_network=inference_net,
    summary_network=summary_net,
    checkpoint_filepath='checkpoints',
    checkpoint_name=network_name,
    inference_variables=model_specs["param_names"]
)

# file_path = '../data/data_offline_training/data_offline_training_' + network_name + '.pickle'

# train_data = simulator.sample(50000)

# with open(file_path, 'wb') as file:
#     pickle.dump(train_data, file)

# with open(file_path, 'rb') as file:
#     train_data = pickle.load(file)
    

val_file_path = 'data/data_offline_training/data_offline_validation_' + network_name + '.pickle'
    
val_data = simulator.sample(200)

with open(val_file_path, 'wb') as file:
    pickle.dump(val_data, file)

# with open(val_file_path, 'rb') as file:
#     val_data = pickle.load(file)


_ = adapter(val_data, strict=True, stage="inference")


history = workflow.fit_online(epochs=epochs, num_batches_per_epoch=num_batches_per_epoch, batch_size=model_specs["batch_size"], validation_data=val_data)

# approximator = keras.saving.load_model("../checkpoints/" + network_name)
