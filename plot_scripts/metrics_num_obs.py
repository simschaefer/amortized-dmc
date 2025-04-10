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
from dmc import DMC


network_name = 'testrun'

file_path = 'model_specs/model_specs_' + network_name + '.pickle'

with open(file_path, 'wb') as file:
    pickle.load(file)

model_specs = {"prior_means": np.array([16., 111., 0.5, 322., 75.]),
                       "prior_sds": np.array([10., 47., 0.13, 40., 23.]),
                        "tmax": 1500,
                        "contamination_probability": None}

simulator = DMC(
    prior_means=model_specs["prior_means"], 
    prior_sds=model_specs["prior_sds"],
    tmax=model_specs["tmax"],
    contamination_probability=model_specs["contamination_probability"]
)

approximator = keras.saving.load_model("/data/training_checkpoints/" + network_name + ".keras")