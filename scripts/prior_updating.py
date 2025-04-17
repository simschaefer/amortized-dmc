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


parent_dir = '/home/administrator/Documents/bf_dmc'


import bayesflow as bf
from dmc import DMC, dmc_helpers

import pandas as pd



parent_dir = '/home/administrator/Documents/bf_dmc'

network_name = 'dmc_optimized_winsim_priors'

model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
with open(model_specs_path, 'rb') as file:
    model_specs = pickle.load(file)

simulator, adapter, inference_net, summary_net, workflow = dmc_helpers.load_model_specs(model_specs, network_name)
## Load Approximator

approximator = keras.saving.load_model(parent_dir +"/data/training_checkpoints/" + network_name + ".keras")
approximator.compile()


narrow_data = pd.read_csv(parent_dir + '/data/empirical_data/experiment_data_narrow.csv')
wide_data = pd.read_csv(parent_dir +'/data/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat([narrow_data, wide_data])


empirical_samples_narrow = dmc_helpers.fit_empirical_data(narrow_data, approximator)

empirical_samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator)

empirical_samples_complete = dmc_helpers.fit_empirical_data(empirical_data, approximator)

empirical_samples_narrow.agg(['mean', 'std'])

empirical_samples_wide.agg(['mean', 'std'])

empirical_samples_complete.agg(['mean', 'std'])

