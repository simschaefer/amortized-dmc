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

from dmc import DMC, dmc_helpers

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D

network_names = [
    'updated_priors_sdr_fixed',
    'updated_priors_sdr_estimated',
    'initial_priors_sdr_fixed',
    'initial_priors_sdr_estimated',
]

host = 'local'

parent_dir = os.path.dirname(os.getcwd())

# Load Empirical Data and double check participants

narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')

narrow_data = narrow_data[narrow_data['participant'].isin(included_parts)]

wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

wide_data = wide_data[wide_data['participant'].isin(included_parts)]


for network_name in network_names:

    # load specific approximator
    approximator = keras.saving.load_model(parent_dir + "/training_checkpoints/" + network_name + '.keras')

    # apply approximator on narrow data
    samples_narrow = dmc_helpers.fit_empirical_data(narrow_data, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

    samples_narrow["spacing"]="narrow"

    # apply approximator on wide data
    samples_wide = dmc_helpers.fit_empirical_data(wide_data, approximator, id_name='participant', rt='rt', accuracy='accuracy', congruency='congruency_num')

    samples_wide["spacing"]="wide"

    samples_complete = pd.concat((samples_wide, samples_narrow))

    data_path = parent_dir + '/data_complete/empirical_estimates/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save samples
    #samples_complete.to_csv(data_path + network_name + '.csv')

    print(network_name, 'estimation completed')





