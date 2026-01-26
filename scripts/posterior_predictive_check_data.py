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
import time 

import bayesflow as bf
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


num_resims = 100

network_names = [
    'updated_priors_sdr_fixed',
    'updated_priors_sdr_estimated',
    'initial_priors_sdr_fixed',
    'initial_priors_sdr_estimated',
]

host = 'local'

parent_dir = os.path.dirname(os.getcwd())


from dmc import DMC, dmc_helpers


narrow_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_narrow.csv')

wide_data = pd.read_csv(parent_dir + '/empirical_data/experiment_data_wide.csv')

empirical_data = pd.concat((narrow_data, wide_data))


lst_data = [] 

for network_name in network_names:

    model_specs_path = parent_dir + '/model_specs/model_specs_' + network_name + '.pickle'
    with open(model_specs_path, 'rb') as file:
        model_specs = pickle.load(file)

    simulator = DMC(**model_specs['simulation_settings'])

    df_samples = pd.read_csv(parent_dir + '/data_complete/empirical_estimates/' + network_name + '.csv')

    parts = df_samples['participant'].unique()

    for spacing, spacing_num in zip(['narrow', 'wide'], [1,0]):
        
        for part in parts:
            
            num_obs = empirical_data[(empirical_data['spacing_num'] == spacing_num) & (empirical_data['participant'] == part)].shape[0]

            # filter sample data for given participant and narrow spacing
            part_data_samples = df_samples[df_samples["participant"]==part]

            part_data_samples = part_data_samples[part_data_samples["spacing"] == spacing]

            # resimulate data
            data_resimulated = dmc_helpers.resim_data(part_data_samples, num_obs=num_obs, simulator=simulator, part=part, param_names=model_specs['simulation_settings']['param_names'], id_name='participant')
            
            # exclude non-convergents
            data_resimulated = data_resimulated[data_resimulated["rt"] != -1]

            # recode congruency
            data_resimulated["condition_label"] = data_resimulated["conditions"].map({0.0: "congruent", 1.0: "incongruent"})

            data_resimulated['model'] = network_name

            data_resimulated['spacing'] = spacing

            lst_data.append(data_resimulated)


df_complete = pd.concat(lst_data)

data_path = parent_dir + '/data_complete/ppc_data/'

if not os.path.exists(data_path):
      os.makedirs(data_path)

df_complete.to_csv(data_path + '/ppc_data_raw.csv')
